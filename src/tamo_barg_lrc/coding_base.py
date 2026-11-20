"""Lightweight MIT-licensed linear coding-theory base classes (zero SageMath dependency).

Provides drop-in native replacements for SageMath's ``AbstractLinearCode``,
``Encoder``, ``Decoder``, ``DecodingError``, ``cached_method``, and
``cartesian_product`` / ``CartesianProductSpace``.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ._ff.numpy_backend import FFMatrix, FFVector, FiniteField, VectorSpace
from ._ff.protocol import DecodingError


def cached_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Lightweight per-instance method memoizer compatible with Sage's ``@cached_method``."""
    cache_attr = f"__cached_method_{fn.__name__}__"

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        cache: dict[Any, Any] | None = self.__dict__.get(cache_attr)
        if cache is None:
            cache = {}
            self.__dict__[cache_attr] = cache
        key = (args, tuple(sorted(kwargs.items()))) if kwargs else args
        if key not in cache:
            cache[key] = fn(self, *args, **kwargs)
        return cache[key]

    return wrapper


class CartesianProductSpace:
    """Thin Cartesian product of ambient spaces (e.g. ``GF(q)^n x GF(2)^n``)."""

    def __init__(self, spaces: Sequence[Any]) -> None:
        self._spaces: tuple[Any, ...] = tuple(spaces)

    def cartesian_factors(self) -> tuple[Any, ...]:
        return self._spaces

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, tuple) or len(item) != len(self._spaces):
            return False
        return all(x in sp for x, sp in zip(item, self._spaces, strict=True))

    def __call__(self, item: Any) -> tuple[Any, ...]:
        if not isinstance(item, (tuple, list)) or len(item) != len(self._spaces):
            raise TypeError(f"Expected a tuple of length {len(self._spaces)}, got {item!r}")
        return tuple(sp(x) for sp, x in zip(self._spaces, item, strict=True))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CartesianProductSpace) and self._spaces == other._spaces

    def __hash__(self) -> int:
        return hash(self._spaces)

    def __repr__(self) -> str:
        inner = " x ".join(repr(s) for s in self._spaces)
        return f"The cartesian product of ({inner})"


def cartesian_product(spaces: Sequence[Any]) -> CartesianProductSpace:
    """Construct a ``CartesianProductSpace`` from a sequence of spaces."""
    return CartesianProductSpace(spaces)


class Encoder:
    """Base class for linear code encoders."""

    def __init__(self, code: AbstractLinearCode) -> None:
        self._code = code
        self._unencode_cache: tuple[list[int], FFMatrix] | None = None

    def code(self) -> AbstractLinearCode:
        return self._code

    def message_space(self) -> VectorSpace:
        return VectorSpace(self._code.base_field(), self._code.dimension())

    def generator_matrix(self) -> FFMatrix:
        raise NotImplementedError("Subclasses must implement generator_matrix()")

    def encode(self, word: FFVector | Sequence[Any]) -> FFVector:
        msg_space = self.message_space()
        w = word if isinstance(word, FFVector) else msg_space(word)
        if w not in msg_space:
            raise ValueError(f"Word {word!r} is not in message space {msg_space!r}")
        G = self.generator_matrix()
        return w * G

    def _get_unencode_data(self) -> tuple[list[int], FFMatrix]:
        if self._unencode_cache is None:
            G = self.generator_matrix()
            pivs = list(G.pivots())
            if len(pivs) != G.nrows():
                raise ValueError("Generator matrix does not have full row rank")
            G_piv = G.matrix_from_columns(pivs)
            G_piv_inv = G_piv.inverse()
            self._unencode_cache = (pivs, G_piv_inv)
        return self._unencode_cache

    def unencode_nocheck(self, c: FFVector | Sequence[Any]) -> FFVector:
        amb = self._code.ambient_space()
        v = c if isinstance(c, FFVector) else amb(c)
        if v not in amb:
            raise ValueError(f"Vector is not in ambient space {amb!r}")
        pivs, G_piv_inv = self._get_unencode_data()
        f = self._code.base_field()
        c_piv = FFVector(f, np.ascontiguousarray(v._arr[pivs], dtype=f._dtype))
        return c_piv * G_piv_inv

    def unencode(self, c: FFVector | Sequence[Any], nocheck: bool = False) -> FFVector:
        amb = self._code.ambient_space()
        v = c if isinstance(c, FFVector) else amb(c)
        if not nocheck and v not in self._code:
            raise ValueError("The vector is not a codeword of this code")
        return self.unencode_nocheck(v)

    def __repr__(self) -> str:
        if hasattr(self, "_repr_"):
            return str(self._repr_())
        return f"{type(self).__name__} for {self._code!r}"


class Decoder:
    """Base class for linear code decoders."""

    _decoder_type: set[str] = set()

    def __init__(
        self,
        code: AbstractLinearCode,
        input_space: Any,
        connected_encoder_name: str,
    ) -> None:
        self._code = code
        self._input_space = input_space
        self._connected_encoder_name = connected_encoder_name

    def code(self) -> AbstractLinearCode:
        return self._code

    def input_space(self) -> Any:
        return self._input_space

    def connected_encoder(self) -> Encoder:
        return self._code.encoder(self._connected_encoder_name)

    def decoder_type(self) -> set[str]:
        return set(self._decoder_type)

    def decode_to_code(self, r: Any) -> Any:
        msg = self.decode_to_message(r)
        return self.connected_encoder().encode(msg)

    def decode_to_message(self, r: Any) -> Any:
        c = self.decode_to_code(r)
        return self.connected_encoder().unencode_nocheck(c)

    def __repr__(self) -> str:
        if hasattr(self, "_repr_"):
            return str(self._repr_())
        return f"{type(self).__name__} for {self._code!r}"


class AbstractLinearCode:
    """Lightweight base class for linear codes over ``GF(q)``."""

    _registered_encoders: dict[str, type[Encoder]] = {}
    _registered_decoders: dict[str, type[Decoder]] = {}

    def __init__(
        self,
        base_field: FiniteField,
        length: int,
        default_encoder_name: str,
        default_decoder_name: str,
    ) -> None:
        if not isinstance(base_field, FiniteField):
            raise TypeError(f"base_field must be a FiniteField, got {type(base_field).__name__}")
        n = int(length)
        if n <= 0:
            raise ValueError(f"Code length must be positive, got {n}")
        self._base_field = base_field
        self._length = n
        self._default_encoder_name = default_encoder_name
        self._default_decoder_name = default_decoder_name
        if "_registered_encoders" not in self.__dict__:
            self._registered_encoders = dict(type(self)._registered_encoders)
        if "_registered_decoders" not in self.__dict__:
            self._registered_decoders = dict(type(self)._registered_decoders)
        self._encoder_cache: dict[str, Encoder] = {}
        self._decoder_cache: dict[str, Decoder] = {}
        self._ambient_space = VectorSpace(base_field, n)
        self._pcm_cache: FFMatrix | None = None

    def length(self) -> int:
        return self._length

    def dimension(self) -> int:
        return self.generator_matrix().nrows()

    def base_field(self) -> FiniteField:
        return self._base_field

    def ambient_space(self) -> VectorSpace:
        return self._ambient_space

    def encoders_available(self) -> list[str]:
        return sorted(self._registered_encoders.keys())

    def decoders_available(self) -> list[str]:
        return sorted(self._registered_decoders.keys())

    def encoder(self, encoder_name: str | None = None, *args: Any, **kwargs: Any) -> Encoder:
        name = self._default_encoder_name if encoder_name is None else encoder_name
        if name not in self._registered_encoders:
            raise ValueError(
                f"There is no encoder named {name!r}. "
                f"Available encoders: {self.encoders_available()}"
            )
        if not args and not kwargs:
            if name not in self._encoder_cache:
                self._encoder_cache[name] = self._registered_encoders[name](self)
            return self._encoder_cache[name]
        return self._registered_encoders[name](self, *args, **kwargs)

    def decoder(self, decoder_name: str | None = None, *args: Any, **kwargs: Any) -> Decoder:
        name = self._default_decoder_name if decoder_name is None else decoder_name
        if name not in self._registered_decoders:
            raise ValueError(
                f"There is no decoder named {name!r}. "
                f"Available decoders: {self.decoders_available()}"
            )
        if not args and not kwargs:
            if name not in self._decoder_cache:
                self._decoder_cache[name] = self._registered_decoders[name](self)
            return self._decoder_cache[name]
        return self._registered_decoders[name](self, *args, **kwargs)

    def generator_matrix(self, encoder_name: str | None = None) -> FFMatrix:
        return self.encoder(encoder_name).generator_matrix()

    def parity_check_matrix(self) -> FFMatrix:
        if self._pcm_cache is None:
            G = self.generator_matrix()
            H = G.right_kernel_matrix()
            H.set_immutable()
            self._pcm_cache = H
        return self._pcm_cache

    def encode(self, word: FFVector | Sequence[Any], encoder_name: str | None = None) -> FFVector:
        return self.encoder(encoder_name).encode(word)

    def unencode(
        self,
        c: FFVector | Sequence[Any],
        encoder_name: str | None = None,
        nocheck: bool = False,
    ) -> FFVector:
        return self.encoder(encoder_name).unencode(c, nocheck=nocheck)

    def decode_to_code(
        self,
        r: Any,
        decoder_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.decoder(decoder_name, *args, **kwargs).decode_to_code(r)

    def decode_to_message(
        self,
        r: Any,
        decoder_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.decoder(decoder_name, *args, **kwargs).decode_to_message(r)

    def __contains__(self, c: object) -> bool:
        if not isinstance(c, FFVector) or c not in self._ambient_space:
            return False
        H = self.parity_check_matrix()
        if H.nrows() == 0:
            return True
        return (H * c).is_zero()

    def weight_distribution(self) -> list[int]:
        """Exact weight distribution ``[A_0, A_1, ..., A_n]`` via projective enumeration."""
        f = self._base_field
        n = self._length
        counts = [0] * (n + 1)
        counts[0] = 1
        G_rref = self.generator_matrix().echelon_form()
        rows = [row._arr for row in G_rref.rows() if not row.is_zero()]
        k = len(rows)
        if k == 0:
            return counts
        q = f.order()
        qm1 = q - 1
        G_mat = np.vstack(rows)
        # Projective enumeration: fix first non-zero coordinate to 1, scale count by q - 1
        for i in range(k):
            base_row = G_mat[i]
            tail_dim = k - 1 - i
            if tail_dim == 0:
                wt = int(np.count_nonzero(base_row))
                counts[wt] += qm1
            else:
                tail_mat = np.ascontiguousarray(G_mat[i + 1 :].T)
                for coeffs in itertools.product(range(q), repeat=tail_dim):
                    c_arr = np.array(coeffs, dtype=f._dtype)
                    cw = f.add_arr(base_row, f.matvec_arr(tail_mat, c_arr))
                    wt = int(np.count_nonzero(cw))
                    counts[wt] += qm1
        return counts

    def minimum_distance(self) -> int:
        """Exact minimum Hamming distance of the code."""
        dist = self.weight_distribution()
        for w in range(1, len(dist)):
            if dist[w] > 0:
                return w
        return self._length + 1

    def __repr__(self) -> str:
        if hasattr(self, "_repr_"):
            return str(self._repr_())
        return f"[{self._length}, {self.dimension()}] linear code over {self._base_field!r}"


__all__ = [
    "AbstractLinearCode",
    "CartesianProductSpace",
    "Decoder",
    "DecodingError",
    "Encoder",
    "cached_method",
    "cartesian_product",
]
