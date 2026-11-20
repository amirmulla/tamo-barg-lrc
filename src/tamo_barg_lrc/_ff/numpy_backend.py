"""Custom MIT-licensed NumPy finite-field, linear-algebra, polynomial, and RS-decoding backend.

Zero SageMath dependencies. Supports:
- ``GF(2)`` and ``GF(2^m)`` (branchless log/antilog LUT up to ``2^16``, polynomial fallback above)
- ``GF(p)`` for arbitrary prime ``p`` (LUT up to ``65536``, modular int64/Python-int fallback above)
- ``GF(p^m)`` for odd prime powers (base-``p`` integer encoding matching SageMath's
  ``to_integer()`` / ``fetch_int()``, LUT + Zech log up to ``65536``, polynomial fallback above)

Automatically dispatches ``GF(2^m)`` (``q <= 65536``) hot kernels to the optional
Cython extension ``_ff_cy`` when compiled and ``TAMO_BARG_NO_CYTHON != "1"``.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from .conway import (
    divisors,
    factor_prime_power,
    find_primitive_root_mod_p,
    get_conway_polynomial,
)
from .protocol import DecodingError

# Optional Cython accelerator (disabled when TAMO_BARG_NO_CYTHON=1)
_ff_cy: Any = None
if os.environ.get("TAMO_BARG_NO_CYTHON") != "1":
    try:
        from . import _ff_cy as _ff_cy_mod

        _ff_cy = _ff_cy_mod
    except ImportError:
        _ff_cy = None

HAS_CYTHON: bool = _ff_cy is not None

# Module-level default RNG (seeded via set_random_seed)
_DEFAULT_RNG: np.random.Generator = np.random.default_rng()


def set_random_seed(seed: int | None = None) -> np.random.Generator:
    """Seed the module-level finite-field RNG and return it."""
    global _DEFAULT_RNG
    _DEFAULT_RNG = np.random.default_rng(seed)
    return _DEFAULT_RNG


def get_rng() -> np.random.Generator:
    """Return the current module-level finite-field RNG."""
    return _DEFAULT_RNG


# ---------------------------------------------------------------------------
# Large-field polynomial arithmetic fallbacks (q > 65536)
# ---------------------------------------------------------------------------


def _gf2_mul_mod(a: int, b: int, mod: int, m: int) -> int:
    """Carry-less multiplication of ``a`` and ``b`` modulo ``mod`` of degree ``m`` in GF(2^m)."""
    res = 0
    high_bit = 1 << m
    aa = a
    bb = b
    while bb:
        if bb & 1:
            res ^= aa
        aa <<= 1
        if aa & high_bit:
            aa ^= mod
        bb >>= 1
    return res


def _gf2_inv_mod(a: int, mod: int) -> int:
    """Extended binary Euclidean inverse of ``a`` modulo ``mod`` in GF(2)[x]."""
    if a == 0:
        raise ZeroDivisionError("Inverse of zero in finite field")
    r0, r1 = mod, a
    t0, t1 = 0, 1
    while r1 != 0:
        d0 = r0.bit_length() - 1
        d1 = r1.bit_length() - 1
        if d0 < d1:
            r0, r1 = r1, r0
            t0, t1 = t1, t0
            d0, d1 = d1, d0
        shift = d0 - d1
        r0 ^= r1 << shift
        t0 ^= t1 << shift
    return t0


def _int_to_base_p(val: int, p: int, m: int) -> list[int]:
    digits = [0] * m
    rem = val
    for i in range(m):
        digits[i] = rem % p
        rem //= p
    return digits


def _base_p_to_int(digits: Sequence[int], p: int) -> int:
    res = 0
    pw = 1
    for d in digits:
        res += (d % p) * pw
        pw *= p
    return res


def _fp_poly_trim(a: list[int]) -> list[int]:
    i = len(a) - 1
    while i >= 0 and a[i] == 0:
        i -= 1
    return a[: i + 1]


def _fp_poly_divmod(a: list[int], b: list[int], p: int) -> tuple[list[int], list[int]]:
    a = _fp_poly_trim(a)
    b = _fp_poly_trim(b)
    if not b:
        raise ZeroDivisionError("Polynomial division by zero")
    da, db = len(a) - 1, len(b) - 1
    if da < db:
        return [], list(a)
    rem = list(a)
    q = [0] * (da - db + 1)
    inv_lc = pow(b[db], -1, p)
    for shift in range(da - db, -1, -1):
        c = rem[shift + db]
        if c:
            fac = (c * inv_lc) % p
            q[shift] = fac
            for j in range(db + 1):
                rem[shift + j] = (rem[shift + j] - fac * b[j]) % p
    return _fp_poly_trim(q), _fp_poly_trim(rem)


def _fp_poly_mul_mod(a: list[int], b: list[int], mod: tuple[int, ...], p: int) -> list[int]:
    m = len(mod) - 1
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    res[i + j] = (res[i + j] + ai * bj) % p
    for deg in range(len(res) - 1, m - 1, -1):
        c = res[deg]
        if c:
            for j in range(m):
                res[deg - m + j] = (res[deg - m + j] - c * mod[j]) % p
            res[deg] = 0
    return res[:m]


def _fp_poly_inv_mod(a: list[int], mod: tuple[int, ...], p: int) -> list[int]:
    m = len(mod) - 1
    r0 = list(mod)
    r1 = _fp_poly_trim(a)
    if not r1:
        raise ZeroDivisionError("Inverse of zero in finite field")
    t0: list[int] = []
    t1: list[int] = [1]
    while r1:
        q, r2 = _fp_poly_divmod(r0, r1, p)
        # t2 = t0 - q * t1
        qt = [0] * (max(len(q) + len(t1) - 1, 0))
        for i, qi in enumerate(q):
            if qi:
                for j, tj in enumerate(t1):
                    if tj:
                        qt[i + j] = (qt[i + j] + qi * tj) % p
        n_len = max(len(t0), len(qt))
        t2 = [0] * n_len
        for i in range(len(t0)):
            t2[i] = t0[i]
        for i in range(len(qt)):
            t2[i] = (t2[i] - qt[i]) % p
        t2 = _fp_poly_trim(t2)
        r0, r1 = r1, r2
        t0, t1 = t1, t2
    inv_c = pow(r0[0], -1, p)
    out = [(c * inv_c) % p for c in t0]
    if len(out) < m:
        out.extend([0] * (m - len(out)))
    return out[:m]


# ---------------------------------------------------------------------------
# L1: FiniteField & FiniteFieldElement
# ---------------------------------------------------------------------------


class FiniteFieldElement:
    """Immutable element of ``GF(p^m)`` encoded as an integer in ``[0, q - 1]``."""

    __slots__ = ("_field", "_val")

    def __init__(self, field: FiniteField, int_val: int) -> None:
        self._field = field
        self._val = int(int_val)

    def parent(self) -> FiniteField:
        return self._field

    def is_zero(self) -> bool:
        return self._val == 0

    def to_integer(self) -> int:
        """Deterministic base-``p`` integer encoding matching SageMath's ``to_integer()``."""
        return self._val

    def to_int(self) -> int:
        return self._val

    def __int__(self) -> int:
        return self._val

    def __bool__(self) -> bool:
        return self._val != 0

    def _coerce(self, other: Any) -> int | None:
        if isinstance(other, FiniteFieldElement):
            if other._field == self._field:
                return other._val
            if other._field._is_prime and other._field._p == self._field._p:
                return other._val
            return None
        if isinstance(other, (int, np.integer)) and not isinstance(other, bool):
            return int(other) % self._field._p
        return None

    def __add__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.add_s(self._val, ov))

    def __radd__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.add_s(ov, self._val))

    def __sub__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.sub_s(self._val, ov))

    def __rsub__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.sub_s(ov, self._val))

    def __neg__(self) -> FiniteFieldElement:
        return FiniteFieldElement(self._field, self._field.neg_s(self._val))

    def __pos__(self) -> FiniteFieldElement:
        return self

    def __mul__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.mul_s(self._val, ov))

    def __rmul__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.mul_s(ov, self._val))

    def __truediv__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.div_s(self._val, ov))

    def __rtruediv__(self, other: Any) -> FiniteFieldElement:
        ov = self._coerce(other)
        if ov is None:
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.div_s(ov, self._val))

    def __pow__(self, exp: Any) -> FiniteFieldElement:
        if not isinstance(exp, (int, np.integer)) or isinstance(exp, bool):
            return NotImplemented
        return FiniteFieldElement(self._field, self._field.pow_s(self._val, int(exp)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FiniteFieldElement):
            if self._field == other._field:
                return self._val == other._val
            if other._field._is_prime and other._field._p == self._field._p:
                return self._val == other._val
            if self._field._is_prime and self._field._p == other._field._p:
                return self._val == other._val
            return False
        if isinstance(other, (int, np.integer)) and not isinstance(other, bool):
            return self._val == (int(other) % self._field._p)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        if self._val < self._field._p:
            return hash(self._val)
        return hash((self._field._order, self._val))

    def __repr__(self) -> str:
        f = self._field
        if f._is_prime or f._repr_mode == "int":
            return str(self._val)
        if self._val == 0:
            return "0"
        digits = _int_to_base_p(self._val, f._p, f._m)
        name = f._name
        terms: list[str] = []
        for deg in range(f._m - 1, -1, -1):
            c = digits[deg]
            if c == 0:
                continue
            if deg == 0:
                terms.append(str(c))
            elif deg == 1:
                terms.append(name if c == 1 else f"{c}*{name}")
            else:
                terms.append(f"{name}^{deg}" if c == 1 else f"{c}*{name}^{deg}")
        return " + ".join(terms) if terms else "0"


class FiniteField:
    """Finite field ``GF(q)``, ``q = p^m >= 2`` with LUT acceleration up to ``q <= 65536``."""

    def __init__(self, order: int, name: str = "a", repr_mode: str = "int") -> None:
        p, m = factor_prime_power(order)
        self._order: int = int(order)
        self._p: int = p
        self._m: int = m
        self._name: str = str(name)
        self._repr_mode: str = str(repr_mode)
        self._is_binary: bool = p == 2
        self._is_prime: bool = m == 1
        self._use_lut: bool = self._order <= 65536
        self._dtype: Any = (
            np.uint16 if self._use_lut else (np.int64 if self._order < (1 << 62) else object)
        )
        self._can_use_cy: bool = bool(self._is_binary and self._use_lut and _ff_cy is not None)

        self._modulus_coeffs: tuple[int, ...] = ()
        self._modulus_int: int = 0

        if self._is_prime:
            self._gen_int = 1
            self._prim_int = find_primitive_root_mod_p(self._p)
        else:
            self._modulus_coeffs = get_conway_polynomial(self._p, self._m)
            self._gen_int = self._p
            self._prim_int = self._p
            if self._is_binary:
                mod_bits = 0
                for i, c in enumerate(self._modulus_coeffs):
                    if c:
                        mod_bits |= 1 << i
                self._modulus_int = mod_bits

        self.exp_table: np.ndarray | None = None
        self.exp_i32: np.ndarray | None = None
        self.log_table: np.ndarray | None = None
        self.log_u16: np.ndarray | None = None
        self.inv_table: np.ndarray | None = None
        self.neg_table: np.ndarray | None = None
        self.mul_table: np.ndarray | None = None
        self.add_table: np.ndarray | None = None
        self.zech_table: np.ndarray | None = None

        if self._use_lut:
            self._build_tables()

        self._zero_elem = FiniteFieldElement(self, 0)
        self._one_elem = FiniteFieldElement(self, 1)
        self._gen_elem = FiniteFieldElement(self, self._gen_int)
        self._prim_elem = FiniteFieldElement(self, self._prim_int)

    def _build_tables(self) -> None:
        q = self._order
        qm1 = q - 1
        exp = np.zeros(4 * qm1 + 1, dtype=np.int32)
        log = np.full(q, 2 * qm1, dtype=np.int32)

        if self._is_binary and self._m > 1:
            mod = self._modulus_int
            x = 1
            for i in range(qm1):
                exp[i] = x
                exp[i + qm1] = x
                log[x] = i
                x <<= 1
                if x & q:
                    x ^= mod
        elif self._is_prime:
            g = self._prim_int
            x = 1
            for i in range(qm1):
                exp[i] = x
                exp[i + qm1] = x
                log[x] = i
                x = (x * g) % q
        else:
            p, m = self._p, self._m
            mod = self._modulus_coeffs
            vec = [0] * m
            vec[0] = 1
            p_powers = [p**j for j in range(m)]
            for i in range(qm1):
                val = sum(vec[j] * p_powers[j] for j in range(m))
                exp[i] = val
                exp[i + qm1] = val
                log[val] = i
                top = vec[m - 1]
                new_vec = [0] + vec[: m - 1]
                if top:
                    for j in range(m):
                        new_vec[j] = (new_vec[j] - top * mod[j]) % p
                vec = new_vec

        self.exp_table = np.ascontiguousarray(exp, dtype=np.uint16)
        self.exp_i32 = exp
        self.log_table = np.ascontiguousarray(log, dtype=np.int32)
        log_u16 = np.zeros(q, dtype=np.uint16)
        if q > 1:
            log_u16[1:] = log[1:].astype(np.uint16)
        self.log_u16 = np.ascontiguousarray(log_u16, dtype=np.uint16)

        inv = np.zeros(q, dtype=np.uint16)
        if qm1 > 0:
            idx = np.arange(1, q, dtype=np.int32)
            inv[1:] = self.exp_table[qm1 - log[idx]]
        self.inv_table = np.ascontiguousarray(inv, dtype=np.uint16)

        if self._is_binary:
            self.neg_table = np.arange(q, dtype=np.uint16)
        elif self._is_prime:
            neg = np.zeros(q, dtype=np.uint16)
            neg[1:] = (q - np.arange(1, q, dtype=np.int32)).astype(np.uint16)
            self.neg_table = neg
        else:
            p, m = self._p, self._m
            neg = np.zeros(q, dtype=np.uint16)
            for a in range(1, q):
                rem = a
                res = 0
                pw = 1
                for _ in range(m):
                    d = rem % p
                    rem //= p
                    if d:
                        res += (p - d) * pw
                    pw *= p
                neg[a] = res
            self.neg_table = neg

        if q <= 256:
            a_grid = log[:, None]
            b_grid = log[None, :]
            self.mul_table = np.ascontiguousarray(self.exp_table[a_grid + b_grid], dtype=np.uint16)
        else:
            self.mul_table = None

        if not self._is_binary and not self._is_prime:
            p, m = self._p, self._m
            zech = np.full(qm1, 2 * qm1, dtype=np.int32)
            for d in range(qm1):
                val = int(exp[d])
                c0 = val % p
                added = val - c0 + ((c0 + 1) % p)
                zech[d] = int(log[added])
            self.zech_table = zech
            if q <= 256:
                add_t = np.zeros((q, q), dtype=np.uint16)
                for a in range(q):
                    for b in range(q):
                        ra, rb = a, b
                        res = 0
                        pw = 1
                        for _ in range(m):
                            res += ((ra % p) + (rb % p)) % p * pw
                            ra //= p
                            rb //= p
                            pw *= p
                        add_t[a, b] = res
                self.add_table = add_t

    # --- Public Sage-compatible API ---

    def order(self) -> int:
        return self._order

    def cardinality(self) -> int:
        return self._order

    def characteristic(self) -> int:
        return self._p

    def degree(self) -> int:
        return self._m

    def is_finite(self) -> bool:
        return True

    def is_field(self) -> bool:
        return True

    def zero(self) -> FiniteFieldElement:
        return self._zero_elem

    def one(self) -> FiniteFieldElement:
        return self._one_elem

    def gen(self) -> FiniteFieldElement:
        return self._gen_elem

    def primitive_element(self) -> FiniteFieldElement:
        return self._prim_elem

    def multiplicative_generator(self) -> FiniteFieldElement:
        return self._prim_elem

    def random_element(self, rng: np.random.Generator | None = None) -> FiniteFieldElement:
        r = _DEFAULT_RNG if rng is None else rng
        if self._order < (1 << 62):
            return FiniteFieldElement(self, int(r.integers(0, self._order)))
        # Large field fallback via Python random bits
        val = int(r.integers(0, self._p))
        for _ in range(self._m - 1):
            val = val * self._p + int(r.integers(0, self._p))
        return FiniteFieldElement(self, val)

    def random_nonzero(self, rng: np.random.Generator | None = None) -> FiniteFieldElement:
        r = _DEFAULT_RNG if rng is None else rng
        if self._order < (1 << 62):
            return FiniteFieldElement(self, int(r.integers(1, self._order)))
        while True:
            elem = self.random_element(r)
            if not elem.is_zero():
                return elem

    def from_integer(self, i: int) -> FiniteFieldElement:
        if isinstance(i, bool):
            raise ValueError(f"Element integer must be in [0, {self._order}), got {i!r}")
        idx = int(i)
        if not (0 <= idx < self._order):
            raise ValueError(f"Element integer {idx} out of range [0, {self._order})")
        if idx == 0:
            return self._zero_elem
        if idx == 1:
            return self._one_elem
        return FiniteFieldElement(self, idx)

    def fetch_int(self, i: int) -> FiniteFieldElement:
        return self.from_integer(i)

    def __call__(self, x: Any) -> FiniteFieldElement:
        if isinstance(x, FiniteFieldElement):
            if x._field == self:
                return x
            if x._field._is_prime and x._field._p == self._p:
                return self.from_integer(x._val)
            raise TypeError(f"Cannot coerce element of {x._field!r} into {self!r}")
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            v = int(x) % self._p
            if v == 0:
                return self._zero_elem
            if v == 1:
                return self._one_elem
            return FiniteFieldElement(self, v)
        raise TypeError(f"Cannot coerce {x!r} into {self!r}")

    def __iter__(self) -> Iterator[FiniteFieldElement]:
        for i in range(self._order):
            yield FiniteFieldElement(self, i)

    def __len__(self) -> int:
        return self._order

    def __contains__(self, x: object) -> bool:
        return isinstance(x, FiniteFieldElement) and x._field == self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FiniteField):
            return False
        return self._order == other._order and self._modulus_coeffs == other._modulus_coeffs

    def __hash__(self) -> int:
        return hash((self._order, self._modulus_coeffs))

    def __repr__(self) -> str:
        if self._is_prime:
            return f"Finite Field of size {self._order}"
        return f"Finite Field in {self._name} of size {self._p}^{self._m}"

    # --- Fast scalar integer arithmetic ---

    def add_s(self, a: int, b: int) -> int:
        if self._is_binary:
            return a ^ b
        if self._is_prime:
            s = a + b
            return s - self._p if s >= self._p else s
        if self.add_table is not None:
            return int(self.add_table[a, b])
        if self._use_lut:
            if a == 0:
                return b
            if b == 0:
                return a
            assert (
                self.log_table is not None
                and self.zech_table is not None
                and self.exp_i32 is not None
            )
            la = int(self.log_table[a])
            lb = int(self.log_table[b])
            if la < lb:
                la, lb = lb, la
            z = int(self.zech_table[la - lb])
            if z >= 2 * (self._order - 1):
                return 0
            return int(self.exp_i32[lb + z])
        da = _int_to_base_p(a, self._p, self._m)
        db = _int_to_base_p(b, self._p, self._m)
        return _base_p_to_int([(x + y) % self._p for x, y in zip(da, db, strict=True)], self._p)

    def neg_s(self, a: int) -> int:
        if self._is_binary or a == 0:
            return a
        if self.neg_table is not None:
            return int(self.neg_table[a])
        if self._is_prime:
            return self._p - a
        da = _int_to_base_p(a, self._p, self._m)
        return _base_p_to_int([(-x) % self._p for x in da], self._p)

    def sub_s(self, a: int, b: int) -> int:
        if self._is_binary:
            return a ^ b
        return self.add_s(a, self.neg_s(b))

    def mul_s(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        if self.mul_table is not None:
            return int(self.mul_table[a, b])
        if self._use_lut:
            assert self.exp_i32 is not None and self.log_table is not None
            return int(self.exp_i32[int(self.log_table[a]) + int(self.log_table[b])])
        if self._is_prime:
            return (a * b) % self._p
        if self._is_binary:
            return _gf2_mul_mod(a, b, self._modulus_int, self._m)
        da = _int_to_base_p(a, self._p, self._m)
        db = _int_to_base_p(b, self._p, self._m)
        return _base_p_to_int(_fp_poly_mul_mod(da, db, self._modulus_coeffs, self._p), self._p)

    def inv_s(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Inverse of zero in finite field")
        if self.inv_table is not None:
            return int(self.inv_table[a])
        if self._is_prime:
            return pow(a, -1, self._p)
        if self._is_binary:
            return _gf2_inv_mod(a, self._modulus_int)
        da = _int_to_base_p(a, self._p, self._m)
        return _base_p_to_int(_fp_poly_inv_mod(da, self._modulus_coeffs, self._p), self._p)

    def div_s(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("Division by zero in finite field")
        if a == 0:
            return 0
        if self._use_lut:
            assert self.exp_i32 is not None and self.log_table is not None
            return int(
                self.exp_i32[int(self.log_table[a]) + (self._order - 1) - int(self.log_table[b])]
            )
        return self.mul_s(a, self.inv_s(b))

    def pow_s(self, a: int, exp: int) -> int:
        if exp == 0:
            return 1
        if a == 0:
            if exp < 0:
                raise ZeroDivisionError("Zero cannot be raised to a negative power")
            return 0
        if self._use_lut:
            assert self.exp_i32 is not None and self.log_table is not None
            qm1 = self._order - 1
            return int(self.exp_i32[(int(self.log_table[a]) * (exp % qm1)) % qm1])
        if self._is_prime:
            return pow(a, exp, self._p)
        if exp < 0:
            a = self.inv_s(a)
            exp = -exp
        res = 1
        base = a
        e = exp
        while e > 0:
            if e & 1:
                res = self.mul_s(res, base)
            base = self.mul_s(base, base)
            e >>= 1
        return res

    # --- Fast NumPy array primitives ---

    def coerce_to_arr(self, items: Iterable[Any]) -> np.ndarray:
        if isinstance(items, FFVector):
            if items._field != self:
                raise TypeError(f"Field mismatch: {items._field!r} != {self!r}")
            return items._arr.copy()
        vals = [self(x)._val for x in items]
        return np.array(vals, dtype=self._dtype)

    def add_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._is_binary:
            return np.bitwise_xor(a, b)
        if self._is_prime:
            return ((a.astype(np.int64) + b.astype(np.int64)) % self._p).astype(self._dtype)
        if self.add_table is not None:
            return self.add_table[a, b]
        if self._use_lut:
            p, m = self._p, self._m
            a32 = a.astype(np.int32, copy=False)
            b32 = b.astype(np.int32, copy=False)
            res = np.zeros(np.broadcast_shapes(a32.shape, b32.shape), dtype=np.int32)
            pw = 1
            ra, rb = a32, b32
            for _ in range(m):
                res += (((ra % p) + (rb % p)) % p) * pw
                ra = ra // p
                rb = rb // p
                pw *= p
            return res.astype(self._dtype)
        ba, bb = np.broadcast_arrays(a, b)
        out = np.empty(ba.shape, dtype=self._dtype)
        for idx in np.ndindex(ba.shape):
            out[idx] = self.add_s(int(ba[idx]), int(bb[idx]))
        return out

    def neg_arr(self, a: np.ndarray) -> np.ndarray:
        if self._is_binary:
            return a.copy()
        if self.neg_table is not None:
            return self.neg_table[a]
        if self._is_prime:
            return ((-a.astype(np.int64)) % self._p).astype(self._dtype)
        out = np.empty_like(a)
        for idx in np.ndindex(a.shape):
            out[idx] = self.neg_s(int(a[idx]))
        return out

    def sub_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._is_binary:
            return np.bitwise_xor(a, b)
        return self.add_arr(a, self.neg_arr(b))

    def mul_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.mul_table is not None:
            return self.mul_table[a, b]
        if self._use_lut:
            assert self.exp_table is not None and self.log_table is not None
            return self.exp_table[self.log_table[a] + self.log_table[b]]
        if self._is_prime and self._p < (1 << 31):
            return ((a.astype(np.int64) * b.astype(np.int64)) % self._p).astype(self._dtype)
        ba, bb = np.broadcast_arrays(a, b)
        out = np.empty(ba.shape, dtype=self._dtype)
        for idx in np.ndindex(ba.shape):
            out[idx] = self.mul_s(int(ba[idx]), int(bb[idx]))
        return out

    def inv_arr(self, a: np.ndarray) -> np.ndarray:
        if self.inv_table is not None:
            return self.inv_table[a]
        out = np.empty_like(a)
        for idx in np.ndindex(a.shape):
            v = int(a[idx])
            out[idx] = 0 if v == 0 else self.inv_s(v)
        return out

    def div_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.mul_table is not None and self.inv_table is not None:
            return self.mul_table[a, self.inv_table[b]]
        if self._use_lut:
            assert self.exp_table is not None and self.log_table is not None
            return self.exp_table[self.log_table[a] + (self._order - 1) - self.log_table[b]]
        return self.mul_arr(a, self.inv_arr(b))

    def pow_arr(self, a: np.ndarray, exp: int) -> np.ndarray:
        exp_i = int(exp)
        if exp_i == 0:
            return np.ones_like(a, dtype=self._dtype)
        if self._use_lut:
            assert self.exp_table is not None and self.log_table is not None
            qm1 = self._order - 1
            e_mod = exp_i % qm1
            out = np.zeros_like(a, dtype=self._dtype)
            nz = a != 0
            if np.any(nz):
                out[nz] = self.exp_table[(self.log_table[a[nz]] * e_mod) % qm1]
            return out
        out = np.empty_like(a)
        for idx in np.ndindex(a.shape):
            out[idx] = self.pow_s(int(a[idx]), exp_i)
        return out

    # --- Linear algebra kernels over GF(q) ---

    def matvec_arr(self, M: np.ndarray, v: np.ndarray) -> np.ndarray:
        m_rows, n_cols = M.shape
        if m_rows == 0:
            return np.zeros(0, dtype=self._dtype)
        if n_cols == 0:
            return np.zeros(m_rows, dtype=self._dtype)
        if self._can_use_cy and hasattr(_ff_cy, "matvec_cy"):
            assert self.exp_table is not None and self.log_u16 is not None
            return _ff_cy.matvec_cy(
                np.ascontiguousarray(M, dtype=np.uint16),
                np.ascontiguousarray(v, dtype=np.uint16),
                self.exp_table,
                self.log_u16,
                self._order,
            )
        if self._is_binary:
            prods = self.mul_arr(M, v[None, :])
            return np.bitwise_xor.reduce(prods, axis=1)
        if self._is_prime and n_cols * (self._p - 1) * (self._p - 1) < (1 << 62):
            return ((M.astype(np.int64) @ v.astype(np.int64)) % self._p).astype(self._dtype)
        prods = self.mul_arr(M, v[None, :])
        acc = prods[:, 0].copy()
        for j in range(1, n_cols):
            acc = self.add_arr(acc, prods[:, j])
        return acc

    def matmul_arr(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        m_rows, k_dim = A.shape
        k2, n_cols = B.shape
        if k_dim != k2:
            raise ValueError(f"Matrix shape mismatch: {A.shape} vs {B.shape}")
        if m_rows == 0 or n_cols == 0 or k_dim == 0:
            return np.zeros((m_rows, n_cols), dtype=self._dtype)
        if self._can_use_cy and hasattr(_ff_cy, "matmul_cy"):
            assert self.exp_table is not None and self.log_u16 is not None
            return _ff_cy.matmul_cy(
                np.ascontiguousarray(A, dtype=np.uint16),
                np.ascontiguousarray(B, dtype=np.uint16),
                self.exp_table,
                self.log_u16,
                self._order,
            )
        if self._is_prime and k_dim * (self._p - 1) * (self._p - 1) < (1 << 62):
            return ((A.astype(np.int64) @ B.astype(np.int64)) % self._p).astype(self._dtype)
        out = np.zeros((m_rows, n_cols), dtype=self._dtype)
        if self._is_binary:
            for t in range(k_dim):
                col = A[:, t]
                row = B[t, :]
                nz_r = np.flatnonzero(col)
                if nz_r.size:
                    out[nz_r] ^= self.mul_arr(col[nz_r, None], row[None, :])
            return out
        for t in range(k_dim):
            col = A[:, t]
            row = B[t, :]
            out = self.add_arr(out, self.mul_arr(col[:, None], row[None, :]))
        return out

    def rref_arr(self, M: np.ndarray) -> tuple[np.ndarray, list[int]]:
        nrows, ncols = M.shape
        if nrows == 0 or ncols == 0:
            return np.array(M, dtype=self._dtype, copy=True), []
        if self._can_use_cy and hasattr(_ff_cy, "rref_cy"):
            assert self.exp_table is not None and self.log_u16 is not None
            R, pivs = _ff_cy.rref_cy(
                np.ascontiguousarray(M, dtype=np.uint16),
                self.exp_table,
                self.log_u16,
                self._order,
            )
            return R, [int(p) for p in pivs]
        A = np.array(M, dtype=self._dtype, copy=True)
        pivots: list[int] = []
        r = 0
        for c in range(ncols):
            if r >= nrows:
                break
            nz = np.flatnonzero(A[r:, c])
            if nz.size == 0:
                continue
            piv_row = r + int(nz[0])
            if piv_row != r:
                A[[r, piv_row]] = A[[piv_row, r]]
            piv_val = int(A[r, c])
            if piv_val != 1:
                inv_p = self.inv_s(piv_val)
                A[r, c:] = self.mul_arr(A[r, c:], np.array(inv_p, dtype=self._dtype))
            factors = A[:, c].copy()
            factors[r] = 0
            elim_rows = np.flatnonzero(factors)
            if elim_rows.size:
                scaled = self.mul_arr(factors[elim_rows, None], A[r : r + 1, c:])
                if self._is_binary:
                    A[elim_rows, c:] ^= scaled
                else:
                    A[elim_rows, c:] = self.sub_arr(A[elim_rows, c:], scaled)
            pivots.append(c)
            r += 1
        return A, pivots

    def left_kernel_arr(self, M: np.ndarray) -> np.ndarray:
        nrows, ncols = M.shape
        if nrows == 0:
            return np.zeros((0, 0), dtype=self._dtype)
        if ncols == 0:
            return np.eye(nrows, dtype=self._dtype)
        if self._can_use_cy and hasattr(_ff_cy, "left_kernel_cy"):
            assert self.exp_table is not None and self.log_u16 is not None
            return _ff_cy.left_kernel_cy(
                np.ascontiguousarray(M, dtype=np.uint16),
                self.exp_table,
                self.log_u16,
                self._order,
            )
        MT_rref, pivots = self.rref_arr(M.T)
        piv_set = set(pivots)
        free_cols = [j for j in range(nrows) if j not in piv_set]
        if not free_cols:
            return np.zeros((0, nrows), dtype=self._dtype)
        K = np.zeros((len(free_cols), nrows), dtype=self._dtype)
        for idx, fc in enumerate(free_cols):
            K[idx, fc] = 1
            for row_i, pc in enumerate(pivots):
                val = int(MT_rref[row_i, fc])
                if val:
                    K[idx, pc] = self.neg_s(val)
        return K

    def right_kernel_arr(self, M: np.ndarray) -> np.ndarray:
        nrows, ncols = M.shape
        if ncols == 0:
            return np.zeros((0, 0), dtype=self._dtype)
        if nrows == 0:
            return np.eye(ncols, dtype=self._dtype)
        R, pivots = self.rref_arr(M)
        piv_set = set(pivots)
        free_cols = [j for j in range(ncols) if j not in piv_set]
        if not free_cols:
            return np.zeros((0, ncols), dtype=self._dtype)
        K = np.zeros((len(free_cols), ncols), dtype=self._dtype)
        for idx, fc in enumerate(free_cols):
            K[idx, fc] = 1
            for row_i, pc in enumerate(pivots):
                val = int(R[row_i, fc])
                if val:
                    K[idx, pc] = self.neg_s(val)
        return K

    def inverse_arr(self, M: np.ndarray) -> np.ndarray:
        n, m = M.shape
        if n != m:
            raise ValueError("Matrix must be square")
        if n == 0:
            return np.zeros((0, 0), dtype=self._dtype)
        aug = np.concatenate(
            [M.astype(self._dtype, copy=False), np.eye(n, dtype=self._dtype)], axis=1
        )
        R, pivots = self.rref_arr(aug)
        if len(pivots) < n or pivots[n - 1] >= n:
            raise ValueError("Matrix is singular")
        return np.ascontiguousarray(R[:, n:])

    # --- Polynomial array kernels over GF(q)[x] ---

    def poly_trim(self, p: np.ndarray) -> np.ndarray:
        nz = np.flatnonzero(p)
        if nz.size == 0:
            return np.zeros(0, dtype=self._dtype)
        return p[: int(nz[-1]) + 1]

    def poly_add_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        la, lb = a.size, b.size
        n = max(la, lb)
        if n == 0:
            return np.zeros(0, dtype=self._dtype)
        out = np.zeros(n, dtype=self._dtype)
        if la:
            out[:la] = a
        if lb:
            out[:lb] = self.add_arr(out[:lb], b)
        return self.poly_trim(out)

    def poly_sub_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._is_binary:
            return self.poly_add_arr(a, b)
        return self.poly_add_arr(a, self.neg_arr(b))

    def poly_scale_arr(self, a: np.ndarray, c: int) -> np.ndarray:
        if c == 0 or a.size == 0:
            return np.zeros(0, dtype=self._dtype)
        if c == 1:
            return a.copy()
        return self.mul_arr(a, np.array(c, dtype=self._dtype))

    def poly_mul_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = self.poly_trim(a)
        b = self.poly_trim(b)
        la, lb = a.size, b.size
        if la == 0 or lb == 0:
            return np.zeros(0, dtype=self._dtype)
        if la > lb:
            a, b = b, a
            la, lb = lb, la
        if la >= 64:
            return self._poly_mul_karatsuba(a, b)
        out = np.zeros(la + lb - 1, dtype=self._dtype)
        if self._is_binary:
            for i in range(la):
                ai = int(a[i])
                if ai:
                    out[i : i + lb] ^= self.mul_arr(b, np.array(ai, dtype=self._dtype))
        else:
            for i in range(la):
                ai = int(a[i])
                if ai:
                    out[i : i + lb] = self.add_arr(
                        out[i : i + lb], self.mul_arr(b, np.array(ai, dtype=self._dtype))
                    )
        return self.poly_trim(out)

    def _poly_mul_karatsuba(self, a: np.ndarray, b: np.ndarray, cutoff: int = 32) -> np.ndarray:
        la, lb = a.size, b.size
        if la == 0 or lb == 0:
            return np.zeros(0, dtype=self._dtype)
        if min(la, lb) <= cutoff:
            if la > lb:
                a, b = b, a
                la, lb = lb, la
            out = np.zeros(la + lb - 1, dtype=self._dtype)
            if self._is_binary:
                for i in range(la):
                    ai = int(a[i])
                    if ai:
                        out[i : i + lb] ^= self.mul_arr(b, np.array(ai, dtype=self._dtype))
            else:
                for i in range(la):
                    ai = int(a[i])
                    if ai:
                        out[i : i + lb] = self.add_arr(
                            out[i : i + lb], self.mul_arr(b, np.array(ai, dtype=self._dtype))
                        )
            return self.poly_trim(out)
        m = max(la, lb) // 2
        a0, a1 = a[:m], a[m:]
        b0, b1 = b[:m], b[m:]
        z0 = self._poly_mul_karatsuba(a0, b0, cutoff)
        z2 = self._poly_mul_karatsuba(a1, b1, cutoff)
        a01 = self.poly_add_arr(a0, a1)
        b01 = self.poly_add_arr(b0, b1)
        z1_full = self._poly_mul_karatsuba(a01, b01, cutoff)
        z1 = self.poly_sub_arr(self.poly_sub_arr(z1_full, z0), z2)
        res = np.zeros(la + lb - 1, dtype=self._dtype)
        if z0.size:
            res[: z0.size] = z0
        if z1.size:
            res[m : m + z1.size] = self.add_arr(res[m : m + z1.size], z1)
        if z2.size:
            res[2 * m : 2 * m + z2.size] = self.add_arr(res[2 * m : 2 * m + z2.size], z2)
        return self.poly_trim(res)

    def poly_divmod_arr(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.poly_trim(a)
        b = self.poly_trim(b)
        if b.size == 0:
            raise ZeroDivisionError("Polynomial division by zero")
        da = a.size - 1
        db = b.size - 1
        if da < db:
            return np.zeros(0, dtype=self._dtype), a.copy()
        rem = a.copy()
        q = np.zeros(da - db + 1, dtype=self._dtype)
        inv_lc = self.inv_s(int(b[db]))
        for shift in range(da - db, -1, -1):
            c = int(rem[shift + db])
            if c:
                factor = self.mul_s(c, inv_lc)
                q[shift] = factor
                term = self.mul_arr(b, np.array(factor, dtype=self._dtype))
                if self._is_binary:
                    rem[shift : shift + db + 1] ^= term
                else:
                    rem[shift : shift + db + 1] = self.sub_arr(rem[shift : shift + db + 1], term)
        return self.poly_trim(q), self.poly_trim(rem)

    def poly_gcd_arr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r0 = self.poly_trim(a)
        r1 = self.poly_trim(b)
        while r1.size > 0:
            _, r2 = self.poly_divmod_arr(r0, r1)
            r0, r1 = r1, r2
        if r0.size > 0:
            r0 = self.poly_scale_arr(r0, self.inv_s(int(r0[-1])))
        return r0

    def poly_xgcd_arr(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r0 = self.poly_trim(a)
        r1 = self.poly_trim(b)
        s0 = np.array([1], dtype=self._dtype)
        s1 = np.zeros(0, dtype=self._dtype)
        t0 = np.zeros(0, dtype=self._dtype)
        t1 = np.array([1], dtype=self._dtype)
        while r1.size > 0:
            q, r2 = self.poly_divmod_arr(r0, r1)
            s2 = self.poly_sub_arr(s0, self.poly_mul_arr(q, s1))
            t2 = self.poly_sub_arr(t0, self.poly_mul_arr(q, t1))
            r0, r1 = r1, r2
            s0, s1 = s1, s2
            t0, t1 = t1, t2
        if r0.size > 0:
            inv_lc = self.inv_s(int(r0[-1]))
            r0 = self.poly_scale_arr(r0, inv_lc)
            s0 = self.poly_scale_arr(s0, inv_lc)
            t0 = self.poly_scale_arr(t0, inv_lc)
        return r0, s0, t0

    def poly_eval_scalar(self, p: np.ndarray, x: int) -> int:
        p = self.poly_trim(p)
        if p.size == 0:
            return 0
        acc = int(p[-1])
        for i in range(p.size - 2, -1, -1):
            acc = self.add_s(self.mul_s(acc, x), int(p[i]))
        return acc

    def poly_eval_many_arr(self, p: np.ndarray, xs: np.ndarray) -> np.ndarray:
        p = self.poly_trim(p)
        if p.size == 0 or xs.size == 0:
            return np.zeros(xs.size, dtype=self._dtype)
        if self._can_use_cy and hasattr(_ff_cy, "poly_eval_batch_cy"):
            assert self.exp_table is not None and self.log_u16 is not None
            return _ff_cy.poly_eval_batch_cy(
                np.ascontiguousarray(p, dtype=np.uint16),
                np.ascontiguousarray(xs, dtype=np.uint16),
                self.exp_table,
                self.log_u16,
                self._order,
            )
        res = np.full(xs.size, int(p[-1]), dtype=self._dtype)
        if self._is_binary:
            for i in range(p.size - 2, -1, -1):
                res = self.mul_arr(res, xs) ^ int(p[i])
        else:
            for i in range(p.size - 2, -1, -1):
                res = self.add_arr(
                    self.mul_arr(res, xs), np.full(xs.size, int(p[i]), dtype=self._dtype)
                )
        return res

    def poly_interpolate_arr(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        n = xs.size
        if n == 0:
            return np.zeros(0, dtype=self._dtype)
        dd = np.array(ys, dtype=self._dtype, copy=True)
        for j in range(1, n):
            num = self.sub_arr(dd[j:], dd[j - 1 : n - 1])
            den = self.sub_arr(xs[j:], xs[: n - j])
            if np.any(den == 0):
                raise ValueError("Interpolation x-coordinates must be distinct")
            dd[j:] = self.div_arr(num, den)
        p = np.array([int(dd[n - 1])], dtype=self._dtype)
        for i in range(n - 2, -1, -1):
            neg_xi = self.neg_s(int(xs[i]))
            shifted = np.zeros(p.size + 1, dtype=self._dtype)
            shifted[1:] = p
            if neg_xi:
                shifted[: p.size] = self.add_arr(
                    shifted[: p.size], self.mul_arr(p, np.array(neg_xi, dtype=self._dtype))
                )
            d_i = int(dd[i])
            if d_i:
                shifted[0] = self.add_s(int(shifted[0]), d_i)
            p = shifted
        return self.poly_trim(p)

    def vanishing_poly_arr(self, roots: np.ndarray) -> np.ndarray:
        n = roots.size
        if n == 0:
            return np.array([1], dtype=self._dtype)
        polys = [np.array([self.neg_s(int(r)), 1], dtype=self._dtype) for r in roots]
        while len(polys) > 1:
            nxt: list[np.ndarray] = []
            for i in range(0, len(polys), 2):
                if i + 1 < len(polys):
                    nxt.append(self.poly_mul_arr(polys[i], polys[i + 1]))
                else:
                    nxt.append(polys[i])
            polys = nxt
        return polys[0]


_FIELD_CACHE: dict[tuple[int, str, str], FiniteField] = {}


def GF(order: int, name: str = "a", repr: str = "int") -> FiniteField:
    """Construct (or retrieve cached) finite field ``GF(order)``."""
    # Validate early so bad inputs raise ValueError before caching
    factor_prime_power(order)
    key = (int(order), str(name), str(repr))
    if key not in _FIELD_CACHE:
        _FIELD_CACHE[key] = FiniteField(int(order), name=name, repr_mode=repr)
    return _FIELD_CACHE[key]


# ---------------------------------------------------------------------------
# L2: FFVector, VectorSpace, VectorSubspace
# ---------------------------------------------------------------------------


class FFVector:
    """1D vector over ``GF(q)`` backed by a contiguous 1D NumPy array."""

    __slots__ = ("_field", "_arr", "_immutable")

    def __init__(self, field: FiniteField, arr: np.ndarray, *, immutable: bool = False) -> None:
        self._field = field
        self._arr = np.ascontiguousarray(arr, dtype=field._dtype)
        self._immutable = False
        if immutable:
            self.set_immutable()

    def base_ring(self) -> FiniteField:
        return self._field

    def parent(self) -> VectorSpace:
        return VectorSpace(self._field, int(self._arr.size))

    def set_immutable(self) -> None:
        self._immutable = True
        self._arr.flags.writeable = False

    def is_immutable(self) -> bool:
        return self._immutable

    def copy(self) -> FFVector:
        return FFVector(self._field, self._arr.copy(), immutable=False)

    def __copy__(self) -> FFVector:
        return self.copy()

    def __len__(self) -> int:
        return int(self._arr.size)

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return FFVector(self._field, self._arr[idx].copy(), immutable=False)
        return FiniteFieldElement(self._field, int(self._arr[idx]))

    def __setitem__(self, idx: int | slice, val: Any) -> None:
        if self._immutable:
            raise ValueError("vector is immutable; please change a copy instead")
        if isinstance(idx, slice):
            vals = [self._field(x)._val for x in val]
            self._arr[idx] = np.array(vals, dtype=self._field._dtype)
            return
        self._arr[idx] = self._field(val)._val

    def __iter__(self) -> Iterator[FiniteFieldElement]:
        f = self._field
        for v in self._arr:
            yield FiniteFieldElement(f, int(v))

    def list(self) -> list[FiniteFieldElement]:
        f = self._field
        return [FiniteFieldElement(f, int(v)) for v in self._arr]

    def is_zero(self) -> bool:
        return not bool(np.any(self._arr))

    def hamming_weight(self) -> int:
        return int(np.count_nonzero(self._arr))

    def nonzero_positions(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self._arr)]

    def __add__(self, other: Any) -> FFVector:
        if not isinstance(other, FFVector) or other._field != self._field:
            return NotImplemented
        if self._arr.size != other._arr.size:
            raise TypeError("Vector lengths do not match")
        return FFVector(self._field, self._field.add_arr(self._arr, other._arr))

    def __sub__(self, other: Any) -> FFVector:
        if not isinstance(other, FFVector) or other._field != self._field:
            return NotImplemented
        if self._arr.size != other._arr.size:
            raise TypeError("Vector lengths do not match")
        return FFVector(self._field, self._field.sub_arr(self._arr, other._arr))

    def __neg__(self) -> FFVector:
        return FFVector(self._field, self._field.neg_arr(self._arr))

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, FFMatrix):
            if other._field != self._field:
                raise TypeError("Field mismatch between vector and matrix")
            if self._arr.size != other._arr.shape[0]:
                raise TypeError(
                    f"Vector length {self._arr.size} != matrix nrows {other._arr.shape[0]}"
                )
            # v * M == (M^T * v^T)^T
            res = self._field.matvec_arr(np.ascontiguousarray(other._arr.T), self._arr)
            return FFVector(self._field, res)
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            c = self._field(other)._val
            if c == 0:
                return FFVector(self._field, np.zeros(self._arr.size, dtype=self._field._dtype))
            if c == 1:
                return self.copy()
            return FFVector(
                self._field,
                self._field.mul_arr(self._arr, np.array(c, dtype=self._field._dtype)),
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            return self.__mul__(other)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FFVector):
            return False
        return (
            self._field == other._field
            and self._arr.shape == other._arr.shape
            and bool(np.array_equal(self._arr, other._arr))
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        if not self._immutable:
            raise TypeError("mutable vectors are unhashable")
        return hash((self._field, self._arr.tobytes()))

    def __repr__(self) -> str:
        inner = ", ".join(repr(FiniteFieldElement(self._field, int(v))) for v in self._arr)
        return f"({inner})"


class VectorSubspace:
    """Linear subspace of ``VectorSpace(field, ambient_dim)`` with RREF basis matrix."""

    def __init__(self, ambient: VectorSpace, basis_arr: np.ndarray) -> None:
        self._ambient = ambient
        self._field = ambient._field
        self._basis_arr = np.ascontiguousarray(basis_arr, dtype=self._field._dtype)

    def dimension(self) -> int:
        return int(self._basis_arr.shape[0])

    def ambient_space(self) -> VectorSpace:
        return self._ambient

    def basis_matrix(self) -> FFMatrix:
        return FFMatrix(self._field, self._basis_arr.copy())

    def __len__(self) -> int:
        return int(self._field._order ** self._basis_arr.shape[0])

    def __iter__(self) -> Iterator[FFVector]:
        f = self._field
        d = int(self._basis_arr.shape[0])
        n = int(self._basis_arr.shape[1])
        if d == 0:
            yield FFVector(f, np.zeros(n, dtype=f._dtype))
            return
        if f._order == 2:
            # Fast bitmask span enumeration for GF(2)
            row_masks = [0] * d
            for r in range(d):
                mask = 0
                for j in range(n):
                    if self._basis_arr[r, j]:
                        mask |= 1 << j
                row_masks[r] = mask
            for bits in range(1 << d):
                vec_mask = 0
                for r in range(d):
                    if (bits >> r) & 1:
                        vec_mask ^= row_masks[r]
                arr = np.array([(vec_mask >> j) & 1 for j in range(n)], dtype=f._dtype)
                yield FFVector(f, arr)
            return
        for coeffs in itertools.product(range(f._order), repeat=d):
            c_arr = np.array(coeffs, dtype=f._dtype)
            vec_arr = f.matvec_arr(np.ascontiguousarray(self._basis_arr.T), c_arr)
            yield FFVector(f, vec_arr)

    def __repr__(self) -> str:
        return (
            f"Vector space of degree {self._ambient._dim} and dimension "
            f"{self.dimension()} over {self._field!r}"
        )


class VectorSpace:
    """Ambient vector space ``GF(q)^dim``."""

    def __init__(self, field: FiniteField, dim: int) -> None:
        if not isinstance(field, FiniteField):
            raise TypeError(f"Expected FiniteField, got {type(field).__name__}")
        dim_i = int(dim)
        if dim_i < 0:
            raise ValueError(f"Dimension must be non-negative, got {dim_i}")
        self._field = field
        self._dim = dim_i

    def dimension(self) -> int:
        return self._dim

    def base_field(self) -> FiniteField:
        return self._field

    def base_ring(self) -> FiniteField:
        return self._field

    def zero(self) -> FFVector:
        return FFVector(self._field, np.zeros(self._dim, dtype=self._field._dtype))

    def random_element(self, rng: np.random.Generator | None = None) -> FFVector:
        r = _DEFAULT_RNG if rng is None else rng
        if self._dim == 0:
            return self.zero()
        if self._field._order < (1 << 62):
            arr = r.integers(0, self._field._order, size=self._dim, dtype=np.int64).astype(
                self._field._dtype
            )
            return FFVector(self._field, arr)
        vals = [self._field.random_element(r)._val for _ in range(self._dim)]
        return FFVector(self._field, np.array(vals, dtype=self._field._dtype))

    def __call__(self, elems: Any) -> FFVector:
        if isinstance(elems, FFVector):
            if elems._field == self._field and elems._arr.size == self._dim:
                return elems.copy()
            if elems._arr.size != self._dim:
                raise ValueError(f"Expected vector of length {self._dim}, got {elems._arr.size}")
            return FFVector(self._field, self._field.coerce_to_arr(elems))
        if isinstance(elems, (int, np.integer)) and int(elems) == 0:
            return self.zero()
        arr = self._field.coerce_to_arr(elems)
        if arr.size != self._dim:
            raise ValueError(f"Expected vector of length {self._dim}, got {arr.size}")
        return FFVector(self._field, arr)

    def __contains__(self, x: object) -> bool:
        return isinstance(x, FFVector) and x._field == self._field and int(x._arr.size) == self._dim

    def __len__(self) -> int:
        return int(self._field._order**self._dim)

    def __iter__(self) -> Iterator[FFVector]:
        f = self._field
        if self._dim == 0:
            yield self.zero()
            return
        for tup in itertools.product(range(f._order), repeat=self._dim):
            yield FFVector(f, np.array(tup, dtype=f._dtype))

    def subspaces(self, d: int) -> list[VectorSubspace]:
        """Enumerate all ``d``-dimensional subspaces via canonical RREF pivot enumeration."""
        d_i = int(d)
        m = self._dim
        f = self._field
        if d_i < 0 or d_i > m:
            return []
        if d_i == 0:
            return [VectorSubspace(self, np.zeros((0, m), dtype=f._dtype))]
        q = f._order
        result: list[VectorSubspace] = []
        for pivots in itertools.combinations(range(m), d_i):
            piv_set = set(pivots)
            free_slots: list[tuple[int, int]] = []
            for row_i, pc in enumerate(pivots):
                for col_j in range(pc + 1, m):
                    if col_j not in piv_set:
                        free_slots.append((row_i, col_j))
            n_free = len(free_slots)
            if q == 2:
                # Fast bitmask enumeration for GF(2)
                for bits in range(1 << n_free):
                    basis = np.zeros((d_i, m), dtype=f._dtype)
                    for row_i, pc in enumerate(pivots):
                        basis[row_i, pc] = 1
                    for k_i, (row_i, col_j) in enumerate(free_slots):
                        if (bits >> k_i) & 1:
                            basis[row_i, col_j] = 1
                    result.append(VectorSubspace(self, basis))
            else:
                for vals in itertools.product(range(q), repeat=n_free):
                    basis = np.zeros((d_i, m), dtype=f._dtype)
                    for row_i, pc in enumerate(pivots):
                        basis[row_i, pc] = 1
                    for (row_i, col_j), v in zip(free_slots, vals, strict=True):
                        basis[row_i, col_j] = v
                    result.append(VectorSubspace(self, basis))
        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VectorSpace)
            and self._field == other._field
            and self._dim == other._dim
        )

    def __hash__(self) -> int:
        return hash((self._field, self._dim))

    def __repr__(self) -> str:
        return f"Vector space of dimension {self._dim} over {self._field!r}"


def vector(field_or_elems: Any, elems: Any = None) -> FFVector:
    """Construct an ``FFVector``: ``vector(F, seq)`` or ``vector(seq)``."""
    if elems is not None:
        if not isinstance(field_or_elems, FiniteField):
            raise TypeError(f"Expected FiniteField, got {type(field_or_elems).__name__}")
        return FFVector(field_or_elems, field_or_elems.coerce_to_arr(elems))
    if isinstance(field_or_elems, FFVector):
        return field_or_elems.copy()
    seq = list(field_or_elems)
    if not seq:
        raise ValueError("Cannot infer field from empty vector; pass field explicitly")
    first = seq[0]
    if not isinstance(first, FiniteFieldElement):
        raise TypeError("Cannot infer FiniteField from non-FiniteFieldElement entries")
    f = first._field
    return FFVector(f, f.coerce_to_arr(seq))


def zero_vector(field: FiniteField, length: int) -> FFVector:
    """Return a mutable zero ``FFVector`` of given ``length`` over ``field``."""
    if not isinstance(field, FiniteField):
        raise TypeError(f"Expected FiniteField, got {type(field).__name__}")
    n = int(length)
    if n < 0:
        raise ValueError(f"Length must be non-negative, got {n}")
    return FFVector(field, np.zeros(n, dtype=field._dtype))


# ---------------------------------------------------------------------------
# L2: FFMatrix, MatrixSpace, matrix
# ---------------------------------------------------------------------------


class _KernelSpace:
    """Lightweight kernel space wrapper supporting ``.basis_matrix()`` and ``.basis()``."""

    def __init__(self, basis_mat: FFMatrix) -> None:
        self._basis_mat = basis_mat

    def basis_matrix(self) -> FFMatrix:
        return self._basis_mat

    def basis(self) -> list[FFVector]:
        return self._basis_mat.rows()

    def dimension(self) -> int:
        return self._basis_mat.nrows()


class FFMatrix:
    """Dense 2D matrix over ``GF(q)`` backed by a contiguous 2D NumPy array."""

    __slots__ = ("_field", "_arr", "_immutable")

    def __init__(self, field: FiniteField, arr: np.ndarray, *, immutable: bool = False) -> None:
        self._field = field
        arr2 = np.ascontiguousarray(arr, dtype=field._dtype)
        if arr2.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr2.shape}")
        self._arr = arr2
        self._immutable = False
        if immutable:
            self.set_immutable()

    def base_ring(self) -> FiniteField:
        return self._field

    def base_field(self) -> FiniteField:
        return self._field

    def parent(self) -> MatrixSpace:
        return MatrixSpace(self._field, self.nrows(), self.ncols())

    def nrows(self) -> int:
        return int(self._arr.shape[0])

    def ncols(self) -> int:
        return int(self._arr.shape[1])

    def dimensions(self) -> tuple[int, int]:
        return int(self._arr.shape[0]), int(self._arr.shape[1])

    def set_immutable(self) -> None:
        self._immutable = True
        self._arr.flags.writeable = False

    def is_immutable(self) -> bool:
        return self._immutable

    def copy(self) -> FFMatrix:
        return FFMatrix(self._field, self._arr.copy(), immutable=False)

    def __copy__(self) -> FFMatrix:
        return self.copy()

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            r, c = key
            if isinstance(r, (int, np.integer)) and isinstance(c, (int, np.integer)):
                return FiniteFieldElement(self._field, int(self._arr[r, c]))
            sub = self._arr[r, c]
            if sub.ndim == 1:
                return FFVector(self._field, sub.copy())
            return FFMatrix(self._field, sub.copy())
        if isinstance(key, (int, np.integer)):
            return self.row(int(key))
        return FFMatrix(self._field, self._arr[key].copy())

    def __setitem__(self, key: Any, val: Any) -> None:
        if self._immutable:
            raise ValueError("matrix is immutable; please change a copy instead")
        if isinstance(key, tuple) and len(key) == 2:
            r, c = key
            self._arr[r, c] = self._field(val)._val
            return
        raise TypeError("Matrix item assignment requires (row, col) index")

    def row(self, i: int) -> FFVector:
        return FFVector(self._field, self._arr[i].copy())

    def rows(self) -> list[FFVector]:
        return [FFVector(self._field, self._arr[i].copy()) for i in range(self.nrows())]

    def column(self, j: int) -> FFVector:
        return FFVector(self._field, self._arr[:, j].copy())

    def columns(self) -> list[FFVector]:
        return [FFVector(self._field, self._arr[:, j].copy()) for j in range(self.ncols())]

    def __iter__(self) -> Iterator[FFVector]:
        for i in range(self.nrows()):
            yield FFVector(self._field, self._arr[i].copy())

    def transpose(self) -> FFMatrix:
        return FFMatrix(self._field, np.ascontiguousarray(self._arr.T))

    @property
    def T(self) -> FFMatrix:
        return self.transpose()

    def is_zero(self) -> bool:
        return not bool(np.any(self._arr))

    def echelon_form(self) -> FFMatrix:
        R, _ = self._field.rref_arr(self._arr)
        return FFMatrix(self._field, R)

    def rref(self) -> FFMatrix:
        return self.echelon_form()

    def pivots(self) -> tuple[int, ...]:
        _, pivs = self._field.rref_arr(self._arr)
        return tuple(pivs)

    def rank(self) -> int:
        _, pivs = self._field.rref_arr(self._arr)
        return len(pivs)

    def left_kernel_matrix(self) -> FFMatrix:
        return FFMatrix(self._field, self._field.left_kernel_arr(self._arr))

    def left_kernel(self) -> _KernelSpace:
        return _KernelSpace(self.left_kernel_matrix())

    def right_kernel_matrix(self) -> FFMatrix:
        return FFMatrix(self._field, self._field.right_kernel_arr(self._arr))

    def right_kernel(self) -> _KernelSpace:
        return _KernelSpace(self.right_kernel_matrix())

    def inverse(self) -> FFMatrix:
        return FFMatrix(self._field, self._field.inverse_arr(self._arr))

    def stack(self, other: FFMatrix) -> FFMatrix:
        if not isinstance(other, FFMatrix) or other._field != self._field:
            raise TypeError("Cannot stack matrices over different fields")
        if self.ncols() != other.ncols():
            raise ValueError(f"Column count mismatch: {self.ncols()} vs {other.ncols()}")
        return FFMatrix(self._field, np.vstack([self._arr, other._arr]))

    def augment(self, other: FFMatrix | FFVector) -> FFMatrix:
        if isinstance(other, FFVector):
            if other._field != self._field:
                raise TypeError("Field mismatch")
            if self.nrows() != len(other):
                raise ValueError(f"Row count mismatch: {self.nrows()} vs {len(other)}")
            return FFMatrix(self._field, np.hstack([self._arr, other._arr[:, None]]))
        if not isinstance(other, FFMatrix) or other._field != self._field:
            raise TypeError("Cannot augment matrices over different fields")
        if self.nrows() != other.nrows():
            raise ValueError(f"Row count mismatch: {self.nrows()} vs {other.nrows()}")
        return FFMatrix(self._field, np.hstack([self._arr, other._arr]))

    def matrix_from_columns(self, cols: Sequence[int]) -> FFMatrix:
        col_list = [int(c) for c in cols]
        if not col_list:
            return FFMatrix(self._field, np.zeros((self.nrows(), 0), dtype=self._field._dtype))
        return FFMatrix(self._field, np.ascontiguousarray(self._arr[:, col_list]))

    def matrix_from_rows(self, rows: Sequence[int]) -> FFMatrix:
        row_list = [int(r) for r in rows]
        if not row_list:
            return FFMatrix(self._field, np.zeros((0, self.ncols()), dtype=self._field._dtype))
        return FFMatrix(self._field, np.ascontiguousarray(self._arr[row_list, :]))

    def submatrix(
        self, r: int = 0, c: int = 0, nr: int | None = None, nc: int | None = None
    ) -> FFMatrix:
        r_end = self.nrows() if nr is None else r + int(nr)
        c_end = self.ncols() if nc is None else c + int(nc)
        return FFMatrix(self._field, np.ascontiguousarray(self._arr[r:r_end, c:c_end]))

    def solve_right(self, b: FFVector | FFMatrix) -> FFVector | FFMatrix:
        """Solve ``self * x = b`` over ``GF(q)``; raise ``ValueError`` if no solution exists."""
        is_vec = isinstance(b, FFVector)
        if is_vec:
            if b._field != self._field or len(b) != self.nrows():
                raise ValueError("Incompatible vector for solve_right")
            b_arr = b._arr[:, None]
            nrhs = 1
        elif isinstance(b, FFMatrix):
            if b._field != self._field or b.nrows() != self.nrows():
                raise ValueError("Incompatible matrix for solve_right")
            b_arr = b._arr
            nrhs = b.ncols()
        else:
            raise TypeError(f"Expected FFVector or FFMatrix, got {type(b).__name__}")

        m_rows, n_cols = self.dimensions()
        aug = np.hstack([self._arr, b_arr])
        R, pivots = self._field.rref_arr(aug)
        for p in pivots:
            if p >= n_cols:
                raise ValueError("matrix equation has no solutions")
        x = np.zeros((n_cols, nrhs), dtype=self._field._dtype)
        for row_i, pc in enumerate(pivots):
            x[pc, :] = R[row_i, n_cols:]
        if is_vec:
            return FFVector(self._field, x[:, 0])
        return FFMatrix(self._field, x)

    def __neg__(self) -> FFMatrix:
        return FFMatrix(self._field, self._field.neg_arr(self._arr))

    def __add__(self, other: Any) -> FFMatrix:
        if not isinstance(other, FFMatrix) or other._field != self._field:
            return NotImplemented
        if self.dimensions() != other.dimensions():
            raise TypeError("Matrix dimensions do not match")
        return FFMatrix(self._field, self._field.add_arr(self._arr, other._arr))

    def __sub__(self, other: Any) -> FFMatrix:
        if not isinstance(other, FFMatrix) or other._field != self._field:
            return NotImplemented
        if self.dimensions() != other.dimensions():
            raise TypeError("Matrix dimensions do not match")
        return FFMatrix(self._field, self._field.sub_arr(self._arr, other._arr))

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, FFMatrix):
            if other._field != self._field:
                raise TypeError("Field mismatch in matrix multiplication")
            return FFMatrix(self._field, self._field.matmul_arr(self._arr, other._arr))
        if isinstance(other, FFVector):
            if other._field != self._field:
                raise TypeError("Field mismatch in matrix-vector multiplication")
            if self.ncols() != len(other):
                raise TypeError(f"Matrix ncols {self.ncols()} != vector length {len(other)}")
            return FFVector(self._field, self._field.matvec_arr(self._arr, other._arr))
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            c = self._field(other)._val
            if c == 0:
                return FFMatrix(self._field, np.zeros(self._arr.shape, dtype=self._field._dtype))
            if c == 1:
                return self.copy()
            return FFMatrix(
                self._field,
                self._field.mul_arr(self._arr, np.array(c, dtype=self._field._dtype)),
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            return self.__mul__(other)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FFMatrix):
            return False
        return (
            self._field == other._field
            and self._arr.shape == other._arr.shape
            and bool(np.array_equal(self._arr, other._arr))
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"{self.nrows()} x {self.ncols()} dense matrix over {self._field!r}"


class MatrixSpace:
    """Space of ``nrows x ncols`` dense matrices over ``GF(q)``."""

    def __init__(self, field: FiniteField, nrows: int, ncols: int | None = None) -> None:
        if not isinstance(field, FiniteField):
            raise TypeError(f"Expected FiniteField, got {type(field).__name__}")
        self._field = field
        self._nrows = int(nrows)
        self._ncols = self._nrows if ncols is None else int(ncols)

    def nrows(self) -> int:
        return self._nrows

    def ncols(self) -> int:
        return self._ncols

    def base_ring(self) -> FiniteField:
        return self._field

    def zero(self) -> FFMatrix:
        return FFMatrix(self._field, np.zeros((self._nrows, self._ncols), dtype=self._field._dtype))

    def __call__(self, entries: Any = None) -> FFMatrix:
        return matrix(self._field, self._nrows, self._ncols, entries)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MatrixSpace)
            and self._field == other._field
            and self._nrows == other._nrows
            and self._ncols == other._ncols
        )

    def __hash__(self) -> int:
        return hash((self._field, self._nrows, self._ncols))

    def __repr__(self) -> str:
        return (
            f"Full MatrixSpace of {self._nrows} by {self._ncols} "
            f"dense matrices over {self._field!r}"
        )


def matrix(
    field_or_rows: Any,
    nrows: Any = None,
    ncols: Any = None,
    entries: Any = None,
) -> FFMatrix:
    """Construct an ``FFMatrix`` following Sage's ``matrix(...)`` calling conventions."""
    if isinstance(field_or_rows, FiniteField):
        f = field_or_rows
        # matrix(F, nrows, ncols, entries) or matrix(F, nrows, ncols)
        if isinstance(nrows, (int, np.integer)) and isinstance(ncols, (int, np.integer)):
            nr, nc = int(nrows), int(ncols)
            if entries is None:
                return FFMatrix(f, np.zeros((nr, nc), dtype=f._dtype))
            if isinstance(entries, np.ndarray):
                return FFMatrix(f, entries.reshape(nr, nc).astype(f._dtype, copy=True))
            seq = list(entries)
            if nr == 0 or nc == 0:
                return FFMatrix(f, np.zeros((nr, nc), dtype=f._dtype))
            if len(seq) == nr and isinstance(seq[0], (list, tuple, FFVector, np.ndarray)):
                arr = np.zeros((nr, nc), dtype=f._dtype)
                for i, row in enumerate(seq):
                    arr[i] = f.coerce_to_arr(row)
                return FFMatrix(f, arr)
            flat = f.coerce_to_arr(seq)
            return FFMatrix(f, flat.reshape(nr, nc))
        # matrix(F, rows)
        if nrows is not None and ncols is None and entries is None:
            rows = list(nrows)
            if not rows:
                return FFMatrix(f, np.zeros((0, 0), dtype=f._dtype))
            first = rows[0]
            nc = len(first)
            arr = np.zeros((len(rows), nc), dtype=f._dtype)
            for i, row in enumerate(rows):
                arr[i] = f.coerce_to_arr(row)
            return FFMatrix(f, arr)
        raise TypeError("Unsupported arguments to matrix()")

    # matrix(rows) inferring field from entries
    rows = list(field_or_rows)
    if not rows:
        raise ValueError("Cannot infer field from empty matrix")
    first_row = rows[0]
    if isinstance(first_row, FFVector):
        f = first_row._field
    else:
        first_list = list(first_row)
        if not first_list or not isinstance(first_list[0], FiniteFieldElement):
            raise TypeError("Cannot infer FiniteField from matrix entries")
        f = first_list[0]._field
    return matrix(f, rows)


# ---------------------------------------------------------------------------
# L3: FFPoly & PolynomialRing
# ---------------------------------------------------------------------------


class FFPoly:
    """Univariate polynomial in ``GF(q)[x]`` with trimmed ascending coefficient array."""

    __slots__ = ("_ring", "_field", "_arr")

    def __init__(self, ring: PolynomialRing, coeffs_arr: np.ndarray) -> None:
        self._ring = ring
        self._field = ring._field
        self._arr = self._field.poly_trim(
            np.ascontiguousarray(coeffs_arr, dtype=self._field._dtype)
        )

    def parent(self) -> PolynomialRing:
        return self._ring

    def base_ring(self) -> FiniteField:
        return self._field

    def is_zero(self) -> bool:
        return self._arr.size == 0

    def degree(self) -> int:
        return int(self._arr.size) - 1

    def leading_coefficient(self) -> FiniteFieldElement:
        if self._arr.size == 0:
            return self._field.zero()
        return FiniteFieldElement(self._field, int(self._arr[-1]))

    def monic(self) -> FFPoly:
        if self._arr.size == 0:
            raise ZeroDivisionError("Zero polynomial cannot be made monic")
        lc = int(self._arr[-1])
        if lc == 1:
            return self
        inv_lc = self._field.inv_s(lc)
        return FFPoly(self._ring, self._field.poly_scale_arr(self._arr, inv_lc))

    def list(self) -> list[FiniteFieldElement]:
        f = self._field
        return [FiniteFieldElement(f, int(c)) for c in self._arr]

    def padded_list(self, n: int) -> list[FiniteFieldElement]:
        n_i = int(n)
        f = self._field
        if n_i <= 0:
            return []
        if self._arr.size >= n_i:
            return [FiniteFieldElement(f, int(c)) for c in self._arr[:n_i]]
        out = [FiniteFieldElement(f, int(c)) for c in self._arr]
        out.extend([f.zero()] * (n_i - self._arr.size))
        return out

    def shift(self, k: int) -> FFPoly:
        k_i = int(k)
        if self._arr.size == 0 or k_i == 0:
            return self
        if k_i > 0:
            out = np.zeros(self._arr.size + k_i, dtype=self._field._dtype)
            out[k_i:] = self._arr
            return FFPoly(self._ring, out)
        # Negative shift drops lower terms
        drop = -k_i
        if drop >= self._arr.size:
            return self._ring.zero()
        return FFPoly(self._ring, self._arr[drop:].copy())

    def derivative(self) -> FFPoly:
        if self._arr.size <= 1:
            return self._ring.zero()
        f = self._field
        out = np.zeros(self._arr.size - 1, dtype=f._dtype)
        if f._is_binary:
            for i in range(1, self._arr.size, 2):
                out[i - 1] = self._arr[i]
        else:
            p = f._p
            for i in range(1, self._arr.size):
                c_mod = i % p
                if c_mod:
                    out[i - 1] = f.mul_s(int(self._arr[i]), c_mod)
        return FFPoly(self._ring, out)

    def __getitem__(self, idx: int) -> FiniteFieldElement:
        i = int(idx)
        if i < 0 or i >= self._arr.size:
            return self._field.zero()
        return FiniteFieldElement(self._field, int(self._arr[i]))

    def __call__(self, x: Any) -> FiniteFieldElement:
        xv = self._field(x)._val
        return FiniteFieldElement(self._field, self._field.poly_eval_scalar(self._arr, xv))

    def eval_points(self, xs: Sequence[Any] | FFVector | np.ndarray) -> FFVector:
        """Vectorized evaluation of ``self`` at all points in ``xs``."""
        if isinstance(xs, np.ndarray):
            xs_arr = np.ascontiguousarray(xs, dtype=self._field._dtype)
        elif isinstance(xs, FFVector):
            if xs._field != self._field:
                raise TypeError("Field mismatch in eval_points")
            xs_arr = xs._arr
        else:
            xs_arr = self._field.coerce_to_arr(xs)
        return FFVector(self._field, self._field.poly_eval_many_arr(self._arr, xs_arr))

    def eval_points_arr(self, xs_arr: np.ndarray) -> np.ndarray:
        """Raw NumPy vectorized evaluation returning integer-encoded field elements."""
        return self._field.poly_eval_many_arr(self._arr, xs_arr)

    def _coerce_other(self, other: Any) -> np.ndarray | None:
        if isinstance(other, FFPoly):
            if other._field != self._field:
                return None
            return other._arr
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            v = self._field(other)._val
            if v == 0:
                return np.zeros(0, dtype=self._field._dtype)
            return np.array([v], dtype=self._field._dtype)
        return None

    def __add__(self, other: Any) -> FFPoly:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        return FFPoly(self._ring, self._field.poly_add_arr(self._arr, oa))

    def __radd__(self, other: Any) -> FFPoly:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        return FFPoly(self._ring, self._field.poly_add_arr(oa, self._arr))

    def __sub__(self, other: Any) -> FFPoly:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        return FFPoly(self._ring, self._field.poly_sub_arr(self._arr, oa))

    def __rsub__(self, other: Any) -> FFPoly:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        return FFPoly(self._ring, self._field.poly_sub_arr(oa, self._arr))

    def __neg__(self) -> FFPoly:
        return FFPoly(self._ring, self._field.neg_arr(self._arr))

    def __mul__(self, other: Any) -> FFPoly:
        if isinstance(other, FFPoly):
            if other._field != self._field:
                return NotImplemented
            return FFPoly(self._ring, self._field.poly_mul_arr(self._arr, other._arr))
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            v = self._field(other)._val
            return FFPoly(self._ring, self._field.poly_scale_arr(self._arr, v))
        return NotImplemented

    def __rmul__(self, other: Any) -> FFPoly:
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            v = self._field(other)._val
            return FFPoly(self._ring, self._field.poly_scale_arr(self._arr, v))
        return NotImplemented

    def __divmod__(self, other: Any) -> tuple[FFPoly, FFPoly]:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        q, r = self._field.poly_divmod_arr(self._arr, oa)
        return FFPoly(self._ring, q), FFPoly(self._ring, r)

    def quo_rem(self, other: Any) -> tuple[FFPoly, FFPoly]:
        return divmod(self, other)

    def __floordiv__(self, other: Any) -> FFPoly:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        q, _ = self._field.poly_divmod_arr(self._arr, oa)
        return FFPoly(self._ring, q)

    def __mod__(self, other: Any) -> FFPoly:
        oa = self._coerce_other(other)
        if oa is None:
            return NotImplemented
        _, r = self._field.poly_divmod_arr(self._arr, oa)
        return FFPoly(self._ring, r)

    def __pow__(self, exp: Any) -> FFPoly:
        if not isinstance(exp, (int, np.integer)) or isinstance(exp, bool):
            return NotImplemented
        e = int(exp)
        if e < 0:
            raise ValueError("Negative polynomial exponents are not supported")
        if e == 0:
            return self._ring.one()
        if self._arr.size == 0:
            return self._ring.zero()
        # Fast path for monomial x^e
        if self._arr.size == 2 and int(self._arr[0]) == 0 and int(self._arr[1]) == 1:
            out = np.zeros(e + 1, dtype=self._field._dtype)
            out[e] = 1
            return FFPoly(self._ring, out)
        res = np.array([1], dtype=self._field._dtype)
        base = self._arr
        while e > 0:
            if e & 1:
                res = self._field.poly_mul_arr(res, base)
            e >>= 1
            if e:
                base = self._field.poly_mul_arr(base, base)
        return FFPoly(self._ring, res)

    def gcd(self, other: FFPoly) -> FFPoly:
        if not isinstance(other, FFPoly) or other._field != self._field:
            raise TypeError("gcd requires another FFPoly over the same field")
        return FFPoly(self._ring, self._field.poly_gcd_arr(self._arr, other._arr))

    def xgcd(self, other: FFPoly) -> tuple[FFPoly, FFPoly, FFPoly]:
        if not isinstance(other, FFPoly) or other._field != self._field:
            raise TypeError("xgcd requires another FFPoly over the same field")
        g, s, t = self._field.poly_xgcd_arr(self._arr, other._arr)
        return FFPoly(self._ring, g), FFPoly(self._ring, s), FFPoly(self._ring, t)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FFPoly):
            return self._field == other._field and bool(np.array_equal(self._arr, other._arr))
        if isinstance(other, (FiniteFieldElement, int, np.integer)) and not isinstance(other, bool):
            v = self._field(other)._val
            if v == 0:
                return self._arr.size == 0
            return self._arr.size == 1 and int(self._arr[0]) == v
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self._field, self._arr.tobytes()))

    def __repr__(self) -> str:
        if self._arr.size == 0:
            return "0"
        name = self._ring._name
        f = self._field
        terms: list[str] = []
        for deg in range(self._arr.size - 1, -1, -1):
            cv = int(self._arr[deg])
            if cv == 0:
                continue
            cs = repr(FiniteFieldElement(f, cv))
            if deg == 0:
                terms.append(cs)
            elif deg == 1:
                terms.append(name if cv == 1 else f"({cs})*{name}" if "+" in cs else f"{cs}*{name}")
            else:
                terms.append(
                    f"{name}^{deg}"
                    if cv == 1
                    else (f"({cs})*{name}^{deg}" if "+" in cs else f"{cs}*{name}^{deg}")
                )
        return " + ".join(terms) if terms else "0"


class PolynomialRing:
    """Univariate polynomial ring ``GF(q)[x]``."""

    def __init__(
        self, field: FiniteField, names: str = "x", implementation: str | None = None
    ) -> None:
        if not isinstance(field, FiniteField):
            raise TypeError(f"Expected FiniteField, got {type(field).__name__}")
        self._field = field
        self._name = str(names)
        self._implementation = implementation
        self._zero = FFPoly(self, np.zeros(0, dtype=field._dtype))
        self._one = FFPoly(self, np.array([1], dtype=field._dtype))
        self._gen = FFPoly(self, np.array([0, 1], dtype=field._dtype))

    def base_ring(self) -> FiniteField:
        return self._field

    def base_field(self) -> FiniteField:
        return self._field

    def gen(self) -> FFPoly:
        return self._gen

    def zero(self) -> FFPoly:
        return self._zero

    def one(self) -> FFPoly:
        return self._one

    def __call__(self, x: Any) -> FFPoly:
        if isinstance(x, FFPoly):
            if x._field == self._field:
                return x if x._ring == self else FFPoly(self, x._arr)
            raise TypeError(f"Cannot coerce polynomial over {x._field!r} into {self!r}")
        if isinstance(x, (FiniteFieldElement, int, np.integer)) and not isinstance(x, bool):
            v = self._field(x)._val
            if v == 0:
                return self._zero
            return FFPoly(self, np.array([v], dtype=self._field._dtype))
        if isinstance(x, np.ndarray):
            return FFPoly(self, x.astype(self._field._dtype, copy=False))
        if isinstance(x, Iterable):
            arr = self._field.coerce_to_arr(x)
            return FFPoly(self, arr)
        raise TypeError(f"Cannot coerce {x!r} into {self!r}")

    def lagrange_polynomial(self, points: Sequence[tuple[Any, Any]]) -> FFPoly:
        """Unique interpolating polynomial of degree ``< len(points)`` through ``(x_i, y_i)``."""
        if not points:
            return self._zero
        xs = self._field.coerce_to_arr([pt[0] for pt in points])
        ys = self._field.coerce_to_arr([pt[1] for pt in points])
        return FFPoly(self, self._field.poly_interpolate_arr(xs, ys))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PolynomialRing)
            and self._field == other._field
            and self._name == other._name
        )

    def __hash__(self) -> int:
        return hash((self._field, self._name))

    def __repr__(self) -> str:
        return f"Univariate Polynomial Ring in {self._name} over {self._field!r}"


# ---------------------------------------------------------------------------
# L4: Plain Reed-Solomon Codec (Unified Sugiyama + Chien + Forney)
# ---------------------------------------------------------------------------


class RSCodec:
    """Plain ``[n, k, d = n - k + 1]`` Reed-Solomon code on arbitrary distinct eval points.

    Uses a unified syndrome + Sugiyama (extended Euclidean) + Chien + Forney
    error-and-erasure decoder with mandatory syndrome self-verification.
    Handles ``alpha == 0``, ``e == d - 1``, ``e >= d``, ``d == 2``, and ``n == k``.
    """

    def __init__(
        self,
        field: FiniteField,
        eval_points: Sequence[Any] | FFVector | np.ndarray,
        k: int,
    ) -> None:
        if not isinstance(field, FiniteField):
            raise TypeError(f"Expected FiniteField, got {type(field).__name__}")
        self._field = field
        if isinstance(eval_points, np.ndarray):
            alphas = np.ascontiguousarray(eval_points, dtype=field._dtype)
        elif isinstance(eval_points, FFVector):
            if eval_points._field != field:
                raise TypeError("Evaluation points field mismatch")
            alphas = eval_points._arr.copy()
        else:
            alphas = field.coerce_to_arr(eval_points)

        self._alphas_arr = np.ascontiguousarray(alphas, dtype=field._dtype)
        self._n = int(self._alphas_arr.size)
        self._k = int(k)
        if not (1 <= self._k <= self._n):
            raise ValueError(f"Invalid RS dimension k={self._k} for length n={self._n}")
        if len(set(int(a) for a in self._alphas_arr)) != self._n:
            raise ValueError("Evaluation points must be distinct")

        self._d = self._n - self._k + 1
        self._t = (self._d - 1) // 2

        self._etas_arr = self._compute_etas()
        self._H_arr = self._compute_pcm_arr()
        self._H_mat: FFMatrix | None = None
        self._G_mat: FFMatrix | None = None

        zero_locs = np.flatnonzero(self._alphas_arr == 0)
        self._zero_idx: int = int(zero_locs[0]) if zero_locs.size else -1

        inv_alphas = np.zeros(self._n, dtype=field._dtype)
        nz = self._alphas_arr != 0
        if np.any(nz):
            inv_alphas[nz] = field.inv_arr(self._alphas_arr[nz])
        self._inv_alphas_arr = np.ascontiguousarray(inv_alphas, dtype=field._dtype)

    def _compute_etas(self) -> np.ndarray:
        f = self._field
        n = self._n
        etas = np.ones(n, dtype=f._dtype)
        for j in range(n):
            aj = int(self._alphas_arr[j])
            diffs = f.sub_arr(np.full(n, aj, dtype=f._dtype), self._alphas_arr)
            diffs[j] = 1
            prod_val = 1
            for dv in diffs:
                prod_val = f.mul_s(prod_val, int(dv))
            etas[j] = f.inv_s(prod_val)
        return np.ascontiguousarray(etas, dtype=f._dtype)

    def _compute_pcm_arr(self) -> np.ndarray:
        f = self._field
        m = self._n - self._k
        if m == 0:
            return np.zeros((0, self._n), dtype=f._dtype)
        H = np.zeros((m, self._n), dtype=f._dtype)
        H[0] = self._etas_arr
        for i in range(1, m):
            H[i] = f.mul_arr(H[i - 1], self._alphas_arr)
        return np.ascontiguousarray(H, dtype=f._dtype)

    def base_field(self) -> FiniteField:
        return self._field

    def length(self) -> int:
        return self._n

    def dimension(self) -> int:
        return self._k

    def minimum_distance(self) -> int:
        return self._d

    def evaluation_points(self) -> list[FiniteFieldElement]:
        f = self._field
        return [FiniteFieldElement(f, int(a)) for a in self._alphas_arr]

    def parity_check_matrix(self) -> FFMatrix:
        if self._H_mat is None:
            self._H_mat = FFMatrix(self._field, self._H_arr.copy(), immutable=True)
        return self._H_mat

    def generator_matrix(self) -> FFMatrix:
        if self._G_mat is None:
            f = self._field
            G = np.zeros((self._k, self._n), dtype=f._dtype)
            G[0] = 1
            for i in range(1, self._k):
                G[i] = f.mul_arr(G[i - 1], self._alphas_arr)
            self._G_mat = FFMatrix(f, G, immutable=True)
        return self._G_mat

    def _extract_arr(self, word: FFVector | np.ndarray | Sequence[Any]) -> tuple[np.ndarray, bool]:
        if isinstance(word, FFVector):
            if word._field != self._field:
                raise ValueError("Received word field does not match RSCodec field")
            return word._arr, True
        if isinstance(word, np.ndarray):
            return np.ascontiguousarray(word, dtype=self._field._dtype), False
        return self._field.coerce_to_arr(word), False

    def syndrome(self, r_arr: FFVector | np.ndarray | Sequence[Any]) -> Any:
        arr, was_vec = self._extract_arr(r_arr)
        if arr.size != self._n:
            raise ValueError(f"Expected vector of length {self._n}, got {arr.size}")
        if self._n == self._k:
            out = np.zeros(0, dtype=self._field._dtype)
        else:
            out = self._field.matvec_arr(self._H_arr, arr)
        return FFVector(self._field, out) if was_vec else out

    def has_zero_syndrome(self, r_arr: FFVector | np.ndarray | Sequence[Any]) -> bool:
        arr, _ = self._extract_arr(r_arr)
        if arr.size != self._n:
            raise ValueError(f"Expected vector of length {self._n}, got {arr.size}")
        if self._n == self._k:
            return True
        return not bool(np.any(self._field.matvec_arr(self._H_arr, arr)))

    def _decode_cy(self, r_arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        f = self._field
        assert f.exp_table is not None and f.log_u16 is not None
        r_c = np.ascontiguousarray(r_arr, dtype=np.uint16)
        if mask is not None:
            mask_c = np.ascontiguousarray(mask, dtype=np.uint8)
            out = _ff_cy.rs_decode_ee_cy(
                r_c,
                mask_c,
                self._alphas_arr,
                self._etas_arr,
                self._k,
                f.exp_table,
                f.log_u16,
                f._order,
            )
        else:
            out = _ff_cy.rs_decode_syndrome_cy(
                r_c,
                self._alphas_arr,
                self._etas_arr,
                self._k,
                f.exp_table,
                f.log_u16,
                f._order,
            )
        if out is None:
            raise DecodingError("RS decoding failed")
        return out

    def decode_errors(self, r_arr: FFVector | np.ndarray | Sequence[Any]) -> Any:
        arr, was_vec = self._extract_arr(r_arr)
        if arr.size != self._n:
            raise ValueError(f"Expected vector of length {self._n}, got {arr.size}")
        if self._k == self._n:
            out = arr.copy()
            return FFVector(self._field, out) if was_vec else out
        if self._field._can_use_cy and hasattr(_ff_cy, "rs_decode_syndrome_cy"):
            out = self._decode_cy(arr, None)
            return FFVector(self._field, out) if was_vec else out
        S = self._field.matvec_arr(self._H_arr, arr)
        if not np.any(S):
            out = arr.copy()
            return FFVector(self._field, out) if was_vec else out
        if self._t == 0:
            raise DecodingError("Unrecoverable error (t = 0)")
        out = self._decode_from_syndromes(arr, S, erasure_indices=())
        return FFVector(self._field, out) if was_vec else out

    def decode_errors_erasures(
        self,
        r_arr: FFVector | np.ndarray | Sequence[Any],
        era_mask: Sequence[Any] | FFVector | np.ndarray,
    ) -> Any:
        arr, was_vec = self._extract_arr(r_arr)
        if isinstance(era_mask, FFVector):
            mask = (era_mask._arr != 0).astype(np.uint8)
        elif isinstance(era_mask, np.ndarray):
            mask = (era_mask != 0).astype(np.uint8)
        else:
            mask = np.array([1 if int(x) != 0 else 0 for x in era_mask], dtype=np.uint8)
        if arr.size != self._n or mask.size != self._n:
            raise ValueError(
                f"Length mismatch: expected {self._n}, got word={arr.size}, mask={mask.size}"
            )
        erased = np.flatnonzero(mask)
        e_cnt = int(erased.size)
        if e_cnt >= self._d:
            raise DecodingError(f"Too many erasures ({e_cnt} >= d={self._d})")
        if e_cnt == 0:
            return self.decode_errors(r_arr)
        if self._k == self._n:
            # e_cnt == 0 handled above; e_cnt >= 1 >= d=1 handled above
            out = arr.copy()
            return FFVector(self._field, out) if was_vec else out
        if self._field._can_use_cy and (
            hasattr(_ff_cy, "rs_decode_ee_cy") or hasattr(_ff_cy, "rs_decode_syndrome_cy")
        ):
            out = self._decode_cy(arr, mask)
            return FFVector(self._field, out) if was_vec else out

        r_tilde = arr.copy()
        r_tilde[erased] = 0
        S = self._field.matvec_arr(self._H_arr, r_tilde)
        if not np.any(S):
            return FFVector(self._field, r_tilde) if was_vec else r_tilde
        out = self._decode_from_syndromes(r_tilde, S, erasure_indices=tuple(int(i) for i in erased))
        return FFVector(self._field, out) if was_vec else out

    def _decode_from_syndromes(
        self,
        r_tilde: np.ndarray,
        S: np.ndarray,
        erasure_indices: tuple[int, ...],
    ) -> np.ndarray:
        f = self._field
        m = self._d - 1
        e_cnt = len(erasure_indices)
        T = (m - e_cnt) // 2

        erasure_set = set(erasure_indices)
        u_star = [j for j in erasure_indices if j != self._zero_idx]
        gamma = np.array([1], dtype=f._dtype)
        for j in u_star:
            aj = int(self._alphas_arr[j])
            neg_aj = f.neg_s(aj)
            factor = np.array([1, neg_aj], dtype=f._dtype)
            gamma = f.poly_mul_arr(gamma, factor)

        S_poly = f.poly_trim(S)
        xi = f.poly_mul_arr(gamma, S_poly)
        if xi.size > m:
            xi = f.poly_trim(xi[:m])

        r0 = np.zeros(m + 1, dtype=f._dtype)
        r0[m] = 1
        r1 = xi
        t0 = np.zeros(0, dtype=f._dtype)
        t1 = np.array([1], dtype=f._dtype)
        stop_deg = T + e_cnt
        while r1.size - 1 >= stop_deg:
            q_poly, r2 = f.poly_divmod_arr(r0, r1)
            t2 = f.poly_sub_arr(t0, f.poly_mul_arr(q_poly, t1))
            r0, r1 = r1, r2
            t0, t1 = t1, t2

        if t1.size == 0 or int(t1[0]) == 0:
            raise DecodingError("Degenerate error locator polynomial")
        deg_lam = t1.size - 1
        if deg_lam > T:
            raise DecodingError("Error locator degree exceeds decoding radius")

        inv_t10 = f.inv_s(int(t1[0]))
        lam = f.poly_scale_arr(t1, inv_t10)
        omega = f.poly_scale_arr(r1, inv_t10)

        e_star: list[int] = []
        if deg_lam > 0:
            vals = f.poly_eval_many_arr(lam, self._inv_alphas_arr)
            cand = np.flatnonzero(vals == 0)
            for idx in cand:
                j = int(idx)
                if j != self._zero_idx and j not in erasure_set:
                    e_star.append(j)
            if len(e_star) != deg_lam:
                raise DecodingError("Error locator does not split over valid non-erased positions")

        psi = f.poly_mul_arr(gamma, lam) if u_star else lam
        psi_deriv = np.zeros(max(psi.size - 1, 0), dtype=f._dtype)
        if f._is_binary:
            for i in range(1, psi.size, 2):
                psi_deriv[i - 1] = psi[i]
        else:
            for i in range(1, psi.size):
                c_mod = i % f._p
                if c_mod:
                    psi_deriv[i - 1] = f.mul_s(int(psi[i]), c_mod)
        psi_deriv = f.poly_trim(psi_deriv)

        err_vals: dict[int, int] = {}
        active_nz = u_star + e_star
        if active_nz:
            idx_arr = np.array(active_nz, dtype=np.int32)
            x_invs = self._inv_alphas_arr[idx_arr]
            om_vals = f.poly_eval_many_arr(omega, x_invs)
            pd_vals = f.poly_eval_many_arr(psi_deriv, x_invs)
            for k_i, j in enumerate(active_nz):
                den = f.mul_s(int(self._etas_arr[j]), int(pd_vals[k_i]))
                if den == 0:
                    raise DecodingError("Zero denominator in Forney formula")
                num = f.mul_s(int(self._alphas_arr[j]), int(om_vals[k_i]))
                ej = f.div_s(num, den)
                if not f._is_binary:
                    ej = f.neg_s(ej)
                err_vals[j] = ej

        if self._zero_idx >= 0:
            j0 = self._zero_idx
            acc = 0
            for j, ej in err_vals.items():
                acc = f.add_s(acc, f.mul_s(ej, int(self._etas_arr[j])))
            rem0 = f.sub_s(int(S[0]), acc)
            ej0 = f.div_s(rem0, int(self._etas_arr[j0]))
            if ej0 != 0:
                if j0 not in erasure_set and (len(e_star) + 1 > T):
                    raise DecodingError("Total unmarked errors exceed decoding radius")
                err_vals[j0] = ej0

        unmarked_nz = sum(1 for j, ej in err_vals.items() if j not in erasure_set and ej != 0)
        if unmarked_nz > T:
            raise DecodingError("Unmarked error weight exceeds radius")

        out = r_tilde.copy()
        for j, ej in err_vals.items():
            if ej:
                out[j] = f.sub_s(int(out[j]), ej)

        # Mandatory syndrome self-verification
        if err_vals:
            act_idx = np.array(list(err_vals.keys()), dtype=np.int32)
            act_e = np.array([err_vals[int(j)] for j in act_idx], dtype=f._dtype)
            S_check = f.matvec_arr(self._H_arr[:, act_idx], act_e)
            if not np.array_equal(S_check, S):
                raise DecodingError("Candidate codeword failed syndrome verification")
        return out


# ---------------------------------------------------------------------------
# L5: Utilities (random_error_vector, balanced product tree)
# ---------------------------------------------------------------------------


def random_error_vector(
    n: int,
    field: FiniteField,
    positions: Sequence[int],
    rng: np.random.Generator | None = None,
) -> FFVector:
    """Return length-``n`` vector over ``field`` with random non-zero entries at ``positions``."""
    r = _DEFAULT_RNG if rng is None else rng
    arr = np.zeros(int(n), dtype=field._dtype)
    pos_list = [int(p) for p in positions]
    if pos_list:
        if field._order < (1 << 62):
            arr[pos_list] = r.integers(1, field._order, size=len(pos_list), dtype=np.int64).astype(
                field._dtype
            )
        else:
            for p in pos_list:
                arr[p] = field.random_nonzero(r)._val
    return FFVector(field, arr)


def prod(factors: Iterable[Any], start: Any = None) -> Any:
    """Balanced binary product tree (``O(M(n) log n)`` for polynomial factors)."""
    items = list(factors)
    if not items:
        return 1 if start is None else start
    while len(items) > 1:
        nxt: list[Any] = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                nxt.append(items[i] * items[i + 1])
            else:
                nxt.append(items[i])
        items = nxt
    res = items[0]
    return res if start is None else start * res


__all__ = [
    "HAS_CYTHON",
    "DecodingError",
    "FFMatrix",
    "FFPoly",
    "FFVector",
    "FiniteField",
    "FiniteFieldElement",
    "GF",
    "MatrixSpace",
    "PolynomialRing",
    "RSCodec",
    "VectorSpace",
    "VectorSubspace",
    "divisors",
    "get_rng",
    "matrix",
    "prod",
    "random_error_vector",
    "set_random_seed",
    "vector",
    "zero_vector",
]
