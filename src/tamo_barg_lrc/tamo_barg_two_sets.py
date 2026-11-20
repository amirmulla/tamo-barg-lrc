"""Two-set Tamo-Barg locally recoverable codes, vector encoder, and decoders."""

import numbers
from collections.abc import Sequence
from copy import copy
from functools import lru_cache
from typing import Any, Literal

import networkx as nx
import numpy as np

from tamo_barg_lrc._ff import (
    GF,
    MatrixSpace,
    PolynomialRing,
    RSCodec,
    VectorSpace,
    matrix,
)
from tamo_barg_lrc.coding_base import (
    AbstractLinearCode,
    Decoder,
    DecodingError,
    Encoder,
    cached_method,
    cartesian_product,
)
from tamo_barg_lrc.expander import create_bipartite_coset_graph
from tamo_barg_lrc.polynomials import (
    build_algebra_basis,
    build_encoding_basis,
    build_good_polynomial,
    evaluate_message_polynomial,
    intersect_encoding_bases,
)
from tamo_barg_lrc.subgroups import (
    SubgroupType,
    is_closed_under_addition,
    list_canonical_additive_subgroups,
    list_multiplicative_subgroups,
    list_subgroup_cosets,
    shift_additive_subgroup,
    sort_field_elements,
)

__all__ = [
    "TamoBargCodeTwoSets",
    "TamoBargGlobalDecoder",
    "TamoBargGlobalEEDecoder",
    "TamoBargIterativeDecoder",
    "TamoBargIterativeEEDecoder",
    "TamoBargTwoSetsVectorEncoder",
    "TamoBargTwoStepDecoder",
    "TamoBargTwoStepEEDecoder",
]

_POPCOUNT_LUT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _is_int_like(value: Any) -> bool:
    """Return True if ``value`` is an integer (not ``bool``)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _int_log_base(size: int, base: int) -> int:
    """Compute exact integer ``log_{base}(size)`` for a prime-power subgroup size."""
    exp = 0
    rem = size
    while rem > 1:
        if rem % base != 0:
            raise ValueError(f"Subgroup size {size} is not a power of characteristic {base}")
        rem //= base
        exp += 1
    return exp


@lru_cache(maxsize=16)
def _gf2m_tables(field: Any) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Build and cache ``(exp_tbl, log_tbl, int_to_elem)`` for ``GF(2^m)`` (``q <= 65536``)."""
    q = int(field.cardinality())
    dtype = np.uint8 if q <= 256 else np.uint16
    exp_tbl = np.zeros(2 * (q - 1), dtype=dtype)
    log_tbl = np.full(q, -1, dtype=np.int32)
    int_to_elem: list[Any] = [field.zero()] * q
    if q == 2:
        exp_tbl[0] = 1
        exp_tbl[1] = 1
        log_tbl[1] = 0
        int_to_elem[1] = field.one()
        return exp_tbl, log_tbl, int_to_elem

    gen = field.multiplicative_generator()
    cur = field.one()
    for i in range(q - 1):
        val = int(cur.to_integer())
        exp_tbl[i] = val
        exp_tbl[i + (q - 1)] = val
        log_tbl[val] = i
        int_to_elem[val] = cur
        cur *= gen
    return exp_tbl, log_tbl, int_to_elem


def _gf2m_min_distance(gen_mat: Any, field: Any, k: int, n: int, stop_bound: int) -> int:
    """Compute exact minimum Hamming distance over ``GF(2^m)`` (``q <= 65536``) via NumPy LUTs."""
    q = int(field.cardinality())
    exp_tbl, log_tbl, _ = _gf2m_tables(field)
    dtype = np.uint8 if q <= 256 else np.uint16
    if hasattr(gen_mat, "_arr"):
        g_int = gen_mat._arr.astype(dtype, copy=False)
    elif q == 2:
        g_int = np.array([[int(gen_mat[i, j]) for j in range(n)] for i in range(k)], dtype=dtype)
    else:
        g_int = np.array(
            [[int(gen_mat[i, j].to_integer()) for j in range(n)] for i in range(k)],
            dtype=dtype,
        )

    # Check the highest-degree basis row first (tail_dim == 0) before allocating LUT tables.
    min_wt = n
    wt_last = int(np.count_nonzero(g_int[k - 1]))
    if 0 < wt_last < min_wt:
        min_wt = wt_last
        if min_wt <= stop_bound:
            return min_wt
    if k == 1:
        return min_wt

    # Precompute scalar multiples mul_rows[r, s] = s * G[r] when table fits comfortably in RAM.
    if k * q * n <= 16_000_000:
        s_logs = log_tbl[np.arange(1, q)]
        mul_rows = np.zeros((k, q, n), dtype=dtype)
        for r_idx in range(k):
            nz = g_int[r_idx] != 0
            if np.any(nz):
                g_logs = log_tbl[g_int[r_idx, nz]]
                mul_rows[r_idx][1:, nz] = exp_tbl[s_logs[:, None] + g_logs[None, :]]

        # Bit-sliced uint64 path for n <= 64 and q <= 256 with k >= 3 (batches 2 inner coords).
        if n <= 64 and q <= 256 and k >= 3:
            m_bits = int(field.degree())
            bit_shifts = np.uint64(1) << np.arange(n, dtype=np.uint64)
            mul_bits = np.zeros((k, q, m_bits), dtype=np.uint64)
            for b in range(m_bits):
                plane = ((mul_rows >> b) & 1).astype(np.uint64)
                mul_bits[:, :, b] = (plane * bit_shifts[None, None, :]).sum(axis=2, dtype=np.uint64)

            def _min_popcount(planes: np.ndarray) -> int:
                nz_masks = np.bitwise_or.reduce(planes, axis=1)
                as_bytes = nz_masks.view(np.uint8).reshape(-1, 8)
                return int(_POPCOUNT_LUT8[as_bytes].sum(axis=1).min())

            pair_planes = (mul_bits[k - 2, :, None, :] ^ mul_bits[k - 1, None, :, :]).reshape(
                q * q, m_bits
            )

            for lead in range(k - 2, -1, -1):
                lead_bits = mul_bits[lead, 1]
                tail_dim = k - lead - 1
                if tail_dim == 1:
                    mw = _min_popcount(lead_bits[None, :] ^ mul_bits[lead + 1])
                    if 0 < mw < min_wt:
                        min_wt = mw
                        if min_wt <= stop_bound:
                            return min_wt
                else:
                    outer_dim = tail_dim - 2
                    if outer_dim == 0:
                        mw = _min_popcount(lead_bits[None, :] ^ pair_planes)
                        if 0 < mw < min_wt:
                            min_wt = mw
                            if min_wt <= stop_bound:
                                return min_wt
                    else:
                        digits = [0] * outer_dim
                        acc = lead_bits.copy()
                        for _ in range(q**outer_dim):
                            mw = _min_popcount(acc[None, :] ^ pair_planes)
                            if 0 < mw < min_wt:
                                min_wt = mw
                                if min_wt <= stop_bound:
                                    return min_wt
                            for pos in range(outer_dim - 1, -1, -1):
                                old_d = digits[pos]
                                new_d = old_d + 1
                                row_idx = lead + 1 + pos
                                if new_d < q:
                                    acc ^= mul_bits[row_idx, old_d] ^ mul_bits[row_idx, new_d]
                                    digits[pos] = new_d
                                    break
                                acc ^= mul_bits[row_idx, old_d]
                                digits[pos] = 0
            return min_wt

        # General uint8/uint16 batched XOR over the innermost coordinate.
        last_mults = mul_rows[k - 1]
        for lead in range(k - 2, -1, -1):
            lead_row = g_int[lead]
            tail_dim = k - lead - 1
            if tail_dim == 1:
                mw = int(np.count_nonzero(lead_row[None, :] ^ last_mults, axis=1).min())
                if 0 < mw < min_wt:
                    min_wt = mw
                    if min_wt <= stop_bound:
                        return min_wt
            else:
                outer_dim = tail_dim - 1
                digits = [0] * outer_dim
                acc = lead_row.copy()
                for _ in range(q**outer_dim):
                    mw = int(np.count_nonzero(acc[None, :] ^ last_mults, axis=1).min())
                    if 0 < mw < min_wt:
                        min_wt = mw
                        if min_wt <= stop_bound:
                            return min_wt
                    for pos in range(outer_dim - 1, -1, -1):
                        old_d = digits[pos]
                        new_d = old_d + 1
                        row_idx = lead + 1 + pos
                        if new_d < q:
                            acc ^= mul_rows[row_idx, old_d] ^ mul_rows[row_idx, new_d]
                            digits[pos] = new_d
                            break
                        acc ^= mul_rows[row_idx, old_d]
                        digits[pos] = 0
        return min_wt

    # Chunked fallback for very large GF(2^m) (e.g., q = 65536) where full mul_rows is huge.
    chunk_size = 256
    last_row = g_int[k - 1]
    last_nz = last_row != 0
    last_g_logs = log_tbl[last_row[last_nz]]
    row_nz = [g_int[r] != 0 for r in range(k)]
    row_g_logs = [log_tbl[g_int[r, row_nz[r]]] for r in range(k)]

    def _scale_row(row_idx: int, scalar: int) -> np.ndarray:
        out = np.zeros(n, dtype=dtype)
        if scalar != 0 and np.any(row_nz[row_idx]):
            out[row_nz[row_idx]] = exp_tbl[log_tbl[scalar] + row_g_logs[row_idx]]
        return out

    def _min_wt_with_last(acc_row: np.ndarray) -> int:
        best = int(np.count_nonzero(acc_row))
        if not np.any(last_nz):
            return best
        for start in range(1, q, chunk_size):
            end = min(start + chunk_size, q)
            s_logs_c = log_tbl[np.arange(start, end)]
            batch = np.broadcast_to(acc_row, (end - start, n)).copy()
            batch[:, last_nz] ^= exp_tbl[s_logs_c[:, None] + last_g_logs[None, :]]
            mw_c = int(np.count_nonzero(batch, axis=1).min())
            if 0 < mw_c < best:
                best = mw_c
                if best <= stop_bound:
                    return best
        return best

    for lead in range(k - 2, -1, -1):
        lead_row = g_int[lead]
        tail_dim = k - lead - 1
        if tail_dim == 1:
            mw = _min_wt_with_last(lead_row)
            if 0 < mw < min_wt:
                min_wt = mw
                if min_wt <= stop_bound:
                    return min_wt
        else:
            outer_dim = tail_dim - 1
            digits = [0] * outer_dim
            acc = lead_row.copy()
            for _ in range(q**outer_dim):
                mw = _min_wt_with_last(acc)
                if 0 < mw < min_wt:
                    min_wt = mw
                    if min_wt <= stop_bound:
                        return min_wt
                for pos in range(outer_dim - 1, -1, -1):
                    old_d = digits[pos]
                    new_d = old_d + 1
                    row_idx = lead + 1 + pos
                    if new_d < q:
                        acc ^= _scale_row(row_idx, old_d) ^ _scale_row(row_idx, new_d)
                        digits[pos] = new_d
                        break
                    acc ^= _scale_row(row_idx, old_d)
                    digits[pos] = 0
    return min_wt


def _partition_is_syndrome_clean(
    part_idx: int,
    work_arr: np.ndarray,
    evalpts_idx_arr: list[list[np.ndarray]],
    local_codes: list[list[RSCodec]],
) -> bool:
    """Return True if every coset in partition ``part_idx`` has zero local syndrome."""
    codes = local_codes[part_idx]
    for j, idxs in enumerate(evalpts_idx_arr[part_idx]):
        if not codes[j].has_zero_syndrome(work_arr[idxs]):
            return False
    return True


class TamoBargCodeTwoSets(AbstractLinearCode):
    """Optimal locally recoverable Tamo-Barg code with two disjoint-repair-set partitions.

    Constructs evaluation codes over ``GF(q)`` partitioned simultaneously by two
    (additive or multiplicative) subgroups ``H_0`` and ``H_1``, enabling iterative,
    global, and two-step error and erasure-and-error decoding.
    """

    _registered_encoders: dict[str, type[Encoder]] = {}
    _registered_decoders: dict[str, type[Decoder]] = {}

    def __init__(
        self,
        base_field: Any,
        length: int,
        dimension: int | None,
        locality: Sequence[int],
        local_minimum_distance: Sequence[int] | int = (2, 2),
        sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"),
        shift_add: bool = False,
        subgroup: Sequence[Sequence[Any] | None] | None = None,
    ) -> None:
        """Initialize a two-set Tamo-Barg LRC.

        Args:
            base_field: Finite field ``GF(q)`` over which the code is defined.
            length: Codeword length ``n``.
            dimension: Message dimension ``k`` (``1 <= k <= max_dimension()``), or
                ``None`` to default to ``max_dimension()`` without a probe code.
            locality: Two-element sequence ``[r_0, r_1]`` of partition localities.
            local_minimum_distance: Scalar ``d`` (expanded to ``[d, d]``) or
                two-element sequence ``[d_0, d_1]`` of local minimum distances (``>= 2``).
            sub_group_type: Two-element sequence of ``"add"`` or ``"mult"`` specifying
                each partition's subgroup structure.
            shift_add: Whether to scale default additive subgroups by a primitive
                element power to alter coset alignment.
            subgroup: Optional two-element sequence ``[sg_0, sg_1]`` overriding default
                subgroup selection where entries are non-``None``.

        Raises:
            ValueError: If ``base_field`` is not finite, parameters have invalid shapes
                or ranges, required subgroups do not exist, coset sizes do not divide
                the partition domain, or ``dimension`` exceeds ``max_dimension()``.
        """
        if not hasattr(base_field, "is_finite") or not base_field.is_finite():
            raise ValueError("base_field must be a finite field")

        if not _is_int_like(length) or int(length) < 1:
            raise ValueError(f"length must be a positive integer, got {length!r}")
        if dimension is not None and (not _is_int_like(dimension) or int(dimension) < 1):
            raise ValueError(f"dimension must be a positive integer or None, got {dimension!r}")
        length_int = int(length)
        dim_int: int | None = None if dimension is None else int(dimension)

        if not isinstance(locality, (list, tuple)) or len(locality) != 2:
            raise ValueError("locality must be a 2-element sequence [r1, r2]")
        if not (_is_int_like(locality[0]) and _is_int_like(locality[1])):
            raise ValueError(f"locality elements must be integers, got {locality!r}")
        self._num_of_sets: int = 2
        self._locality: list[int] = [int(locality[0]), int(locality[1])]
        if any(r_val < 1 for r_val in self._locality):
            raise ValueError(f"locality values must be >= 1, got {self._locality}")

        if _is_int_like(local_minimum_distance):
            d_val = int(local_minimum_distance)
            self._local_minimum_distance: list[int] = [d_val, d_val]
        elif isinstance(local_minimum_distance, (list, tuple)) and len(local_minimum_distance) == 2:
            if not (
                _is_int_like(local_minimum_distance[0]) and _is_int_like(local_minimum_distance[1])
            ):
                raise ValueError(
                    "local_minimum_distance elements must be integers, "
                    f"got {local_minimum_distance!r}"
                )
            self._local_minimum_distance = [
                int(local_minimum_distance[0]),
                int(local_minimum_distance[1]),
            ]
        else:
            raise ValueError(
                "local_minimum_distance must be an integer or a 2-element sequence [d1, d2]"
            )
        if any(d_val < 2 for d_val in self._local_minimum_distance):
            raise ValueError(
                f"local_minimum_distance values must be >= 2, got {self._local_minimum_distance}"
            )

        if not isinstance(sub_group_type, (list, tuple)) or len(sub_group_type) != 2:
            raise ValueError("sub_group_type must be a 2-element sequence of 'add' or 'mult'")
        self._sub_group_type: list[str] = [str(sub_group_type[0]), str(sub_group_type[1])]
        if any(sgt not in ("add", "mult") for sgt in self._sub_group_type):
            raise ValueError(
                f"sub_group_type elements must be 'add' or 'mult', got {self._sub_group_type!r}"
            )

        if subgroup is None:
            norm_subgroup: list[Sequence[Any] | None] = [None, None]
        elif isinstance(subgroup, (list, tuple)) and len(subgroup) == 2:
            norm_subgroup = [subgroup[0], subgroup[1]]
        else:
            raise ValueError("subgroup must be None or a 2-element sequence [sg0, sg1]")

        coset_sizes = [self._locality[i] + self._local_minimum_distance[i] - 1 for i in range(2)]
        mult_subgroups = list_multiplicative_subgroups(base_field)
        add_subgroups = list_canonical_additive_subgroups(base_field)
        p = int(base_field.characteristic())

        self._sub_group: list[list[Any]] = []
        for i in (0, 1):
            target_size = coset_sizes[i]
            sgt = self._sub_group_type[i]
            custom_sg = norm_subgroup[i]

            if custom_sg is not None:
                if len(custom_sg) != target_size:
                    raise ValueError(
                        f"Custom subgroup {i} has size {len(custom_sg)}, expected {target_size}"
                    )
                coerced_sg = sort_field_elements(base_field(h) for h in custom_sg)
                if len(set(coerced_sg)) != target_size:
                    raise ValueError(f"Custom subgroup {i} contains duplicate elements")
                if sgt == "add":
                    if not is_closed_under_addition(coerced_sg):
                        raise ValueError(
                            f"Custom subgroup {i} is not closed under addition over {base_field}"
                        )
                else:
                    if (
                        target_size not in mult_subgroups
                        or coerced_sg != mult_subgroups[target_size]
                    ):
                        raise ValueError(
                            f"Custom subgroup {i} is not a valid multiplicative subgroup "
                            f"of order {target_size} over {base_field}"
                        )
                self._sub_group.append(coerced_sg)
            elif sgt == "mult":
                if target_size not in mult_subgroups:
                    raise ValueError(
                        f"Cannot find multiplicative subgroup of size {target_size} over "
                        f"{base_field}; available sizes: {sorted(mult_subgroups)}"
                    )
                self._sub_group.append(mult_subgroups[target_size])
            else:
                if target_size not in add_subgroups:
                    raise ValueError(
                        f"Cannot find additive subgroup of size {target_size} over "
                        f"{base_field}; available sizes: {sorted(add_subgroups)}"
                    )
                sg = add_subgroups[target_size]
                if self._sub_group_type == ["add", "add"]:
                    if i == 0:
                        if shift_add:
                            sg = shift_additive_subgroup(base_field, sg, 1)
                    else:
                        t_0 = _int_log_base(len(self._sub_group[0]), p)
                        shift_exp = (1 + t_0) if shift_add else t_0
                        sg = shift_additive_subgroup(base_field, sg, shift_exp)
                elif shift_add:
                    shift_exp = _int_log_base(len(sg), p)
                    sg = shift_additive_subgroup(base_field, sg, shift_exp)
                self._sub_group.append(sg)

        q = int(base_field.cardinality())
        has_add = "add" in self._sub_group_type
        self._partition_size: list[int] = []
        self._partition: list[list[list[Any]]] = []

        for i in (0, 1):
            sgt_lit: SubgroupType = "add" if self._sub_group_type[i] == "add" else "mult"
            c_size = coset_sizes[i]
            if sgt_lit == "add":
                if length_int > q or length_int % c_size != 0:
                    raise ValueError(
                        f"Code length ({length_int}) must be <= {q} and divisible by "
                        f"additive coset size ({c_size}) for partition {i}"
                    )
                p_size = length_int // c_size
            elif has_add and length_int == q:
                if (q - 1) % c_size != 0:
                    raise ValueError(
                        f"Field multiplicative order ({q - 1}) must be divisible by "
                        f"multiplicative coset size ({c_size}) for mixed code of length {q}"
                    )
                p_size = (q - 1) // c_size
            else:
                if length_int > q - 1 or length_int % c_size != 0:
                    raise ValueError(
                        f"Code length ({length_int}) must be <= {q - 1} and divisible by "
                        f"multiplicative coset size ({c_size}) for partition {i}"
                    )
                p_size = length_int // c_size

            self._partition_size.append(p_size)
            cosets = list_subgroup_cosets(base_field, self._sub_group[i], sgt_lit)[:p_size]
            self._partition.append(cosets)

        all_pts = {pt for part in self._partition for coset in part for pt in coset}
        self._evaluation_points: list[Any] = sort_field_elements(all_pts)
        if len(self._evaluation_points) != length_int:
            raise ValueError(
                f"Union of partition cosets has {len(self._evaluation_points)} evaluation points, "
                f"expected code length {length_int}"
            )

        pt_to_idx = {pt: idx for idx, pt in enumerate(self._evaluation_points)}
        self._evaluation_points_idx: list[list[list[int]]] = [
            [[pt_to_idx[h] for h in coset] for coset in part] for part in self._partition
        ]

        self._bipartite_graph_cache: tuple[nx.MultiGraph, set[int], set[int]] | None = None
        self._hash_cache: int | None = None
        self._design_distance_cache: int | None = None
        self._default_encoder_cache: Encoder | None = None
        self._encoder_cache: dict[tuple[Any, ...], Encoder] = {}

        self._registered_encoders = self._registered_encoders.copy()
        self._registered_decoders = self._registered_decoders.copy()
        super().__init__(base_field, length_int, "VectorEncoder", "IterativeDecoder")
        self._dimension = dim_int

        max_dim = self.max_dimension()
        if dim_int is None:
            self._dimension = max_dim
        elif dim_int < 1 or dim_int > max_dim:
            raise ValueError(f"Code dimension ({dim_int}) must satisfy 1 <= dimension <= {max_dim}")

    def dimension(self) -> int:
        """Return the message dimension ``k`` of the code."""
        return self._dimension  # type: ignore[return-value]

    def encoder(self, encoder_name: str | None = None, *args: Any, **kwargs: Any) -> Encoder:
        """Return a cached encoder, unifying ``encoder()`` and ``encoder('VectorEncoder')``."""
        if not self._default_encoder_name:
            raise NotImplementedError("No encoder implemented for this code.")
        name = self._default_encoder_name if encoder_name is None else encoder_name
        if not args and not kwargs and name == self._default_encoder_name:
            if self._default_encoder_cache is None:
                enc_cls = self._registered_encoders[name]
                self._default_encoder_cache = enc_cls(self)
            return self._default_encoder_cache
        key = (name, args, tuple(sorted(kwargs.items())))
        if key not in self._encoder_cache:
            if name not in self._registered_encoders:
                raise ValueError(
                    f"There is no Encoder named '{name}'. The known Encoders are: "
                    f"{self.encoders_available()}"
                )
            enc_cls = self._registered_encoders[name]
            self._encoder_cache[key] = enc_cls(self, *args, **kwargs)
        return self._encoder_cache[key]

    def __eq__(self, other: object) -> bool:
        """Return True if ``other`` is a two-set Tamo-Barg code with identical parameters."""
        return (
            isinstance(other, TamoBargCodeTwoSets)
            and self.base_field() == other.base_field()
            and self.length() == other.length()
            and self.dimension() == other.dimension()
            and self.locality() == other.locality()
            and self.local_minimum_distance() == other.local_minimum_distance()
            and self.sub_group() == other.sub_group()
        )

    def __hash__(self) -> int:
        """Return a deterministic hash consistent with ``__eq__``."""
        if self._hash_cache is None:
            sgs, sg_types = self.sub_group()
            self._hash_cache = hash(
                (
                    self.base_field(),
                    self.length(),
                    self.dimension(),
                    tuple(self.locality()),
                    tuple(self.local_minimum_distance()),
                    tuple(tuple(sg) for sg in sgs),
                    tuple(sg_types),
                )
            )
        return self._hash_cache

    def _repr_(self) -> str:
        """Return a human-readable summary of the code parameters."""
        return (
            f"[{self.length()}, {self.dimension()}, {self.locality()}] "
            f"Tamo-Barg Code over GF({self.base_field().cardinality()})"
        )

    def bipartite_graph(self) -> tuple[nx.MultiGraph, set[int], set[int]]:
        """Return ``(graph, left_nodes, right_nodes)`` for the two coset partitions."""
        if self._bipartite_graph_cache is None:
            self._bipartite_graph_cache = create_bipartite_coset_graph(
                self._partition, self._evaluation_points_idx, self._evaluation_points
            )
        return self._bipartite_graph_cache

    def partition(self) -> tuple[list[list[list[Any]]], list[int]]:
        """Return ``(partitions, partition_sizes)`` across both recovery sets."""
        return self._partition, self._partition_size

    def sub_group(self) -> tuple[list[list[Any]], list[str]]:
        """Return ``(subgroups, subgroup_types)`` defining the two partitions."""
        return self._sub_group, self._sub_group_type

    def evaluation_points_idx(self) -> list[list[list[int]]]:
        """Return evaluation-point indices for each coset of each partition."""
        return self._evaluation_points_idx

    def evaluation_points(self) -> list[Any]:
        """Return the canonically sorted list of codeword evaluation points."""
        return self._evaluation_points

    def locality(self) -> list[int]:
        """Return the two-element list ``[r_0, r_1]`` of localities."""
        return self._locality

    def num_of_sets(self) -> int:
        """Return the number of disjoint-repair-set partitions (always ``2``)."""
        return self._num_of_sets

    def local_minimum_distance(self) -> list[int]:
        """Return the two-element list ``[d_0, d_1]`` of local minimum distances."""
        return self._local_minimum_distance

    def design_distance(self) -> int:
        """Return the polynomial degree-bound design distance ``n - max_deg``."""
        if self._design_distance_cache is None:
            basis = self.encoder().enc_basis()
            max_degree = int(basis[-1].degree()) if basis else 0
            self._design_distance_cache = int(self.length()) - max_degree
        return self._design_distance_cache

    def max_dimension(self) -> int:
        """Return the maximum message dimension supported by the intersection basis."""
        return int(self.encoder().max_dimension())

    def max_rate(self) -> Any:
        """Return the maximum achievable code rate ``max_dimension() / length()``."""
        return self.max_dimension() / self.length()

    def minimum_distance_py(self) -> int:
        """Compute the exact minimum Hamming weight by enumerating monic messages.

        Iterates only over projectively normalized non-zero message vectors in
        ``GF(q)^k`` whose first non-zero coordinate equals ``1``, achieving a
        ``(q - 1)x`` speedup over full message-space enumeration. Uses vectorized
        NumPy ``GF(2^m)`` LUTs with reverse ``lead`` order and ``design_distance()``
        early termination when ``char(F) == 2`` and ``q <= 65536``.

        Returns:
            Exact minimum Hamming distance of the code.
        """
        gen_mat = self.generator_matrix()
        field = self.base_field()
        k = int(self.dimension())
        n = int(self.length())
        stop_bound = max(1, self.design_distance())

        if int(field.characteristic()) == 2 and int(field.cardinality()) <= 65536:
            return _gf2m_min_distance(gen_mat, field, k, n, stop_bound)

        min_wt = n
        for lead in range(k - 1, -1, -1):
            lead_row = gen_mat.row(lead)
            tail_dim = k - lead - 1
            if tail_dim == 0:
                wt = int(lead_row.hamming_weight())
                if 0 < wt < min_wt:
                    min_wt = wt
                    if min_wt <= stop_bound:
                        return min_wt
                continue
            sub_gen = gen_mat.matrix_from_rows(list(range(lead + 1, k)))
            for tail_vec in VectorSpace(field, tail_dim):
                cw = lead_row + tail_vec * sub_gen
                wt = int(cw.hamming_weight())
                if 0 < wt < min_wt:
                    min_wt = wt
                    if min_wt <= stop_bound:
                        return min_wt
        return min_wt


class TamoBargTwoSetsVectorEncoder(Encoder):
    """Polynomial intersection vector encoder for ``TamoBargCodeTwoSets``."""

    def __init__(self, code: TamoBargCodeTwoSets) -> None:
        """Construct per-partition Tamo-Barg bases and their degree-minimal intersection.

        Args:
            code: Associated ``TamoBargCodeTwoSets`` instance.

        Raises:
            ValueError: If ``code.dimension()`` is outside ``1 .. max_dimension``.
        """
        super().__init__(code)
        self._num_of_sets: int = code.num_of_sets()
        self._partition, self._partition_size = code.partition()
        self._sub_group, self._sub_group_type = code.sub_group()
        self._base_poly_ring = PolynomialRing(code.base_field(), "x", implementation="generic")

        self._good_poly: list[Any] = []
        self._algebra_basis: list[list[Any]] = []
        self._enc_basis: list[list[Any]] = []

        for i in range(self._num_of_sets):
            sgt_lit: SubgroupType = "add" if self._sub_group_type[i] == "add" else "mult"
            good_p = build_good_polynomial(self._base_poly_ring, self._sub_group[i], sgt_lit)
            alg_b = build_algebra_basis(good_p, self._partition_size[i])
            enc_b = build_encoding_basis(self._base_poly_ring, alg_b, code.locality()[i])
            self._good_poly.append(good_p)
            self._algebra_basis.append(alg_b)
            self._enc_basis.append(enc_b)

        full_comb_basis = intersect_encoding_bases(
            self._base_poly_ring,
            self._enc_basis[0],
            self._enc_basis[1],
            int(code.length()),
        )
        self._max_dimension: int = len(full_comb_basis)
        raw_dim = code.dimension()
        k = self._max_dimension if raw_dim is None else int(raw_dim)
        if k < 1 or k > self._max_dimension:
            raise ValueError(
                f"Code dimension ({k}) must satisfy 1 <= dimension <= {self._max_dimension}"
            )
        self._comb_enc_basis: list[Any] = full_comb_basis[:k]
        self._row_matrix_space = MatrixSpace(code.base_field(), 1, k)

    def _repr_(self) -> str:
        """Return string representation of the vector encoder."""
        return f"Vector Encoder for {self.code()}"

    def enc_basis(self) -> list[Any]:
        """Return the ``k`` combined encoding basis polynomials of minimal degrees."""
        return self._comb_enc_basis

    def enc_basis_ext(self) -> list[list[Any]]:
        """Return the per-partition full encoding bases ``[basis_0, basis_1]``."""
        return self._enc_basis

    def enc_algebra_basis(self) -> list[list[Any]]:
        """Return the per-partition good-polynomial power bases."""
        return self._algebra_basis

    def enc_good_poly(self) -> list[Any]:
        """Return ``[g_0(x), g_1(x)]``, the good polynomials of both partitions."""
        return self._good_poly

    def enc_poly(self, message: Sequence[Any]) -> Any:
        """Evaluate ``sum_{i=0}^{k-1} message[i] * basis[i](x)`` in ``GF(q)['x']``.

        Args:
            message: Message vector or sequence of length ``k`` over ``GF(q)``.

        Returns:
            Univariate message polynomial in ``GF(q)['x']``.
        """
        return evaluate_message_polynomial(self._base_poly_ring, self._comb_enc_basis, message)

    def max_dimension(self) -> int:
        """Return the dimension of the full intersection encoding subspace."""
        return self._max_dimension

    def encode(self, word: Any) -> Any:
        """Encode ``word`` into a codeword via a vector-matrix product with ``G``."""
        msg_space = self.message_space()
        if word not in msg_space:
            raise ValueError(f"The value to encode must be in {msg_space}")
        return word * self.generator_matrix()

    @cached_method
    def generator_matrix(self) -> Any:
        """Build and cache the immutable ``k x n`` generator matrix over ``GF(q)``."""
        code = self.code()
        field = code.base_field()
        k = int(code.dimension())
        n = int(code.length())
        alphas = code.evaluation_points()

        alphas_arr = field.coerce_to_arr(alphas)
        rows_arr = np.zeros((k, n), dtype=field._dtype)
        for i, poly in enumerate(self._comb_enc_basis):
            rows_arr[i] = poly.eval_points_arr(alphas_arr)
        gen_mat = matrix(field, k, n, rows_arr)
        gen_mat.set_immutable()
        return gen_mat


class TamoBargIterativeDecoder(Decoder):
    """Iterative hard-decision local RS decoder across both coset partitions."""

    def __init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10) -> None:
        """Precompute local RS codecs and parity-check matrices.

        Args:
            code: Associated ``TamoBargCodeTwoSets`` instance.
            max_num_of_itr: Maximum number of alternating two-partition iterations.
        """
        if not _is_int_like(max_num_of_itr) or int(max_num_of_itr) < 0:
            raise ValueError(
                f"max_num_of_itr must be a non-negative integer, got {max_num_of_itr!r}"
            )
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._max_num_of_itr: int = int(max_num_of_itr)
        self._F = code.base_field()
        self._ambient_space = code.ambient_space()
        self._partitions, _ = code.partition()
        self._evalpts_idx = code.evaluation_points_idx()
        self._evalpts_idx_arr: list[list[np.ndarray]] = [
            [np.array(idxs, dtype=np.intp) for idxs in part_idxs] for part_idxs in self._evalpts_idx
        ]

        self._partitions_local_codes: list[list[RSCodec]] = []
        self._local_parity_checks: list[list[Any]] = []

        localities = code.locality()
        for i, partition in enumerate(self._partitions):
            code_tmp: list[RSCodec] = []
            pcm_tmp: list[Any] = []
            for coset in partition:
                local_rs = RSCodec(self._F, coset, localities[i])
                code_tmp.append(local_rs)
                pcm_tmp.append(local_rs.parity_check_matrix())
            self._partitions_local_codes.append(code_tmp)
            self._local_parity_checks.append(pcm_tmp)

    def _repr_(self) -> str:
        """Return string representation of the iterative decoder."""
        return f"Iterative Decoder for {self.code()}"

    def _partition_is_syndrome_clean(self, part_idx: int, work_arr: np.ndarray) -> bool:
        """Return True if every coset in partition ``part_idx`` has zero local syndrome."""
        return _partition_is_syndrome_clean(
            part_idx,
            work_arr,
            self._evalpts_idx_arr,
            self._partitions_local_codes,
        )

    def _decode_with_status(self, r: Any) -> tuple[Any, int, bool]:
        """Iteratively decode ``r`` and return ``(work, num_itr, converged)``."""
        work = copy(self._ambient_space(r))
        work_arr = work._arr
        num_itr = 0
        converged = False
        prev_state_1: bytes | None = work_arr.tobytes()
        prev_state_2: bytes | None = None

        while num_itr < self._max_num_of_itr:
            uncorrectable_err = [False, False]
            changed_in_itr = False
            changed_in_p1 = False

            for i in (0, 1):
                local_codes = self._partitions_local_codes[i]
                for j, idxs in enumerate(self._evalpts_idx_arr[i]):
                    local_arr = work_arr[idxs]
                    if local_codes[j].has_zero_syndrome(local_arr):
                        continue
                    try:
                        corr_arr = local_codes[j].decode_errors(local_arr)
                    except (DecodingError, ValueError):
                        uncorrectable_err[i] = True
                        continue

                    if not np.array_equal(corr_arr, local_arr):
                        changed_in_itr = True
                        if i == 1:
                            changed_in_p1 = True
                        work_arr[idxs] = corr_arr

            num_itr += 1
            if not uncorrectable_err[0] and not uncorrectable_err[1]:
                if not changed_in_p1 or self._partition_is_syndrome_clean(0, work_arr):
                    converged = True
                    break
            elif not changed_in_itr:
                break

            curr_state = work_arr.tobytes()
            if curr_state == prev_state_2:
                break
            prev_state_2 = prev_state_1
            prev_state_1 = curr_state

        return work, num_itr, converged

    def decode_to_code(self, r: Any) -> tuple[Any, int]:
        """Iteratively decode cosets of both partitions without mutating ``r``.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``(corrected_word, num_itr)``.
        """
        work, num_itr, _ = self._decode_with_status(r)
        return work, num_itr

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        corr_c, _, converged = self._decode_with_status(r)
        return self.code().unencode(corr_c, nocheck=converged)


class TamoBargIterativeEEDecoder(Decoder):
    """Iterative erasure-and-error local RS decoder across both coset partitions."""

    def __init__(
        self,
        code: TamoBargCodeTwoSets,
        max_num_of_itr: int = 10,
        erasure_start_itr: int | None = None,
    ) -> None:
        """Precompute local RS error-and-erasure codecs and parity-check matrices.

        Args:
            code: Associated ``TamoBargCodeTwoSets`` instance.
            max_num_of_itr: Maximum number of alternating two-partition iterations.
            erasure_start_itr: Iteration index (1-based) at which DUE coset erasure
                marking begins; if ``None``, triggers immediately on hard-decision stall
                (or at ``max_num_of_itr // 2``, whichever comes first).
        """
        if not _is_int_like(max_num_of_itr) or int(max_num_of_itr) < 0:
            raise ValueError(
                f"max_num_of_itr must be a non-negative integer, got {max_num_of_itr!r}"
            )
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._max_num_of_itr: int = int(max_num_of_itr)
        self._dynamic_erasure_start: bool = erasure_start_itr is None
        if erasure_start_itr is None:
            self._erasure_start_itr: int = self._max_num_of_itr // 2
        else:
            if not _is_int_like(erasure_start_itr) or int(erasure_start_itr) < 0:
                raise ValueError(
                    f"erasure_start_itr must be a non-negative integer, got {erasure_start_itr!r}"
                )
            self._erasure_start_itr = int(erasure_start_itr)

        self._F = code.base_field()
        self._ambient_space = code.ambient_space()
        self._era_ambient_space = VectorSpace(GF(2), int(code.length()))
        self._partitions, _ = code.partition()
        self._evalpts_idx = code.evaluation_points_idx()
        self._evalpts_idx_arr: list[list[np.ndarray]] = [
            [np.array(idxs, dtype=np.intp) for idxs in part_idxs] for part_idxs in self._evalpts_idx
        ]
        self._local_min_distances = code.local_minimum_distance()

        self._partitions_local_codes: list[list[RSCodec]] = []
        self._local_parity_checks: list[list[Any]] = []

        localities = code.locality()
        for i, partition in enumerate(self._partitions):
            code_tmp: list[RSCodec] = []
            pcm_tmp: list[Any] = []
            for coset in partition:
                local_rs = RSCodec(self._F, coset, localities[i])
                code_tmp.append(local_rs)
                pcm_tmp.append(local_rs.parity_check_matrix())
            self._partitions_local_codes.append(code_tmp)
            self._local_parity_checks.append(pcm_tmp)

    def _repr_(self) -> str:
        """Return string representation of the iterative erasure-and-error decoder."""
        return f"Iterative Decoder Error-and-Erasure for {self.code()}"

    def _partition_is_syndrome_clean(self, part_idx: int, work_arr: np.ndarray) -> bool:
        """Return True if every coset in partition ``part_idx`` has zero local syndrome."""
        return _partition_is_syndrome_clean(
            part_idx,
            work_arr,
            self._evalpts_idx_arr,
            self._partitions_local_codes,
        )

    def _decode_with_status(self, r: Any) -> tuple[tuple[Any, Any], int, bool]:
        """Iteratively decode ``r`` and return ``((work, era_vec), num_itr, converged)``."""
        work = copy(self._ambient_space(r))
        work_arr = work._arr
        erasure_flags = np.zeros(len(work), dtype=np.uint8)
        num_itr = 0
        converged = False
        erasures_activated = False

        while num_itr < self._max_num_of_itr:
            prev_work = work_arr.tobytes()
            prev_erasures = erasure_flags.tobytes()
            uncorrectable_err = [False, False]
            changed_in_p1 = False
            due_status: list[list[bool]] = []

            for i in (0, 1):
                d_local = self._local_min_distances[i]
                local_codes = self._partitions_local_codes[i]
                due_tmp: list[bool] = []
                for j, idxs in enumerate(self._evalpts_idx_arr[i]):
                    local_arr = work_arr[idxs]
                    era_sub = erasure_flags[idxs]
                    era_wt = int(era_sub.sum())

                    if era_wt == 0:
                        if local_codes[j].has_zero_syndrome(local_arr):
                            due_tmp.append(False)
                            continue
                        try:
                            corr_arr = local_codes[j].decode_errors(local_arr)
                        except (DecodingError, ValueError):
                            due_tmp.append(True)
                            uncorrectable_err[i] = True
                        else:
                            if not np.array_equal(corr_arr, local_arr):
                                if i == 1:
                                    changed_in_p1 = True
                                work_arr[idxs] = corr_arr
                            due_tmp.append(False)
                    elif era_wt >= d_local:
                        due_tmp.append(True)
                        uncorrectable_err[i] = True
                    else:
                        try:
                            corr_arr = local_codes[j].decode_errors_erasures(local_arr, era_sub)
                        except (DecodingError, ValueError):
                            due_tmp.append(True)
                            uncorrectable_err[i] = True
                        else:
                            if not np.array_equal(corr_arr, local_arr):
                                if i == 1:
                                    changed_in_p1 = True
                                work_arr[idxs] = corr_arr
                            due_tmp.append(False)

                due_status.append(due_tmp)

            num_itr += 1

            if (
                not uncorrectable_err[0]
                and not uncorrectable_err[1]
                and (not changed_in_p1 or self._partition_is_syndrome_clean(0, work_arr))
            ):
                for s in (0, 1):
                    for t, idxs in enumerate(self._evalpts_idx_arr[s]):
                        if not due_status[s][t]:
                            erasure_flags[idxs] = 0
                converged = True
                break

            work_stalled = work_arr.tobytes() == prev_work
            trigger_erasures = num_itr >= self._erasure_start_itr or (
                self._dynamic_erasure_start and (erasures_activated or work_stalled)
            )
            if trigger_erasures:
                erasures_activated = True
                for t, idxs in enumerate(self._evalpts_idx_arr[0]):
                    erasure_flags[idxs] = 1 if due_status[0][t] else 0
                for t, idxs in enumerate(self._evalpts_idx_arr[1]):
                    if not due_status[1][t]:
                        erasure_flags[idxs] = 0

            if (
                (
                    num_itr > self._erasure_start_itr
                    or (self._dynamic_erasure_start and erasures_activated)
                )
                and work_stalled
                and erasure_flags.tobytes() == prev_erasures
            ):
                break

        return (work, self._era_ambient_space(erasure_flags.tolist())), num_itr, converged

    def decode_to_code(self, r: Any) -> tuple[tuple[Any, Any], int]:
        """Iteratively decode errors and DUE-marked erasures without mutating ``r``.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``((corrected_word, erasure_vector), num_itr)``.
        """
        pair, num_itr, _ = self._decode_with_status(r)
        return pair, num_itr

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        (corr_c, _), _, converged = self._decode_with_status(r)
        return self.code().unencode(corr_c, nocheck=converged)


class TamoBargGlobalDecoder(Decoder):
    """Global RS hard-decision decoder using the code's polynomial design distance."""

    def __init__(self, code: TamoBargCodeTwoSets) -> None:
        """Initialize the ambient global RS codec and parity-check matrix."""
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._F = code.base_field()
        self._ambient_space = code.ambient_space()
        self._global_rs = RSCodec(
            self._F,
            code.evaluation_points(),
            int(code.length()) - code.design_distance() + 1,
        )
        self._global_pcm = self._global_rs.parity_check_matrix()

    def _repr_(self) -> str:
        """Return string representation of the global decoder."""
        return f"Global Decoder for {self.code()}"

    def decode_to_code(self, r: Any) -> Any:
        """Decode ``r`` via the global enveloping RS code without mutating ``r``.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Decoded codeword vector in ``GF(q)^n``.
        """
        work = copy(self._ambient_space(r))
        if self._global_rs.has_zero_syndrome(work):
            return work
        return self._global_rs.decode_errors(work)


class TamoBargGlobalEEDecoder(Decoder):
    """Global RS erasure-and-error decoder over ``GF(q)^n x GF(2)^n``."""

    def __init__(self, code: TamoBargCodeTwoSets) -> None:
        """Initialize the Cartesian input space and global RS error-and-erasure codec."""
        era_space = VectorSpace(GF(2), int(code.length()))
        input_space = cartesian_product([code.ambient_space(), era_space])
        super().__init__(code, input_space, "VectorEncoder")
        self._F = code.base_field()
        self._ambient_space = code.ambient_space()
        self._era_space = era_space
        self._global_rs = RSCodec(
            self._F,
            code.evaluation_points(),
            int(code.length()) - code.design_distance() + 1,
        )
        self._global_pcm = self._global_rs.parity_check_matrix()
        self._d_glob: int = int(self._global_rs.minimum_distance())

    def _repr_(self) -> str:
        """Return string representation of the global erasure-and-error decoder."""
        return f"Global Erasure-and-Error Decoder for {self.code()}"

    def decode_to_code(self, word_and_erasure_vector: Any) -> Any:
        """Decode ``(word, erasure_vector)`` globally without mutating inputs.

        Args:
            word_and_erasure_vector: Pair ``(word, erasure_vector)`` in
                ``GF(q)^n x GF(2)^n``.

        Returns:
            Decoded codeword vector in ``GF(q)^n``.

        Raises:
            DecodingError: If ``erasure_vector`` exceeds ``d_design - 1`` erasures
                or error-and-erasure decoding fails.
        """
        word, erasure_vector = word_and_erasure_vector
        work = copy(self._ambient_space(word))
        era_vec = copy(self._era_space(erasure_vector))
        era_wt = int(era_vec.hamming_weight())

        if era_wt == 0:
            if self._global_rs.has_zero_syndrome(work):
                return work
            return self._global_rs.decode_errors(work)
        if era_wt >= self._d_glob:
            raise DecodingError("Too many erasures for global decoder")
        return self._global_rs.decode_errors_erasures(work, era_vec)


class TamoBargTwoStepDecoder(Decoder):
    """Two-step decoder combining iterative local decoding with global GRS cleanup."""

    def __init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10) -> None:
        """Initialize the inner iterative decoder and outer global decoder."""
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._iter_decoder = TamoBargIterativeDecoder(code, max_num_of_itr)
        self._global_decoder = TamoBargGlobalDecoder(code)
        self._is_full_dim: bool = int(code.dimension()) == code.max_dimension()

    def _repr_(self) -> str:
        """Return string representation of the two-step decoder."""
        return f"Two-Steps Decoder for {self.code()}"

    def _decode_with_status(self, r: Any) -> tuple[Any, int, bool]:
        """Run iterative local decoding and optional global cleanup, returning status."""
        tmp, num_itr, converged = self._iter_decoder._decode_with_status(r)
        if self._is_full_dim and converged:
            return tmp, num_itr, True
        try:
            result = self._global_decoder.decode_to_code(tmp)
        except (DecodingError, ValueError):
            return tmp, num_itr, converged
        return result, num_itr, True

    def decode_to_code(self, r: Any) -> tuple[Any, int]:
        """Run iterative local decoding followed by global GRS decoding.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``(corrected_word, num_itr)``.
        """
        result, num_itr, _ = self._decode_with_status(r)
        return result, num_itr

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        corr_c, _, decoded = self._decode_with_status(r)
        return self.code().unencode(corr_c, nocheck=decoded)


class TamoBargTwoStepEEDecoder(Decoder):
    """Two-step erasure-and-error decoder combining iterative EE with global EE cleanup."""

    def __init__(
        self,
        code: TamoBargCodeTwoSets,
        max_num_of_itr: int = 10,
        erasure_start_itr: int | None = None,
    ) -> None:
        """Initialize the inner iterative EE decoder and outer global EE decoder."""
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._iter_ee_decoder = TamoBargIterativeEEDecoder(code, max_num_of_itr, erasure_start_itr)
        self._global_ee_decoder = TamoBargGlobalEEDecoder(code)
        self._is_full_dim: bool = int(code.dimension()) == code.max_dimension()
        self._d_glob: int = self._global_ee_decoder._d_glob

    def _repr_(self) -> str:
        """Return string representation of the two-step erasure-and-error decoder."""
        return f"Two-Steps Erasure-and-Error Decoder for {self.code()}"

    def _decode_with_status(self, r: Any) -> tuple[Any, int, int, bool]:
        """Run iterative EE decoding and optional global EE cleanup, returning status."""
        (word_after_iter, era_vec), num_itr, converged = self._iter_ee_decoder._decode_with_status(
            r
        )
        num_rem_era = int(era_vec.hamming_weight())
        if self._is_full_dim and converged:
            return word_after_iter, num_itr, num_rem_era, True
        if num_rem_era >= self._d_glob:
            return word_after_iter, num_itr, num_rem_era, False
        try:
            result = self._global_ee_decoder.decode_to_code((word_after_iter, era_vec))
        except (DecodingError, ValueError):
            return word_after_iter, num_itr, num_rem_era, converged
        return result, num_itr, num_rem_era, True

    def decode_to_code(self, r: Any) -> tuple[Any, int, int]:
        """Run iterative EE decoding followed by global EE decoding.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``(corrected_word, num_itr, num_rem_era)``.
        """
        result, num_itr, num_rem_era, _ = self._decode_with_status(r)
        return result, num_itr, num_rem_era

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        corr_c, _, _, decoded = self._decode_with_status(r)
        return self.code().unencode(corr_c, nocheck=decoded)


TamoBargCodeTwoSets._registered_encoders["VectorEncoder"] = TamoBargTwoSetsVectorEncoder
TamoBargCodeTwoSets._registered_decoders["IterativeDecoder"] = TamoBargIterativeDecoder
TamoBargCodeTwoSets._registered_decoders["TwoStepsDecoder"] = TamoBargTwoStepDecoder
TamoBargCodeTwoSets._registered_decoders["GlobalDecoder"] = TamoBargGlobalDecoder
TamoBargCodeTwoSets._registered_decoders["IterativeErasureErrorDecoder"] = (
    TamoBargIterativeEEDecoder
)
TamoBargCodeTwoSets._registered_decoders["GlobalErasureErrorDecoder"] = TamoBargGlobalEEDecoder
TamoBargCodeTwoSets._registered_decoders["TwoStepsErasureErrorDecoder"] = TamoBargTwoStepEEDecoder
