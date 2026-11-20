"""Single-set Tamo-Barg optimal locally recoverable codes, encoders, and decoders."""

from typing import Any, Literal

from tamo_barg_lrc._ff import (
    GF,
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
from tamo_barg_lrc.polynomials import (
    build_algebra_basis,
    build_encoding_basis,
    build_good_polynomial,
    evaluate_message_polynomial,
)
from tamo_barg_lrc.subgroups import (
    SubgroupType,
    list_canonical_additive_subgroups,
    list_multiplicative_subgroups,
    list_subgroup_cosets,
)

__all__ = [
    "TamoBargCode",
    "TamoBargVectorEncoder",
    "TamoBargErasureDecoder",
    "TamoBargErrorDecoder",
]


def _coerce_int(value: Any, name: str) -> int:
    """Validate and convert an integer-valued code parameter to ``int``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        int_val = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if int_val != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int_val


def _make_poly_ring(base_field: Any) -> Any:
    """Construct univariate polynomial ring ``base_field['x']``."""
    return PolynomialRing(base_field, "x")


def _build_coset_alignments(
    sub_group: list[Any],
    sub_group_type: SubgroupType,
    partition: list[list[Any]],
) -> tuple[list[list[int]], list[list[int]], bool]:
    """Precompute index permutations aligning each coset to ``sub_group`` order.

    For each coset ``A_c``, ``aligned[l] = A_c[perms[c][l]]`` corresponds to
    ``sub_group[l]`` under the coset's affine map (translation for ``"add"``
    or scaling for ``"mult"``), and ``inv_perms[c]`` is the inverse map.
    """
    m = len(sub_group)
    h_index = {h: idx for idx, h in enumerate(sub_group)}
    identity = list(range(m))
    perms: list[list[int]] = []
    inv_perms: list[list[int]] = []
    all_identity = True

    for coset in partition:
        beta = coset[0]
        perm = [0] * m
        if sub_group_type == "add":
            for pos, elem in enumerate(coset):
                perm[h_index[elem - beta]] = pos
        else:
            inv_beta = 1 / beta
            for pos, elem in enumerate(coset):
                perm[h_index[elem * inv_beta]] = pos

        inv_perm = [0] * m
        for l_idx, pos in enumerate(perm):
            inv_perm[pos] = l_idx

        if perm != identity:
            all_identity = False
        perms.append(perm)
        inv_perms.append(inv_perm)

    return perms, inv_perms, all_identity


def _build_canonical_rs_pcm_rows(
    base_field: Any, eval_points: list[Any], num_checks: int
) -> tuple[list[list[Any]], list[Any]]:
    """Build generalized Vandermonde parity-check rows for ``RS(eval_points, m - num_checks)``.

    Returns ``(h0_rows, inv_v)`` where ``h0_rows[u][l] = v_l * eval_points[l]^u``
    with ``v_l = (prod_{j != l} (eval_points[l] - eval_points[j]))^(-1)`` and
    ``inv_v[l] = 1 / v_l``.
    """
    m = len(eval_points)
    one = base_field.one()
    inv_v: list[Any] = []
    v_row: list[Any] = []
    for l_idx in range(m):
        prod_diff = one
        a_l = eval_points[l_idx]
        for j_idx in range(m):
            if j_idx != l_idx:
                prod_diff *= a_l - eval_points[j_idx]
        inv_v.append(prod_diff)
        v_row.append(one / prod_diff)

    rows: list[list[Any]] = [v_row]
    for _ in range(1, num_checks):
        prev = rows[-1]
        rows.append([prev[l_idx] * eval_points[l_idx] for l_idx in range(m)])
    return rows, inv_v


def _solve_erasure_system(
    h0_rows: list[list[Any]],
    inv_v: list[Any],
    aligned: list[Any],
    erased: list[int],
    zero: Any,
) -> None:
    """Recover ``erased`` positions of ``aligned`` in place via ``H_{0, E} x_E = -H_{0, K} x_K``."""
    m = len(aligned)
    num_erased = len(erased)

    if num_erased == 1:
        e0 = erased[0]
        row0 = h0_rows[0]
        acc = zero
        for j in range(m):
            if j != e0:
                acc += row0[j] * aligned[j]
        aligned[e0] = (-acc) * inv_v[e0]
        return

    if num_erased == 2:
        e0, e1 = erased[0], erased[1]
        row0 = h0_rows[0]
        row1 = h0_rows[1]
        s0 = zero
        s1 = zero
        for j in range(m):
            if j != e0 and j != e1:
                xj = aligned[j]
                s0 += row0[j] * xj
                s1 += row1[j] * xj
        s0 = -s0
        s1 = -s1
        a00 = row0[e0]
        a01 = row0[e1]
        a10 = row1[e0]
        a11 = row1[e1]
        inv_det = 1 / (a00 * a11 - a01 * a10)
        aligned[e0] = (s0 * a11 - a01 * s1) * inv_det
        aligned[e1] = (a00 * s1 - s0 * a10) * inv_det
        return

    erased_set = set(erased)
    aug: list[list[Any]] = []
    for u in range(num_erased):
        row_u = h0_rows[u]
        rhs = zero
        for j in range(m):
            if j not in erased_set:
                rhs += row_u[j] * aligned[j]
        aug_row = [row_u[idx] for idx in erased]
        aug_row.append(-rhs)
        aug.append(aug_row)

    for col in range(num_erased):
        piv = col
        while aug[piv][col].is_zero():
            piv += 1
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        piv_row = aug[col]
        inv_p = 1 / piv_row[col]
        for j in range(col, num_erased + 1):
            piv_row[j] *= inv_p
        for r_idx in range(num_erased):
            if r_idx != col:
                target_row = aug[r_idx]
                factor = target_row[col]
                if not factor.is_zero():
                    for j in range(col, num_erased + 1):
                        target_row[j] -= factor * piv_row[j]

    for u, idx in enumerate(erased):
        aligned[idx] = aug[u][num_erased]


class TamoBargCode(AbstractLinearCode):
    """Single-set Tamo-Barg optimal locally recoverable code over ``GF(q)``.

    Constructs an ``(r, d_local)``-locally recoverable evaluation code whose
    evaluation points are partitioned into disjoint cosets of size
    ``r + d_local - 1`` of a multiplicative or additive subgroup of ``GF(q)``.
    """

    _registered_encoders: dict[str, type[Encoder]] = {}
    _registered_decoders: dict[str, type[Decoder]] = {}

    def __init__(
        self,
        base_field: Any,
        length: int,
        dimension: int,
        locality: int,
        local_minimum_distance: int = 2,
        sub_group_type: Literal["any", "mult", "add"] = "any",
    ) -> None:
        """Initialize a single-set Tamo-Barg code.

        Args:
            base_field: Finite field ``GF(q)``.
            length: Codeword block length ``n`` (divisible by ``r + d_local - 1``).
            dimension: Message dimension ``k`` (``1 <= k <= max_dimension``).
            locality: Locality parameter ``r >= 1``.
            local_minimum_distance: Local code minimum distance ``d_local >= 2``.
            sub_group_type: Subgroup family selector: ``"any"``, ``"mult"``, or ``"add"``.

        Raises:
            ValueError: If any parameter is invalid or no suitable subgroup exists.
        """
        if (
            not hasattr(base_field, "is_finite")
            or not base_field.is_finite()
            or (hasattr(base_field, "is_field") and not base_field.is_field())
        ):
            raise ValueError("base_field must be a finite field")

        n_int = _coerce_int(length, "length")
        k_int = _coerce_int(dimension, "dimension")
        r_int = _coerce_int(locality, "locality")
        d_local_int = _coerce_int(local_minimum_distance, "local_minimum_distance")

        if n_int < 1:
            raise ValueError(f"length ({n_int}) must be at least 1")
        if r_int < 1:
            raise ValueError(f"locality ({r_int}) must be at least 1")
        if d_local_int < 2:
            raise ValueError(f"local_minimum_distance ({d_local_int}) must be at least 2")
        if sub_group_type not in ("any", "mult", "add"):
            raise ValueError(
                f"sub_group_type must be 'any', 'mult', or 'add', got {sub_group_type!r}"
            )

        coset_size = r_int + d_local_int - 1
        if n_int % coset_size != 0:
            raise ValueError(f"Code length {n_int} must be divisible by coset size {coset_size}")

        q = int(base_field.cardinality())

        if sub_group_type == "any":
            if n_int <= q - 1 and (q - 1) % coset_size == 0:
                mult_subgroups = list_multiplicative_subgroups(base_field)
                self._sub_group = mult_subgroups[coset_size]
                self._sub_group_type: SubgroupType = "mult"
            elif n_int <= q and q % coset_size == 0:
                add_subgroups = list_canonical_additive_subgroups(base_field)
                if coset_size in add_subgroups:
                    self._sub_group = add_subgroups[coset_size]
                    self._sub_group_type = "add"
                else:
                    raise ValueError(
                        f"No multiplicative or additive subgroup of size {coset_size} in "
                        f"{base_field} can support code length {n_int}"
                    )
            else:
                raise ValueError(
                    f"No multiplicative or additive subgroup of size {coset_size} in "
                    f"{base_field} can support code length {n_int}"
                )
        elif sub_group_type == "mult":
            if (q - 1) % coset_size != 0:
                raise ValueError(
                    f"No multiplicative subgroup of size {coset_size} exists in {base_field}"
                )
            mult_subgroups = list_multiplicative_subgroups(base_field)
            if coset_size not in mult_subgroups:
                raise ValueError(
                    f"No multiplicative subgroup of size {coset_size} exists in {base_field}"
                )
            if n_int > q - 1:
                raise ValueError(
                    f"Code length {n_int} exceeds multiplicative group order {q - 1} "
                    f"of {base_field}"
                )
            self._sub_group = mult_subgroups[coset_size]
            self._sub_group_type = "mult"
        else:
            if q % coset_size != 0:
                raise ValueError(
                    f"No canonical additive subgroup of size {coset_size} exists in {base_field}"
                )
            add_subgroups = list_canonical_additive_subgroups(base_field)
            if coset_size not in add_subgroups:
                raise ValueError(
                    f"No canonical additive subgroup of size {coset_size} exists in {base_field}"
                )
            if n_int > q:
                raise ValueError(
                    f"Code length {n_int} exceeds field cardinality {q} of {base_field}"
                )
            self._sub_group = add_subgroups[coset_size]
            self._sub_group_type = "add"

        self._partition_size = n_int // coset_size
        all_cosets = list_subgroup_cosets(base_field, self._sub_group, self._sub_group_type)
        if len(all_cosets) < self._partition_size:
            raise ValueError(
                f"Only {len(all_cosets)} cosets available, need {self._partition_size}"
            )
        self._partition = all_cosets[: self._partition_size]
        self._evaluation_points = [pt for coset in self._partition for pt in coset]

        self._locality = r_int
        self._local_minimum_distance = d_local_int
        max_dim = self._partition_size * self._locality
        if not (1 <= k_int <= max_dim):
            raise ValueError(f"Code dimension {k_int} must satisfy 1 <= dimension <= {max_dim}")

        super().__init__(base_field, n_int, "VectorEncoder", "ErasureDecoder")
        self._dimension = k_int
        self._cached_hash: int | None = None

    def __eq__(self, other: object) -> bool:
        """Check structural equality between two single-set Tamo-Barg codes."""
        return (
            isinstance(other, TamoBargCode)
            and self.base_field() == other.base_field()
            and self.length() == other.length()
            and self.dimension() == other.dimension()
            and self.locality() == other.locality()
            and self.local_minimum_distance() == other.local_minimum_distance()
            and self.evaluation_points() == other.evaluation_points()
        )

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        if self._cached_hash is None:
            self._cached_hash = hash(
                (
                    self.base_field(),
                    int(self.length()),
                    self._dimension,
                    self._locality,
                    self._local_minimum_distance,
                    tuple(self._evaluation_points),
                )
            )
        return self._cached_hash

    def _repr_(self) -> str:
        """Return string representation of the code."""
        return (
            f"[{self.length()}, {self.dimension()}, {self.locality()}] "
            f"Tamo-Barg Code over GF({self.base_field().cardinality()})"
        )

    def dimension(self) -> int:
        """Return the code dimension ``k``."""
        return self._dimension

    def partition(self) -> tuple[list[list[Any]], int]:
        """Return ``(coset_partition, partition_size)``."""
        return self._partition, self._partition_size

    def sub_group(self) -> tuple[list[Any], str]:
        """Return ``(subgroup_elements, sub_group_type)``."""
        return self._sub_group, self._sub_group_type

    def evaluation_points(self) -> list[Any]:
        """Return flattened list of ``n`` distinct evaluation points in ``GF(q)``."""
        return self._evaluation_points

    def locality(self) -> int:
        """Return the locality parameter ``r``."""
        return self._locality

    def local_minimum_distance(self) -> int:
        """Return the local repair group minimum distance ``d_local``."""
        return self._local_minimum_distance

    def max_dimension(self) -> int:
        """Return the maximum supported dimension ``partition_size * locality``."""
        return self._partition_size * self._locality

    def minimum_distance(self, algorithm: Any = None) -> int:
        """Return the Singleton-like optimal LRC minimum distance of the code."""
        del algorithm
        num_blocks = (self._dimension + self._locality - 1) // self._locality
        return (
            int(self.length())
            - self._dimension
            + 1
            - (num_blocks - 1) * (self._local_minimum_distance - 1)
        )


class TamoBargVectorEncoder(Encoder):
    """Polynomial evaluation vector encoder for :class:`TamoBargCode`."""

    def __init__(self, code: TamoBargCode) -> None:
        """Initialize the encoder and construct the ``k``-element encoding polynomial basis.

        Args:
            code: Associated :class:`TamoBargCode` instance.

        Raises:
            ValueError: If ``code.dimension()`` exceeds the maximum basis size.
        """
        super().__init__(code)
        self._partition, self._partition_size = code.partition()
        sub_group, sub_group_type = code.sub_group()
        self._sub_group = sub_group
        self._sub_group_type = sub_group_type
        self._base_poly_ring = _make_poly_ring(code.base_field())
        if sub_group_type not in ("add", "mult"):
            raise ValueError(f"sub_group_type must be 'add' or 'mult', got {sub_group_type!r}")
        self._good_poly = build_good_polynomial(
            self._base_poly_ring,
            self._sub_group,
            sub_group_type,
        )
        self._algebra_basis = build_algebra_basis(
            self._good_poly,
            self._partition_size,
        )
        full_enc_basis = build_encoding_basis(
            self._base_poly_ring,
            self._algebra_basis,
            code.locality(),
        )
        self._max_dimension = len(full_enc_basis)
        if code.dimension() > self._max_dimension:
            raise ValueError(
                f"Code dimension {code.dimension()} exceeds maximum dimension {self._max_dimension}"
            )
        self._enc_basis = full_enc_basis[: code.dimension()]

    def __eq__(self, other: object) -> bool:
        """Check equality between two :class:`TamoBargVectorEncoder` instances."""
        return isinstance(other, TamoBargVectorEncoder) and self.code() == other.code()

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash((type(self), self.code()))

    def _repr_(self) -> str:
        """Return string representation of the encoder."""
        return f"Vector Encoder for {self.code()}"

    def enc_basis(self) -> list[Any]:
        """Return the ``k`` encoding basis polynomials of minimum ascending degrees."""
        return self._enc_basis

    def enc_algebra_basis(self) -> list[Any]:
        """Return powers ``[1, g, ..., g^(partition_size - 1)]`` of the good polynomial."""
        return self._algebra_basis

    def enc_good_poly(self) -> Any:
        """Return the good polynomial ``g(x)`` constant on each subgroup coset."""
        return self._good_poly

    def enc_poly(self, message: Any) -> Any:
        """Construct the message encoding polynomial ``sum_{i=0}^{k-1} m_i * b_i(x)``.

        Args:
            message: Message vector or sequence of length ``k`` over ``GF(q)``.

        Returns:
            Univariate encoding polynomial in ``GF(q)['x']``.
        """
        return evaluate_message_polynomial(self._base_poly_ring, self._enc_basis, message)

    def max_dimension(self) -> int:
        """Return the total size of the unsliced encoding basis."""
        return self._max_dimension

    @cached_method
    def generator_matrix(self) -> Any:
        """Return the immutable ``k x n`` generator matrix over ``GF(q)``."""
        code = self.code()
        base_field = code.base_field()
        k = code.dimension()
        n = int(code.length())
        r = code.locality()
        s = self._partition_size
        m = n // s
        alphas = code.evaluation_points()
        one = base_field.one()
        zero = base_field.zero()

        max_i = min(r, k)
        alpha_powers: list[list[Any]] = [[one] * n]
        for _ in range(1, max_i):
            prev = alpha_powers[-1]
            alpha_powers.append([prev[col] * alphas[col] for col in range(n)])

        num_blocks = (k + r - 1) // r
        g_vals = [
            coset[0] ** m if self._sub_group_type == "mult" else self._good_poly(coset[0])
            for coset in self._partition
        ]
        g_powers: list[list[Any]] = [[one] * s]
        for _ in range(1, num_blocks):
            prev_g = g_powers[-1]
            g_powers.append([prev_g[c] * g_vals[c] for c in range(s)])

        rows: list[list[Any]] = []
        for j in range(num_blocks):
            block_rows = min(r, k - j * r)
            if j == 0:
                rows.extend(alpha_powers[:block_rows])
            else:
                g_j = g_powers[j]
                zero_block = [zero] * m
                for i in range(block_rows):
                    ap = alpha_powers[i]
                    row: list[Any] = []
                    for c in range(s):
                        gc = g_j[c]
                        if gc.is_zero():
                            row.extend(zero_block)
                        else:
                            base = c * m
                            row.extend(ap[base + l_idx] * gc for l_idx in range(m))
                    rows.append(row)

        g = matrix(base_field, k, n, rows)
        g.set_immutable()
        return g

    def encode(self, word: Any) -> Any:
        """Encode a message vector into a codeword using dense vector-matrix multiplication."""
        message_space = self.message_space()
        if word not in message_space:
            raise ValueError(f"The value to encode must be in {message_space}")
        g = self.generator_matrix()
        return word * g


class TamoBargErasureDecoder(Decoder):
    """Local coset parity-check erasure decoder for :class:`TamoBargCode`."""

    def __init__(self, code: TamoBargCode) -> None:
        """Initialize the local erasure decoder.

        Args:
            code: Associated :class:`TamoBargCode` instance.
        """
        ambient_space = code.ambient_space()
        input_space = cartesian_product([ambient_space, VectorSpace(GF(2), code.length())])
        super().__init__(code, input_space, "VectorEncoder")
        self._ambient_space = ambient_space
        self._coset_size = code.locality() + code.local_minimum_distance() - 1
        sub_group, sub_group_type = code.sub_group()
        partition, partition_size = code.partition()
        self._partition_size = partition_size
        self._coset_perms, self._coset_inv_perms, self._all_identity = _build_coset_alignments(
            sub_group, sub_group_type, partition
        )
        self._h0_rows, self._inv_v = _build_canonical_rs_pcm_rows(
            code.base_field(), sub_group, code.local_minimum_distance() - 1
        )

    def __eq__(self, other: object) -> bool:
        """Check equality between two :class:`TamoBargErasureDecoder` instances."""
        return isinstance(other, TamoBargErasureDecoder) and self.code() == other.code()

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash((type(self), self.code()))

    def _repr_(self) -> str:
        """Return string representation of the erasure decoder."""
        return f"Erasure Decoder for {self.code()}"

    def decode_to_code(self, word_and_erasure_vector: Any) -> Any:
        """Recover up to ``d_local - 1`` erasures per repair coset without mutating inputs.

        Args:
            word_and_erasure_vector: Pair ``(word, erasure_vector)`` in
                ``GF(q)^n x GF(2)^n``.

        Returns:
            Corrected codeword vector in ``GF(q)^n``.

        Raises:
            DecodingError: If any coset contains ``>= d_local`` erasures.
            ValueError: If ``word`` or ``erasure_vector`` does not have length ``n``.
        """
        code = self.code()
        base_field = code.base_field()
        n = int(code.length())
        word, erasure_vector = word_and_erasure_vector
        if len(word) != n or len(erasure_vector) != n:
            raise ValueError(
                f"word and erasure_vector must have length {n}, "
                f"got {len(word)} and {len(erasure_vector)}"
            )

        if getattr(word, "base_ring", lambda: None)() == base_field:
            word_list = list(word)
        else:
            word_list = [base_field(x) for x in word]

        m = self._coset_size
        d_local = code.local_minimum_distance()
        zero = base_field.zero()
        h0_rows = self._h0_rows
        inv_v = self._inv_v
        all_identity = self._all_identity
        corrected_flat: list[Any] = []

        for c in range(self._partition_size):
            offset = c * m
            erased_local = [l_idx for l_idx in range(m) if int(erasure_vector[offset + l_idx]) != 0]
            era_wt = len(erased_local)
            if era_wt >= d_local:
                raise DecodingError("Too many erasures in the received word")
            if era_wt == 0:
                corrected_flat.extend(word_list[offset : offset + m])
            elif all_identity:
                aligned = word_list[offset : offset + m]
                _solve_erasure_system(h0_rows, inv_v, aligned, erased_local, zero)
                corrected_flat.extend(aligned)
            else:
                perm = self._coset_perms[c]
                inv_perm = self._coset_inv_perms[c]
                aligned = [word_list[offset + idx] for idx in perm]
                erased_aligned = [inv_perm[idx] for idx in erased_local]
                _solve_erasure_system(h0_rows, inv_v, aligned, erased_aligned, zero)
                corrected_flat.extend(aligned[idx] for idx in inv_perm)

        return self._ambient_space(corrected_flat)


class TamoBargErrorDecoder(Decoder):
    """Local Reed-Solomon error decoder for :class:`TamoBargCode`."""

    def __init__(self, code: TamoBargCode) -> None:
        """Initialize the local error decoder with a shared canonical RS codec and PCM.

        Args:
            code: Associated :class:`TamoBargCode` instance.
        """
        ambient_space = code.ambient_space()
        super().__init__(code, ambient_space, "VectorEncoder")
        base_field = code.base_field()
        m = code.locality() + code.local_minimum_distance() - 1
        sub_group, sub_group_type = code.sub_group()
        partition, partition_size = code.partition()

        self._ambient_space = ambient_space
        self._local_space = VectorSpace(base_field, m)
        self._coset_size = m
        self._partition_size = partition_size
        self._num_checks = code.local_minimum_distance() - 1
        self._max_correctable_errors = self._num_checks // 2
        self._coset_perms, self._coset_inv_perms, self._all_identity = _build_coset_alignments(
            sub_group, sub_group_type, partition
        )

        # Pick a canonical coset avoiding 0 when possible for uniform affine alignment.
        if sub_group_type == "add" and partition_size >= 2:
            beta_ref = partition[1][0]
            canonical_pts = [beta_ref + h for h in sub_group]
        else:
            canonical_pts = list(sub_group)

        self._local_codec = RSCodec(base_field, canonical_pts, code.locality())
        self._local_pcm = self._local_codec.parity_check_matrix()

    def __eq__(self, other: object) -> bool:
        """Check equality between two :class:`TamoBargErrorDecoder` instances."""
        return isinstance(other, TamoBargErrorDecoder) and self.code() == other.code()

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash((type(self), self.code()))

    def _repr_(self) -> str:
        """Return string representation of the error decoder."""
        return f"Error Decoder for {self.code()}"

    def decode_to_code(self, r: Any) -> Any:
        """Locally decode each coset via batched syndrome check + shared RS codec.

        Args:
            r: Received vector in ``GF(q)^n``.

        Returns:
            Locally decoded vector in ``GF(q)^n``.

        Raises:
            ValueError: If ``r`` does not have length ``n``.
        """
        code = self.code()
        base_field = code.base_field()
        n = int(code.length())
        if len(r) != n:
            raise ValueError(f"Received word must have length {n}, got {len(r)}")

        if getattr(r, "base_ring", lambda: None)() == base_field:
            r_list = list(r)
        else:
            r_list = [base_field(x) for x in r]

        m = self._coset_size
        s = self._partition_size
        all_identity = self._all_identity
        if all_identity:
            r_mat = matrix(
                base_field,
                m,
                s,
                [r_list[c * m + l_idx] for l_idx in range(m) for c in range(s)],
            )
        else:
            perms = self._coset_perms
            r_mat = matrix(
                base_field,
                m,
                s,
                [r_list[c * m + perms[c][l_idx]] for l_idx in range(m) for c in range(s)],
            )

        syndromes = self._local_pcm * r_mat
        local_space = self._local_space
        local_codec = self._local_codec
        can_correct = self._max_correctable_errors > 0
        corrected: list[Any] = []

        for c in range(s):
            offset = c * m
            if not can_correct or syndromes.column(c).is_zero():
                corrected.extend(r_list[offset : offset + m])
                continue

            if all_identity:
                local_vec = local_space(r_list[offset : offset + m])
            else:
                perm = self._coset_perms[c]
                local_vec = local_space([r_list[offset + idx] for idx in perm])

            try:
                local_corr = local_codec.decode_errors(local_vec)
                if all_identity:
                    corrected.extend(local_corr)
                else:
                    inv_perm = self._coset_inv_perms[c]
                    corrected.extend(local_corr[idx] for idx in inv_perm)
            except (DecodingError, ValueError):
                corrected.extend(r_list[offset : offset + m])

        return self._ambient_space(corrected)


TamoBargCode._registered_encoders["VectorEncoder"] = TamoBargVectorEncoder
TamoBargCode._registered_decoders["ErasureDecoder"] = TamoBargErasureDecoder
TamoBargCode._registered_decoders["ErrorDecoder"] = TamoBargErrorDecoder
