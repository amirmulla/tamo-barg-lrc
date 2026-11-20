"""Univariate polynomial algebra utilities for Tamo-Barg encoding and erasure recovery."""

from collections.abc import Sequence
from typing import Any, Literal

from tamo_barg_lrc._ff import matrix, prod
from tamo_barg_lrc.subgroups import field_element_to_int

__all__ = [
    "build_annihilator_polynomial",
    "build_good_polynomial",
    "build_algebra_basis",
    "build_encoding_basis",
    "intersect_encoding_bases",
    "evaluate_message_polynomial",
    "interpolate_erasures",
]


def build_annihilator_polynomial(poly_ring: Any, evaluation_points: Sequence[Any]) -> Any:
    """Construct the monic vanishing polynomial ``prod_{pt}(x - pt)`` via a product tree.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        evaluation_points: Sequence of evaluation points in ``GF(q)``.

    Returns:
        Monic polynomial in ``poly_ring`` vanishing on ``evaluation_points``.
    """
    x = poly_ring.gen()
    base_ring = poly_ring.base_ring()
    n_pts = len(evaluation_points)
    if hasattr(base_ring, "is_finite") and base_ring.is_finite():
        q = int(base_ring.cardinality())
        if n_pts == q:
            pts = [base_ring(pt) for pt in evaluation_points]
            if len(set(pts)) == q:
                return x**q - x
            return prod([x - pt for pt in pts], poly_ring.one())
        if n_pts == q - 1:
            pts = [base_ring(pt) for pt in evaluation_points]
            unique_pts = set(pts)
            if len(unique_pts) == q - 1 and base_ring.zero() not in unique_pts:
                return x ** (q - 1) - poly_ring.one()
            return prod([x - pt for pt in pts], poly_ring.one())
    return prod([x - base_ring(pt) for pt in evaluation_points], poly_ring.one())


def _build_linearized_subspace_polynomial(poly_ring: Any, subgroup: Sequence[Any]) -> Any:
    """Build ``prod_{h in H}(x - h)`` via linearized p-polynomial recurrence for GF(p)-subspaces."""
    x = poly_ring.gen()
    base_ring = poly_ring.base_ring()
    sg_coerced = [base_ring(h) for h in subgroup]
    if len(sg_coerced) == 1:
        return x - sg_coerced[0]

    if hasattr(base_ring, "is_finite") and base_ring.is_finite():
        p = int(base_ring.characteristic())
        zero = base_ring.zero()
        sg_set = set(sg_coerced)
        if zero in sg_set and len(sg_set) == len(sg_coerced):
            span_list = [zero]
            span_set = {zero}
            basis_elems: list[Any] = []
            is_subspace = True
            for h in sg_coerced:
                if h not in span_set:
                    basis_elems.append(h)
                    base_len = len(span_list)
                    if base_len * p > len(sg_set):
                        is_subspace = False
                        break
                    new_elems = [elem + c * h for c in range(1, p) for elem in span_list[:base_len]]
                    if not all(e in sg_set for e in new_elems):
                        is_subspace = False
                        break
                    span_list.extend(new_elems)
                    span_set.update(new_elems)
                    if len(span_set) == len(sg_set):
                        break

            if is_subspace and len(span_set) == len(sg_set):
                # L_0(x) = x, L_{i+1}(x) = L_i(x)^p - L_i(b_i)^(p-1) * L_i(x)
                # coeffs[k] holds the coefficient of x^(p^k)
                coeffs = [base_ring.one()]
                for b in basis_elems:
                    val = zero
                    bp = b
                    for c in coeffs:
                        val += c * bp
                        bp = bp**p
                    factor = val ** (p - 1)
                    new_coeffs = [zero] * (len(coeffs) + 1)
                    for k, c in enumerate(coeffs):
                        new_coeffs[k + 1] += c**p
                        new_coeffs[k] -= factor * c
                    coeffs = new_coeffs

                deg = p ** len(basis_elems)
                dense = [zero] * (deg + 1)
                for k, c in enumerate(coeffs):
                    dense[p**k] = c
                return poly_ring(dense)

    return prod([x - h for h in sg_coerced], poly_ring.one())


def build_good_polynomial(
    poly_ring: Any,
    subgroup: Sequence[Any],
    subgroup_type: Literal["add", "mult"],
    annihilator: Any | None = None,
) -> Any:
    """Construct a Tamo-Barg good polynomial constant on cosets of ``subgroup``.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        subgroup: Non-empty additive or multiplicative subgroup of ``GF(q)``.
        subgroup_type: ``"add"`` (uses ``prod_{h in H}(x - h)``) or ``"mult"``
            (uses ``x^|H|``).
        annihilator: Optional vanishing polynomial used to reduce degree if needed.

    Returns:
        Good polynomial in ``poly_ring``.

    Raises:
        ValueError: If ``subgroup_type`` is not ``"add"`` or ``"mult"``, or if
            ``subgroup`` is empty.
    """
    if subgroup_type not in ("add", "mult"):
        raise ValueError(f"subgroup_type must be 'add' or 'mult', got {subgroup_type!r}")
    if len(subgroup) < 1:
        raise ValueError("subgroup must be non-empty")

    x = poly_ring.gen()
    if subgroup_type == "mult":
        good = x ** len(subgroup)
    else:
        good = _build_linearized_subspace_polynomial(poly_ring, subgroup)

    if annihilator is not None and good.degree() >= annihilator.degree():
        reduced = good % annihilator
        if not reduced.is_zero():
            good = reduced
    return good


def build_algebra_basis(
    good_poly: Any, partition_size: int, annihilator: Any | None = None
) -> list[Any]:
    """Generate powers ``[1, g, g^2, ..., g^(partition_size - 1)]`` of a good polynomial.

    Args:
        good_poly: Good polynomial ``g(x)`` in ``GF(q)['x']``.
        partition_size: Number of coset blocks in the partition.
        annihilator: Optional vanishing polynomial used for modular reduction.

    Returns:
        List of ``partition_size`` powers of ``good_poly``.

    Raises:
        ValueError: If ``partition_size < 0``.
    """
    count = int(partition_size)
    if count < 0:
        raise ValueError(f"partition_size ({count}) must be non-negative")
    if count == 0:
        return []

    poly_ring = good_poly.parent()
    curr = poly_ring.one()
    if annihilator is not None and curr.degree() >= annihilator.degree():
        curr = curr % annihilator

    basis = [curr]
    for _ in range(1, count):
        curr = curr * good_poly
        if annihilator is not None and curr.degree() >= annihilator.degree():
            curr = curr % annihilator
        basis.append(curr)
    return basis


def build_encoding_basis(
    poly_ring: Any,
    algebra_basis: Sequence[Any],
    locality: int,
    annihilator: Any | None = None,
) -> list[Any]:
    """Construct the Tamo-Barg encoding basis ``{x^i * g(x)^j}`` sorted by degree.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        algebra_basis: Powers ``[1, g, ..., g^(s - 1)]`` of the good polynomial.
        locality: Locality parameter ``r`` (exponents ``0 <= i < r``).
        annihilator: Optional vanishing polynomial used for modular reduction.

    Returns:
        Deterministically sorted list of ``locality * len(algebra_basis)`` basis
        polynomials ordered primarily by ascending degree.

    Raises:
        ValueError: If ``locality < 0``.
    """
    r_int = int(locality)
    if r_int < 0:
        raise ValueError(f"locality ({r_int}) must be non-negative")
    if r_int == 0 or not algebra_basis:
        return []

    enc_basis: list[Any] = []
    for shift_deg in range(r_int):
        for p in algebra_basis:
            prod_poly = p.shift(shift_deg) if shift_deg > 0 else p
            if annihilator is not None and prod_poly.degree() >= annihilator.degree():
                prod_poly = prod_poly % annihilator
            enc_basis.append(prod_poly)

    enc_basis.sort(key=lambda poly: poly.degree())
    if any(enc_basis[i].degree() == enc_basis[i + 1].degree() for i in range(len(enc_basis) - 1)):
        enc_basis.sort(
            key=lambda poly: (
                poly.degree(),
                [field_element_to_int(c) for c in reversed(poly.list())],
            )
        )
    return enc_basis


def intersect_encoding_bases(
    poly_ring: Any,
    basis_a: Sequence[Any],
    basis_b: Sequence[Any],
    length: int,
) -> list[Any]:
    """Compute a degree-minimizing basis of ``span(basis_a) & span(basis_b)``.

    Uses Zassenhaus subspace intersection followed by reverse-column reduced row
    echelon form so pivots eliminate high-degree terms first, yielding basis
    polynomials of strictly ascending, minimal degrees.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        basis_a: First polynomial basis sequence.
        basis_b: Second polynomial basis sequence.
        length: Code length upper-bounding polynomial degrees (``deg < length``).

    Returns:
        Basis of ``span(basis_a) & span(basis_b)`` in ``GF(q)[x]_{< length}``,
        sorted in strictly ascending polynomial degree.

    Raises:
        ValueError: If ``length < 0``.
    """
    n_len = int(length)
    if n_len < 0:
        raise ValueError(f"length ({n_len}) must be non-negative")
    if n_len == 0 or not basis_a or not basis_b:
        return []

    base_field = poly_ring.base_ring()
    num_cols = max(
        n_len,
        max((int(p.degree()) + 1 for p in basis_a), default=0),
        max((int(p.degree()) + 1 for p in basis_b), default=0),
    )

    mat_a_rev = matrix(
        base_field,
        len(basis_a),
        num_cols,
        [p.padded_list(num_cols)[::-1] for p in basis_a],
    )
    mat_b_rev = matrix(
        base_field,
        len(basis_b),
        num_cols,
        [p.padded_list(num_cols)[::-1] for p in basis_b],
    )

    if len(basis_a) + len(basis_b) > num_cols:
        # Dual orthogonal method: U cap V = (U^perp + V^perp)^perp.
        # Put the intersection basis in RREF (reversed column order) for minimal degrees.
        h_a = mat_a_rev.right_kernel_matrix()
        h_b = mat_b_rev.right_kernel_matrix()
        rref_rev = h_a.stack(h_b).right_kernel_matrix().echelon_form()
    else:
        ker = mat_a_rev.stack(-mat_b_rev).left_kernel_matrix()
        if ker.nrows() == 0:
            return []
        basis_mat_rev = ker.matrix_from_columns(list(range(mat_a_rev.nrows()))) * mat_a_rev
        rref_rev = basis_mat_rev.echelon_form()

    last_col = num_cols - 1
    zero = base_field.zero()
    polys: list[Any] = []
    # rref_rev has leading pivots in strictly increasing column order (decreasing degree),
    # so iterating in reverse yields polynomials in strictly ascending degree.
    for row_idx in range(rref_rev.nrows() - 1, -1, -1):
        r_rev = rref_rev[row_idx]
        nz = r_rev.nonzero_positions()
        if not nz:
            continue
        deg = last_col - nz[0]
        if deg < n_len:
            coeffs: list[Any] = [zero] * (deg + 1)
            for idx in nz:
                coeffs[last_col - idx] = r_rev[idx]
            polys.append(poly_ring(coeffs))

    return polys


def evaluate_message_polynomial(
    poly_ring: Any, basis: Sequence[Any], message: Sequence[Any]
) -> Any:
    """Form the message encoding polynomial ``sum_i message[i] * basis[i]``.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        basis: Encoding polynomial basis of length ``k``.
        message: Message vector or sequence of length ``k`` over ``GF(q)``.

    Returns:
        Encoding polynomial in ``poly_ring``.

    Raises:
        ValueError: If ``len(message) != len(basis)``.
    """
    if len(message) != len(basis):
        raise ValueError(f"Message length ({len(message)}) must match basis length ({len(basis)})")
    base_ring = poly_ring.base_ring()
    return sum(
        (base_ring(message[i]) * basis[i] for i in range(len(basis))),
        poly_ring.zero(),
    )


def interpolate_erasures(
    poly_ring: Any,
    eval_points: Sequence[Any],
    values: Sequence[Any],
    erasure_mask: Sequence[Any],
) -> list[Any]:
    """Recover erased evaluations via univariate Lagrange interpolation.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        eval_points: Evaluation points ``[alpha_0, ..., alpha_{m - 1}]``.
        values: Received symbol values aligned with ``eval_points``.
        erasure_mask: Binary/integer erasure indicators (``0`` = known, non-zero = erased).

    Returns:
        New list of values with erased positions replaced by interpolated evaluations.

    Raises:
        ValueError: If ``eval_points``, ``values``, and ``erasure_mask`` differ in length.
    """
    n = len(eval_points)
    if len(values) != n or len(erasure_mask) != n:
        raise ValueError("eval_points, values, and erasure_mask must all have the same length")

    base_ring = poly_ring.base_ring()
    xs: list[Any] = []
    ys: list[Any] = []
    erased_indices: list[int] = []
    for i in range(n):
        if int(erasure_mask[i]) == 0:
            xs.append(base_ring(eval_points[i]))
            ys.append(base_ring(values[i]))
        else:
            erased_indices.append(i)

    out = list(values)
    if not erased_indices:
        return out

    s = len(xs)
    if s == 0:
        raise ValueError("Cannot interpolate erasures with no known evaluation points")

    # 1D in-place Newton divided-difference table: dd[j] becomes f[x_0, ..., x_j].
    dd = list(ys)
    try:
        for j in range(1, s):
            for i in range(s - 1, j - 1, -1):
                dd[i] = (dd[i] - dd[i - 1]) / (xs[i] - xs[i - j])
    except ZeroDivisionError as exc:
        raise ValueError("eval_points must be distinct at known positions") from exc

    # Direct Horner evaluation in Newton basis at each erased point (no monomial expansion).
    for idx in erased_indices:
        z = base_ring(eval_points[idx])
        acc = dd[s - 1]
        for i in range(s - 2, -1, -1):
            acc = acc * (z - xs[i]) + dd[i]
        out[idx] = acc
    return out
