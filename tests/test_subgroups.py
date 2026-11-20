"""Unit, property, and regression tests for tamo_barg_lrc.subgroups and polynomials."""

import pytest

from tamo_barg_lrc import GF, PolynomialRing
from tamo_barg_lrc.polynomials import (
    build_algebra_basis,
    build_annihilator_polynomial,
    build_encoding_basis,
    build_good_polynomial,
    evaluate_message_polynomial,
    interpolate_erasures,
    intersect_encoding_bases,
)
from tamo_barg_lrc.subgroups import (
    field_element_to_int,
    find_additive_subgroups,
    find_additive_subgroups_bruteforce,
    find_subspaces,
    is_closed_under_addition,
    list_canonical_additive_subgroups,
    list_multiplicative_subgroups,
    list_subgroup_cosets,
    shift_additive_subgroup,
    sort_field_elements,
)


class _InfiniteRing:
    """Lightweight non-finite-field stub replacing SageMath's ZZ in validation tests."""

    def is_finite(self) -> bool:
        return False

    def is_field(self) -> bool:
        return False


def test_field_element_to_int_and_sorting() -> None:
    """Verify deterministic integer bijection and canonical sorting across fields."""
    f13 = GF(13)
    assert field_element_to_int(f13.zero()) == 0
    assert [field_element_to_int(x) for x in sort_field_elements(f13)] == list(range(13))

    f16 = GF(16, "a")
    sorted_elems = sort_field_elements(f16)
    assert sorted_elems[0] == f16.zero()
    assert [field_element_to_int(x) for x in sorted_elems] == list(range(16))


def test_is_closed_under_addition() -> None:
    """Test additive closure check on valid subgroups, non-zero cosets, and non-subspaces."""
    f16 = GF(16, "a")
    a = f16.gen()

    # Empty subset is not a subgroup.
    assert not is_closed_under_addition([])

    # Valid subgroups of sizes 1, 2, 4, and 16.
    assert is_closed_under_addition([f16.zero()])
    assert is_closed_under_addition([f16.zero(), f16.one()])
    assert is_closed_under_addition([f16.zero(), f16.one(), a, a + 1])
    assert is_closed_under_addition(list(f16))

    # Subset missing 0 (non-zero coset) must return False.
    assert not is_closed_under_addition([f16.one(), a + 1])

    # Subset containing 0 that is not closed under addition must return False.
    assert not is_closed_under_addition([f16.zero(), f16.one(), a])

    # Characteristic 3 subset {0, 1} missing 2 = 1 + 1 must return False.
    f9 = GF(9, "a")
    assert not is_closed_under_addition([f9.zero(), f9.one()])
    assert is_closed_under_addition([f9.zero(), f9.one(), f9(2)])


def test_find_subspaces_and_validation() -> None:
    """Verify GF(p) subspace enumeration and input validation."""
    f16 = GF(16, "a")
    assert len(find_subspaces(0, 4, f16)) == 1
    assert len(find_subspaces(4, 4, f16)) == 1
    assert len(find_subspaces(2, 4, f16)) == 35

    with pytest.raises(ValueError, match="finite field"):
        find_subspaces(1, 2, _InfiniteRing())
    with pytest.raises(ValueError, match="0 <= dim <="):
        find_subspaces(5, 4, f16)
    with pytest.raises(ValueError, match="0 <= dim <="):
        find_subspaces(-1, 4, f16)


def test_find_additive_subgroups_gaussian_counts_and_bruteforce_parity() -> None:
    """Verify Gaussian binomial subgroup counts and parity with brute-force search (P3)."""
    f16 = GF(16, "a")
    # [4 choose 2]_2 = (16 - 1)(16 - 2) / ((4 - 1)(4 - 2)) = 35
    subgroups_16_4 = find_additive_subgroups(f16, 4)
    assert len(subgroups_16_4) == 35
    assert all(len(sg) == 4 and is_closed_under_addition(sg) for sg in subgroups_16_4)

    bruteforce_16_4 = find_additive_subgroups_bruteforce(f16, 4)
    assert subgroups_16_4 == bruteforce_16_4

    capped = find_additive_subgroups_bruteforce(f16, 4, max_subgroups=5)
    assert len(capped) == 5
    assert all(is_closed_under_addition(sg) for sg in capped)

    # [4 choose 2]_3 = (81 - 1)(81 - 3) / ((9 - 1)(9 - 3)) = 130
    f81 = GF(81, "a")
    subgroups_81_9 = find_additive_subgroups(f81, 9)
    assert len(subgroups_81_9) == 130
    assert all(len(sg) == 9 and is_closed_under_addition(sg) for sg in subgroups_81_9)


def test_regression_b7_large_prime_characteristics_and_invalid_sizes() -> None:
    """Regression B7: support p >= 11 and p > 36 without base-p string parsing bugs."""
    f121 = GF(11**2, "a")
    sgs_11 = find_additive_subgroups(f121, 11)
    # [2 choose 1]_11 = (121 - 1) / (11 - 1) = 12
    assert len(sgs_11) == 12
    assert all(len(sg) == 11 and is_closed_under_addition(sg) for sg in sgs_11)

    f1369 = GF(37**2, "a")
    sgs_37 = find_additive_subgroups(f1369, 37)
    # [2 choose 1]_37 = (1369 - 1) / (37 - 1) = 38
    assert len(sgs_37) == 38
    assert all(len(sg) == 37 and is_closed_under_addition(sg) for sg in sgs_37)

    f16 = GF(16, "a")
    for bad_size in (0, 6, 12, 32):
        with pytest.raises(ValueError, match="Additive subgroup size"):
            find_additive_subgroups(f16, bad_size)

    with pytest.raises(ValueError, match="positive divisor"):
        find_additive_subgroups_bruteforce(f16, 6)
    with pytest.raises(ValueError, match="max_subgroups"):
        find_additive_subgroups_bruteforce(f16, 4, max_subgroups=0)


def test_regression_b8_canonical_additive_subgroups_and_deterministic_cosets() -> None:
    """Regression B8: include trivial/full subgroups and deterministic {0} coset ordering."""
    f16 = GF(16, "a")
    canonical = list_canonical_additive_subgroups(f16)
    assert set(canonical.keys()) == {1, 2, 4, 8, 16}
    assert canonical[1] == [f16.zero()]
    assert len(canonical[16]) == 16

    sizes = [1, 2, 4, 8, 16]
    assert all(is_closed_under_addition(canonical[s]) for s in sizes)
    for s_small, s_large in zip(sizes[:-1], sizes[1:], strict=True):
        assert set(canonical[s_small]).issubset(set(canonical[s_large]))

    trivial_cosets_1 = list_subgroup_cosets(f16, [f16.zero()], "add")
    trivial_cosets_2 = list_subgroup_cosets(f16, [f16.zero()], "add")
    assert len(trivial_cosets_1) == 16
    assert [f16.zero()] in trivial_cosets_1
    assert trivial_cosets_1 == trivial_cosets_2
    assert {pt for coset in trivial_cosets_1 for pt in coset} == set(f16)

    with pytest.raises(ValueError, match="non-empty"):
        list_subgroup_cosets(f16, [], "add")
    with pytest.raises(ValueError, match="subgroup_type"):
        list_subgroup_cosets(f16, [f16.zero()], "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot contain zero"):
        list_subgroup_cosets(f16, [f16.zero(), f16.one()], "mult")


def test_regression_b13_multiplicative_subgroups_and_coset_intersections() -> None:
    """Regression B13: exact multiplicative subgroup orders over GF(81) and coset partitions."""
    f81 = GF(81, "a")
    mult_sgs = list_multiplicative_subgroups(f81)
    assert sorted(mult_sgs.keys()) == [1, 2, 4, 5, 8, 10, 16, 20, 40, 80]

    for order, sg in mult_sgs.items():
        assert len(sg) == order
        assert f81.one() in sg
        assert f81.zero() not in sg
        assert all(elem**order == f81.one() for elem in sg)

    # Check multiplicative (size 4) and additive (size 9) coset partitions from Subgroups_Test.py.
    m_cosets = list_subgroup_cosets(f81, mult_sgs[4], "mult")
    assert len(m_cosets) == 20
    assert {pt for coset in m_cosets for pt in coset} == set(f81) - {f81.zero()}

    # Trivial multiplicative subgroup cosets intersect every additive coset in <= 1 element.
    m1_cosets = list_subgroup_cosets(f81, mult_sgs[1], "mult")
    canonical_81 = list_canonical_additive_subgroups(f81)
    a9_cosets = list_subgroup_cosets(f81, canonical_81[9], "add")
    assert all(len(set(mc) & set(ac)) <= 1 for mc in m1_cosets for ac in a9_cosets)

    # Shifted complementary size-9 additive subgroups over GF(81) have |C_0 cap C_1| == 1.
    shifted_a9 = shift_additive_subgroup(f81, canonical_81[9], shift_exponent=2)
    assert is_closed_under_addition(shifted_a9)
    assert set(canonical_81[9]) & set(shifted_a9) == {f81.zero()}
    shifted_cosets = list_subgroup_cosets(f81, shifted_a9, "add")
    assert all(len(set(c0) & set(c1)) == 1 for c0 in a9_cosets for c1 in shifted_cosets)


def test_polynomial_algebra_and_lagrange_interpolation_helpers() -> None:
    """Verify univariate polynomial basis construction, intersection, and erasure recovery (P5)."""
    f16 = GF(16, "a")
    poly_ring = PolynomialRing(f16, "x")
    canonical = list_canonical_additive_subgroups(f16)
    sg0 = canonical[4]
    sg1 = shift_additive_subgroup(f16, sg0, 2)

    eval_pts = sort_field_elements(f16)
    ann = build_annihilator_polynomial(poly_ring, eval_pts)
    assert ann.degree() == 16
    assert all(ann(pt) == f16.zero() for pt in eval_pts)

    g0 = build_good_polynomial(poly_ring, sg0, "add", annihilator=ann)
    for coset in list_subgroup_cosets(f16, sg0, "add"):
        vals = {g0(pt) for pt in coset}
        assert len(vals) == 1

    alg0 = build_algebra_basis(g0, 4, annihilator=ann)
    enc0 = build_encoding_basis(poly_ring, alg0, 2, annihilator=ann)
    assert len(enc0) == 8

    g1 = build_good_polynomial(poly_ring, sg1, "add", annihilator=ann)
    enc1 = build_encoding_basis(poly_ring, build_algebra_basis(g1, 4, annihilator=ann), 2, ann)
    inter = intersect_encoding_bases(poly_ring, enc0, enc1, 16)
    degrees = [int(p.degree()) for p in inter]
    assert degrees == sorted(degrees)
    assert len(set(degrees)) == len(degrees)

    msg = [f16.one(), f16.gen()]
    msg_poly = evaluate_message_polynomial(poly_ring, inter[:2], msg)
    coset = list_subgroup_cosets(f16, sg0, "add")[0]
    true_vals = [msg_poly(pt) for pt in coset]
    damaged = [f16.zero(), true_vals[1], f16.zero(), true_vals[3]]
    recovered = interpolate_erasures(poly_ring, coset, damaged, [1, 0, 1, 0])
    assert recovered == true_vals
    assert interpolate_erasures(poly_ring, coset, true_vals, [0, 0, 0, 0]) == true_vals

    with pytest.raises(ValueError, match="same length"):
        interpolate_erasures(poly_ring, coset, true_vals[:3], [0, 0, 0, 0])
