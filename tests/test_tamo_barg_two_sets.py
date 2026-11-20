"""Unit, property, and bug-regression tests for two-set TamoBargCodeTwoSets."""

from typing import Any

import pytest

from tamo_barg_lrc import (
    GF,
    DecodingError,
    TamoBargCodeTwoSets,
    VectorSpace,
    find_additive_subgroups,
    vector,
)


def _local_and_global_pos(
    idx0: list[list[int]], idx1: list[list[int]], row_idx: int, col_idx: int
) -> tuple[int, int]:
    """Return (local_pos_in_row, global_pos) for the unique intersection of row and col."""
    common = set(idx0[row_idx]) & set(idx1[col_idx])
    assert len(common) == 1
    global_pos = next(iter(common))
    return idx0[row_idx].index(global_pos), global_pos


def _weight5_local_codeword(
    local_grs: Any, idx0: list[list[int]], idx1: list[list[int]], row_idx: int, zero_cols: list[int]
) -> Any:
    """Construct a weight-5 local GRS codeword in an [8, 4, 5] coset vanishing on zero_cols."""
    gen = local_grs.generator_matrix()
    zero_locs = [_local_and_global_pos(idx0, idx1, row_idx, c)[0] for c in zero_cols]
    ker_vec = gen.matrix_from_columns(zero_locs).left_kernel().basis()[0]
    lcw = ker_vec * gen
    assert int(lcw.hamming_weight()) == 5
    return lcw


def test_all_four_subgroup_combinations_and_regression_b10() -> None:
    """Test ['add','add'], ['mult','mult'], ['add','mult'], ['mult','add'] (B10) & validation."""
    f64 = GF(64, "a")

    # 1. ["add", "add"] (n = 64)
    code_aa = TamoBargCodeTwoSets(f64, 64, 8, [2, 2], [3, 3], ["add", "add"])
    assert len(code_aa.evaluation_points()) == 64
    assert code_aa.num_of_sets() == 2

    # 2. ["mult", "mult"] (n = 63 over GF(64), from legacy tests/Readme.md)
    code_mm = TamoBargCodeTwoSets(f64, 63, 35, [5, 7], [3, 3], ["mult", "mult"])
    assert len(code_mm.evaluation_points()) == 63
    assert f64.zero() not in code_mm.evaluation_points()
    cw_mm = code_mm.encode(VectorSpace(f64, 35).random_element())
    r_mm = vector(f64, list(cw_mm))
    for pos in (0, 15, 31, 48):
        r_mm[pos] += f64.gen() ** (pos % 7 + 1)
    corr_mm, _ = code_mm.decoder("IterativeDecoder", 10).decode_to_code(r_mm)
    assert corr_mm == cw_mm

    # 3. ["add", "mult"] (n = 64 over GF(64), including custom additive subgroup override)
    custom_add = find_additive_subgroups(f64, 4)[0]
    code_am = TamoBargCodeTwoSets(
        f64, 64, 12, [2, 5], [3, 3], ["add", "mult"], subgroup=[custom_add, None]
    )
    assert len(code_am.evaluation_points()) == 64
    assert f64.zero() in code_am.evaluation_points()

    # 4. Regression B10: ["mult", "add"] (n = 64 over GF(64); previously crashed on missing 0)
    code_ma = TamoBargCodeTwoSets(f64, 64, 12, [5, 2], [3, 3], ["mult", "add"])
    assert len(code_ma.evaluation_points()) == 64
    assert f64.zero() in code_ma.evaluation_points()
    msg_ma = VectorSpace(f64, 12).random_element()
    cw_ma = code_ma.encode(msg_ma)
    r_ma = vector(f64, list(cw_ma))
    r_ma[0] += f64.one()  # Error right at evaluation point 0 (only in Partition 1)
    r_ma[10] += f64.gen()
    corr_ma, _ = code_ma.decoder("TwoStepsDecoder", 10).decode_to_code(r_ma)
    assert corr_ma == cw_ma

    # Regression B10 & B11: 1-set locality and invalid parameters raise ValueError upfront.
    with pytest.raises(ValueError, match="2-element sequence"):
        TamoBargCodeTwoSets(f64, 64, 8, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2-element sequence"):
        TamoBargCodeTwoSets(f64, 64, 8, [3])
    with pytest.raises(ValueError, match="1 <= dimension <="):
        TamoBargCodeTwoSets(f64, 64, 60, [2, 2], [3, 3], ["add", "add"])
    with pytest.raises(ValueError, match="not closed under addition"):
        TamoBargCodeTwoSets(
            f64,
            64,
            8,
            [2, 2],
            [3, 3],
            ["add", "add"],
            subgroup=[[f64.zero(), f64.one(), f64.gen(), f64.gen() ** 2], None],
        )


def test_regression_b9_shift_add_produces_disjoint_subgroups() -> None:
    """Regression B9: shift_add=True with ['add', 'add'] must not make both subgroups identical."""
    f64 = GF(64, "a")
    code_shifted = TamoBargCodeTwoSets(f64, 64, 8, [2, 2], [3, 3], ["add", "add"], shift_add=True)
    sg_shifted, _ = code_shifted.sub_group()
    assert sg_shifted[0] != sg_shifted[1]
    assert set(sg_shifted[0]) & set(sg_shifted[1]) == {f64.zero()}

    code_unshifted = TamoBargCodeTwoSets(
        f64, 64, 8, [2, 2], [3, 3], ["add", "add"], shift_add=False
    )
    sg_unshifted, _ = code_unshifted.sub_group()
    assert set(sg_unshifted[0]) & set(sg_unshifted[1]) == {f64.zero()}
    assert sg_shifted[0] != sg_unshifted[0]

    # Mixed modes with shift_add=True must also shift the single additive subgroup cleanly.
    code_am_shift = TamoBargCodeTwoSets(
        f64, 64, 12, [2, 5], [3, 3], ["add", "mult"], shift_add=True
    )
    assert code_am_shift.sub_group()[0][0] != sg_unshifted[0]


def test_regression_b1_iterative_decoder_partition1_uncorrectable_and_multi_iteration() -> None:
    """Regression B1: IterativeDecoder must not ignore Partition 1 uncorrectable errors."""
    f64 = GF(64, "a")
    code = TamoBargCodeTwoSets(f64, 64, 6, [4, 4], [5, 5], ["add", "add"])
    idx0, idx1 = code.evaluation_points_idx()
    cw = code.encode(VectorSpace(f64, 6).random_element())
    dec_iter = code.decoder("IterativeDecoder", 10)

    # Case A: 3 errors in Partition 0 coset 0 -> P0 fails in pass 1, P1 resolves in pass 1,
    # and pass 2 confirms convergence (num_itr == 2).
    r_case_a = vector(f64, list(cw))
    for k_i, pos in enumerate(idx0[0][:3]):
        r_case_a[pos] += f64.gen() ** (k_i + 1)
    corr_a, itr_a = dec_iter.decode_to_code(r_case_a)
    assert corr_a == cw
    assert itr_a > 1

    # Case B (exact B1 trigger): Rows 0, 1, 2 each have 3 errors at distance 2 from a weight-5
    # local codeword, so Partition 0 misdecodes all 3 rows WITHOUT raising DecodingError
    # (`uncorrectable_err[0] == False`). After Partition 0 misdecodes:
    # - Column 0 has 3 errors -> Partition 1 raises DecodingError (`uncorrectable_err[1] == True`)!
    # - Columns 1..7 each have <= 2 errors -> Partition 1 corrects all of them in pass 1!
    # Legacy bitwise-AND convergence check ignored `uncorrectable_err[1]` and evaluated to True,
    # and terminated after pass 1 leaving 3 errors in Column 0. Fixed code runs pass 2 and recovers.
    local_codes_p0 = dec_iter._partitions_local_codes[0]
    lcw0 = _weight5_local_codeword(local_codes_p0[0], idx0, idx1, 0, [5, 6, 7])
    lcw1 = _weight5_local_codeword(local_codes_p0[1], idx0, idx1, 1, [3, 4, 7])
    lcw2 = (f64.gen()) * _weight5_local_codeword(local_codes_p0[2], idx0, idx1, 2, [1, 2, 6])

    r_case_b = vector(f64, list(cw))
    for col in (1, 2, 3):
        lp, gp = _local_and_global_pos(idx0, idx1, 0, col)
        r_case_b[gp] += lcw0[lp]
    for col in (1, 5, 6):
        lp, gp = _local_and_global_pos(idx0, idx1, 1, col)
        r_case_b[gp] += lcw1[lp]
    for col in (3, 5, 7):
        lp, gp = _local_and_global_pos(idx0, idx1, 2, col)
        r_case_b[gp] += lcw2[lp]

    corr_after_1, itr_1 = code.decoder("IterativeDecoder", 1).decode_to_code(r_case_b)
    assert itr_1 == 1
    assert int((corr_after_1 - cw).hamming_weight()) == 3

    corr_full, itr_full = dec_iter.decode_to_code(r_case_b)
    assert corr_full == cw
    assert itr_full == 2


def test_regression_b2_erasure_error_decoders_d_local_minus_one_lagrange_recovery() -> None:
    """Regression B2: d_local - 1 DUE erasures recover via Lagrange and clear erasure flags."""
    f64 = GF(64, "a")
    code = TamoBargCodeTwoSets(f64, 64, 6, [4, 4], [5, 5], ["add", "add"])
    idx0, idx1 = code.evaluation_points_idx()
    cw = code.encode(VectorSpace(f64, 6).random_element())

    # Inject 12 errors across rows 0..3 x cols 0..3 (3 errors per row and 3 per col > t_local = 2).
    # Hard-decision IterativeDecoder stalls completely, whereas with erasure_start_itr=1, pass 1
    # marks rows 0..3 and unmarks clean cols 4..7, leaving exactly 4 == d_local - 1 erasures per row
    # for pass 2 Lagrange interpolation!
    r = vector(f64, list(cw))
    row_col_sets = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    for row_i, cols in enumerate(row_col_sets):
        for k_i, col in enumerate(cols):
            _, gp = _local_and_global_pos(idx0, idx1, row_i, col)
            r[gp] += f64.gen() ** (7 * row_i + 3 * k_i + 1)

    hard_corr, _ = code.decoder("IterativeDecoder", 10).decode_to_code(r)
    assert hard_corr != cw

    dec_iter_ee = code.decoder("IterativeErasureErrorDecoder", 10, 1)
    (corr_ee, era_vec), itr_ee = dec_iter_ee.decode_to_code(r)
    assert corr_ee == cw
    assert int(era_vec.hamming_weight()) == 0
    assert itr_ee == 2

    dec_2step_ee = code.decoder("TwoStepsErasureErrorDecoder", 10, 1)
    corr_2step, itr_2step, rem_era = dec_2step_ee.decode_to_code(r)
    assert corr_2step == cw
    assert rem_era == 0
    assert itr_2step == 2


def test_regression_b3_and_b5_global_ee_decoder_lagrange_branch_and_input_space() -> None:
    """Regression B3 & B5: GlobalErasureErrorDecoder handles d_design - 1 erasures & input_space."""
    f64 = GF(64, "a")
    code = TamoBargCodeTwoSets(f64, 64, 8, [2, 2], [3, 3], ["add", "add"])
    cw = code.encode(VectorSpace(f64, 8).random_element())
    d_design = code.design_distance()
    dec_glob_ee = code.decoder("GlobalErasureErrorDecoder")

    # Exactly d_design - 1 erasures hits `if era_wt == d_glob - 1:` calling `evaluation_points()`.
    num_era = d_design - 1
    damaged = vector(f64, list(cw))
    era_flags = [0] * 64
    for idx in range(num_era):
        damaged[idx] = f64.zero()
        era_flags[idx] = 1
    era_vec = vector(GF(2), era_flags)

    assert dec_glob_ee.decode_to_code((damaged, era_vec)) == cw

    # High-level C.decode_to_code validates Cartesian product input_space (B5).
    pair_elem = dec_glob_ee.input_space()((damaged, era_vec))
    assert code.decode_to_code(pair_elem, "GlobalErasureErrorDecoder") == cw

    # Combined erasures + errors within d_design bound: 2 erasures + 1 error when d_design >= 5.
    assert d_design >= 5
    mixed_word = vector(f64, list(cw))
    mixed_word[0] = f64.zero()
    mixed_word[1] = f64.zero()
    mixed_word[10] += f64.gen()
    mixed_era = vector(GF(2), [1, 1] + [0] * 62)
    assert dec_glob_ee.decode_to_code((mixed_word, mixed_era)) == cw

    # d_design erasures exceeds global GRS capability and must raise DecodingError.
    too_many_era = vector(GF(2), [1] * d_design + [0] * (64 - d_design))
    with pytest.raises(DecodingError, match="Too many erasures"):
        dec_glob_ee.decode_to_code((damaged, too_many_era))


def test_regression_b4_and_canonical_return_contracts_all_six_decoders() -> None:
    """Regression B4: immutable inputs and exact return shapes across all 6 decoders."""
    f64 = GF(64, "a")
    code = TamoBargCodeTwoSets(f64, 64, 8, [2, 2], [3, 3], ["add", "add"])
    msg = VectorSpace(f64, 8).random_element()
    cw = code.encode(msg)

    r = vector(f64, list(cw))
    r[5] += f64.gen() ** 3
    r_snapshot = list(r)
    r.set_immutable()

    era_vec = vector(GF(2), [0] * 64)
    era_vec.set_immutable()

    # 1. GlobalDecoder -> Vector
    out_glob = code.decoder("GlobalDecoder").decode_to_code(r)
    assert out_glob == cw
    assert out_glob.parent() == code.ambient_space()

    # 2. GlobalErasureErrorDecoder -> Vector
    out_glob_ee = code.decoder("GlobalErasureErrorDecoder").decode_to_code((r, era_vec))
    assert out_glob_ee == cw
    assert out_glob_ee.parent() == code.ambient_space()

    # 3. IterativeDecoder -> (Vector, int)
    dec_iter = code.decoder("IterativeDecoder", 8)
    out_iter = dec_iter.decode_to_code(r)
    assert isinstance(out_iter, tuple) and len(out_iter) == 2
    assert out_iter[0] == cw and isinstance(out_iter[1], int)
    assert dec_iter.decode_to_message(r) == msg

    # 4. IterativeErasureErrorDecoder -> ((Vector, Vector), int)
    dec_iter_ee = code.decoder("IterativeErasureErrorDecoder", 8, 4)
    out_iter_ee = dec_iter_ee.decode_to_code(r)
    assert isinstance(out_iter_ee, tuple) and len(out_iter_ee) == 2
    (corr_iee, out_era), itr_iee = out_iter_ee
    assert corr_iee == cw
    assert out_era.parent() == VectorSpace(GF(2), 64)
    assert isinstance(itr_iee, int)
    assert dec_iter_ee.decode_to_message(r) == msg

    # 5. TwoStepsDecoder -> (Vector, int)
    dec_2s = code.decoder("TwoStepsDecoder", 8)
    out_2s = dec_2s.decode_to_code(r)
    assert isinstance(out_2s, tuple) and len(out_2s) == 2
    assert out_2s[0] == cw and isinstance(out_2s[1], int)
    assert dec_2s.decode_to_message(r) == msg

    # 6. TwoStepsErasureErrorDecoder -> (Vector, int, int)
    dec_2s_ee = code.decoder("TwoStepsErasureErrorDecoder", 8, 4)
    out_2s_ee = dec_2s_ee.decode_to_code(r)
    assert isinstance(out_2s_ee, tuple) and len(out_2s_ee) == 3
    assert out_2s_ee[0] == cw
    assert isinstance(out_2s_ee[1], int) and isinstance(out_2s_ee[2], int)
    assert dec_2s_ee.decode_to_message(r) == msg

    # Confirm r remained immutable and unmodified after all 6 decoders ran.
    assert r.is_immutable()
    assert list(r) == r_snapshot


def test_regression_p1_p2_p5_p6_basis_degrees_convergence_and_minimum_distance_py() -> None:
    """Verify basis degrees (P5), stall convergence (P1/P2), and minimum_distance_py (P6)."""
    f64 = GF(64, "a")
    code = TamoBargCodeTwoSets(f64, 64, 8, [2, 2], [3, 3], ["add", "add"])
    enc = code.encoder()
    assert code.dimension() < code.max_dimension()

    # P5 & B6: enc_basis has strictly increasing polynomial degrees and matches encode(msg).
    basis = enc.enc_basis()
    assert len(basis) == 8
    degrees = [int(poly.degree()) for poly in basis]
    assert all(degrees[i] < degrees[i + 1] for i in range(len(degrees) - 1))

    msg = VectorSpace(f64, 8).random_element()
    cw = enc.encode(msg)
    msg_poly = enc.enc_poly(msg)
    assert [msg_poly(pt) for pt in code.evaluation_points()] == list(cw)

    # P1 & P2: clean codeword finishes in 1 iteration; stalled high-error word halts early.
    _, clean_itr = code.decoder("IterativeDecoder", 10).decode_to_code(cw)
    assert clean_itr == 1
    (_, _), clean_ee_itr = code.decoder("IterativeErasureErrorDecoder", 10, 2).decode_to_code(cw)
    assert clean_ee_itr == 1

    # Inject 2 errors into every row/col of the first 4x4 component so local decoders stall.
    idx0, idx1 = code.evaluation_points_idx()
    stalled_r = vector(f64, list(cw))
    for r_i in range(4):
        for c_i in range(4):
            common = set(idx0[r_i]) & set(idx1[c_i])
            if common:
                stalled_r[next(iter(common))] += f64.gen() ** (r_i + c_i + 1)
    (_, _), stall_itr = code.decoder("IterativeErasureErrorDecoder", 20, 1).decode_to_code(
        stalled_r
    )
    assert stall_itr < 20

    # P6: projective minimum_distance_py on GF(16), k=3 matches exact minimum codeword weight.
    f16 = GF(16, "a")
    small_code = TamoBargCodeTwoSets(f16, 16, 3, [2, 2], [3, 3], ["add", "add"])
    d_py = small_code.minimum_distance_py()
    assert d_py >= small_code.design_distance()

    gen_mat = small_code.generator_matrix()
    exact_d = min(
        int((m_vec * gen_mat).hamming_weight())
        for m_vec in VectorSpace(f16, 3)
        if not m_vec.is_zero()
    )
    assert d_py == exact_d

    try:
        sage_d = int(small_code.minimum_distance())
    except ImportError:
        # Modular passagemath environments without libgap use exact enumeration above.
        pass
    else:
        assert d_py == sage_d
