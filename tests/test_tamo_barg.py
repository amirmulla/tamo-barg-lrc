"""Unit, property, and bug-regression tests for single-set TamoBargCode."""

import pytest

from tamo_barg_lrc import (
    GF,
    DecodingError,
    TamoBargCode,
    TamoBargErasureDecoder,
    TamoBargErrorDecoder,
    TamoBargVectorEncoder,
    VectorSpace,
    vector,
)


class _InfiniteField:
    """Lightweight non-finite-field stub replacing SageMath's QQ in validation tests."""

    def is_finite(self) -> bool:
        return False

    def is_field(self) -> bool:
        return True


def test_single_set_codes_parameters_and_properties() -> None:
    """Test [12, 8, 3] over GF(13) and [16, 5, 2] over GF(16) parameter invariants."""
    f13 = GF(13)
    code_13 = TamoBargCode(f13, 12, 8, 3, local_minimum_distance=2)
    assert code_13.length() == 12
    assert code_13.dimension() == 8
    assert code_13.locality() == 3
    assert code_13.local_minimum_distance() == 2
    assert code_13.max_dimension() == 9
    assert code_13.minimum_distance() == 3
    assert "[12, 8, 3] Tamo-Barg Code over GF(13)" in repr(code_13)

    sg_13, sg_type_13 = code_13.sub_group()
    assert sg_type_13 == "mult"
    assert len(sg_13) == 4
    parts_13, part_size_13 = code_13.partition()
    assert part_size_13 == 3
    assert len(parts_13) == 3
    assert len(code_13.evaluation_points()) == 12
    assert code_13 == TamoBargCode(f13, 12, 8, 3, local_minimum_distance=2)
    assert hash(code_13) == hash(TamoBargCode(f13, 12, 8, 3, local_minimum_distance=2))

    f16 = GF(16, "a")
    code_16 = TamoBargCode(f16, 16, 5, 2, local_minimum_distance=3)
    assert code_16.length() == 16
    assert code_16.dimension() == 5
    assert code_16.locality() == 2
    assert code_16.local_minimum_distance() == 3
    assert code_16.max_dimension() == 8
    assert code_16.minimum_distance() == 8
    sg_16, sg_type_16 = code_16.sub_group()
    assert sg_type_16 == "add"
    assert len(sg_16) == 4

    # Verify Singleton-optimal minimum_distance matches exact codeword minimum weight on [16, 2, 7].
    code_small = TamoBargCode(f16, 16, 2, 7, local_minimum_distance=2)
    gen_mat = code_small.generator_matrix()
    exact_min_wt = min(
        int((msg * gen_mat).hamming_weight()) for msg in VectorSpace(f16, 2) if not msg.is_zero()
    )
    assert exact_min_wt == code_small.minimum_distance()


def test_local_erasure_and_error_decoding() -> None:
    """Verify coset-wise local erasure interpolation and Berlekamp-Welch error decoding."""
    f13 = GF(13)
    code_13 = TamoBargCode(f13, 12, 8, 3, local_minimum_distance=2)
    msg_13 = VectorSpace(f13, 8).random_element()
    cw_13 = code_13.encode(msg_13)

    # 1 erasure in each of the 3 cosets of size 4 (3 erasures total) must be recovered locally.
    damaged_13 = vector(f13, list(cw_13))
    era_13 = vector(GF(2), [0] * 12)
    for coset_idx in range(3):
        pos = coset_idx * 4 + coset_idx
        damaged_13[pos] = f13.zero()
        era_13[pos] = GF(2).one()

    era_dec_13 = code_13.decoder("ErasureDecoder")
    assert isinstance(era_dec_13, TamoBargErasureDecoder)
    assert era_dec_13.decode_to_code((damaged_13, era_13)) == cw_13

    # 2 >= d_local erasures in a single coset must raise DecodingError.
    too_many_era = vector(GF(2), [1, 1] + [0] * 10)
    with pytest.raises(DecodingError, match="Too many erasures"):
        era_dec_13.decode_to_code((damaged_13, too_many_era))

    # [16, 5, 2] over GF(16) with d_local = 3 corrects 2 erasures or 1 error per coset.
    f16 = GF(16, "a")
    code_16 = TamoBargCode(f16, 16, 5, 2, local_minimum_distance=3)
    msg_16 = VectorSpace(f16, 5).random_element()
    cw_16 = code_16.encode(msg_16)

    damaged_16 = vector(f16, list(cw_16))
    era_16 = vector(GF(2), [0] * 16)
    for coset_idx in range(4):
        for offset in (0, 2):
            pos = coset_idx * 4 + offset
            damaged_16[pos] = f16.zero()
            era_16[pos] = GF(2).one()
    assert code_16.decoder("ErasureDecoder").decode_to_code((damaged_16, era_16)) == cw_16

    # Inject 1 error in 3 cosets and leave coset 1 error-free (exercising syndrome fast-path P1).
    err_word = vector(f16, list(cw_16))
    for coset_idx in (0, 2, 3):
        pos = coset_idx * 4 + 1
        err_word[pos] += f16.gen() ** (coset_idx + 1)
    err_dec_16 = code_16.decoder("ErrorDecoder")
    assert isinstance(err_dec_16, TamoBargErrorDecoder)
    assert err_dec_16.decode_to_code(err_word) == cw_16


def test_regression_b4_immutable_inputs_not_mutated() -> None:
    """Regression B4: ErasureDecoder and ErrorDecoder must never mutate received vectors."""
    f16 = GF(16, "a")
    code = TamoBargCode(f16, 16, 5, 2, local_minimum_distance=3)
    cw = code.encode(VectorSpace(f16, 5).random_element())

    word = vector(f16, list(cw))
    word[0] = f16.zero()
    word[5] = f16.zero()
    era_vec = vector(GF(2), [1 if idx in (0, 5) else 0 for idx in range(16)])
    word_snapshot = list(word)
    era_snapshot = list(era_vec)
    word.set_immutable()
    era_vec.set_immutable()

    corr_era = code.decoder("ErasureDecoder").decode_to_code((word, era_vec))
    assert corr_era == cw
    assert word.is_immutable()
    assert list(word) == word_snapshot
    assert list(era_vec) == era_snapshot

    r = vector(f16, list(cw))
    r[2] += f16.gen()
    r[9] += f16.one()
    r_snapshot = list(r)
    r.set_immutable()

    corr_err = code.decoder("ErrorDecoder").decode_to_code(r)
    assert corr_err == cw
    assert r.is_immutable()
    assert list(r) == r_snapshot


def test_regression_b5_decoder_repr_and_high_level_decode_to_code() -> None:
    """Regression B5: ErrorDecoder has ambient input_space and 'Error' repr."""
    f16 = GF(16, "a")
    code = TamoBargCode(f16, 16, 5, 2, local_minimum_distance=3)
    err_dec = code.decoder("ErrorDecoder")
    era_dec = code.decoder("ErasureDecoder")

    assert "Error" in repr(err_dec)
    assert "Erasure" not in repr(err_dec)
    assert "Erasure" in repr(era_dec)

    cw = code.encode(VectorSpace(f16, 5).random_element())
    r = vector(f16, list(cw))
    r[3] += f16.gen() ** 2

    # High-level C.decode_to_code validates `r in err_dec.input_space()`.
    assert code.decode_to_code(r, "ErrorDecoder") == cw

    era_pair = era_dec.input_space()((r, vector(GF(2), [0, 0, 0, 1] + [0] * 12)))
    assert code.decode_to_code(era_pair, "ErasureDecoder") == cw


def test_regression_b6_encoder_basis_slicing_and_enc_poly() -> None:
    """Regression B6: enc_basis slices to k < k_max and enc_poly evaluates to encode(msg)."""
    for field, n, k, r, d_loc in [(GF(13), 12, 8, 3, 2), (GF(16, "a"), 16, 5, 2, 3)]:
        code = TamoBargCode(field, n, k, r, local_minimum_distance=d_loc)
        enc = code.encoder("VectorEncoder")
        assert isinstance(enc, TamoBargVectorEncoder)
        assert enc.max_dimension() == code.max_dimension()
        assert k < enc.max_dimension()
        assert len(enc.enc_basis()) == k
        assert enc.generator_matrix().nrows() == k
        assert enc.generator_matrix().ncols() == n
        assert enc.generator_matrix().is_immutable()

        # Good polynomial must be constant on every coset of the partition.
        good_poly = enc.enc_good_poly()
        for coset in code.partition()[0]:
            assert len({good_poly(pt) for pt in coset}) == 1

        msg = VectorSpace(field, k).random_element()
        poly = enc.enc_poly(msg)
        evaluated = [poly(pt) for pt in code.evaluation_points()]
        assert evaluated == list(enc.encode(msg))


def test_regression_b11_upfront_parameter_validation() -> None:
    """Regression B11: invalid length, missing subgroup, or k > k_max raises ValueError."""
    f13 = GF(13)
    f16 = GF(16, "a")

    # Non-divisible length (11 not divisible by 3 + 2 - 1 = 4).
    with pytest.raises(ValueError, match="divisible by coset size"):
        TamoBargCode(f13, 11, 6, 3, 2)

    # Unavailable multiplicative subgroup of size 6 in GF(16)^* (|GF(16)^*| = 15).
    with pytest.raises(ValueError, match="No multiplicative subgroup"):
        TamoBargCode(f16, 12, 4, 5, 2, sub_group_type="mult")

    # Unavailable additive subgroup of size 6 in GF(16).
    with pytest.raises(ValueError, match="No canonical additive subgroup"):
        TamoBargCode(f16, 12, 4, 5, 2, sub_group_type="add")

    # Unavailable subgroup of size 6 under "any".
    with pytest.raises(ValueError, match="No multiplicative or additive subgroup"):
        TamoBargCode(f16, 12, 4, 5, 2, sub_group_type="any")

    # Dimension exceeding max_dimension (max_dimension = 9 for [12, k, 3] over GF(13)).
    with pytest.raises(ValueError, match="1 <= dimension <="):
        TamoBargCode(f13, 12, 10, 3, 2)
    with pytest.raises(ValueError, match="1 <= dimension <="):
        TamoBargCode(f13, 12, 0, 3, 2)

    # Invalid field, locality, local_minimum_distance, bool parameter, and sub_group_type.
    with pytest.raises(ValueError, match="finite field"):
        TamoBargCode(_InfiniteField(), 12, 6, 3, 2)
    with pytest.raises(ValueError, match="locality"):
        TamoBargCode(f13, 12, 6, 0, 2)
    with pytest.raises(ValueError, match="local_minimum_distance"):
        TamoBargCode(f13, 12, 6, 3, 1)
    with pytest.raises(ValueError, match="must be an integer"):
        TamoBargCode(f13, True, 6, 3, 2)
    with pytest.raises(ValueError, match="sub_group_type"):
        TamoBargCode(f13, 12, 6, 3, 2, sub_group_type="invalid")  # type: ignore[arg-type]

    # Decoder length mismatch raises ValueError.
    code = TamoBargCode(f13, 12, 8, 3, 2)
    with pytest.raises(ValueError, match="length 12"):
        code.decoder("ErrorDecoder").decode_to_code(vector(f13, [0] * 11))
    with pytest.raises(ValueError, match="length 12"):
        code.decoder("ErasureDecoder").decode_to_code(
            (vector(f13, [0] * 11), vector(GF(2), [0] * 12))
        )
