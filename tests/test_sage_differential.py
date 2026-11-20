"""Differential tests verifying bit-exact equivalence with SageMath and Cython/NumPy parity.

Automatically skipped when SageMath / passagemath is not installed.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from typing import Any

import pytest

# Skip entire module cleanly in zero-Sage environments.
sage_ff_mod = pytest.importorskip("sage.rings.finite_rings.finite_field_constructor")
try:
    import sage.all  # noqa: F401
except ModuleNotFoundError:
    import sage.all__sagemath_modules  # noqa: F401

from sage.coding.decoder import DecodingError as SageDecodingError  # noqa: E402
from sage.coding.grs_code import GeneralizedReedSolomonCode  # noqa: E402
from sage.matrix.constructor import matrix as sage_matrix  # noqa: E402
from sage.modules.free_module import VectorSpace as SageVectorSpace  # noqa: E402
from sage.modules.free_module_element import vector as sage_vector  # noqa: E402
from sage.rings.finite_rings.finite_field_constructor import GF as SageGF  # noqa: E402
from sage.rings.polynomial.polynomial_ring_constructor import (  # noqa: E402
    PolynomialRing as SagePolynomialRing,
)

from tamo_barg_lrc import (  # noqa: E402
    GF,
    DecodingError,
    RSCodec,
    TamoBargCode,
    TamoBargCodeTwoSets,
    vector,
)
from tamo_barg_lrc.simulation.runner import compute_code_weight_distribution  # noqa: E402


def _make_sage_field(q: int) -> Any:
    """Construct SageMath GF(q) matching our Conway-polynomial representation."""
    if q in (2, 13):
        return SageGF(q)
    return SageGF(q, "a")


def _sage_to_int(elem: Any) -> int:
    """Convert a SageMath finite-field element to its canonical integer encoding."""
    if hasattr(elem, "to_integer"):
        return int(elem.to_integer())
    return int(elem)


def _sage_from_int(field: Any, val: int) -> Any:
    """Convert a canonical integer in [0, q - 1] to a SageMath finite-field element."""
    if hasattr(field, "from_integer"):
        return field.from_integer(int(val))
    return field(int(val))


def _mat_to_int_rows(mat: Any) -> list[list[int]]:
    """Convert either an FFMatrix or Sage matrix into a nested list of ints."""
    return [[_sage_to_int(mat[r, c]) for c in range(mat.ncols())] for r in range(mat.nrows())]


def _vec_to_ints(vec: Any) -> list[int]:
    """Convert either an FFVector or Sage vector into a list of ints."""
    return [_sage_to_int(x) for x in vec]


# ---------------------------------------------------------------------------
# 1. Finite-field arithmetic & to_integer() differential tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [2, 9, 13, 16, 64, 81, 121, 256, 1369])
def test_field_arithmetic_and_to_integer_matches_sage(q: int) -> None:
    """Verify bit-exact field arithmetic and to_integer() encoding against SageMath."""
    f_ours = GF(q, "a") if q not in (2, 13) else GF(q)
    f_sage = _make_sage_field(q)

    assert f_ours.order() == int(f_sage.order())
    assert f_ours.characteristic() == int(f_sage.characteristic())
    assert f_ours.degree() == int(f_sage.degree())
    assert f_ours.zero().to_integer() == _sage_to_int(f_sage.zero()) == 0
    assert f_ours.one().to_integer() == _sage_to_int(f_sage.one()) == 1
    assert f_ours.gen().to_integer() == _sage_to_int(f_sage.gen())

    # Check powers of the generator agree bit-for-bit.
    g_ours = f_ours.gen()
    g_sage = f_sage.gen()
    for exp in range(min(q, 128)):
        assert (g_ours**exp).to_integer() == _sage_to_int(g_sage**exp)

    # Check all pairs for small fields, or deterministic boundary + random pairs for large fields.
    if q <= 16:
        pairs = [(a, b) for a in range(q) for b in range(q)]
    else:
        rng = random.Random(1000 + q)
        p = f_ours.characteristic()
        special = sorted({0, 1, p - 1, p % q, q - 1})
        pairs = [(a, b) for a in special for b in special]
        pairs.extend((rng.randrange(q), rng.randrange(q)) for _ in range(80))

    for a_i, b_i in pairs:
        a_o, b_o = f_ours.from_integer(a_i), f_ours.from_integer(b_i)
        a_s, b_s = _sage_from_int(f_sage, a_i), _sage_from_int(f_sage, b_i)

        assert (a_o + b_o).to_integer() == _sage_to_int(a_s + b_s)
        assert (a_o - b_o).to_integer() == _sage_to_int(a_s - b_s)
        assert (-a_o).to_integer() == _sage_to_int(-a_s)
        assert (a_o * b_o).to_integer() == _sage_to_int(a_s * b_s)
        if b_i != 0:
            assert (a_o / b_o).to_integer() == _sage_to_int(a_s / b_s)
        for exp in (0, 1, 2, 5):
            assert (a_o**exp).to_integer() == _sage_to_int(a_s**exp)
        if a_i != 0:
            for exp in (-1, -3):
                assert (a_o**exp).to_integer() == _sage_to_int(a_s**exp)


# ---------------------------------------------------------------------------
# 2. RSCodec error & error-erasure decoding vs Sage GeneralizedReedSolomonCode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "include_zero", "n", "k"),
    [
        (16, True, 16, 8),  # binary extension with alpha = 0, d = 9
        (16, True, 16, 15),  # binary extension with alpha = 0, d = 2 (t = 0)
        (13, True, 12, 6),  # prime field with alpha = 0, d = 7
        (13, True, 12, 11),  # prime field with alpha = 0, d = 2 (t = 0)
        (9, True, 9, 4),  # odd prime power with alpha = 0, d = 6
        (64, True, 16, 8),  # GF(64) with alpha = 0, d = 9
        (64, False, 15, 7),  # GF(64) without alpha = 0, d = 9
    ],
)
def test_rscodec_matches_sage_grs_gao_and_error_erasure(
    q: int, include_zero: bool, n: int, k: int
) -> None:
    """Verify RSCodec matches Sage GRS (Gao + ErrorErasure) on edge and random patterns."""
    f_ours = GF(q, "a") if q not in (2, 13) else GF(q)
    f_sage = _make_sage_field(q)

    start = 0 if include_zero else 1
    eval_ints = list(range(start, start + n))
    eval_ours = [f_ours.from_integer(i) for i in eval_ints]
    eval_sage = [_sage_from_int(f_sage, i) for i in eval_ints]

    rs_ours = RSCodec(f_ours, eval_ours, k)
    grs_sage = GeneralizedReedSolomonCode(eval_sage, k)
    d = n - k + 1
    t = (d - 1) // 2

    # Generator matrix bit-exact parity.
    assert _mat_to_int_rows(rs_ours.generator_matrix()) == _mat_to_int_rows(
        grs_sage.generator_matrix()
    )

    rng = random.Random(2000 + q * 31 + n * 7 + k)
    msg_ints = [rng.randrange(q) for _ in range(k)]
    cw_ours = (
        vector(f_ours, [f_ours.from_integer(x) for x in msg_ints]) * rs_ours.generator_matrix()
    )
    cw_sage = (
        sage_vector(f_sage, [_sage_from_int(f_sage, x) for x in msg_ints])
        * grs_sage.generator_matrix()
    )
    cw_ints = _vec_to_ints(cw_ours)
    assert cw_ints == _vec_to_ints(cw_sage)

    gao_dec = grs_sage.decoder("Gao")
    ee_dec = grs_sage.decoder("ErrorErasure")
    gf2_ours = GF(2)
    gf2_sage = SageGF(2)

    # --- Pure error decoding: 0 .. t errors (explicitly including alpha = 0 when present) ---
    for wt in range(0, t + 1):
        positions = list(range(wt)) if (include_zero and wt > 0) else rng.sample(range(n), wt)
        r_ints = list(cw_ints)
        for pos in positions:
            delta = rng.randrange(1, q)
            r_ints[pos] = (
                f_ours.from_integer(r_ints[pos]) + f_ours.from_integer(delta)
            ).to_integer()

        r_ours = vector(f_ours, [f_ours.from_integer(x) for x in r_ints])
        r_sage = sage_vector(f_sage, [_sage_from_int(f_sage, x) for x in r_ints])

        dec_o = rs_ours.decode_errors(r_ours)
        dec_s = gao_dec.decode_to_code(r_sage)
        assert _vec_to_ints(dec_o) == cw_ints
        assert _vec_to_ints(dec_o) == _vec_to_ints(dec_s)

    # --- d == 2 (t == 0): 1 error must raise DecodingError in both ---
    if d == 2:
        r_ints = list(cw_ints)
        r_ints[0] = (f_ours.from_integer(r_ints[0]) + f_ours.one()).to_integer()
        r_ours = vector(f_ours, [f_ours.from_integer(x) for x in r_ints])
        r_sage = sage_vector(f_sage, [_sage_from_int(f_sage, x) for x in r_ints])
        with pytest.raises(DecodingError):
            rs_ours.decode_errors(r_ours)
        with pytest.raises(SageDecodingError):
            gao_dec.decode_to_code(r_sage)

    # --- Beyond-radius error decoding: mandatory syndrome verification ---
    for extra in (1, 2):
        wt = t + extra
        if wt > n:
            continue
        for _ in range(5):
            r_ints = list(cw_ints)
            for pos in rng.sample(range(n), wt):
                delta = rng.randrange(1, q)
                r_ints[pos] = (
                    f_ours.from_integer(r_ints[pos]) + f_ours.from_integer(delta)
                ).to_integer()
            r_ours = vector(f_ours, [f_ours.from_integer(x) for x in r_ints])
            r_sage = sage_vector(f_sage, [_sage_from_int(f_sage, x) for x in r_ints])

            try:
                dec_o = rs_ours.decode_errors(r_ours)
            except DecodingError:
                dec_o = None
            try:
                dec_s = gao_dec.decode_to_code(r_sage)
            except SageDecodingError:
                dec_s = None

            if dec_o is not None:
                assert rs_ours.has_zero_syndrome(dec_o)
                assert int((dec_o - r_ours).hamming_weight()) <= t
                assert dec_s is not None
                assert _vec_to_ints(dec_o) == _vec_to_ints(dec_s)
            else:
                assert dec_s is None

    # --- Pure erasure decoding: e = d - 1 (including alpha = 0) ---
    era_positions = list(range(d - 1))
    r_ints = list(cw_ints)
    era_flags = [0] * n
    for pos in era_positions:
        r_ints[pos] = 0
        era_flags[pos] = 1
    r_ours = vector(f_ours, [f_ours.from_integer(x) for x in r_ints])
    r_sage = sage_vector(f_sage, [_sage_from_int(f_sage, x) for x in r_ints])
    e_ours = vector(gf2_ours, era_flags)
    e_sage = sage_vector(gf2_sage, era_flags)

    dec_o = rs_ours.decode_errors_erasures(r_ours, e_ours)
    dec_s = ee_dec.decode_to_code((r_sage, e_sage))
    assert _vec_to_ints(dec_o) == cw_ints
    assert _vec_to_ints(dec_o) == _vec_to_ints(dec_s)

    # --- Too many erasures: e == d must raise DecodingError in both ---
    if d <= n:
        too_many_flags = [1] * d + [0] * (n - d)
        with pytest.raises(DecodingError):
            rs_ours.decode_errors_erasures(r_ours, vector(gf2_ours, too_many_flags))
        with pytest.raises(SageDecodingError):
            ee_dec.decode_to_code((r_sage, sage_vector(gf2_sage, too_many_flags)))

    # --- Combined errors + erasures: 2*v + e <= d - 1 (testing both alpha=0 erased & errored) ---
    if d >= 5:
        for zero_role in ("erased", "errored"):
            e_cnt = 2
            v_cnt = (d - 1 - e_cnt) // 2
            if zero_role == "erased":
                e_pos = [0, 2]
                v_pos = list(range(3, 3 + v_cnt))
            else:
                e_pos = [1, 2]
                v_pos = [0] + list(range(3, 3 + v_cnt - 1))

            r_ints = list(cw_ints)
            era_flags = [0] * n
            for pos in e_pos:
                r_ints[pos] = 0
                era_flags[pos] = 1
            for pos in v_pos:
                delta = rng.randrange(1, q)
                r_ints[pos] = (
                    f_ours.from_integer(r_ints[pos]) + f_ours.from_integer(delta)
                ).to_integer()

            r_ours = vector(f_ours, [f_ours.from_integer(x) for x in r_ints])
            r_sage = sage_vector(f_sage, [_sage_from_int(f_sage, x) for x in r_ints])
            dec_o = rs_ours.decode_errors_erasures(r_ours, vector(gf2_ours, era_flags))
            dec_s = ee_dec.decode_to_code((r_sage, sage_vector(gf2_sage, era_flags)))
            assert _vec_to_ints(dec_o) == cw_ints
            assert _vec_to_ints(dec_o) == _vec_to_ints(dec_s)


# ---------------------------------------------------------------------------
# 3. TamoBargCode & TamoBargCodeTwoSets vs SageMath + Cython vs Pure-NumPy parity
# ---------------------------------------------------------------------------


def test_tamo_barg_single_set_and_two_sets_against_sage() -> None:
    """Verify generator matrices, polynomial evaluation, and minimum distance against SageMath."""
    # Single-set TamoBargCode on GF(13) & GF(16): generator rows match Sage polynomial evaluation.
    for q, n, k, r, d_loc in [(13, 12, 8, 3, 2), (16, 16, 5, 2, 3)]:
        f_ours = GF(q, "a") if q != 13 else GF(q)
        f_sage = _make_sage_field(q)
        code = TamoBargCode(f_ours, n, k, r, local_minimum_distance=d_loc)
        enc = code.encoder("VectorEncoder")

        eval_sage = [_sage_from_int(f_sage, x.to_integer()) for x in code.evaluation_points()]
        ring_sage = SagePolynomialRing(f_sage, "x")
        x_s = ring_sage.gen()

        # Reconstruct each basis polynomial in Sage and evaluate at Sage evaluation points.
        sage_rows: list[list[Any]] = []
        for poly in enc.enc_basis():
            coeffs = [int(c.to_integer()) for c in poly.list()]
            p_sage = sum(_sage_from_int(f_sage, c) * (x_s**deg) for deg, c in enumerate(coeffs))
            sage_rows.append([p_sage(pt) for pt in eval_sage])

        G_sage = sage_matrix(f_sage, sage_rows)
        assert _mat_to_int_rows(code.generator_matrix()) == _mat_to_int_rows(G_sage)
        assert int(G_sage.rank()) == k

    # Two-set TamoBargCodeTwoSets on GF(16) [16, 3]: compare minimum_distance_py and weight spectrum
    # against exhaustive SageMath VectorSpace enumeration over the Sage generator matrix.
    f16_o = GF(16, "a")
    f16_s = _make_sage_field(16)
    code_16 = TamoBargCodeTwoSets(f16_o, 16, 3, [2, 2], [3, 3], ["add", "add"])
    G_ints = _mat_to_int_rows(code_16.generator_matrix())
    G_sage = sage_matrix(f16_s, [[_sage_from_int(f16_s, v) for v in row] for row in G_ints])
    assert int(G_sage.rank()) == 3

    sage_wt_dist = [0] * 17
    for msg in SageVectorSpace(f16_s, 3):
        cw = msg * G_sage
        sage_wt_dist[int(cw.hamming_weight())] += 1

    sage_min_d = next(w for w in range(1, 17) if sage_wt_dist[w] > 0)
    assert code_16.minimum_distance_py() == sage_min_d
    assert code_16.minimum_distance() == sage_min_d
    assert compute_code_weight_distribution(code_16) == sage_wt_dist


def test_cython_and_pure_numpy_modes_bit_exact_parity() -> None:
    """Verify Cython-enabled and TAMO_BARG_NO_CYTHON=1 modes match bit-for-bit across all codes."""
    probe_script = r"""
import json
import random
from tamo_barg_lrc import (
    GF,
    HAS_CYTHON,
    TamoBargCode,
    TamoBargCodeTwoSets,
    VectorSpace,
    set_random_seed,
    vector,
)
from tamo_barg_lrc.simulation import VALID_DECODER_NAMES, build_decoder, invoke_decoder
from tamo_barg_lrc.simulation.runner import compute_code_weight_distribution

random.seed(1234)
set_random_seed(1234)

def mat_ints(m):
    return [[int(m[r, c].to_integer()) for c in range(m.ncols())] for r in range(m.nrows())]

def vec_ints(v):
    return [int(x.to_integer()) for x in v]

out = {"has_cython": bool(HAS_CYTHON)}

# 1. Single-set TamoBargCode over GF(16)
f16 = GF(16, "a")
c1 = TamoBargCode(f16, 16, 5, 2, local_minimum_distance=3)
out["c1_G"] = mat_ints(c1.generator_matrix())
cw1 = c1.encode(VectorSpace(f16, 5).random_element())
r1 = vector(f16, list(cw1))
r1[1] += f16.gen()
r1[9] += f16.gen() ** 3
out["c1_err_dec"] = vec_ints(c1.decoder("ErrorDecoder").decode_to_code(r1))
r1_era = vector(f16, list(cw1))
r1_era[0] = f16.zero()
r1_era[2] = f16.zero()
e1 = vector(GF(2), [1, 0, 1] + [0] * 13)
out["c1_era_dec"] = vec_ints(c1.decoder("ErasureDecoder").decode_to_code((r1_era, e1)))

# 2. Two-set TamoBargCodeTwoSets over GF(16) (weight dist & minimum_distance_py)
c2_16 = TamoBargCodeTwoSets(f16, 16, 3, [2, 2], [3, 3], ["add", "add"])
out["c2_16_G"] = mat_ints(c2_16.generator_matrix())
out["c2_16_d"] = int(c2_16.minimum_distance_py())
out["c2_16_wd"] = compute_code_weight_distribution(c2_16)

# 3. Two-set TamoBargCodeTwoSets over GF(64) across all 6 decoders
f64 = GF(64, "a")
c2_64 = TamoBargCodeTwoSets(f64, 64, 6, [4, 4], [5, 5], ["add", "add"])
out["c2_64_G"] = mat_ints(c2_64.generator_matrix())
cw64 = c2_64.encode(VectorSpace(f64, 6).random_element())
r64 = vector(f64, list(cw64))
for pos in (0, 5, 13, 27, 42):
    r64[pos] += f64.gen() ** (pos % 11 + 1)

dec_results = {}
for name in VALID_DECODER_NAMES:
    dec = build_decoder(c2_64, name, max_num_of_itr=8, erasure_start_itr=1)
    res = invoke_decoder(dec, name, r64)
    dec_results[name] = {
        "cw": vec_ints(res.codeword),
        "itr": int(res.num_iterations),
        "rem_era": int(res.remaining_erasures),
    }
out["c2_64_decoders"] = dec_results
print(json.dumps(out))
"""

    def _run_mode(no_cython: str) -> dict[str, Any]:
        env = os.environ.copy()
        env["TAMO_BARG_NO_CYTHON"] = no_cython
        proc = subprocess.run(
            [sys.executable, "-c", probe_script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(proc.stdout)

    res_cy = _run_mode("0")
    res_np = _run_mode("1")

    assert res_np["has_cython"] is False
    for key in ("c1_G", "c1_err_dec", "c1_era_dec", "c2_16_G", "c2_16_d", "c2_16_wd", "c2_64_G"):
        assert res_cy[key] == res_np[key], f"Mismatch between Cython and pure-NumPy on {key}"
    assert res_cy["c2_64_decoders"] == res_np["c2_64_decoders"]
