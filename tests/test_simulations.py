"""Unit, smoke, and bug-regression tests for tamo_barg_lrc.simulation and CLI scripts."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from simulation import (
    error_weight,
    expander_sim,
    global_decoding_sim,
    iterative_decoding_sim,
    process_results,
    two_step_decoding_sim,
    weight_distribution,
)

from tamo_barg_lrc import GF, TamoBargCodeTwoSets, VectorSpace, vector
from tamo_barg_lrc.simulation import (
    VALID_DECODER_NAMES,
    DecodeOutcome,
    build_candidate_error_positions,
    build_decoder,
    compute_print_frequency,
    invoke_decoder,
    process_error_distribution_csv,
    run_error_weight_simulation,
    sample_fixed_weight_error,
)


def test_regression_b14_process_error_distribution_csv_operator_precedence(
    tmp_path: Path,
) -> None:
    """Regression B14: (design_distance - 1) // 2 accepts weight 7 and rejects 8 and 15."""
    csv_path = tmp_path / "synthetic_err.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Num_of_error_symbols", "remain_err_weight", "count", "dist"])
        # Error weight 1: 40 trials with 0 residual + 60 trials with 7 residual -> 100% success.
        writer.writerow([1, 0, 40, 0.4])
        writer.writerow([1, 7, 60, 0.6])
        # Error weight 2: 100 trials with residual weight 8 (> 7) -> 0% success!
        # Legacy unparenthesized `design_distance - 1 // 2` evaluated `15 - 0 = 15` and counted 8.
        writer.writerow([2, 8, 100, 1.0])
        # Error weight 3: 100 trials with residual weight 15 (> 7) -> 0% success!
        writer.writerow([3, 15, 100, 1.0])

    result = process_error_distribution_csv(
        err_csv_path=csv_path,
        design_distance=15,
        real_distance=4,
        sim_itr=100,
        threshold_pct=99.0,
    )
    assert result["succ_count"] == [100, 0, 0, 0]
    assert result["prob"] == [100.0, 0.0, 0.0, 0.0]
    assert result["threshold_index"] == 1

    # When every bin meets threshold_pct, threshold_index must be None.
    all_pass_csv = tmp_path / "all_pass_err.csv"
    with open(all_pass_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Num_of_error_symbols", "remain_err_weight", "count", "dist"])
        writer.writerow([1, 7, 100, 1.0])
    all_pass = process_error_distribution_csv(all_pass_csv, 15, 1, sim_itr=100, threshold_pct=99.0)
    assert all_pass["threshold_index"] is None

    with pytest.raises(ValueError, match="design_distance"):
        process_error_distribution_csv(csv_path, 0, 4, sim_itr=100)
    with pytest.raises(ValueError, match="real_distance"):
        process_error_distribution_csv(csv_path, 15, 0, sim_itr=100)
    with pytest.raises(ValueError, match="sim_itr"):
        process_error_distribution_csv(csv_path, 15, 4, sim_itr=0)


@pytest.mark.parametrize("decoder_name", VALID_DECODER_NAMES)
def test_regression_b15_and_b16_build_invoke_and_runner_all_six_decoders(
    decoder_name: str, tmp_path: Path
) -> None:
    """Regression B15 & B16: all 6 decoders build/invoke cleanly and run with sim_itr < 5."""
    assert compute_print_frequency(sim_itr=2, print_freq_factor=5) == 1

    f64 = GF(64, "a")
    code = TamoBargCodeTwoSets(f64, 64, 6, [2, 2], [3, 3], ["add", "add"])
    cw = code.encode(VectorSpace(f64, 6).random_element())
    r = vector(f64, list(cw))
    r[3] += f64.gen()
    r.set_immutable()

    decoder = build_decoder(code, decoder_name, max_num_of_itr=6, erasure_start_itr=2)
    outcome = invoke_decoder(decoder, decoder_name, r)
    assert isinstance(outcome, DecodeOutcome)
    assert outcome.codeword == cw
    assert outcome.num_iterations >= 1
    assert outcome.remaining_erasures == 0

    # Exercise run_error_weight_simulation with sim_itr=2 < print_freq_factor=5 across all 6 keys.
    run_error_weight_simulation(
        q=64,
        n=64,
        k=6,
        locality=(2, 2),
        local_minimum_distance=(3, 3),
        sub_group_type=("add", "add"),
        decoder_name=decoder_name,
        sim_itr=2,
        max_err=2,
        print_freq_factor=5,
        output_dir=tmp_path,
        seed=42,
    )


def test_shortened_mixed_subgroup_error_sampling_and_invalid_decoder_name() -> None:
    """Verify mixed-subgroup candidate positions exclude 0 and sample exact Hamming weight."""
    f64 = GF(64, "a")
    code_am = TamoBargCodeTwoSets(f64, 64, 12, [2, 5], [3, 3], ["add", "mult"])
    zero_idx = code_am.evaluation_points().index(f64.zero())

    candidates = build_candidate_error_positions(code_am)
    assert len(candidates) == 63
    assert zero_idx not in candidates

    for _ in range(10):
        err = sample_fixed_weight_error(f64, 64, 5, candidates)
        assert int(err.hamming_weight()) == 5
        assert err[zero_idx] == f64.zero()

    with pytest.raises(ValueError, match="Unknown decoder_name"):
        build_decoder(code_am, "NonExistentDecoder")
    with pytest.raises(ValueError, match="Unknown decoder_name"):
        invoke_decoder(object(), "NonExistentDecoder", vector(f64, [0] * 64))


def test_cli_simulation_entrypoints_smoke_and_resource_cleanup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Smoke-test main(argv) across all 7 CLI simulation scripts without leaking figures (B18)."""
    plt.close("all")
    out_dir = str(tmp_path)

    # 1. error_weight.py
    error_weight.main(["--sim-itr", "2", "--max-err", "2", "--output-dir", out_dir, "--seed", "7"])

    # 2. expander_sim.py (both uniform and biased modes on GF(64))
    expander_sim.main(
        [
            "--q",
            "64",
            "--n",
            "64",
            "--sim-itr",
            "2",
            "--max-err",
            "2",
            "--biased",
            "--error-boundary",
            "16",
            "--output-dir",
            out_dir,
            "--seed",
            "7",
        ]
    )

    # 3. global_decoding_sim.py (produces *_err.csv for process_results.py)
    global_decoding_sim.main(
        [
            "--q",
            "64",
            "--n",
            "64",
            "--sim-itr",
            "2",
            "--max-err",
            "2",
            "--output-dir",
            out_dir,
            "--seed",
            "7",
        ]
    )

    # 4. process_results.py (post-process the generated GF_64_add_4_add_8_err.csv)
    err_csv = tmp_path / "GF_64_add_4_add_8" / "GF_64_add_4_add_8_err.csv"
    assert err_csv.is_file()
    process_results.main(
        [
            "--csv-path",
            str(err_csv),
            "--design-distance",
            "15",
            "--real-distance",
            "3",
            "--sim-itr",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert "1:" in captured.out

    # 5. iterative_decoding_sim.py
    iterative_decoding_sim.main(
        [
            "--q",
            "64",
            "--n",
            "64",
            "--r",
            "2",
            "5",
            "--local-min-dist",
            "3",
            "3",
            "--sim-itr",
            "2",
            "--max-err",
            "2",
            "--output-dir",
            out_dir,
            "--seed",
            "7",
        ]
    )

    # 6. two_step_decoding_sim.py
    two_step_decoding_sim.main(
        ["--sim-itr", "2", "--max-err", "2", "--output-dir", out_dir, "--seed", "7"]
    )

    # 7. weight_distribution.py (k=2 -> 65 projective vectors, fast & verifies no duplicate header)
    weight_distribution.main(["--q", "64", "--n", "64", "--k", "2", "--output-dir", out_dir])
    wd_csv_files = list(tmp_path.rglob("*weight*.csv")) + list(
        (tmp_path / "GF_64_add_4_mult_7").glob("*.csv")
    )
    assert len(wd_csv_files) >= 1
    with open(wd_csv_files[0], newline="") as handle:
        rows = list(csv.reader(handle))
    header_labels = [row[0] for row in rows if row and row[0].endswith(": ")]
    assert len(header_labels) == len(set(header_labels))

    png_files = list(tmp_path.rglob("*.png"))
    assert len(png_files) >= 1
    assert all(p.stat().st_size > 0 for p in png_files)
    assert len(plt.get_fignums()) == 0
