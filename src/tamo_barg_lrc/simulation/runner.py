"""Reusable simulation runners, decoder dispatchers, and CLI utilities."""

import csv
import math
import multiprocessing as mp
import pickle
import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tamo_barg_lrc._ff import (
    GF,
    FFMatrix,
    FiniteField,
    VectorSpace,
    random_error_vector,
    set_random_seed,
    zero_vector,
)
from tamo_barg_lrc.coding_base import DecodingError
from tamo_barg_lrc.expander import compute_expansion_metrics, save_bipartite_graph_plot
from tamo_barg_lrc.simulation import VALID_DECODER_NAMES, add_common_simulation_args
from tamo_barg_lrc.subgroups import SubgroupType, find_additive_subgroups
from tamo_barg_lrc.tamo_barg_two_sets import TamoBargCodeTwoSets

__all__ = [
    "DecodeOutcome",
    "VALID_DECODER_NAMES",
    "add_common_simulation_args",
    "build_candidate_error_positions",
    "build_decoder",
    "compute_code_weight_distribution",
    "compute_print_frequency",
    "invoke_decoder",
    "run_decoding_success_simulation",
    "run_error_weight_simulation",
    "run_global_threshold_simulation",
    "run_weight_distribution_analysis",
    "sample_fixed_weight_error",
    "seed_simulation_rngs",
    "write_code_metadata_csv",
]

# Module-level caches to avoid redundant per-trial finite-field allocations.
_ZERO_GF2_CACHE: dict[int, Any] = {}
_ZERO_VEC_CACHE: dict[tuple[Any, int], Any] = {}
_NONZERO_ELEMS_CACHE: dict[Any, tuple[Any, ...]] = {}

# Worker-process shared state populated before fork() or via spawn initializer.
_WORKER_STATE: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class DecodeOutcome:
    """Normalized decoding outcome across all two-set Tamo-Barg decoders.

    Attributes:
        codeword: Decoded codeword vector in ``GF(q)^n``.
        num_iterations: Number of decoder iterations executed (``1`` for global).
        remaining_erasures: Count of unresolved DUE erasures after decoding.
    """

    codeword: Any
    num_iterations: int
    remaining_erasures: int = 0


def build_decoder(
    code: TamoBargCodeTwoSets,
    decoder_name: str,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
) -> Any:
    """Instantiate a registered two-set Tamo-Barg decoder with matching arguments.

    Args:
        code: Two-set Tamo-Barg code instance.
        decoder_name: Registered decoder key in ``VALID_DECODER_NAMES``.
        max_num_of_itr: Maximum iterations for iterative and two-step decoders.
        erasure_start_itr: Optional iteration index at which DUE erasure marking
            starts for erasure-and-error iterative/two-step decoders.

    Returns:
        Configured ``Decoder`` instance.

    Raises:
        ValueError: If ``decoder_name`` is not a recognized two-set decoder name.
    """
    if decoder_name in ("GlobalDecoder", "GlobalErasureErrorDecoder"):
        return code.decoder(decoder_name)
    if decoder_name in ("IterativeDecoder", "TwoStepsDecoder"):
        return code.decoder(decoder_name, max_num_of_itr)
    if decoder_name in ("IterativeErasureErrorDecoder", "TwoStepsErasureErrorDecoder"):
        return code.decoder(decoder_name, max_num_of_itr, erasure_start_itr)
    raise ValueError(
        f"Unknown decoder_name {decoder_name!r}; expected one of {VALID_DECODER_NAMES}"
    )


def invoke_decoder(
    decoder: Any,
    decoder_name: str,
    received_word: Any,
) -> DecodeOutcome:
    """Execute ``decoder`` on ``received_word`` and normalize its return value.

    Args:
        decoder: Instantiated two-set Tamo-Barg decoder.
        decoder_name: Registered decoder key matching ``decoder``.
        received_word: Received vector in ``GF(q)^n``.

    Returns:
        Normalized :class:`DecodeOutcome` containing the decoded vector, iteration
        count, and remaining erasure weight.

    Raises:
        ValueError: If ``decoder_name`` is not a recognized two-set decoder name.
    """
    if decoder_name == "GlobalDecoder":
        try:
            corr = decoder.decode_to_code(received_word)
        except (DecodingError, ValueError):
            corr = copy(received_word)
        return DecodeOutcome(codeword=corr, num_iterations=1, remaining_erasures=0)

    if decoder_name == "GlobalErasureErrorDecoder":
        n_len = len(received_word)
        era_vec = _ZERO_GF2_CACHE.get(n_len)
        if era_vec is None:
            era_vec = zero_vector(GF(2), n_len)
            era_vec.set_immutable()
            _ZERO_GF2_CACHE[n_len] = era_vec
        try:
            corr = decoder.decode_to_code((received_word, era_vec))
        except (DecodingError, ValueError):
            corr = copy(received_word)
        return DecodeOutcome(codeword=corr, num_iterations=1, remaining_erasures=0)

    if decoder_name in ("IterativeDecoder", "TwoStepsDecoder"):
        corr, num_itr = decoder.decode_to_code(received_word)
        return DecodeOutcome(codeword=corr, num_iterations=int(num_itr), remaining_erasures=0)

    if decoder_name == "IterativeErasureErrorDecoder":
        (corr, era_vec), num_itr = decoder.decode_to_code(received_word)
        return DecodeOutcome(
            codeword=corr,
            num_iterations=int(num_itr),
            remaining_erasures=int(era_vec.hamming_weight()),
        )

    if decoder_name == "TwoStepsErasureErrorDecoder":
        corr, num_itr, rem_era = decoder.decode_to_code(received_word)
        return DecodeOutcome(
            codeword=corr,
            num_iterations=int(num_itr),
            remaining_erasures=int(rem_era),
        )

    raise ValueError(
        f"Unknown decoder_name {decoder_name!r}; expected one of {VALID_DECODER_NAMES}"
    )


def seed_simulation_rngs(seed: int | None) -> None:
    """Seed both Python and finite-field pseudo-random number generators if ``seed`` is set.

    Args:
        seed: Optional integer seed; if ``None``, RNG state is left unchanged.
    """
    if seed is not None:
        random.seed(int(seed))
        set_random_seed(int(seed))


def compute_print_frequency(sim_itr: int, print_freq_factor: int = 5) -> int:
    """Return a positive progress-log interval that never divides by zero.

    Args:
        sim_itr: Total number of Monte Carlo trials per error weight.
        print_freq_factor: Target number of progress messages per error weight.

    Returns:
        Positive integer step size ``>= 1``.
    """
    return max(1, int(sim_itr) // max(1, int(print_freq_factor)))


def build_candidate_error_positions(
    code: TamoBargCodeTwoSets,
    base_positions: Sequence[int] | None = None,
    shorten: bool | None = None,
) -> list[int]:
    """Build valid codeword error-injection indices, excluding ``0`` when shortened.

    In mixed additive/multiplicative codes (``sub_group_type[0] != sub_group_type[1]``)
    of length ``q``, the evaluation point ``0`` belongs only to the additive partition.
    Excluding its coordinate upfront ensures every sampled error pattern has exact
    requested Hamming weight without post-hoc overwriting.

    Args:
        code: Two-set Tamo-Barg code instance.
        base_positions: Optional candidate coordinate subset; defaults to ``0 .. n - 1``.
        shorten: Whether to exclude the coordinate of ``0``; defaults to
            ``sub_group_type[0] != sub_group_type[1]``.

    Returns:
        Ordered list of distinct candidate coordinate indices in ``0 .. n - 1``.
    """
    length = int(code.length())
    _, sg_types = code.sub_group()
    if shorten is None:
        shorten = sg_types[0] != sg_types[1]

    raw_positions = range(length) if base_positions is None else base_positions
    candidates = list(dict.fromkeys(int(pos) for pos in raw_positions if 0 <= int(pos) < length))

    if shorten:
        zero_elem = code.base_field().zero()
        eval_pts = code.evaluation_points()
        try:
            zero_idx = eval_pts.index(zero_elem)
        except ValueError:
            zero_idx = -1
        if zero_idx >= 0:
            candidates = [pos for pos in candidates if pos != zero_idx]

    return candidates


def sample_fixed_weight_error(
    field: Any,
    length: int,
    weight: int,
    candidate_positions: Sequence[int],
) -> Any:
    """Sample a random error vector of exact Hamming ``weight`` on ``candidate_positions``.

    Args:
        field: Finite field ``GF(q)`` over which the error vector is defined.
        length: Ambient vector length ``n``.
        weight: Exact Hamming weight (number of non-zero coordinates) to inject.
        candidate_positions: Sequence of distinct valid coordinate indices in ``0 .. n - 1``.
            Passing a pre-validated ``tuple[int, ...]`` avoids redundant per-trial
            deduplication overhead.

    Returns:
        Vector in ``GF(q)^n`` with uniform random non-zero elements at ``weight``
        uniformly sampled positions from ``candidate_positions``.

    Raises:
        ValueError: If ``candidate_positions`` contains out-of-bounds indices or
            ``weight`` is outside ``0 .. len(unique_positions)``.
    """
    if isinstance(candidate_positions, tuple):
        positions: Sequence[int] = candidate_positions
        if positions and (min(positions) < 0 or max(positions) >= length):
            raise ValueError(f"candidate_positions must lie in 0 .. {length - 1}")
    else:
        unique_positions = list(dict.fromkeys(int(pos) for pos in candidate_positions))
        if any(pos < 0 or pos >= length for pos in unique_positions):
            raise ValueError(f"candidate_positions must lie in 0 .. {length - 1}")
        positions = unique_positions

    if weight < 0 or weight > len(positions):
        raise ValueError(
            f"Requested error weight {weight} outside valid range 0 .. {len(positions)}"
        )

    if isinstance(field, FiniteField):
        chosen_positions = random.sample(positions, weight) if weight > 0 else ()
        return random_error_vector(length, field, chosen_positions)

    vec_key = (field, length)
    zero_template = _ZERO_VEC_CACHE.get(vec_key)
    if zero_template is None:
        zero_template = zero_vector(field, length)
        _ZERO_VEC_CACHE[vec_key] = zero_template

    err = zero_template.__copy__()
    if weight == 0:
        return err

    chosen_positions = random.sample(positions, weight)

    nonzero_elems = _NONZERO_ELEMS_CACHE.get(field)
    if nonzero_elems is None and int(field.cardinality()) <= 65536:
        zero_el = field.zero()
        nonzero_elems = tuple(el for el in field if el != zero_el)
        _NONZERO_ELEMS_CACHE[field] = nonzero_elems

    if nonzero_elems is not None:
        for pos, val in zip(chosen_positions, random.choices(nonzero_elems, k=weight), strict=True):
            err[pos] = val
    else:
        for pos in chosen_positions:
            err[pos] = field.random_nonzero()
    return err


def _compute_runner_expansion_metrics(graph: Any, right_nodes: Any) -> dict[str, Any]:
    """Compute expansion metrics without unused spectral eigenvalues when supported."""
    try:
        return compute_expansion_metrics(graph, right_nodes, include_spectrum=False)
    except TypeError:
        return compute_expansion_metrics(graph, right_nodes)


def _code_metadata_rows(
    code: TamoBargCodeTwoSets,
    graph_metrics: Mapping[str, Any],
    codeword: Any | None = None,
    extra_rows: Sequence[Sequence[Any]] | None = None,
) -> list[list[Any]]:
    """Build standardized metadata key-value rows for CSV and console reporting."""
    subgroups, sg_types = code.sub_group()
    localities = code.locality()
    local_dists = code.local_minimum_distance()
    q = int(code.base_field().cardinality())

    rows: list[list[Any]] = [
        ["GF: ", q],
        ["Code dim n: ", int(code.length())],
        ["Message dim k: ", int(code.dimension())],
        ["k_max: ", code.max_dimension()],
        ["max rate: ", code.max_rate()],
        ["Locality r1: ", localities[0]],
        ["Locality r2: ", localities[1]],
        ["Local Minimum distance d1: ", local_dists[0]],
        ["Local Minimum distance d2: ", local_dists[1]],
        ["Design Distance: ", code.design_distance()],
        ["First Subgroup: ", subgroups[0]],
        ["First Subgroup Type: ", sg_types[0]],
        ["First Subgroup Size: ", len(subgroups[0])],
        ["Second Subgroup: ", subgroups[1]],
        ["Second Subgroup Type: ", sg_types[1]],
        ["Second Subgroup Size: ", len(subgroups[1])],
        ["Evaluation Points: ", code.evaluation_points()],
        ["Partitions Graph is Connected:", graph_metrics["is_connected"]],
        ["Number of disjoint subgraphs:", graph_metrics["num_connected_components"]],
        ["Partitions Graph Edge Expansion:", graph_metrics["edge_expansion"]],
    ]
    if codeword is not None:
        rows.append(["Codeword : ", codeword])
    if extra_rows:
        rows.extend(list(row) for row in extra_rows)
    return rows


def write_code_metadata_csv(
    writer: Any,
    code: TamoBargCodeTwoSets,
    graph_metrics: Mapping[str, Any],
    codeword: Any | None = None,
    extra_rows: Sequence[Sequence[Any]] | None = None,
) -> None:
    """Write standardized code and coset-graph metadata rows to a CSV writer.

    Args:
        writer: Active ``csv.writer`` object.
        code: Two-set Tamo-Barg code instance.
        graph_metrics: Expansion/connectivity metrics dictionary from
            :func:`compute_expansion_metrics`.
        codeword: Optional reference codeword vector included in the header.
        extra_rows: Optional additional ``[label, value]`` rows appended after
            standard metadata.
    """
    for row in _code_metadata_rows(code, graph_metrics, codeword, extra_rows):
        writer.writerow(row)


def _print_code_summary(
    code: TamoBargCodeTwoSets,
    sim_name: str,
    graph_metrics: Mapping[str, Any],
    codeword: Any | None = None,
    extra_rows: Sequence[Sequence[Any]] | None = None,
    stream: Any = None,
    include_banner: bool = True,
) -> None:
    """Print human-readable simulation setup metadata to stdout or a log stream."""
    if include_banner:
        print("## Setup Done ##", file=stream)
        print(f"## Start Simulation {sim_name} ##", file=stream)
    print(code, file=stream)
    for label, value in _code_metadata_rows(code, graph_metrics, codeword, extra_rows):
        print(f"{label}{value}", file=stream)


def _build_sim_name(code: TamoBargCodeTwoSets, suffix: str = "") -> str:
    """Construct canonical simulation identifier ``GF_<q>_<t0>_<s0>_<t1>_<s1>[suffix]``."""
    subgroups, sg_types = code.sub_group()
    q = int(code.base_field().cardinality())
    base = f"GF_{q}_{sg_types[0]}_{len(subgroups[0])}_{sg_types[1]}_{len(subgroups[1])}"
    return f"{base}{suffix}"


def _prepare_code_and_codeword(
    q: int,
    n: int,
    k: int | None,
    locality: Sequence[int],
    local_minimum_distance: Sequence[int],
    sub_group_type: Sequence[Literal["add", "mult"]],
    use_enc: bool,
    subgroup: Sequence[Sequence[Any] | None] | None = None,
) -> tuple[Any, TamoBargCodeTwoSets, Any]:
    """Construct ``GF(q)``, ``TamoBargCodeTwoSets``, and initial transmitted codeword."""
    field = GF(int(q), repr="int")
    sgt_tuple: list[SubgroupType] = [
        "add" if sub_group_type[0] == "add" else "mult",
        "add" if sub_group_type[1] == "add" else "mult",
    ]
    if k is None:
        try:
            code = TamoBargCodeTwoSets(
                field,
                int(n),
                None,  # type: ignore[arg-type]
                locality,
                local_minimum_distance,
                sgt_tuple,
                shift_add=False,
                subgroup=subgroup,
            )
            resolved_k = int(code.dimension())
        except (TypeError, ValueError):
            probe = TamoBargCodeTwoSets(
                field,
                int(n),
                1,
                locality,
                local_minimum_distance,
                sgt_tuple,
                shift_add=False,
                subgroup=subgroup,
            )
            resolved_k = probe.max_dimension()
            code = TamoBargCodeTwoSets(
                field,
                int(n),
                resolved_k,
                locality,
                local_minimum_distance,
                sgt_tuple,
                shift_add=False,
                subgroup=subgroup,
            )
    else:
        resolved_k = int(k)
        code = TamoBargCodeTwoSets(
            field,
            int(n),
            resolved_k,
            locality,
            local_minimum_distance,
            sgt_tuple,
            shift_add=False,
            subgroup=subgroup,
        )

    if use_enc:
        msg_space = VectorSpace(field, resolved_k)
        message = msg_space.random_element()
        codeword = code.encoder("VectorEncoder").encode(message)
    else:
        codeword = zero_vector(field, int(n))

    return field, code, codeword


def _compute_weight_distribution_gf2m(code: TamoBargCodeTwoSets) -> list[int]:
    """Exact projective weight distribution for ``GF(2^m)`` (``q <= 65536``) via NumPy XOR."""
    length = int(code.length())
    field = code.base_field()
    q = int(field.cardinality())

    echelon_mat = code.generator_matrix().echelon_form()
    pivots = list(echelon_mat.pivots())
    rank = len(pivots)
    counts = np.zeros(length + 1, dtype=np.int64)
    counts[0] = 1
    if rank == 0:
        return counts.tolist()

    pivot_set = set(pivots)
    non_pivots = [j for j in range(length) if j not in pivot_set]
    m_rem = len(non_pivots)
    if m_rem == 0:
        scalar_mult = q - 1
        for wt in range(1, rank + 1):
            counts[wt] = math.comb(rank, wt) * (scalar_mult**wt)
        return counts.tolist()

    dtype = np.uint8 if q <= 256 else np.uint16

    # Extract non-pivot columns of the RREF basis into a compact integer array (rank, m_rem).
    if isinstance(echelon_mat, FFMatrix):
        mat_int = np.ascontiguousarray(echelon_mat._arr[:rank, non_pivots], dtype=dtype)
    else:
        to_int = (lambda x: x.to_integer()) if hasattr(field.one(), "to_integer") else int
        mat_int = np.empty((rank, m_rem), dtype=dtype)
        for r_idx in range(rank):
            row = echelon_mat.row(r_idx)
            for c_idx, col in enumerate(non_pivots):
                mat_int[r_idx, c_idx] = to_int(row[col])

    # Precompute all (rank, q, m_rem) scalar-multiplied basis rows via log/antilog addition.
    # Index 0 -> 0; index s in 1..q-1 -> gen^(s - 1) (so s = 1 corresponds to 1).
    tables = np.zeros((rank, q, m_rem), dtype=dtype)
    s_exps = np.arange(q - 1, dtype=np.int32)[None, :, None]
    if (
        isinstance(field, FiniteField)
        and field.exp_table is not None
        and field.log_table is not None
    ):
        sum_logs = field.log_table[mat_int][:, None, :] + s_exps
        tables[:, 1:, :] = field.exp_table[sum_logs].astype(dtype, copy=False)
    else:
        to_int = (lambda x: x.to_integer()) if hasattr(field.one(), "to_integer") else int
        gen = field.multiplicative_generator()
        exp_tbl = np.empty(2 * (q - 1), dtype=dtype)
        log_tbl = np.full(q, -1, dtype=np.int32)
        cur = field.one()
        for exp_idx in range(q - 1):
            val_int = to_int(cur)
            exp_tbl[exp_idx] = val_int
            exp_tbl[exp_idx + q - 1] = val_int
            log_tbl[val_int] = exp_idx
            cur *= gen
        mat_logs = log_tbl[mat_int]
        nz_mask = mat_logs >= 0
        sum_logs = mat_logs[:, None, :] + s_exps
        prod = np.where(nz_mask[:, None, :], exp_tbl[np.maximum(sum_logs, 0)], 0)
        tables[:, 1:, :] = prod

    scalar_mult = q - 1
    batch_dims = 2 if q <= 64 else 1

    for lead in range(rank):
        tail_dim = rank - lead - 1
        base_vec = tables[lead, 1]
        if tail_dim == 0:
            wt = 1 + int(np.count_nonzero(base_vec))
            counts[wt] += scalar_mult
            continue

        inner_d = min(batch_dims, tail_dim)
        outer_d = tail_dim - inner_d

        inner_start = rank - inner_d
        chunk_vecs = tables[inner_start]
        chunk_pwts = np.ones(q, dtype=np.int32)
        chunk_pwts[0] = 0
        for d_idx in range(inner_start + 1, rank):
            t_d = tables[d_idx]
            chunk_vecs = (chunk_vecs[:, None, :] ^ t_d[None, :, :]).reshape(-1, m_rem)
            w_d = np.ones(q, dtype=np.int32)
            w_d[0] = 0
            chunk_pwts = (chunk_pwts[:, None] + w_d[None, :]).reshape(-1)

        if outer_d == 0:
            vecs = chunk_vecs ^ base_vec
            wts = 1 + chunk_pwts + np.count_nonzero(vecs, axis=1)
            counts += np.bincount(wts, minlength=length + 1) * scalar_mult
        else:
            outer_rows = list(range(lead + 1, inner_start))
            _accumulate_gf2m_dfs(
                0,
                base_vec,
                1,
                outer_d,
                outer_rows,
                tables,
                chunk_vecs,
                chunk_pwts,
                counts,
                length,
                scalar_mult,
                q,
            )

    return counts.tolist()


def _accumulate_gf2m_dfs(
    pos: int,
    acc_vec: np.ndarray,
    acc_pwt: int,
    outer_d: int,
    outer_rows: list[int],
    tables: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_pwts: np.ndarray,
    counts: np.ndarray,
    length: int,
    scalar_mult: int,
    q: int,
) -> None:
    """Recursively accumulate non-pivot XOR batches across outer message coordinates."""
    if pos == outer_d:
        vecs = chunk_vecs ^ acc_vec
        wts = acc_pwt + chunk_pwts + np.count_nonzero(vecs, axis=1)
        counts[:] += np.bincount(wts, minlength=length + 1) * scalar_mult
        return
    r_i = outer_rows[pos]
    tbl = tables[r_i]
    _accumulate_gf2m_dfs(
        pos + 1,
        acc_vec,
        acc_pwt,
        outer_d,
        outer_rows,
        tables,
        chunk_vecs,
        chunk_pwts,
        counts,
        length,
        scalar_mult,
        q,
    )
    for s_i in range(1, q):
        _accumulate_gf2m_dfs(
            pos + 1,
            acc_vec ^ tbl[s_i],
            acc_pwt + 1,
            outer_d,
            outer_rows,
            tables,
            chunk_vecs,
            chunk_pwts,
            counts,
            length,
            scalar_mult,
            q,
        )


def compute_code_weight_distribution(code: TamoBargCodeTwoSets) -> list[int]:
    """Compute exact Hamming weight distribution ``[A_0, A_1, ..., A_n]`` of ``code``.

    For characteristic-2 fields with ``q <= 65536``, uses a vectorized NumPy
    log/antilog table and batched bitwise-XOR projective enumeration. For odd
    prime-power or larger fields, delegates to ``code.weight_distribution()``
    with a projective (monic) ``VectorSpace`` fallback achieving a ``(q - 1)x``
    speedup over naive enumeration.

    Args:
        code: Two-set Tamo-Barg code instance.

    Returns:
        List of length ``n + 1`` where entry ``w`` is the number of codewords of
        Hamming weight ``w``.
    """
    field = code.base_field()
    q = int(field.cardinality())
    if int(field.characteristic()) == 2 and q <= 65536:
        return _compute_weight_distribution_gf2m(code)

    try:
        return [int(val) for val in code.weight_distribution()]
    except (AttributeError, NotImplementedError):
        pass

    length = int(code.length())
    counts = [0] * (length + 1)
    counts[0] = 1

    echelon_mat = code.generator_matrix().echelon_form()
    non_zero_row_indices = [idx for idx, row in enumerate(echelon_mat.rows()) if not row.is_zero()]
    rank = len(non_zero_row_indices)
    if rank == 0:
        return counts

    basis_mat = echelon_mat.matrix_from_rows(non_zero_row_indices)
    scalar_mult = q - 1

    for lead in range(rank):
        lead_row = basis_mat.row(lead)
        tail_dim = rank - lead - 1
        if tail_dim == 0:
            wt = int(lead_row.hamming_weight())
            counts[wt] += scalar_mult
            continue
        sub_gen = basis_mat.matrix_from_rows(list(range(lead + 1, rank)))
        for tail_vec in VectorSpace(field, tail_dim):
            cw = lead_row + tail_vec * sub_gen
            wt = int(cw.hamming_weight())
            counts[wt] += scalar_mult

    return counts


def _init_worker_from_pickle(
    payload: bytes,
    decoder_name: str,
    length: int,
    error_positions: tuple[int, ...],
    threshold: int,
) -> None:
    """Populate worker-process state when using non-fork multiprocessing contexts."""
    field, codeword, decoder = pickle.loads(payload)
    _WORKER_STATE["field"] = field
    _WORKER_STATE["codeword"] = codeword
    _WORKER_STATE["decoder"] = decoder
    _WORKER_STATE["decoder_name"] = decoder_name
    _WORKER_STATE["length"] = length
    _WORKER_STATE["error_positions"] = error_positions
    _WORKER_STATE["threshold"] = threshold


def _create_worker_pool(
    workers: int,
    field: Any,
    codeword: Any,
    decoder: Any,
    decoder_name: str,
    length: int,
    error_positions: tuple[int, ...],
    threshold: int = 0,
) -> ProcessPoolExecutor:
    """Create a ``ProcessPoolExecutor`` using ``fork`` on Linux or pickled ``spawn`` elsewhere."""
    _WORKER_STATE["field"] = field
    _WORKER_STATE["codeword"] = codeword
    _WORKER_STATE["decoder"] = decoder
    _WORKER_STATE["decoder_name"] = decoder_name
    _WORKER_STATE["length"] = length
    _WORKER_STATE["error_positions"] = error_positions
    _WORKER_STATE["threshold"] = threshold

    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
        return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)

    ctx = mp.get_context("spawn")
    payload = pickle.dumps((field, codeword, decoder))
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker_from_pickle,
        initargs=(payload, decoder_name, length, error_positions, threshold),
    )


def _build_chunk_tasks(
    n_err: int,
    sim_itr: int,
    workers: int,
    master_ss: np.random.SeedSequence,
) -> list[tuple[int, int, int]]:
    """Split ``sim_itr`` trials at ``n_err`` into independently seeded chunk tasks."""
    num_chunks = min(sim_itr, workers * 2) if sim_itr >= workers * 2 else min(sim_itr, workers)
    child_seeds = master_ss.spawn(num_chunks)
    base = sim_itr // num_chunks
    rem = sim_itr % num_chunks
    tasks: list[tuple[int, int, int]] = []
    for idx in range(num_chunks):
        cnt = base + (1 if idx < rem else 0)
        chunk_seed = int(child_seeds[idx].generate_state(1)[0])
        tasks.append((n_err, cnt, chunk_seed))
    return tasks


def _worker_error_weight_chunk(
    task: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int, int]:
    """Execute one error-weight Monte Carlo chunk inside a worker process."""
    n_err, num_trials, chunk_seed = task
    seed_simulation_rngs(chunk_seed)

    field = _WORKER_STATE["field"]
    codeword = _WORKER_STATE["codeword"]
    decoder = _WORKER_STATE["decoder"]
    decoder_name = _WORKER_STATE["decoder_name"]
    length = _WORKER_STATE["length"]
    error_positions = _WORKER_STATE["error_positions"]
    design_radius = _WORKER_STATE["threshold"]

    twostep_fail_itr = 0
    iter_error_incr_itr = 0
    iter_error_decr_itr = 0
    iter_error_no_change = 0
    overall_remain_err_weight = 0
    max_remain_err_weight = 0
    min_remain_err_weight = length

    for _ in range(num_trials):
        err_vec = sample_fixed_weight_error(field, length, n_err, error_positions)
        received = codeword + err_vec
        outcome = invoke_decoder(decoder, decoder_name, received)
        remain_err_weight = int((codeword - outcome.codeword).hamming_weight())

        if remain_err_weight > max_remain_err_weight:
            max_remain_err_weight = remain_err_weight
        if remain_err_weight < min_remain_err_weight:
            min_remain_err_weight = remain_err_weight

        overall_remain_err_weight += remain_err_weight

        if remain_err_weight > n_err:
            iter_error_incr_itr += 1
        elif remain_err_weight < n_err:
            iter_error_decr_itr += 1
        else:
            iter_error_no_change += 1

        if n_err <= design_radius and remain_err_weight > design_radius:
            twostep_fail_itr += 1

    return (
        iter_error_incr_itr,
        iter_error_decr_itr,
        iter_error_no_change,
        twostep_fail_itr,
        overall_remain_err_weight,
        min_remain_err_weight,
        max_remain_err_weight,
    )


def _worker_global_threshold_chunk(task: tuple[int, int, int]) -> tuple[int, list[int]]:
    """Execute one global-threshold Monte Carlo chunk inside a worker process."""
    n_err, num_trials, chunk_seed = task
    seed_simulation_rngs(chunk_seed)

    field = _WORKER_STATE["field"]
    codeword = _WORKER_STATE["codeword"]
    decoder = _WORKER_STATE["decoder"]
    decoder_name = _WORKER_STATE["decoder_name"]
    length = _WORKER_STATE["length"]
    error_positions = _WORKER_STATE["error_positions"]
    dd_th = _WORKER_STATE["threshold"]

    success_itr = 0
    remain_err_dist = [0] * (length + 1)

    for _ in range(num_trials):
        err_vec = sample_fixed_weight_error(field, length, n_err, error_positions)
        received = codeword + err_vec
        outcome = invoke_decoder(decoder, decoder_name, received)
        correct_c = outcome.codeword

        if correct_c == codeword:
            remain_wt = 0
            success_itr += 1
        else:
            remain_wt = int((codeword - correct_c).hamming_weight())
            if remain_wt <= dd_th:
                success_itr += 1

        remain_err_dist[remain_wt] += 1

    return success_itr, remain_err_dist


def _worker_decoding_success_chunk(task: tuple[int, int, int]) -> tuple[int, int]:
    """Execute one decoding-success Monte Carlo chunk inside a worker process."""
    n_err, num_trials, chunk_seed = task
    seed_simulation_rngs(chunk_seed)

    field = _WORKER_STATE["field"]
    codeword = _WORKER_STATE["codeword"]
    decoder = _WORKER_STATE["decoder"]
    decoder_name = _WORKER_STATE["decoder_name"]
    length = _WORKER_STATE["length"]
    error_positions = _WORKER_STATE["error_positions"]

    success_itr = 0
    overall_num_itr = 0

    for _ in range(num_trials):
        err_vec = sample_fixed_weight_error(field, length, n_err, error_positions)
        received = codeword + err_vec
        outcome = invoke_decoder(decoder, decoder_name, received)
        if outcome.codeword == codeword:
            success_itr += 1
            overall_num_itr += outcome.num_iterations

    return success_itr, overall_num_itr


def run_error_weight_simulation(
    *,
    q: int = 64,
    n: int = 64,
    k: int = 6,
    locality: Sequence[int] = (2, 2),
    local_minimum_distance: Sequence[int] = (3, 3),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"),
    decoder_name: str = "IterativeDecoder",
    sim_itr: int = 10,
    max_err: int = 16,
    start_err: int = 1,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
    use_enc: bool = False,
    output_dir: str | Path = "./results",
    seed: int | None = None,
    print_freq_factor: int = 5,
    biased: bool | None = None,
    error_boundary: int | None = None,
    candidate_positions: Sequence[int] | None = None,
    run_label: str | None = None,
    plot_graph: bool = True,
    workers: int = 1,
    stop_after_zero_runs: int | None = None,
) -> dict[str, Any]:
    """Run residual error-weight transition simulation across injected error weights.

    Shared simulation engine for ``simulation/error_weight.py`` and
    ``simulation/expander_sim.py``.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        decoder_name: Registered two-set decoder name.
        sim_itr: Number of Monte Carlo iterations per error weight (``>= 1``).
        max_err: Maximum error weight to simulate.
        start_err: Starting error weight (default ``1``).
        max_num_of_itr: Maximum iterations for iterative/two-step decoders.
        erasure_start_itr: Optional iteration threshold for DUE erasure marking.
        use_enc: Encode a random message if ``True``, else transmit the zero codeword.
        output_dir: Parent directory under which run artifacts are written.
        seed: Optional integer RNG seed.
        print_freq_factor: Progress print frequency divisor.
        biased: If ``True``, restrict errors to ``0 .. error_boundary - 1``. If
            ``None``, standard ``error_weight`` naming is used; if ``bool``,
            expander naming (``_expander_biased_`` / ``_expander_uniform_``) is used.
        error_boundary: Upper bound ``t`` for biased error support ``[0, t)``.
        candidate_positions: Optional explicit coordinate subset for error injection.
        run_label: Optional custom subdirectory suffix override.
        plot_graph: If ``True`` (default), render and save the bipartite coset-graph PNG.
        workers: Number of worker processes (``1`` runs serially with exact legacy RNG order).
        stop_after_zero_runs: Optional early-stop threshold; stops after this many
            consecutive error weights observe zero error-weight decreases.

    Returns:
        Dictionary with paths (``csv_path``, ``graph_path``) and recorded ``rows``.
    """
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if stop_after_zero_runs is not None and stop_after_zero_runs < 1:
        raise ValueError(f"stop_after_zero_runs must be >= 1, got {stop_after_zero_runs}")

    seed_simulation_rngs(seed)
    field, code, codeword = _prepare_code_and_codeword(
        q, n, k, locality, local_minimum_distance, sub_group_type, use_enc
    )
    decoder = build_decoder(code, decoder_name, max_num_of_itr, erasure_start_itr)

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = _compute_runner_expansion_metrics(graph, right_nodes)
    sim_name = _build_sim_name(code)

    if run_label is not None:
        dir_name = run_label
    elif biased is None:
        dir_name = f"{sim_name}_{decoder_name}"
    elif biased:
        dir_name = f"{sim_name}_expander_biased_{decoder_name}"
    else:
        dir_name = f"{sim_name}_expander_uniform_{decoder_name}"

    res_dir = Path(output_dir) / dir_name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}.csv"
    graph_path: Path | None = (res_dir / f"{sim_name}_graph.png") if plot_graph else None

    if graph_path is not None:
        save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)

    if candidate_positions is not None:
        base_pos: Sequence[int] | None = candidate_positions
    elif biased:
        bound = min(int(n), int(error_boundary) if error_boundary is not None else int(n))
        base_pos = range(max(0, bound))
    else:
        base_pos = None

    error_positions = build_candidate_error_positions(code, base_pos)
    error_positions_tuple = tuple(error_positions)
    effective_max_err = min(int(n) - int(code.dimension()), int(max_err), len(error_positions))
    print_freq = compute_print_frequency(sim_itr, print_freq_factor)
    design_radius = (code.design_distance() - 1) // 2

    _print_code_summary(code, sim_name, graph_metrics, codeword)
    print(f"error_positions: {error_positions}")

    use_pool = workers > 1 and sim_itr > 1
    master_ss = np.random.SeedSequence(seed) if use_pool else None

    recorded_rows: list[list[Any]] = []
    consecutive_zero_runs = 0
    try:
        with ExitStack() as stack:
            res_file = stack.enter_context(open(csv_path, "w", newline=""))
            executor = (
                stack.enter_context(
                    _create_worker_pool(
                        workers,
                        field,
                        codeword,
                        decoder,
                        decoder_name,
                        int(n),
                        error_positions_tuple,
                        threshold=design_radius,
                    )
                )
                if use_pool
                else None
            )

            writer = csv.writer(res_file)
            write_code_metadata_csv(writer, code, graph_metrics, codeword)
            writer.writerow(
                [
                    "Num_of_error_symbols",
                    "probability_of_error_increase",
                    "num_of_error_increase",
                    "probability_of_error_no_change",
                    "num_of_error_no_change",
                    "probability_of_error_decrease",
                    "num_of_error_decrease",
                    "probability_of_failure",
                    "num_of_failure",
                    "avg_remain_err_weight",
                    "min_remain_err_weight",
                    "max_remain_err_weight",
                ]
            )
            res_file.flush()

            for n_err in range(int(start_err), effective_max_err + 1):
                print(f"Error Weight: {n_err}")
                twostep_fail_itr = 0
                iter_error_incr_itr = 0
                iter_error_decr_itr = 0
                iter_error_no_change = 0
                overall_remain_err_weight = 0
                max_remain_err_weight = 0
                min_remain_err_weight = int(n)

                if executor is not None and master_ss is not None:
                    tasks = _build_chunk_tasks(n_err, sim_itr, workers, master_ss)
                    for (
                        c_incr,
                        c_decr,
                        c_no_chg,
                        c_fail,
                        c_overall,
                        c_min,
                        c_max,
                    ) in executor.map(_worker_error_weight_chunk, tasks):
                        iter_error_incr_itr += c_incr
                        iter_error_decr_itr += c_decr
                        iter_error_no_change += c_no_chg
                        twostep_fail_itr += c_fail
                        overall_remain_err_weight += c_overall
                        if c_min < min_remain_err_weight:
                            min_remain_err_weight = c_min
                        if c_max > max_remain_err_weight:
                            max_remain_err_weight = c_max
                    print(f"Iteration {sim_itr} / {sim_itr} Done.")
                else:
                    for i in range(sim_itr):
                        err_vec = sample_fixed_weight_error(
                            field, int(n), n_err, error_positions_tuple
                        )
                        received = codeword + err_vec
                        outcome = invoke_decoder(decoder, decoder_name, received)
                        remain_err_weight = int((codeword - outcome.codeword).hamming_weight())

                        if remain_err_weight > max_remain_err_weight:
                            max_remain_err_weight = remain_err_weight
                        if remain_err_weight < min_remain_err_weight:
                            min_remain_err_weight = remain_err_weight

                        overall_remain_err_weight += remain_err_weight

                        if remain_err_weight > n_err:
                            iter_error_incr_itr += 1
                        elif remain_err_weight < n_err:
                            iter_error_decr_itr += 1
                        else:
                            iter_error_no_change += 1

                        if n_err <= design_radius and remain_err_weight > design_radius:
                            twostep_fail_itr += 1

                        if (i + 1) % print_freq == 0:
                            print(f"Iteration {i + 1} / {sim_itr} Done.")

                row = [
                    n_err,
                    100.0 * (iter_error_incr_itr / sim_itr),
                    iter_error_incr_itr,
                    100.0 * (iter_error_no_change / sim_itr),
                    iter_error_no_change,
                    100.0 * (iter_error_decr_itr / sim_itr),
                    iter_error_decr_itr,
                    100.0 * (twostep_fail_itr / sim_itr),
                    twostep_fail_itr,
                    overall_remain_err_weight / sim_itr,
                    min_remain_err_weight,
                    max_remain_err_weight,
                ]
                writer.writerow(row)
                res_file.flush()
                recorded_rows.append(row)

                if iter_error_decr_itr == 0:
                    consecutive_zero_runs += 1
                else:
                    consecutive_zero_runs = 0
                if (
                    stop_after_zero_runs is not None
                    and consecutive_zero_runs >= stop_after_zero_runs
                ):
                    break
    finally:
        if use_pool:
            _WORKER_STATE.clear()

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "graph_path": graph_path,
        "rows": recorded_rows,
    }


def run_global_threshold_simulation(
    *,
    q: int = 256,
    n: int = 256,
    k: int = 6,
    locality: Sequence[int] = (2, 4),
    local_minimum_distance: Sequence[int] = (3, 5),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"),
    decoder_name: str = "IterativeDecoder",
    sim_itr: int = 5000,
    max_err: int = 250,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
    use_enc: bool = False,
    exact_distance: int | None = None,
    print_log: bool = False,
    output_dir: str | Path = "./results",
    seed: int | None = None,
    print_freq_factor: int = 5,
    plot_graph: bool = True,
    workers: int = 1,
    stop_after_zero_runs: int | None = None,
) -> dict[str, Any]:
    """Run global-cleanup threshold simulation and record residual weight distributions.

    Counts a trial as successful when residual error weight after decoding does
    not exceed ``(target_distance - 1) // 2``, writing both summary success rates
    and ``*_err.csv`` residual error-weight histograms consumed by
    :func:`tamo_barg_lrc.simulation.postprocess.process_error_distribution_csv`.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        decoder_name: Registered two-set decoder name.
        sim_itr: Number of Monte Carlo iterations per error weight (``>= 1``).
        max_err: Maximum error weight to simulate.
        max_num_of_itr: Maximum iterations for iterative/two-step decoders.
        erasure_start_itr: Optional iteration threshold for DUE erasure marking.
        use_enc: Encode a random message if ``True``, else transmit the zero codeword.
        exact_distance: Optional override distance replacing ``code.design_distance()``.
        print_log: If ``True``, write detailed trial logs to ``*_log.txt``.
        output_dir: Parent directory under which run artifacts are written.
        seed: Optional integer RNG seed.
        print_freq_factor: Progress print frequency divisor.
        plot_graph: If ``True`` (default), render and save the bipartite coset-graph PNG.
        workers: Number of worker processes (``1`` runs serially with exact legacy RNG order;
            automatically runs serially when ``print_log=True``).
        stop_after_zero_runs: Optional early-stop threshold; stops after this many
            consecutive error weights observe zero decoding successes.

    Returns:
        Dictionary with paths (``csv_path``, ``err_csv_path``, ``graph_path``,
        ``log_path``) and recorded ``rows``.
    """
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if stop_after_zero_runs is not None and stop_after_zero_runs < 1:
        raise ValueError(f"stop_after_zero_runs must be >= 1, got {stop_after_zero_runs}")

    seed_simulation_rngs(seed)
    field, code, codeword = _prepare_code_and_codeword(
        q, n, k, locality, local_minimum_distance, sub_group_type, use_enc
    )
    decoder = build_decoder(code, decoder_name, max_num_of_itr, erasure_start_itr)

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = _compute_runner_expansion_metrics(graph, right_nodes)
    sim_name = _build_sim_name(code)

    uses_erasure = "Erasure" in decoder_name
    if decoder_name == "IterativeDecoder":
        dir_name = sim_name
    elif decoder_name == "IterativeErasureErrorDecoder":
        dir_name = f"{sim_name}_erasure"
    else:
        dir_name = f"{sim_name}_{decoder_name}"

    res_dir = Path(output_dir) / dir_name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}.csv"
    err_csv_path = res_dir / f"{sim_name}_err.csv"
    graph_path: Path | None = (res_dir / f"{sim_name}_graph.png") if plot_graph else None
    log_path: Path | None = (res_dir / f"{sim_name}_log.txt") if print_log else None

    if graph_path is not None:
        save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)

    extra_rows: list[list[Any]] = [["Erasure Decoder: ", uses_erasure]]
    if exact_distance is not None:
        extra_rows.append(["Code Distance: ", int(exact_distance)])

    _print_code_summary(code, sim_name, graph_metrics, codeword, extra_rows=extra_rows)

    error_positions = build_candidate_error_positions(code)
    error_positions_tuple = tuple(error_positions)
    effective_max_err = min(int(max_err), len(error_positions))
    print_freq = compute_print_frequency(sim_itr, print_freq_factor)
    target_distance = int(exact_distance) if exact_distance is not None else code.design_distance()
    dd_th = (target_distance - 1) // 2

    use_pool = workers > 1 and sim_itr > 1 and not print_log
    master_ss = np.random.SeedSequence(seed) if use_pool else None

    recorded_rows: list[list[Any]] = []
    consecutive_zero_runs = 0
    try:
        with ExitStack() as stack:
            res_file = stack.enter_context(open(csv_path, "w", newline=""))
            err_file = stack.enter_context(open(err_csv_path, "w", newline=""))
            log_handle = (
                stack.enter_context(open(log_path, "w", encoding="utf-8"))
                if log_path is not None
                else None
            )
            executor = (
                stack.enter_context(
                    _create_worker_pool(
                        workers,
                        field,
                        codeword,
                        decoder,
                        decoder_name,
                        int(n),
                        error_positions_tuple,
                        threshold=dd_th,
                    )
                )
                if use_pool
                else None
            )

            writer = csv.writer(res_file)
            err_writer = csv.writer(err_file)

            write_code_metadata_csv(writer, code, graph_metrics, codeword, extra_rows=extra_rows)
            writer.writerow(["Num_of_error_symbols", "probability_of_success"])
            err_writer.writerow(["Num_of_error_symbols", "remain_err_weight", "count", "dist"])
            res_file.flush()
            err_file.flush()

            if log_handle is not None:
                _print_code_summary(
                    code,
                    sim_name,
                    graph_metrics,
                    codeword,
                    extra_rows=extra_rows,
                    stream=log_handle,
                    include_banner=False,
                )

            for n_err in range(1, effective_max_err + 1):
                print(f"Error Weight: {n_err}")
                success_itr = 0
                remain_err_dist = [0] * (int(n) + 1)

                if executor is not None and master_ss is not None:
                    tasks = _build_chunk_tasks(n_err, sim_itr, workers, master_ss)
                    for c_succ, c_dist in executor.map(_worker_global_threshold_chunk, tasks):
                        success_itr += c_succ
                        for wt, cnt in enumerate(c_dist):
                            if cnt:
                                remain_err_dist[wt] += cnt
                    print(f"Iteration {sim_itr} / {sim_itr} Done.")
                else:
                    for i in range(sim_itr):
                        err_vec = sample_fixed_weight_error(
                            field, int(n), n_err, error_positions_tuple
                        )
                        received = codeword + err_vec
                        outcome = invoke_decoder(decoder, decoder_name, received)
                        correct_c = outcome.codeword
                        is_exact_match = correct_c == codeword

                        if log_handle is not None:
                            print(f"Error              : {err_vec}", file=log_handle)
                            print(f"Corrected Codeword : {correct_c}", file=log_handle)
                            print(f"Correction Successful: {is_exact_match}", file=log_handle)

                        if is_exact_match:
                            remain_wt = 0
                            success_itr += 1
                        else:
                            remain_wt = int((codeword - correct_c).hamming_weight())
                            if remain_wt <= dd_th:
                                success_itr += 1

                        remain_err_dist[remain_wt] += 1

                        if (i + 1) % print_freq == 0:
                            print(f"Iteration {i + 1} / {sim_itr} Done.")

                succ_prob = 100.0 * (success_itr / sim_itr)
                row = [n_err, succ_prob]
                writer.writerow(row)
                res_file.flush()
                recorded_rows.append(row)

                for wt, count in enumerate(remain_err_dist):
                    if count != 0:
                        err_writer.writerow([n_err, wt, count, count / sim_itr])
                err_file.flush()

                if log_handle is not None:
                    print(f"Num of error symbols: {n_err}", file=log_handle)
                    print(f"Probability of Success: {succ_prob}", file=log_handle)
                    print("##########################", file=log_handle)

                if success_itr == 0:
                    consecutive_zero_runs += 1
                else:
                    consecutive_zero_runs = 0
                if (
                    stop_after_zero_runs is not None
                    and consecutive_zero_runs >= stop_after_zero_runs
                ):
                    break
    finally:
        if use_pool:
            _WORKER_STATE.clear()

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "err_csv_path": err_csv_path,
        "graph_path": graph_path,
        "log_path": log_path,
        "rows": recorded_rows,
    }


def run_decoding_success_simulation(
    *,
    q: int = 64,
    n: int = 64,
    k: int | None = 6,
    locality: Sequence[int] = (2, 4),
    local_minimum_distance: Sequence[int] = (3, 4),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "mult"),
    decoder_name: str = "TwoStepsDecoder",
    sim_itr: int = 5000,
    max_err: int = 125,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
    use_enc: bool = False,
    print_log: bool = False,
    output_dir: str | Path = "./results",
    seed: int | None = None,
    print_freq_factor: int = 5,
    legacy_iter_dir_naming: bool = False,
    plot_graph: bool = True,
    workers: int = 1,
    stop_after_zero_runs: int | None = None,
) -> dict[str, Any]:
    """Run end-to-end decoding success rate and iteration-count simulation.

    Shared simulation engine for ``simulation/iterative_decoding_sim.py`` and
    ``simulation/two_step_decoding_sim.py``.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension, or ``None`` to use ``code.max_dimension()``.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        decoder_name: Registered two-set decoder name.
        sim_itr: Number of Monte Carlo iterations per error weight (``>= 1``).
        max_err: Maximum error weight to simulate.
        max_num_of_itr: Maximum iterations for iterative/two-step decoders.
        erasure_start_itr: Optional iteration threshold for DUE erasure marking.
        use_enc: Encode a random message if ``True``, else transmit the zero codeword.
        print_log: If ``True``, write detailed trial logs to ``*_log.txt``.
        output_dir: Parent directory under which run artifacts are written.
        seed: Optional integer RNG seed.
        print_freq_factor: Progress print frequency divisor.
        legacy_iter_dir_naming: Use ``<sim_name>`` / ``<sim_name>_erasure`` naming
            when ``decoder_name`` is ``IterativeDecoder`` / ``IterativeErasureErrorDecoder``.
        plot_graph: If ``True`` (default), render and save the bipartite coset-graph PNG.
        workers: Number of worker processes (``1`` runs serially with exact legacy RNG order;
            automatically runs serially when ``print_log=True``).
        stop_after_zero_runs: Optional early-stop threshold; stops after this many
            consecutive error weights observe zero decoding successes.

    Returns:
        Dictionary with paths (``csv_path``, ``graph_path``, ``log_path``) and ``rows``.
    """
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if stop_after_zero_runs is not None and stop_after_zero_runs < 1:
        raise ValueError(f"stop_after_zero_runs must be >= 1, got {stop_after_zero_runs}")

    seed_simulation_rngs(seed)
    field, code, codeword = _prepare_code_and_codeword(
        q, n, k, locality, local_minimum_distance, sub_group_type, use_enc
    )
    decoder = build_decoder(code, decoder_name, max_num_of_itr, erasure_start_itr)

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = _compute_runner_expansion_metrics(graph, right_nodes)
    sim_name = _build_sim_name(code)

    if legacy_iter_dir_naming and decoder_name == "IterativeDecoder":
        dir_name = sim_name
    elif legacy_iter_dir_naming and decoder_name == "IterativeErasureErrorDecoder":
        dir_name = f"{sim_name}_erasure"
    else:
        dir_name = f"{sim_name}_{decoder_name}"

    res_dir = Path(output_dir) / dir_name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}.csv"
    graph_path: Path | None = (res_dir / f"{sim_name}_graph.png") if plot_graph else None
    log_path: Path | None = (res_dir / f"{sim_name}_log.txt") if print_log else None

    if graph_path is not None:
        save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)
    _print_code_summary(code, sim_name, graph_metrics, codeword)

    error_positions = build_candidate_error_positions(code)
    error_positions_tuple = tuple(error_positions)
    effective_max_err = min(int(n) - int(code.dimension()), int(max_err), len(error_positions))
    print_freq = compute_print_frequency(sim_itr, print_freq_factor)

    use_pool = workers > 1 and sim_itr > 1 and not print_log
    master_ss = np.random.SeedSequence(seed) if use_pool else None

    recorded_rows: list[list[Any]] = []
    consecutive_zero_runs = 0
    try:
        with ExitStack() as stack:
            res_file = stack.enter_context(open(csv_path, "w", newline=""))
            log_handle = (
                stack.enter_context(open(log_path, "w", encoding="utf-8"))
                if log_path is not None
                else None
            )
            executor = (
                stack.enter_context(
                    _create_worker_pool(
                        workers,
                        field,
                        codeword,
                        decoder,
                        decoder_name,
                        int(n),
                        error_positions_tuple,
                    )
                )
                if use_pool
                else None
            )

            writer = csv.writer(res_file)
            write_code_metadata_csv(writer, code, graph_metrics, codeword)
            writer.writerow(
                ["Num_of_error_symbols", "average_num_of_iteration", "probability_of_success"]
            )
            res_file.flush()

            if log_handle is not None:
                _print_code_summary(
                    code,
                    sim_name,
                    graph_metrics,
                    codeword,
                    stream=log_handle,
                    include_banner=False,
                )

            for n_err in range(1, effective_max_err + 1):
                print(f"Error Weight: {n_err}")
                success_itr = 0
                overall_num_itr = 0

                if executor is not None and master_ss is not None:
                    tasks = _build_chunk_tasks(n_err, sim_itr, workers, master_ss)
                    for c_succ, c_itr in executor.map(_worker_decoding_success_chunk, tasks):
                        success_itr += c_succ
                        overall_num_itr += c_itr
                    print(f"Iteration {sim_itr} / {sim_itr} Done.")
                else:
                    for i in range(sim_itr):
                        err_vec = sample_fixed_weight_error(
                            field, int(n), n_err, error_positions_tuple
                        )
                        received = codeword + err_vec
                        outcome = invoke_decoder(decoder, decoder_name, received)
                        correct_c = outcome.codeword
                        is_exact_match = correct_c == codeword

                        if log_handle is not None:
                            print(f"Error              : {err_vec}", file=log_handle)
                            print(f"Corrected Codeword : {correct_c}", file=log_handle)
                            print(f"Correction Successful: {is_exact_match}", file=log_handle)

                        if is_exact_match:
                            success_itr += 1
                            overall_num_itr += outcome.num_iterations

                        if (i + 1) % print_freq == 0:
                            print(f"Iteration {i + 1} / {sim_itr} Done.")

                avg_num_of_itr = (
                    overall_num_itr / success_itr if success_itr != 0 else float(max_num_of_itr)
                )
                succ_prob = 100.0 * (success_itr / sim_itr)
                row = [n_err, avg_num_of_itr, succ_prob]
                writer.writerow(row)
                res_file.flush()
                recorded_rows.append(row)

                if log_handle is not None:
                    print(f"Num of error symbols: {n_err}", file=log_handle)
                    print(f"Average Num of Iteration: {avg_num_of_itr}", file=log_handle)
                    print(f"Probability of Success: {succ_prob}", file=log_handle)
                    print("##########################", file=log_handle)

                if success_itr == 0:
                    consecutive_zero_runs += 1
                else:
                    consecutive_zero_runs = 0
                if (
                    stop_after_zero_runs is not None
                    and consecutive_zero_runs >= stop_after_zero_runs
                ):
                    break
    finally:
        if use_pool:
            _WORKER_STATE.clear()

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "graph_path": graph_path,
        "log_path": log_path,
        "rows": recorded_rows,
    }


def run_weight_distribution_analysis(
    *,
    q: int = 64,
    n: int = 64,
    k: int = 6,
    locality: Sequence[int] = (2, 5),
    local_minimum_distance: Sequence[int] = (3, 3),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "mult"),
    add_idx: int | None = None,
    output_dir: str | Path = "./results",
    plot_graph: bool = True,
) -> dict[str, Any]:
    """Compute exact codeword weight distribution and minimum distance of a two-set code.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        add_idx: Optional index into ``find_additive_subgroups`` overriding the
            default canonical additive subgroup.
        output_dir: Parent directory under which run artifacts are written.
        plot_graph: If ``True`` (default), render and save the bipartite coset-graph PNG.

    Returns:
        Dictionary with ``csv_path``, ``graph_path``, ``weight_distribution``, and
        ``code_distance``.
    """
    field = GF(int(q), repr="int")
    subgroup_override: list[Sequence[Any] | None] = [None, None]
    if add_idx is not None:
        if sub_group_type[0] == "add":
            sg_size = int(locality[0]) + int(local_minimum_distance[0]) - 1
            additive_subgroups = find_additive_subgroups(field, sg_size)
            subgroup_override = [additive_subgroups[int(add_idx)], None]
        elif sub_group_type[1] == "add":
            sg_size = int(locality[1]) + int(local_minimum_distance[1]) - 1
            additive_subgroups = find_additive_subgroups(field, sg_size)
            subgroup_override = [None, additive_subgroups[int(add_idx)]]
        else:
            raise ValueError("Cannot specify add_idx when neither subgroup_type is 'add'")

    _, code, _ = _prepare_code_and_codeword(
        q,
        n,
        k,
        locality,
        local_minimum_distance,
        sub_group_type,
        use_enc=False,
        subgroup=subgroup_override,
    )

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = _compute_runner_expansion_metrics(graph, right_nodes)
    suffix = f"_idx_{add_idx}" if add_idx is not None else ""
    sim_name = _build_sim_name(code, suffix=suffix)

    res_dir = Path(output_dir) / f"{sim_name}_weight_dist"
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}_weight_dist.csv"
    graph_path: Path | None = (res_dir / f"{sim_name}_graph.png") if plot_graph else None

    if graph_path is not None:
        save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)
    _print_code_summary(code, sim_name, graph_metrics)

    weight_dist = compute_code_weight_distribution(code)
    code_distance = next((i for i in range(1, len(weight_dist)) if weight_dist[i] != 0), None)

    print(weight_dist)
    if code_distance is not None:
        print(f"Code Distance: {code_distance}")

    with open(csv_path, "w", newline="") as res_file:
        writer = csv.writer(res_file)
        write_code_metadata_csv(writer, code, graph_metrics)
        writer.writerow(["Weight Distribution: ", weight_dist])
        if code_distance is not None:
            writer.writerow(["Code Distance: ", code_distance])
        res_file.flush()

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "graph_path": graph_path,
        "weight_distribution": weight_dist,
        "code_distance": code_distance,
    }
