"""Unit and regression tests for tamo_barg_lrc.expander bipartite coset graphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from tamo_barg_lrc import GF, TamoBargCodeTwoSets
from tamo_barg_lrc.expander import (
    compute_expansion_metrics,
    create_bipartite_coset_graph,
    get_error_position_subgraph,
    save_bipartite_graph_plot,
)


def test_bipartite_coset_graph_counts_and_attributes_add_add_and_add_mult() -> None:
    """Verify node/edge counts and bipartite attributes on ['add', 'add'] and ['add', 'mult']."""
    f64 = GF(64, "a")

    # ["add", "add"] over GF(64), coset sizes 4 and 4 -> 16 left, 16 right, 64 edges.
    code_aa = TamoBargCodeTwoSets(f64, 64, 10, [2, 2], [3, 3], ["add", "add"])
    graph_aa, left_aa, right_aa = code_aa.bipartite_graph()
    assert len(left_aa) == 16
    assert len(right_aa) == 16
    assert left_aa.isdisjoint(right_aa)
    assert graph_aa.number_of_nodes() == 32
    assert graph_aa.number_of_edges() == 64
    assert all(graph_aa.nodes[u]["bipartite"] == 0 for u in left_aa)
    assert all(graph_aa.nodes[v]["bipartite"] == 1 for v in right_aa)

    # ["add", "mult"] over GF(64), coset sizes 4 and 7 -> 16 left, 9 right, 63 edges (0 excluded).
    code_am = TamoBargCodeTwoSets(f64, 64, 12, [2, 5], [3, 3], ["add", "mult"])
    graph_am, left_am, right_am = code_am.bipartite_graph()
    assert len(left_am) == 16
    assert len(right_am) == 9
    assert graph_am.number_of_nodes() == 25
    assert graph_am.number_of_edges() == 63


def test_create_bipartite_coset_graph_validation() -> None:
    """Verify create_bipartite_coset_graph raises ValueError on malformed inputs."""
    f4 = GF(4, "a")
    pts = list(f4)

    with pytest.raises(ValueError, match="length 2"):
        create_bipartite_coset_graph([[[pts[0]]]], [[[0]]], pts)

    with pytest.raises(ValueError, match="out of range"):
        create_bipartite_coset_graph([[[pts[0]]], [[pts[0]]]], [[[4]], [[0]]], pts)

    with pytest.raises(ValueError, match="Duplicate evaluation point index"):
        create_bipartite_coset_graph(
            [[[pts[0]], [pts[1]]], [[pts[0], pts[1]]]],
            [[[0], [0]], [[0, 1]]],
            pts,
        )


def test_regression_b12_error_position_subgraph_polynomial_repr_and_mult_mult() -> None:
    """Regression B12: support default polynomial GF(64, 'a') and n=63 ['mult', 'mult'] indices."""
    # Default polynomial representation (NO repr='int'): str(elem) is a polynomial in 'a'.
    f64 = GF(64, "a")
    code_aa = TamoBargCodeTwoSets(f64, 64, 10, [2, 2], [3, 3], ["add", "add"])
    graph_aa, _, _ = code_aa.bipartite_graph()
    num_comp = nx.number_connected_components(graph_aa)
    assert num_comp == 4

    all_positions: list[int] = []
    for comp_idx in range(num_comp):
        sub_positions = get_error_position_subgraph(graph_aa, comp_idx)
        assert len(sub_positions) == 16
        assert all(0 <= idx < 64 for idx in sub_positions)
        all_positions.extend(sub_positions)
    assert sorted(all_positions) == list(range(64))

    # ["mult", "mult"] of length n=63 over GF(64) where 0 is absent from evaluation_points.
    code_mm = TamoBargCodeTwoSets(f64, 63, 35, [5, 7], [3, 3], ["mult", "mult"])
    assert f64.zero() not in code_mm.evaluation_points()
    graph_mm, _, _ = code_mm.bipartite_graph()
    positions_mm = get_error_position_subgraph(graph_mm, 0)
    assert positions_mm == list(range(63))
    assert all(0 <= idx < 63 for idx in positions_mm)

    # Higher edge connectivity k=2 on connected coset graph and error handling.
    positions_k2 = get_error_position_subgraph(graph_mm, 0, edge_connectivity_k=2)
    assert all(0 <= idx < 63 for idx in positions_k2)

    with pytest.raises(ValueError, match="edge_connectivity_k"):
        get_error_position_subgraph(graph_mm, 0, edge_connectivity_k=0)
    with pytest.raises(IndexError, match="out of range"):
        get_error_position_subgraph(graph_aa, num_comp)


def test_expansion_metrics_and_headless_plot_saving(tmp_path: Path) -> None:
    """Verify expansion metrics and leak-free PNG rendering to tmp_path (B18, P4)."""
    f64 = GF(64, "a")

    # Connected coset graph: coset sizes 8 and 8 span all of GF(64).
    code_conn = TamoBargCodeTwoSets(f64, 64, 12, [6, 6], [3, 3], ["add", "add"])
    g_conn, left_conn, right_conn = code_conn.bipartite_graph()
    metrics_conn = compute_expansion_metrics(g_conn, right_conn)
    assert metrics_conn["is_connected"] is True
    assert metrics_conn["num_connected_components"] == 1
    assert metrics_conn["edge_expansion"] > 0.0
    assert metrics_conn["spectral_gap"] > 0.0
    assert metrics_conn["algebraic_connectivity"] > 0.0
    assert len(metrics_conn["eigenvalues"]) == g_conn.number_of_nodes()

    # Disconnected coset graph: coset sizes 4 and 4 yield 4 components -> algebraic_connectivity 0.
    code_disc = TamoBargCodeTwoSets(f64, 64, 10, [2, 2], [3, 3], ["add", "add"])
    g_disc, _, _ = code_disc.bipartite_graph()
    metrics_disc = compute_expansion_metrics(g_disc)
    assert metrics_disc["is_connected"] is False
    assert metrics_disc["num_connected_components"] == 4
    assert metrics_disc["algebraic_connectivity"] == 0.0

    # Empty graph edge case.
    empty_metrics = compute_expansion_metrics(nx.Graph())
    assert empty_metrics["is_connected"] is False
    assert empty_metrics["num_connected_components"] == 0

    # Regression B18: headless PNG creation with zero leaked matplotlib figures.
    plt.close("all")
    out_png = tmp_path / "plots" / "bipartite_graph.png"
    save_bipartite_graph_plot(g_conn, left_conn, right_conn, out_png)
    assert out_png.is_file()
    raw_bytes = out_png.read_bytes()
    assert len(raw_bytes) > 0
    assert raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(plt.get_fignums()) == 0
