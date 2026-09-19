"""Tests for the graph- and base-ID clustered statistics.

The point of these tests is the reviewer's objection: three budgets on one
frozen graph are repeated measures, so the analysis must average inside the
graph before testing and must resample whole clusters, never individual
records.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rmcd_f.rev import stats as S


def _row(
    graph: str,
    method: str,
    budget: float,
    gap: float,
    *,
    seed: int = 11,
    base: str | None = None,
    experiment: str = "main",
    kappa: float = 10.0,
    undefended: float = 8.0,
) -> dict:
    return {
        "experiment": experiment,
        "instance_id": f"{experiment}__{graph}",
        "graph_id": graph,
        "base_id": base or graph,
        "topology": graph.split("-")[0],
        "N": 42,
        "num_edges": 126,
        "seed": seed,
        "method": method,
        "budget_ratio": budget,
        "relative_gap": gap,
        "kappa": kappa,
        "undefended": undefended,
        "status": "optimal",
        "certificate_mode": "solver_closed",
        "runtime_selection_s": 1.0,
        "runtime_evaluation_s": 0.1,
        "runtime_total_s": 1.1,
    }


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    raw = [0.001, 0.02, 0.03, 0.5]
    adjusted = S.holm_adjust(raw)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    values = [adjusted[index] for index in order]
    assert values == sorted(values)
    assert adjusted[0] == pytest.approx(0.004)
    assert adjusted[3] == pytest.approx(0.5)


def test_holm_is_empty_safe() -> None:
    assert S.holm_adjust([]) == []


def test_rank_biserial_is_bounded_and_signed() -> None:
    assert S.rank_biserial([1, 2, 3]) == pytest.approx(1.0)
    assert S.rank_biserial([-1, -2, -3]) == pytest.approx(-1.0)
    assert S.rank_biserial([]) == 0.0
    assert S.rank_biserial([1e-12, -1e-12]) == 0.0
    mixed = S.rank_biserial([1, -1, 2])
    assert -1.0 <= mixed <= 1.0


def test_paired_wilcoxon_returns_a_small_p_for_a_consistent_shift() -> None:
    shifted = [float(value) for value in range(1, 21)]
    statistic, p_value, nonzero = S.paired_wilcoxon(shifted)
    assert nonzero == 20
    assert p_value < 1e-4
    assert math.isfinite(statistic)


def test_paired_wilcoxon_handles_degenerate_input() -> None:
    statistic, p_value, nonzero = S.paired_wilcoxon([1e-12, 0.0])
    assert nonzero == 0
    assert p_value == 1.0
    assert math.isnan(statistic)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0, 200.0]
    clusters = ["a", "a", "a", "a", "b", "b"]
    low, high = S.cluster_bootstrap_ci(values, clusters, resamples=2000, seed=7)
    # Only two clusters exist, so every resample contains a or b twice; the
    # interval must cover the two cluster means and cannot be narrower than
    # an individual-record bootstrap would be.
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= high
    record_low, record_high = S.cluster_bootstrap_ci(
        values, [str(index) for index in range(len(values))], resamples=2000, seed=7
    )
    assert (high - low) >= (record_high - record_low)


def test_cluster_bootstrap_needs_two_clusters() -> None:
    low, high = S.cluster_bootstrap_ci([1.0, 2.0], ["x", "x"], resamples=100)
    assert math.isnan(low) and math.isnan(high)


def test_panel_averages_budgets_inside_each_graph() -> None:
    rows = []
    for graph in ("a-N42-s11", "b-N42-s22", "c-N42-s33"):
        for budget, gap in ((0.02, 0.10), (0.05, 0.20), (0.10, 0.30)):
            rows.append(_row(graph, "MPCF-Greedy", budget, gap))
            rows.append(_row(graph, "MPCF-Exact", budget, 0.0))
    panel = S.build_panel(
        rows, name="primary", metric="relative_gap",
        methods=["MPCF-Exact", "MPCF-Greedy"], unit="graph",
    )
    assert len(panel.blocks) == 3
    # One block per graph, not one per graph-budget pair.
    assert panel.values[("a-N42-s11", "MPCF-Greedy")] == pytest.approx(0.20)
    blocks, matrix = panel.matrix()
    assert len(blocks) == 3
    assert matrix.shape == (3, 2)


def test_friedman_runs_over_graph_blocks() -> None:
    rows = []
    for index in range(6):
        graph = f"g{index}-N42-s{index}"
        rows.append(_row(graph, "MPCF-Exact", 0.05, 0.0))
        rows.append(_row(graph, "MPCF-Greedy", 0.05, 0.1 + 0.01 * index))
        rows.append(_row(graph, "DegreeProtect", 0.05, 0.4 + 0.01 * index))
    panel = S.build_panel(
        rows, name="p", metric="relative_gap",
        methods=["MPCF-Exact", "MPCF-Greedy", "DegreeProtect"], unit="graph",
    )
    row = S.friedman_row(panel)
    assert row["status"] == "ok"
    assert row["n_blocks"] == 6
    assert row["df"] == 2
    assert row["statistic"] == pytest.approx(12.0)
    assert row["p_value"] < 0.01


def test_friedman_reports_insufficiency_instead_of_crashing() -> None:
    rows = [
        _row("g0", "A", 0.05, 0.1),
        _row("g0", "B", 0.05, 0.2),
    ]
    panel = S.build_panel(rows, name="p", metric="relative_gap", methods=["A", "B"], unit="graph")
    row = S.friedman_row(panel)
    assert row["status"] != "ok"
    assert row["p_value"] == ""


def test_pairwise_rows_carry_raw_holm_effect_and_ci() -> None:
    rows = []
    for index in range(8):
        graph = f"g{index}-N42-s{index}"
        rows.append(_row(graph, "MPCF-Exact", 0.05, 0.0))
        rows.append(_row(graph, "MPCF-Greedy", 0.05, 0.1 * (index + 1)))
        rows.append(_row(graph, "DegreeProtect", 0.05, 0.05 * (index + 1)))
    panel = S.build_panel(
        rows, name="primary", metric="relative_gap",
        methods=["MPCF-Exact", "MPCF-Greedy", "DegreeProtect"], unit="graph",
    )
    pairs = S.apply_holm(S.pairwise_rows(panel, resamples=500, seed=3))
    assert len(pairs) == 2
    for row in pairs:
        assert row["n_pairs"] == 8
        assert row["n_clusters"] == 8
        assert 0.0 <= row["p_raw"] <= 1.0
        assert row["p_holm"] >= row["p_raw"] - 1e-12
        assert -1.0 <= row["rank_biserial"] <= 1.0
        assert row["mean_ci_low"] <= row["mean_ci_high"]
        assert row["unit"] == "graph"
        assert row["metric_better_when"] == "lower"
        # The paired difference is reference minus comparator and a lower
        # relative gap is better, so the orientation text must say so.
        assert "negative values" in row["orientation"]
        assert "favour the reference" in row["orientation"]


def test_role_panel_clusters_on_base_id_not_on_records() -> None:
    rows = []
    for base_index in range(5):
        base = f"base{base_index}-N42-s{base_index}"
        for composition in ("4-3-3-4", "3-4-4-3", "2-4-4-4"):
            instance = f"role__{base}__{composition}"
            for budget, gap in ((0.02, 0.3), (0.05, 0.2), (0.10, 0.1)):
                for method, shift in (("MPCF-Exact", 0.0), ("DegreeProtect", 0.05)):
                    row = _row(
                        base, method, budget, gap + shift,
                        base=base, experiment="role",
                    )
                    row["instance_id"] = instance
                    row["composition"] = composition
                    rows.append(row)
    panel = S.build_panel(
        rows, name="role", metric="relative_gap",
        methods=["MPCF-Exact", "DegreeProtect"], unit="base",
    )
    # 5 bases, not 5 x 3 compositions x 3 budgets = 45 records.
    assert len(panel.blocks) == 5
    assert panel.values[("base0-N42-s0", "DegreeProtect")] == pytest.approx(0.25)
    pairs = S.pairwise_rows(panel, resamples=200)
    assert pairs[0]["n_pairs"] == 5
    assert pairs[0]["n_clusters"] == 5


# ---------------------------------------------------------------------------
# descriptive outputs
# ---------------------------------------------------------------------------


def test_descriptive_rows_split_all_cells_and_effective_cells() -> None:
    rows = [
        _row("g0", "MPCF-Exact", 0.02, 0.0, kappa=8.0, undefended=8.0),
        _row("g0", "MPCF-Exact", 0.05, 0.1, kappa=9.0, undefended=8.0),
        _row("g1", "MPCF-Exact", 0.02, 0.0, kappa=8.0, undefended=8.0),
    ]
    output = S.descriptive_rows(rows, panel="all_cells", methods=["MPCF-Exact"])
    assert len(output) == 1
    row = output[0]
    assert row["n_records"] == 3
    assert row["n_graphs"] == 2
    assert row["relative_gap_mean"] == pytest.approx(0.1 / 3)
    assert row["exact_attainment_rate"] == pytest.approx(2 / 3)
    assert row["runtime_selection_median_s"] == pytest.approx(1.0)


def test_bootstrap_summary_reports_the_cluster_unit() -> None:
    rows = []
    for index in range(6):
        graph = f"g{index}-N42-s{index}"
        for budget in (0.02, 0.05, 0.10):
            rows.append(
                _row(graph, "MPCF-Greedy", budget, 0.1 + 0.05 * index, seed=index)
            )
    panel = S.build_panel(
        rows, name="primary", metric="relative_gap", methods=["MPCF-Greedy"], unit="graph"
    )
    summary = S.bootstrap_summary_rows(panel, metric_label="relative_gap", resamples=500)
    assert len(summary) == 1
    row = summary[0]
    assert row["bootstrap_unit"] == "graph_id"
    assert row["n_clusters"] == 6
    assert row["n_observations"] == 6
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
