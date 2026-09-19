"""Tests for the seed-extension robustness analysis.

The analysis exists to answer "do the conclusions survive more random graph
realizations?", so the tests pin the properties that make it trustworthy: the
inferential unit is the graph, budgets stay repeated measures, the convergence
curve enumerates every subset rather than one accumulation order, and the
prespecified seeds are constants rather than a post-hoc choice.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from rmcd_f.rev import robustness as RB
from rmcd_f.rev.panels import DEFAULT_SEEDS
from rmcd_f.rev.seedext import SEED_EXTENSION_METHODS, SEED_EXTENSION_SEEDS


def _rows(
    seeds,
    *,
    n_per_seed: int = 3,
    methods=("MPCF-Exact", "MPCF-Greedy"),
    gap_of=lambda method, seed, index: 0.0 if method == "MPCF-Exact" else 0.1,
    budgets=(0.02, 0.05, 0.10),
):
    """Synthetic run rows shaped like the real archive."""

    rows = []
    for seed in seeds:
        for index in range(n_per_seed):
            graph = f"centralized-N42-s{seed}-{index}"
            for method in methods:
                for budget in budgets:
                    rows.append(
                        {
                            "experiment": "seedext",
                            "instance_id": f"seedext__{graph}",
                            "graph_id": graph,
                            "base_id": graph,
                            "topology": "centralized",
                            "N": 42,
                            "seed": seed,
                            "method": method,
                            "variant": "test",
                            "budget_ratio": budget,
                            "relative_gap": gap_of(method, seed, index),
                            "kappa": 10.0,
                        }
                    )
    return rows


def test_prespecified_seeds_are_constants_not_a_post_hoc_choice() -> None:
    assert SEED_EXTENSION_SEEDS == (66, 77, 88, 99, 110)
    assert len(DEFAULT_SEEDS) == 5 and len(SEED_EXTENSION_SEEDS) == 5
    assert not set(DEFAULT_SEEDS) & set(SEED_EXTENSION_SEEDS)
    assert len(SEED_EXTENSION_METHODS) == 9
    assert "MPCF-Exact" in SEED_EXTENSION_METHODS
    assert "MPCF-Greedy" in SEED_EXTENSION_METHODS


def test_budgets_are_averaged_inside_each_graph() -> None:
    rows = _rows(
        [11],
        n_per_seed=1,
        budgets=(0.02, 0.05, 0.10),
        gap_of=lambda method, seed, index: 0.1 if method == "MPCF-Greedy" else 0.4,
    )
    gaps = RB.graph_level_gaps(rows, methods=["MPCF-Greedy"])
    # one value per graph, not one per graph-budget pair
    assert len(gaps) == 1
    assert gaps[("centralized-N42-s11-0", "MPCF-Greedy")] == pytest.approx(0.1)


def test_effect_stability_reports_five_and_ten_seed_estimates() -> None:
    rows = _rows(list(DEFAULT_SEEDS) + list(SEED_EXTENSION_SEEDS))
    output = RB.effect_stability_rows(
        rows,
        methods=["MPCF-Exact", "MPCF-Greedy"],
        original_seeds=DEFAULT_SEEDS,
        extension_seeds=SEED_EXTENSION_SEEDS,
    )
    greedy = next(row for row in output if row["method"] == "MPCF-Greedy")
    assert greedy["n_graphs_5seed"] == 5 * 3
    assert greedy["n_graphs_10seed"] == 10 * 3
    assert greedy["relative_gap_mean_5seed"] == pytest.approx(0.1)
    assert greedy["relative_gap_mean_10seed"] == pytest.approx(0.1)
    assert greedy["relative_gap_mean_delta"] == pytest.approx(0.0)
    assert greedy["effect_direction_preserved"] == 1


def test_effect_stability_detects_a_real_shift() -> None:
    def gap(method, seed, index):
        if method == "MPCF-Exact":
            return 0.0
        # extension seeds are materially worse
        return 0.3 if seed in SEED_EXTENSION_SEEDS else 0.1

    rows = _rows(list(DEFAULT_SEEDS) + list(SEED_EXTENSION_SEEDS), gap_of=gap)
    output = RB.effect_stability_rows(
        rows,
        methods=["MPCF-Exact", "MPCF-Greedy"],
        original_seeds=DEFAULT_SEEDS,
        extension_seeds=SEED_EXTENSION_SEEDS,
    )
    greedy = next(row for row in output if row["method"] == "MPCF-Greedy")
    assert greedy["relative_gap_mean_5seed"] == pytest.approx(0.1)
    assert greedy["relative_gap_mean_10seed"] == pytest.approx(0.2)
    assert greedy["relative_gap_mean_delta"] == pytest.approx(0.1)


def test_paired_intervals_narrow_when_graphs_double() -> None:
    # Reference and comparator each get their own per-graph draw: sharing one
    # draw would make the paired difference constant and the interval
    # degenerate, testing nothing about clustering.
    rng = np.random.default_rng(0)
    keys = [
        (seed, index)
        for seed in DEFAULT_SEEDS + SEED_EXTENSION_SEEDS
        for index in range(9)
    ]
    reference_noise = {k: float(rng.normal(0, 0.05)) for k in keys}
    comparator_noise = {k: float(rng.normal(0, 0.05)) for k in keys}

    def gap(method, seed, index):
        if method == "MPCF-Exact":
            return reference_noise[(seed, index)]
        return 0.2 + comparator_noise[(seed, index)]

    rows = _rows(list(DEFAULT_SEEDS) + list(SEED_EXTENSION_SEEDS), n_per_seed=9, gap_of=gap)
    output = RB.paired_interval_rows(
        rows,
        methods=["MPCF-Exact", "MPCF-Greedy"],
        original_seeds=DEFAULT_SEEDS,
        extension_seeds=SEED_EXTENSION_SEEDS,
        resamples=4000,
    )
    row = next(entry for entry in output if entry["method"] == "MPCF-Greedy")
    assert row["n_pairs_5seed"] == 45
    assert row["n_pairs_10seed"] == 90
    assert row["mean_effect_10seed"] == pytest.approx(-0.2, abs=0.02)
    assert row["ci_low_10seed"] < row["mean_effect_10seed"] < row["ci_high_10seed"]
    # Doubling the panel must not report a widening interval.
    assert float(row["ci_width_10seed"]) <= float(row["ci_width_5seed"]) * 1.35
    assert row["ci_width_shrank"] in (0, 1)


def test_convergence_enumerates_every_subset() -> None:
    rows = _rows(list(DEFAULT_SEEDS) + list(SEED_EXTENSION_SEEDS), n_per_seed=1)
    output = RB.convergence_rows(
        rows, methods=["MPCF-Greedy"], seeds=DEFAULT_SEEDS + SEED_EXTENSION_SEEDS
    )
    by_k = {row["n_seeds"]: row for row in output}
    assert set(by_k) == set(range(2, 11))
    for k, row in by_k.items():
        assert row["n_subsets"] == math.comb(10, k)
    assert sum(row["n_subsets"] for row in output) == 1013
    # a constant panel gives a flat curve with a degenerate band
    assert by_k[2]["estimate_mean"] == pytest.approx(0.1)
    assert by_k[10]["estimate_mean"] == pytest.approx(0.1)
    assert by_k[10]["band_low"] == pytest.approx(0.1)


def test_convergence_subset_count_matches_itertools() -> None:
    subsets = sum(math.comb(10, k) for k in range(2, 11))
    enumerated = sum(1 for k in range(2, 11) for _ in itertools.combinations(range(10), k))
    # 2**10 - C(10,0) - C(10,1) = 1013 subsets of size >= 2
    assert subsets == enumerated == 1013


def test_stratified_bootstrap_keeps_strata_fixed() -> None:
    rows = []
    for topology in ("centralized", "modular"):
        for seed in list(DEFAULT_SEEDS) + list(SEED_EXTENSION_SEEDS):
            graph = f"{topology}-N42-s{seed}"
            for method in ("MPCF-Exact", "MPCF-Greedy"):
                for budget in (0.02, 0.05, 0.10):
                    rows.append(
                        {
                            "graph_id": graph,
                            "instance_id": graph,
                            "topology": topology,
                            "N": 42,
                            "seed": seed,
                            "method": method,
                            "budget_ratio": budget,
                            "relative_gap": 0.0 if method == "MPCF-Exact" else 0.2,
                        }
                    )
    output = RB.stratified_bootstrap_rows(
        rows, methods=["MPCF-Exact", "MPCF-Greedy"], resamples=1000
    )
    row = next(entry for entry in output if entry["method"] == "MPCF-Greedy")
    assert row["n_strata"] == 2
    assert row["n_observations"] == 20
    assert row["mean_paired_effect"] == pytest.approx(-0.2)
    assert row["ci_low"] <= row["mean_paired_effect"] <= row["ci_high"]


def test_manifest_records_the_prespecification() -> None:
    payload = RB.manifest_provenance(
        original_seeds=DEFAULT_SEEDS, extension_seeds=SEED_EXTENSION_SEEDS
    )
    assert payload["prespecified"] is True
    assert payload["seeds_per_topology_size_cell"] == {"original": 5, "pooled": 10}
    assert payload["inferential_unit"] == "graph"
    assert "averaged inside each graph" in payload["repeated_measure"]


# ---------------------------------------------------------------------------
# panel wiring: the extension must never silently reuse the benchmark graphs
# ---------------------------------------------------------------------------


def _workspace_temp():
    """Scratch directory inside the repo: the OS temp dir is not writable here."""

    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return Path(tempfile.mkdtemp(prefix="seedext_", dir=root))


def _load_runner():
    """Import scripts/rev_run.py without installing it as a package."""

    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "rev_run_under_test", root / "scripts" / "rev_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seedext_resolves_its_own_seeds_not_the_benchmark_seeds() -> None:
    """Regression: a shared --seeds default once made the extension a duplicate.

    ``--seeds`` has no default, and each panel resolves its own, because a
    default of ``DEFAULT_SEEDS`` silently rebuilt the original 45 graphs and
    labelled them an independent extension.
    """

    runner = _load_runner()
    scratch = _workspace_temp()
    args = runner.build_parser().parse_args(
        ["--experiment", "seedext", "--output", str(scratch)]
    )
    assert args.seeds is None, "the CLI must not pre-fill a seed set"

    extension = runner.build_panel("seedext", scratch, args)
    benchmark = runner.build_panel("main", scratch, args)

    extension_seeds = sorted({spec.seed for spec in extension})
    benchmark_seeds = sorted({spec.seed for spec in benchmark})
    assert extension_seeds == sorted(SEED_EXTENSION_SEEDS)
    assert benchmark_seeds == sorted(DEFAULT_SEEDS)
    assert not set(extension_seeds) & set(benchmark_seeds)
    assert len(extension) == 45 and len(benchmark) == 45
    # the pooled design is ten realizations per topology-size cell
    assert len(set(extension_seeds) | set(benchmark_seeds)) == 10


def test_seedext_graphs_match_the_benchmark_in_every_respected_way() -> None:
    """Only the seed may differ; topology, size, cost and budget must not."""

    runner = _load_runner()
    scratch = _workspace_temp()
    args = runner.build_parser().parse_args(
        ["--experiment", "seedext", "--output", str(scratch)]
    )
    extension = runner.build_panel("seedext", scratch, args)
    benchmark = runner.build_panel("main", scratch, args)

    def profile(specs):
        return sorted(
            (
                spec.topology,
                spec.node_count,
                spec.edge_count,
                spec.composition,
                spec.scale_tier,
                round(spec.total_protection_cost, 6),
                tuple(spec.budget_ratios),
            )
            for spec in specs
        )

    assert profile(extension) == profile(benchmark)
    assert extension[0].experiment == "seedext"
    assert benchmark[0].experiment == "main"
