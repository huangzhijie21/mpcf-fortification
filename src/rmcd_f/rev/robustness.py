"""Seed-extension robustness analysis.

Reviewer 2's concern is not a significance threshold: it is whether five random
realizations per topology-size cell are enough to show that the conclusions are
not driven by a few particular graphs.  The answer is a stability analysis, so
this module deliberately does **not** lead with a new p-value.

Three questions are answered, each against the same prespecified panels:

1. *Is the effect estimate stable?*  The per-method mean and median relative gap
   is reported on the original five seeds and on the pooled ten seeds, together
   with the change between them.
2. *Is the interval stable and does it narrow?*  The prespecified Exact-versus-
   baseline paired effect is recomputed with its graph-clustered 95% interval at
   five and at ten seeds.
3. *Where does the estimate stabilise?*  For every ``k`` from 2 to 10 the mean
   relative gap is computed over **all** ``C(10, k)`` seed subsets, not one
   cumulative ordering, so the curve cannot depend on the order seeds are added.
   The band is the 2.5-97.5% quantile across subsets.

The inferential unit stays the graph.  Budgets remain repeated measures and are
averaged inside each graph before anything is compared.
"""

from __future__ import annotations

import itertools
import math
import statistics as st
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import stats as S

PRIMARY_METRIC = "relative_gap"
REFERENCE = "MPCF-Exact"
MIN_SEEDS = 2
SUBSET_BAND = (2.5, 97.5)


def _num(value: Any) -> float | None:
    if value is None or value == "NA":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def graph_level_gaps(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    metric: str = PRIMARY_METRIC,
) -> dict[tuple[str, str], float]:
    """Average the repeated budgets inside each ``(graph, method)`` cell."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method"))
        if method not in methods:
            continue
        value = _num(row.get(metric))
        if value is None:
            continue
        grouped[(str(row["graph_id"]), method)].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def seed_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """``graph_id -> generator seed``."""

    index: dict[str, int] = {}
    for row in rows:
        seed = row.get("seed")
        if seed in (None, "NA"):
            continue
        index[str(row["graph_id"])] = int(seed)
    return index


def _graphs_for_seeds(seeds_index: Mapping[str, int], seeds: Iterable[int]) -> set[str]:
    wanted = {int(seed) for seed in seeds}
    return {graph for graph, seed in seeds_index.items() if seed in wanted}


def effect_stability_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    original_seeds: Sequence[int],
    extension_seeds: Sequence[int],
    reference: str = REFERENCE,
) -> list[dict[str, Any]]:
    """Mean and median gap at five versus ten seeds, and the change."""

    gaps = graph_level_gaps(rows, methods=methods)
    index = seed_index(rows)
    base_graphs = _graphs_for_seeds(index, original_seeds)
    pooled_graphs = base_graphs | _graphs_for_seeds(index, extension_seeds)

    def values(graphs: set[str], method: str) -> list[float]:
        return [gaps[(g, method)] for g in sorted(graphs) if (g, method) in gaps]

    output: list[dict[str, Any]] = []
    for method in methods:
        five = values(base_graphs, method)
        ten = values(pooled_graphs, method)
        if not five or not ten:
            continue
        reference_ten = values(pooled_graphs, reference)
        paired_ten = [
            gaps[(g, reference)] - gaps[(g, method)]
            for g in sorted(pooled_graphs)
            if (g, reference) in gaps and (g, method) in gaps
        ]
        reference_five = values(base_graphs, reference)
        paired_five = [
            gaps[(g, reference)] - gaps[(g, method)]
            for g in sorted(base_graphs)
            if (g, reference) in gaps and (g, method) in gaps
        ]
        output.append(
            {
                "method": method,
                "n_graphs_5seed": len(five),
                "n_graphs_10seed": len(ten),
                "relative_gap_mean_5seed": float(np.mean(five)),
                "relative_gap_mean_10seed": float(np.mean(ten)),
                "relative_gap_mean_delta": float(np.mean(ten) - np.mean(five)),
                "relative_gap_median_5seed": float(np.median(five)),
                "relative_gap_median_10seed": float(np.median(ten)),
                "relative_gap_median_delta": float(np.median(ten) - np.median(five)),
                "kappa_mean_5seed": float(np.mean(reference_five)) if reference_five else "",
                "paired_effect_vs_reference_5seed": (
                    float(np.mean(paired_five)) if paired_five else ""
                ),
                "paired_effect_vs_reference_10seed": (
                    float(np.mean(paired_ten)) if paired_ten else ""
                ),
                "effect_direction_preserved": int(
                    (np.mean(paired_five) >= 0) == (np.mean(paired_ten) >= 0)
                    if paired_five and paired_ten
                    else 1
                ),
            }
        )
    return output


def paired_interval_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    original_seeds: Sequence[int],
    extension_seeds: Sequence[int],
    reference: str = REFERENCE,
    resamples: int = 10_000,
    seed: int = S.DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Prespecified paired effect with its graph-clustered interval, 5 vs 10 seeds."""

    gaps = graph_level_gaps(rows, methods=methods)
    index = seed_index(rows)
    base_graphs = _graphs_for_seeds(index, original_seeds)
    pooled_graphs = base_graphs | _graphs_for_seeds(index, extension_seeds)

    output: list[dict[str, Any]] = []
    for method in methods:
        if method == reference:
            continue
        entry: dict[str, Any] = {"method": method, "reference_method": reference}
        for label, graphs in (("5seed", base_graphs), ("10seed", pooled_graphs)):
            pairs = [
                (g, gaps[(g, reference)] - gaps[(g, method)])
                for g in sorted(graphs)
                if (g, reference) in gaps and (g, method) in gaps
            ]
            if not pairs:
                continue
            values = [value for _, value in pairs]
            clusters = [graph for graph, _ in pairs]
            low, high = S.cluster_bootstrap_ci(
                values, clusters, resamples=resamples, seed=seed
            )
            entry[f"n_pairs_{label}"] = len(values)
            entry[f"mean_effect_{label}"] = float(np.mean(values))
            entry[f"median_effect_{label}"] = float(np.median(values))
            entry[f"ci_low_{label}"] = low
            entry[f"ci_high_{label}"] = high
            entry[f"ci_width_{label}"] = (
                float(high - low) if math.isfinite(low) and math.isfinite(high) else ""
            )
            entry[f"rank_biserial_{label}"] = S.rank_biserial(values)
        if "ci_width_5seed" in entry and "ci_width_10seed" in entry:
            try:
                entry["ci_width_shrank"] = int(
                    float(entry["ci_width_10seed"]) <= float(entry["ci_width_5seed"])
                )
            except (TypeError, ValueError):
                entry["ci_width_shrank"] = ""
        output.append(entry)
    return output


def convergence_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    metric: str = PRIMARY_METRIC,
    band: Sequence[float] = SUBSET_BAND,
) -> list[dict[str, Any]]:
    """Mean gap against the number of random realizations.

    Every ``k``-subset of the available seeds is enumerated, so the curve is a
    property of the panels rather than of one arbitrary accumulation order.  The
    band is the ``band`` percentile range across subsets.
    """

    gaps = graph_level_gaps(rows, methods=methods, metric=metric)
    index = seed_index(rows)
    available = sorted({int(seed) for seed in seeds})
    output: list[dict[str, Any]] = []
    for k in range(MIN_SEEDS, len(available) + 1):
        subsets = list(itertools.combinations(available, k))
        for method in methods:
            per_subset: list[float] = []
            for subset in subsets:
                graphs = _graphs_for_seeds(index, subset)
                values = [gaps[(g, method)] for g in sorted(graphs) if (g, method) in gaps]
                if values:
                    per_subset.append(float(np.mean(values)))
            if not per_subset:
                continue
            array = np.asarray(per_subset, dtype=float)
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_seeds": k,
                    "n_subsets": len(per_subset),
                    "n_graphs_per_subset": k * (len(index) // len(available)) if available else "",
                    "estimate_mean": float(array.mean()),
                    "estimate_median": float(np.median(array)),
                    "estimate_min": float(array.min()),
                    "estimate_max": float(array.max()),
                    "band_low": float(np.percentile(array, band[0])),
                    "band_high": float(np.percentile(array, band[1])),
                    "band_level": f"{band[0]}-{band[1]}",
                }
            )
    return output


def stratified_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    reference: str = REFERENCE,
    resamples: int = 10_000,
    seed: int = S.DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Bootstrap that resamples graphs *within* each topology-size stratum.

    Plain graph bootstrapping can let an unlucky draw over-represent one
    topology or size.  Resampling inside the nine strata keeps their counts
    fixed, which is the conservative reading of "the panel has a balanced
    design".
    """

    gaps = graph_level_gaps(rows, methods=methods)
    strata: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        strata[f"{row.get('topology')}|{row.get('N')}"].add(str(row["graph_id"]))

    generator = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for method in methods:
        if method == reference:
            continue
        pools = [
            np.asarray(
                [
                    gaps[(g, reference)] - gaps[(g, method)]
                    for g in sorted(graphs)
                    if (g, reference) in gaps and (g, method) in gaps
                ],
                dtype=float,
            )
            for graphs in strata.values()
        ]
        pools = [pool for pool in pools if pool.size]
        if len(pools) < 2:
            continue
        draws = np.empty(resamples, dtype=float)
        for index in range(resamples):
            drawn = [pool[generator.integers(0, pool.size, size=pool.size)] for pool in pools]
            draws[index] = float(np.concatenate(drawn).mean())
        observed = float(np.concatenate(pools).mean())
        output.append(
            {
                "method": method,
                "reference_method": reference,
                "n_strata": len(pools),
                "n_observations": int(sum(pool.size for pool in pools)),
                "mean_paired_effect": observed,
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "ci_level": 0.95,
                "ci_method": f"stratified_cluster_bootstrap_{resamples}",
                "stratification": "topology x N",
                "bootstrap_seed": seed,
            }
        )
    return output


def manifest_provenance(
    *,
    original_seeds: Sequence[int],
    extension_seeds: Sequence[int],
) -> dict[str, Any]:
    """Machine-readable statement that the extension seeds were prespecified."""

    return {
        "design": "seed-extension robustness analysis",
        "original_seeds": list(original_seeds),
        "extension_seeds": list(extension_seeds),
        "seeds_per_topology_size_cell": {
            "original": len(original_seeds),
            "pooled": len(original_seeds) + len(extension_seeds),
        },
        "prespecified": True,
        "prespecification_note": (
            "The extension seeds are declared as constants in "
            "rmcd_f.rev.seedext.SEED_EXTENSION_SEEDS and were fixed before any of "
            "their results were examined; no seed was selected on its outcome."
        ),
        "inferential_unit": "graph",
        "repeated_measure": "budget_ratio (averaged inside each graph)",
        "primary_analysis_unchanged": (
            "the original 45-graph panel remains the frozen full-method benchmark"
        ),
        "convergence": (
            "for every k, the mean is taken over ALL C(10,k) seed subsets, so the "
            "curve does not depend on an accumulation order"
        ),
    }
