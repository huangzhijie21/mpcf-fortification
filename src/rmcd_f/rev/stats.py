"""Graph- and base-ID clustered statistics for the revision.

Reviewer 2's objection is the organising constraint here: the three budget
levels are repeated measures on one frozen graph, so the 135 graph-budget pairs
are **not** 135 independent samples, and the 810 role-composition records are
**not** 810 independent samples either.

Primary analysis (pre-specified)
    1. average the relative gap over the three budgets inside each
       ``(graph, method)`` cell;
    2. Friedman omnibus over the 45 graphs, one block per graph;
    3. pairwise Wilcoxon signed-rank against ``MPCF-Exact`` on the 45
       graph-level paired observations;
    4. Holm-adjust the pairwise family, report rank-biserial and a cluster
       bootstrap 95% CI.

Secondary analysis
    the same tests repeated inside each budget separately (n = 45 each).

Role-composition analysis
    whole ``base_id`` clusters are resampled, carrying all six compositions and
    all three budgets together.

Nothing here reads a Python object produced by a solver: every function takes
the rows of ``runs_long.csv``.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

DEFAULT_REFERENCE = "MPCF-Exact"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260905
DEFAULT_ALPHA = 0.05
MIN_PAIRS = 2

#: The methods that certify their own optimality.  ``MPCF-Greedy`` is not here by
#: construction: it returns a one-step marginal ablation and sets
#: ``optimal = False`` unconditionally, so its certificate is always open and its
#: ``solved_count`` of zero says nothing about the quality of its answers.
CERTIFIED_METHODS = ("MPCF-Exact", "MPCF-CG")

#: Which direction of each metric is good.  ``relative_gap`` is a distance to
#: the optimum, so a *negative* paired difference (reference minus comparator)
#: favours the reference; ``kappa`` is a defended margin, so a positive one
#: does.  The sign convention is stated on every emitted row.
METRIC_BETTER: Mapping[str, str] = {
    "relative_gap": "lower",
    "kappa": "higher",
}


def effect_orientation(metric: str) -> str:
    better = METRIC_BETTER.get(metric, "higher")
    if better == "lower":
        return (
            "paired difference is reference minus comparator on "
            f"{metric}; {metric} is better when lower, so negative values and "
            "negative rank-biserial favour the reference method"
        )
    return (
        "paired difference is reference minus comparator on "
        f"{metric}; {metric} is better when higher, so positive values and "
        "positive rank-biserial favour the reference method"
    )


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment, monotone and capped at one."""

    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = p_values[index] * (count - rank)
        running = max(running, min(1.0, value))
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: Sequence[float], *, tolerance: float = 1e-9) -> float:
    """Matched-pairs rank-biserial correlation on the non-zero differences."""

    values = np.asarray([value for value in differences if abs(value) > tolerance], dtype=float)
    if values.size == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    total = positive + negative
    return (positive - negative) / total if total > 0 else 0.0


def paired_wilcoxon(
    differences: Sequence[float],
    *,
    alternative: str = "two-sided",
    tolerance: float = 1e-9,
) -> tuple[float, float, int]:
    """Return ``(statistic, p_value, n_nonzero)`` for a one-sample signed-rank test."""

    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > tolerance]
    if nonzero.size < MIN_PAIRS:
        return float("nan"), 1.0, int(nonzero.size)
    result = wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
    return float(result.statistic), float(result.pvalue), int(nonzero.size)


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Percentile CI from resampling whole clusters, not individual records.

    A drawn cluster contributes *all* of its repeated observations, so a graph
    that contributed three budgets enters every resample with all three.

    The mean is computed from per-cluster sums with a single vectorised draw,
    which is what makes the 10,000-resample setting affordable; other statistics
    fall back to building each resample explicitly.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            grouped[str(cluster)].append(number)
    keys = sorted(grouped)
    if len(keys) < 2 or resamples < 1:
        return float("nan"), float("nan")

    pools = [np.asarray(grouped[key], dtype=float) for key in keys]
    counts = np.array([pool.size for pool in pools], dtype=float)
    sums = np.array([pool.sum() for pool in pools], dtype=float)
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(pools), size=(resamples, len(pools)))

    if statistic is None:
        # Mean of the pooled sample: sum of drawn cluster sums over total count.
        totals = counts[picks].sum(axis=1)
        draws = np.divide(
            sums[picks].sum(axis=1), totals, out=np.full(resamples, np.nan), where=totals > 0
        )
    else:
        flat = np.concatenate(pools)
        offsets = np.concatenate(([0], np.cumsum(counts).astype(int)))
        slices = [
            (offsets[index], offsets[index + 1]) for index in range(len(pools))
        ]
        draws = np.empty(resamples, dtype=float)
        for position in range(resamples):
            pieces = [
                flat[slices[index][0] : slices[index][1]] for index in picks[position]
            ]
            draws[position] = statistic(np.concatenate(pieces))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    lower = float(np.percentile(draws, 100.0 * alpha / 2.0))
    upper = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


# ---------------------------------------------------------------------------
# panel assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """A block x method view of one metric.

    ``values`` holds the repeated-measure average that the Friedman and
    Wilcoxon tests run on, so each graph contributes exactly one observation.
    ``samples`` keeps the underlying repeated observations per block so that the
    cluster bootstrap can draw a whole graph and carry every one of its budget
    measurements with it, exactly as the revision protocol requires.
    """

    name: str
    unit: str
    metric: str
    blocks: tuple[str, ...]
    methods: tuple[str, ...]
    values: Mapping[tuple[str, str], float]
    cluster_labels: Mapping[str, str]
    samples: Mapping[tuple[str, str], Mapping[str, float]] = field(default_factory=dict)

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """Complete-case block x method matrix (rows with any NaN dropped)."""

        rows: list[str] = []
        data: list[list[float]] = []
        for block in self.blocks:
            row = [self.values.get((block, method), float("nan")) for method in self.methods]
            if any(not math.isfinite(value) for value in row):
                continue
            rows.append(block)
            data.append(row)
        return rows, np.asarray(data, dtype=float)

    def paired_samples(self, reference: str, comparator: str) -> tuple[list[float], list[str]]:
        """Per-observation paired differences with their resampling cluster.

        Without stored samples this degrades to the block-level averages, which
        is the correct behaviour when the metric was already aggregated.
        """

        differences: list[float] = []
        clusters: list[str] = []
        for block in self.blocks:
            left = self.samples.get((block, reference))
            right = self.samples.get((block, comparator))
            cluster = self.cluster_labels.get(block, block)
            if left and right:
                for key in sorted(set(left) & set(right)):
                    differences.append(float(left[key]) - float(right[key]))
                    clusters.append(cluster)
                continue
            left_mean = self.values.get((block, reference))
            right_mean = self.values.get((block, comparator))
            if left_mean is None or right_mean is None:
                continue
            differences.append(float(left_mean) - float(right_mean))
            clusters.append(cluster)
        return differences, clusters


def build_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    metric: str,
    methods: Sequence[str],
    aggregate_over: Sequence[str] = ("budget_ratio",),
    unit: str = "graph",
) -> Panel:
    """Collapse repeated measures and return a block x method metric matrix.

    ``aggregate_over`` names the columns whose variation inside one block is a
    repeated measure and must therefore be averaged before testing.  The metric
    column is either ``relative_gap`` or ``kappa``.
    """

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    clusters: dict[str, str] = {}
    for row in rows:
        if str(row.get("method")) not in methods:
            continue
        value = row.get(metric)
        if value is None or value == "NA":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        block = str(row.get("graph_id") if unit == "graph" else row.get("base_id"))
        repeated = "|".join(f"{row.get(column)}" for column in aggregate_over)
        grouped[(block, str(row["method"]))][repeated].append(number)
        # The resampling unit is the graph/base ID itself, never its generator
        # seed: a seed recurs across topologies and scale tiers and would merge
        # genuinely independent graphs into one cluster.
        clusters[block] = block

    averaged: dict[tuple[str, str], float] = {}
    samples: dict[tuple[str, str], dict[str, float]] = {}
    for key, repeated in grouped.items():
        per_key = {name_: float(np.mean(values)) for name_, values in repeated.items()}
        samples[key] = per_key
        averaged[key] = float(np.mean(list(per_key.values())))
    blocks = tuple(sorted({block for block, _ in averaged}))
    return Panel(
        name=name,
        unit=unit,
        metric=metric,
        blocks=blocks,
        methods=tuple(methods),
        values=averaged,
        cluster_labels=clusters,
        samples=samples,
    )


def build_panels_by_budget(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    methods: Sequence[str],
    budgets: Sequence[float],
    unit: str = "graph",
) -> dict[str, Panel]:
    panels: dict[str, Panel] = {}
    for budget in budgets:
        subset = [
            row for row in rows if abs(float(row["budget_ratio"]) - float(budget)) < 1e-12
        ]
        panels[f"{float(budget):.2f}"] = build_panel(
            subset,
            name=f"budget_{float(budget):.2f}",
            metric=metric,
            methods=methods,
            unit=unit,
        )
    return panels


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def friedman_row(panel: Panel) -> dict[str, Any]:
    """Omnibus Friedman test over the panel's blocks."""

    blocks, matrix = panel.matrix()
    if len(blocks) < 2 or matrix.shape[1] < 3:
        return {
            "panel": panel.name,
            "test": "friedman",
            "metric": panel.metric,
            "unit": panel.unit,
            "n_blocks": len(blocks),
            "n_methods": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "statistic": "",
            "df": "",
            "p_value": "",
            "status": "insufficient_blocks_or_methods",
        }
    statistic, p_value = friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    return {
        "panel": panel.name,
        "test": "friedman",
        "metric": panel.metric,
        "unit": panel.unit,
        "n_blocks": len(blocks),
        "n_methods": int(matrix.shape[1]),
        "statistic": float(statistic),
        "df": int(matrix.shape[1] - 1),
        "p_value": float(p_value),
        "status": "ok",
    }


def pairwise_rows(
    panel: Panel,
    *,
    reference: str = DEFAULT_REFERENCE,
    alternative: str = "two-sided",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Wilcoxon signed-rank of ``reference - comparator`` over panel blocks."""

    comparators = [method for method in panel.methods if method != reference]
    rows: list[dict[str, Any]] = []
    for comparator in comparators:
        # The test statistic uses one observation per graph (the repeated-measure
        # average); the interval resamples whole graphs, carrying all of a
        # graph's budget observations in together.
        block_differences: list[float] = []
        for block in panel.blocks:
            left = panel.values.get((block, reference))
            right = panel.values.get((block, comparator))
            if left is None or right is None:
                continue
            block_differences.append(float(left) - float(right))
        bootstrap_values, bootstrap_clusters = panel.paired_samples(reference, comparator)
        wins = ties = losses = 0
        for delta in block_differences:
            if delta > 1e-9:
                wins += 1
            elif delta < -1e-9:
                losses += 1
            else:
                ties += 1
        statistic, p_value, nonzero = paired_wilcoxon(
            block_differences, alternative=alternative
        )
        mean_effect = (
            float(np.mean(block_differences)) if block_differences else float("nan")
        )
        median_effect = (
            float(np.median(block_differences)) if block_differences else float("nan")
        )
        ci_low, ci_high = cluster_bootstrap_ci(
            bootstrap_values,
            bootstrap_clusters,
            resamples=resamples,
            seed=seed,
        )
        ci_low_median, ci_high_median = cluster_bootstrap_ci(
            bootstrap_values,
            bootstrap_clusters,
            statistic=lambda array: float(np.median(array)),
            resamples=resamples,
            seed=seed,
        )
        rows.append(
            {
                "panel": panel.name,
                "test": "wilcoxon_signed_rank",
                "metric": panel.metric,
                "unit": panel.unit,
                "reference_method": reference,
                "comparator": comparator,
                "alternative": alternative,
                "n_pairs": len(block_differences),
                "n_nonzero_pairs": nonzero,
                "n_clusters": len(set(bootstrap_clusters)),
                "n_resampled_observations": len(bootstrap_values),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "p_holm": "",
                "rank_biserial": rank_biserial(block_differences),
                "mean_paired_effect": mean_effect,
                "median_paired_effect": median_effect,
                "mean_ci_low": ci_low,
                "mean_ci_high": ci_high,
                "median_ci_low": ci_low_median,
                "median_ci_high": ci_high_median,
                "ci_method": f"cluster_percentile_bootstrap_{resamples}",
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "metric_better_when": METRIC_BETTER.get(panel.metric, "higher"),
                "orientation": effect_orientation(panel.metric),
            }
        )
    return rows


def apply_holm(rows: Sequence[dict[str, Any]], *, family_key: str = "panel") -> list[dict[str, Any]]:
    """Holm-adjust ``p_raw`` within each family and write ``p_holm``."""

    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        families[str(row.get(family_key, ""))].append(index)
    output = [dict(row) for row in rows]
    for indices in families.values():
        adjusted = holm_adjust([float(output[index]["p_raw"]) for index in indices])
        for index, value in zip(indices, adjusted):
            output[index]["p_holm"] = value
    return output


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR, reported alongside Holm as a sensitivity."""

    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for rank in range(count - 1, -1, -1):
        index = order[rank]
        running = min(running, p_values[index] * count / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# descriptive audits
# ---------------------------------------------------------------------------


def descriptive_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    methods: Sequence[str],
    metric: str = "relative_gap",
    extra_metric: str = "kappa",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("method")) in methods:
            grouped[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method in methods:
        values = grouped.get(method, [])
        numbers = [
            float(row[metric])
            for row in values
            if row.get(metric) not in (None, "NA") and math.isfinite(float(row[metric]))
        ]
        kappas = [
            float(row[extra_metric])
            for row in values
            if row.get(extra_metric) not in (None, "NA")
            and math.isfinite(float(row[extra_metric]))
        ]
        gains = [
            float(row["kappa"]) - float(row["undefended"])
            for row in values
            if row.get("undefended") is not None
            and row.get("kappa") not in (None, "NA")
        ]
        output.append(
            {
                "panel": panel,
                "method": method,
                "n_records": len(values),
                "n_graphs": len({str(row["graph_id"]) for row in values}),
                "n_instances": len({str(row["instance_id"]) for row in values}),
                "n_budgets": len({float(row["budget_ratio"]) for row in values}),
                "relative_gap_mean": float(np.mean(numbers)) if numbers else "",
                "relative_gap_median": float(np.median(numbers)) if numbers else "",
                "relative_gap_max": float(np.max(numbers)) if numbers else "",
                "kappa_mean": float(np.mean(kappas)) if kappas else "",
                "kappa_median": float(np.median(kappas)) if kappas else "",
                "exact_attainment_rate": (
                    float(np.mean([value <= 1e-9 for value in numbers])) if numbers else ""
                ),
                "runtime_selection_median_s": _median_of(values, "runtime_selection_s"),
                "runtime_selection_max_s": _max_of(values, "runtime_selection_s"),
                "runtime_evaluation_median_s": _median_of(values, "runtime_evaluation_s"),
                "runtime_total_median_s": _median_of(values, "runtime_total_s"),
                "certified_optimal_rate": _rate_of(
                    values, lambda row: row.get("certificate_mode") in {"solver_closed", "cg_closed"}
                ),
                "feasible_rate": _rate_of(
                    values, lambda row: row.get("status") not in {"error", None}
                ),
            }
        )
    return output


def _numbers(values: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    output: list[float] = []
    for row in values:
        value = row.get(key)
        if value is None or value == "NA":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _median_of(values: Sequence[Mapping[str, Any]], key: str) -> float | str:
    numbers = _numbers(values, key)
    return float(np.median(numbers)) if numbers else ""


def _max_of(values: Sequence[Mapping[str, Any]], key: str) -> float | str:
    numbers = _numbers(values, key)
    return float(np.max(numbers)) if numbers else ""


def _rate_of(values: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float | str:
    if not values:
        return ""
    return float(np.mean([bool(predicate(row)) for row in values]))


def number(row: Mapping[str, Any], key: str) -> float | None:
    """One finite float from a run row, or ``None`` when the field is ``NA``."""

    value = row.get(key)
    if value is None or value == "NA":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def instance_optimum_bounds(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, float] | None:
    """Bracket the certified optimum of one ``(instance, budget)`` cell.

    ``kappa_opt`` is known only where an exact solver closed, but a run that hit
    its time limit still leaves two useful numbers: its incumbent is a
    **feasible** defence and therefore lower-bounds ``kappa*``, while its own
    upper bound (``best_bound`` for the MILP, ``cg_U`` for cut generation)
    upper-bounds it.  Together they bracket the optimum even when nothing closed.
    """

    lowers: list[float] = []
    uppers: list[float] = []
    for row in rows:
        if str(row.get("method")) not in CERTIFIED_METHODS:
            continue
        for column, bucket in (
            ("kappa_opt", "both"),
            ("incumbent", "lower"),
            ("cg_L", "lower"),
            ("best_bound", "upper"),
            ("cg_U", "upper"),
        ):
            value = number(row, column)
            if value is None:
                continue
            if bucket in ("both", "lower"):
                lowers.append(value)
            if bucket in ("both", "upper"):
                uppers.append(value)
    if not lowers or not uppers:
        return None
    return max(lowers), min(uppers)


def scaling_gap_bounds(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str = "MPCF-Greedy",
) -> list[dict[str, Any]]:
    """A method's relative gap over **every** instance, as an interval.

    Restricting the gap to the instances that happen to carry a certified
    optimum is a selected subset, and at ``N=1000`` a badly selected one: the
    covered instances there have the *largest* gaps, so the subset median
    overstates the whole-cell median by about 21 points.  Reporting the subset
    alone therefore misleads in the one cell the scaling claim rests on.

    For instance ``i`` with result ``G_i`` and ``L_i <= kappa*_i <= U_i``,

        1 - G_i / L_i  <=  g_i  <=  1 - G_i / U_i

    because ``g_i = 1 - G_i / kappa*_i`` increases in ``kappa*_i``.  The median is
    monotone, so the median of the lower bounds brackets the cell's median gap
    from below and the median of the upper bounds from above.  This is an
    optimisation bound, not a confidence interval.  Where the two coincide the
    median is pinned exactly even if individual instances are not.
    """

    cells: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        cells.setdefault((int(row["N"]), float(row["budget_ratio"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (size, budget), values in sorted(cells.items()):
        by_instance: dict[str, list[Mapping[str, Any]]] = {}
        for row in values:
            by_instance.setdefault(str(row["instance_id"]), []).append(row)

        lowers: list[float] = []
        uppers: list[float] = []
        known: list[float] = []
        considered = 0
        empty_selection = 0
        completed = 0
        for rows_i in by_instance.values():
            target = next(
                (r for r in rows_i if str(r.get("method")) == method), None
            )
            if target is None:
                continue
            value = number(target, "kappa")
            if value is None:
                continue
            considered += 1
            if (number(target, "selected_count") or 0.0) == 0.0:
                empty_selection += 1
            if str(target.get("status")) != "time_limit":
                completed += 1
            bounded = instance_optimum_bounds(rows_i)
            if bounded is None:
                continue
            low, high = bounded
            low = max(low, value)          # its own feasible value is a lower bound
            if low <= 0 or high <= 0:
                continue
            lowers.append(1.0 - value / low)
            uppers.append(1.0 - value / high)
            optimum = number(target, "kappa_opt")
            if optimum is not None and optimum > 0:
                known.append(1.0 - value / optimum)

        if not lowers:
            continue
        lower = float(np.median(lowers))
        upper = float(np.median(uppers))
        # Rounding noise from the arithmetic above must not read as a range.
        if abs(lower) < 1e-12:
            lower = 0.0
        if abs(upper) < 1e-12:
            upper = 0.0
        output.append(
            {
                "N": size,
                "budget_ratio": budget,
                "n_instances": len(by_instance),
                "n_with_known_optimum": len(
                    {str(r["instance_id"]) for r in values
                     if number(r, "kappa_opt") is not None}
                ),
                "method": method,
                "n_results": considered,
                "completed_count": completed,
                "empty_selection_count": empty_selection,
                "gap_median_lower_bound": lower,
                "gap_median_upper_bound": upper,
                "gap_median_interval_exact": int(abs(upper - lower) <= 1e-12),
                "gap_median_known_optimum_only": (
                    float(np.median(known)) if known else ""
                ),
            }
        )
    return output


def bootstrap_summary_rows(
    panel: Panel,
    *,
    metric_label: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Estimate and clustered CI per method, on the panel's analysis unit."""

    output: list[dict[str, Any]] = []
    for method in panel.methods:
        values: list[float] = []
        clusters: list[str] = []
        for block in panel.blocks:
            value = panel.values.get((block, method))
            if value is None:
                continue
            values.append(float(value))
            clusters.append(panel.cluster_labels.get(block, block))
        if not values:
            continue
        low, high = cluster_bootstrap_ci(
            values, clusters, resamples=resamples, seed=seed
        )
        output.append(
            {
                "metric": metric_label,
                "method": method,
                "estimate": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
                "ci_level": 0.95,
                "ci_method": f"cluster_percentile_bootstrap_{resamples}",
                "bootstrap_unit": f"{panel.unit}_id",
                "n_resamples": resamples,
                "n_clusters": len(set(clusters)),
                "n_observations": len(values),
                "bootstrap_seed": seed,
                "test_panel": panel.name,
            }
        )
    return output
