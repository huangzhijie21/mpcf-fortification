#!/usr/bin/env python3
"""Rebuild every paper figure from exported CSV tables.

The plotting layer is deliberately decoupled: it opens ``runs_long.csv`` and the
CSV files under ``statistics/`` and ``tables/`` and never imports a solver.  A
figure can therefore be regenerated from an archived result directory alone.

Journal conventions applied here
-------------------------------
* Figures 3-9 are drawn at 7.2 in double-column width with 8-9 pt type.
* Legends are simplified: long method names are shortened and the legend is
  placed outside the data.
* Panels that mixed scales are split.
* Figure 1 keeps a wide left margin for the stage labels.
* Every figure is written as vector PDF and as 600 dpi PNG.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmcd_f.rev import stats as S  # noqa: E402
from rmcd_f.rev.schema import coerce_row  # noqa: E402

# ---------------------------------------------------------------------------
# house style
# ---------------------------------------------------------------------------

TEXT_WIDTH = 7.2          # double-column width in inches
SINGLE_WIDTH = 3.5
PNG_DPI = 600


def _supports_boxplot_orientation() -> bool:
    """True when ``Axes.boxplot`` takes ``orientation`` instead of ``vert``.

    Matplotlib 3.11 deprecated the ``vert`` flag and 3.13 removes it, so the
    released figures must work on both sides of that change.
    """

    try:
        version = tuple(int(part) for part in matplotlib.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover - non-numeric version string
        return True
    return version >= (3, 11)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

METHOD_COLORS = {
    "MPCF-Exact": "#1b4f8a",
    "MPCF-CG": "#2e8b57",
    "MPCF-Greedy": "#c1560b",
    "NoProtection": "#888888",
    "RandomProtect": "#b0b0b0",
}

SHORT_LABELS = {
    "MPCF-Exact": "MPCF-Exact",
    "MPCF-CG": "MPCF-CG",
    "MPCF-Greedy": "MPCF-Greedy",
    "NoProtection": "No protection",
    "RandomProtect": "Random",
    "DegreeProtect": "Degree",
    "BetweennessProtect": "Betweenness",
    "KCoreProtect": "k-core",
    "PageRankProtect": "PageRank",
    "PathFrequencyProtect": "Path frequency",
    "InitialPathCutProtect": "Init. PathCut",
    "BPDReference-Protect": "BPD reference",
}

PROFILE_LABELS = {
    "homogeneous": "homogeneous",
    "independent": "independent",
    "positive_corr": "positive",
    "negative_corr": "negative",
}


def short(name: str) -> str:
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    if name.endswith("-Protect"):
        return name[: -len("-Protect")].replace("Official", "")
    return name


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def _number(row: Mapping[str, Any], key: str) -> float | None:
    """Finite float from ``row[key]``, or ``None`` when absent/non-numeric."""

    value = row.get(key)
    if value is None or value == "NA":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class FigureWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []

    def save(self, figure: plt.Figure, stem: str) -> None:
        for suffix in (".pdf", ".png"):
            figure.savefig(
                self.directory / f"{stem}{suffix}",
                dpi=PNG_DPI if suffix == ".png" else None,
            )
        plt.close(figure)
        self.written.append(stem)


# ---------------------------------------------------------------------------
# figure 1: study design
# ---------------------------------------------------------------------------


def figure_study_design(writer: FigureWriter, manifest: Sequence[Mapping[str, Any]]) -> None:
    """Panel inventory and clustering structure, with a wide left margin."""

    experiments = [str(row["experiment"]) for row in manifest]
    counts: dict[str, int] = defaultdict(int)
    budgets: dict[str, set[float]] = defaultdict(set)
    for row in manifest:
        counts[str(row["experiment"])] += 1
        for ratio in str(row.get("budget_ratios") or "[]").strip("[]").split(","):
            if ratio.strip():
                budgets[str(row["experiment"])].add(float(ratio.strip()))
    order = [name for name in ("main", "role", "scaling", "heterogeneity") if name in counts]
    if not order:
        return

    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.1), gridspec_kw={"wspace": 0.45})
    axis = axes[0]
    positions = np.arange(len(order))
    axis.barh(positions, [counts[name] for name in order], color="#1b4f8a", height=0.55)
    axis.set_yticks(positions)
    axis.set_yticklabels(order)
    axis.set_xlabel("frozen instances")
    axis.set_title("a  Panel inventory", loc="left")
    for position, name in zip(positions, order):
        axis.text(counts[name] + max(counts.values()) * 0.02, position,
                  f"{counts[name]}", va="center", fontsize=8)
    axis.invert_yaxis()
    axis.margins(x=0.18)

    axis = axes[1]
    tiers = defaultdict(int)
    for row in manifest:
        if str(row["experiment"]) == "main":
            tiers[int(row["N"])] += 1
    if tiers:
        sizes = sorted(tiers)
        axis.bar([str(size) for size in sizes], [tiers[size] for size in sizes],
                 color="#2e8b57", width=0.55)
        axis.set_xlabel("nodes per graph  $N$")
        axis.set_ylabel("independent graphs")
        axis.set_title("b  Main panel: 45 independent graphs", loc="left")
        axis.text(
            0.98, 0.95,
            "repeated measure: 3 budgets per graph",
            transform=axis.transAxes, ha="right", va="top", fontsize=7.5,
        )
    else:
        axis.axis("off")
    figure.subplots_adjust(left=0.14, right=0.98)
    writer.save(figure, "fig1")


# ---------------------------------------------------------------------------
# figure 3: main relative-gap distributions
# ---------------------------------------------------------------------------


def figure_main_gap(writer: FigureWriter, runs: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row
        for row in runs
        if str(row.get("experiment")) == "main" and _number(row, "relative_gap") is not None
    ]
    if not rows:
        return
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(float(row["relative_gap"]))
    reference = "MPCF-Exact"
    others = sorted(
        (name for name in by_method if name != reference),
        key=lambda name: np.median(by_method[name]),
    )
    order = [reference] + others
    if len(order) < 2:
        return
    values = [by_method[name] for name in order]

    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 3.4))
    # ``orientation`` replaced the ``vert`` flag in Matplotlib 3.11 and ``vert``
    # is removed in 3.13, so select whichever this Matplotlib understands.
    _boxplot_kwargs = (
        {"orientation": "horizontal"}
        if _supports_boxplot_orientation()
        else {"vert": False}
    )
    box = axis.boxplot(
        values,
        widths=0.6,
        showfliers=True,
        flierprops={"markersize": 2, "markerfacecolor": "#999999", "markeredgecolor": "none"},
        medianprops={"color": "#c1560b", "linewidth": 1.1},
        boxprops={"linewidth": 0.7},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
        patch_artist=True,
        **_boxplot_kwargs,
    )
    for index, patch in enumerate(box["boxes"]):
        name = order[index]
        patch.set_facecolor(METHOD_COLORS.get(name, "#8fb3d9"))
        patch.set_alpha(1.0 if name == reference else 0.75)
    axis.set_yticks(range(1, len(order) + 1))
    axis.set_yticklabels([short(name) for name in order])
    axis.set_xlabel("relative gap to the certified optimum  $g=(\\kappa^{\\star}-\\kappa)/\\kappa^{\\star}$")
    axis.axvline(0.0, color="#444444", linewidth=0.7, linestyle=":")
    axis.set_title(
        "Main panel: 45 graphs, budgets averaged inside each graph",
        loc="left",
    )
    axis.margins(y=0.02)
    writer.save(figure, "fig3")


# ---------------------------------------------------------------------------
# figure 4: per-budget view
# ---------------------------------------------------------------------------


def figure_gap_by_budget(writer: FigureWriter, runs: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row
        for row in runs
        if str(row.get("experiment")) == "main" and _number(row, "relative_gap") is not None
    ]
    if not rows:
        return
    budgets = sorted({float(row["budget_ratio"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    if len(budgets) < 2 or not methods:
        return
    # Keep the reference plus the five worst performers for a readable legend.
    ranking: list[tuple[float, str]] = []
    for method in methods:
        values = [
            float(row["relative_gap"])
            for row in rows
            if str(row["method"]) == method
        ]
        ranking.append((float(np.median(values)) if values else 0.0, method))
    ranking.sort(reverse=True)
    keep = ["MPCF-Exact"] + [name for _, name in ranking if name != "MPCF-Exact"][:5]
    keep = list(dict.fromkeys(keep))

    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 3.2))
    width = 0.8 / len(keep)
    for index, method in enumerate(keep):
        centres, medians, lows, highs = [], [], [], []
        for budget_index, budget in enumerate(budgets):
            values = [
                float(row["relative_gap"])
                for row in rows
                if str(row["method"]) == method
                and abs(float(row["budget_ratio"]) - budget) < 1e-12
            ]
            if not values:
                continue
            centres.append(budget_index + (index - len(keep) / 2 + 0.5) * width)
            medians.append(float(np.median(values)))
            lows.append(float(np.percentile(values, 25)))
            highs.append(float(np.percentile(values, 75)))
        axis.bar(
            centres,
            medians,
            width=width * 0.9,
            color=METHOD_COLORS.get(method, "#8fb3d9"),
            label=short(method),
            yerr=[np.subtract(medians, lows), np.subtract(highs, medians)],
            error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#555555"},
        )
    axis.set_xticks(range(len(budgets)))
    axis.set_xticklabels([f"{budget * 100:.0f}%" for budget in budgets])
    axis.set_xlabel("protection budget as a share of total fortification cost")
    axis.set_ylabel("median relative gap (IQR)")
    axis.set_title("Budget-resolved view (secondary analysis, $n=45$ per budget)", loc="left")
    axis.legend(ncol=2, loc="upper right", columnspacing=1.0, handlelength=1.2)
    writer.save(figure, "fig4")


# ---------------------------------------------------------------------------
# figures 5 and 6: scaling
# ---------------------------------------------------------------------------


def figure_scaling(writer: FigureWriter, runs: Sequence[Mapping[str, Any]]) -> None:
    """Runtime (fig5) and solution quality (fig6) as two journal-size figures."""

    rows = [row for row in runs if str(row.get("experiment")) == "scaling"]
    if not rows:
        return
    methods = [name for name in ("MPCF-Exact", "MPCF-CG", "MPCF-Greedy")
               if any(str(row["method"]) == name for row in rows)]
    sizes = sorted({int(row["N"]) for row in rows})
    if not methods or not sizes:
        return
    limits = {_number(row, "time_limit_s") for row in rows}
    limits = {value for value in limits if value is not None}
    unified_limit = limits.pop() if len(limits) == 1 else None

    # --- fig5: runtime ----------------------------------------------------
    #
    # A run stopped by the time limit records the wall clock at which it was
    # killed.  That is a real elapsed time but a right-censored measurement of
    # how long the method needs, so the two are drawn differently: the solid
    # curve is the median over runs that finished on their own, and a hollow
    # marker on the budget line marks a size where runs were stopped.
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 3.6))
    for method in methods:
        medians, censored = [], []
        for size in sizes:
            values = [
                row
                for row in rows
                if str(row["method"]) == method and int(row["N"]) == size
            ]
            finished = [
                _number(row, "runtime_selection_s")
                for row in values
                if str(row.get("status")) != "time_limit"
            ]
            finished = [value for value in finished if value is not None]
            stopped = [row for row in values if str(row.get("status")) == "time_limit"]
            medians.append(float(np.median(finished)) if finished else np.nan)
            censored.append(len(stopped))
        colour = METHOD_COLORS.get(method, "#666666")
        axis.plot(
            sizes, medians, marker="o", markersize=3.6, linewidth=1.2,
            color=colour, label=f"{short(method)} median (completed)",
        )
        if unified_limit is not None:
            hit = [
                size for size, count in zip(sizes, censored) if count
            ]
            if hit:
                axis.plot(
                    hit, [unified_limit] * len(hit), linestyle="none", marker="o",
                    markersize=4.2, markerfacecolor="none", markeredgewidth=1.0,
                    markeredgecolor=colour,
                )
                for size in hit:
                    last = [
                        value for value, s in zip(medians, sizes)
                        if s <= size and np.isfinite(value)
                    ]
                    if last:
                        axis.annotate(
                            "", xy=(size, unified_limit), xytext=(size, last[-1]),
                            arrowprops=dict(arrowstyle="->", color=colour,
                                            linewidth=0.7, alpha=0.65),
                        )
    if unified_limit is not None:
        axis.axhline(unified_limit, color="#b03030", linewidth=0.9, linestyle=":")
        axis.text(sizes[0], unified_limit, f" budget {unified_limit:.0f} s",
                  va="bottom", ha="left", fontsize=7.5, color="#b03030")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(size) for size in sizes])
    axis.set_xlabel("nodes per graph  $N$")
    axis.set_ylabel("selection runtime (s)")
    axis.set_title(
        "Scaling under one budget per method; hollow markers were stopped "
        "at the budget",
        loc="left",
    )
    axis.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.6)
    writer.save(figure, "fig5")

    # --- fig6: solution quality -------------------------------------------
    #
    # MPCF-Greedy never certifies its own optimality, so "solved" says nothing
    # about it.  Its gap is drawn over **all** instances as the optimisation
    # interval implied by the saved bounds, and the estimate restricted to the
    # instances that happen to carry a certified optimum is shown separately --
    # at N=1000 that subset is a badly selected one and overstates the gap.
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 3.6))
    for method in methods:
        medians, tops = [], []
        for size in sizes:
            values = [
                _number(row, "relative_gap")
                for row in rows
                if str(row["method"]) == method and int(row["N"]) == size
            ]
            values = [value for value in values if value is not None]
            medians.append(float(np.median(values)) if values else np.nan)
            tops.append(float(np.max(values)) if values else np.nan)
        colour = METHOD_COLORS.get(method, "#666666")
        axis.plot(sizes, medians, marker="o", markersize=3.5, linewidth=1.2,
                  color=colour, label=short(method))
        axis.fill_between(sizes, medians, tops, color=colour, alpha=0.12, linewidth=0)

    bounds = S.scaling_gap_bounds(rows)
    if bounds:
        by_size: dict[int, list[dict[str, Any]]] = {}
        for entry in bounds:
            by_size.setdefault(int(entry["N"]), []).append(entry)
        low, high, restricted = [], [], []
        for size in sizes:
            entries = by_size.get(size, [])
            low.append(float(np.median([e["gap_median_lower_bound"] for e in entries]))
                       if entries else np.nan)
            high.append(float(np.median([e["gap_median_upper_bound"] for e in entries]))
                        if entries else np.nan)
            known = [e["gap_median_known_optimum_only"] for e in entries
                     if e["gap_median_known_optimum_only"] != ""]
            restricted.append(float(np.median(known)) if known else np.nan)
        colour = METHOD_COLORS.get("MPCF-Greedy", "#666666")
        axis.fill_between(sizes, low, high, color=colour, alpha=0.28, linewidth=0,
                          label="Greedy gap: interval over all 9 instances")
        axis.plot(sizes, restricted, linestyle="none", marker="v", markersize=4.0,
                  markerfacecolor="none", markeredgewidth=1.0, markeredgecolor=colour,
                  label="Greedy gap: known-optimum subset only")
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(size) for size in sizes])
    axis.set_xlabel("nodes per graph  $N$")
    axis.set_ylabel("relative gap to the certified optimum")
    axis.set_title(
        "Scaling: solution quality; the Greedy band is an optimisation bound, "
        "not a confidence interval",
        loc="left",
    )
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="upper left", handlelength=1.6)
    writer.save(figure, "fig6")


# ---------------------------------------------------------------------------
# figure 7: heterogeneity
# ---------------------------------------------------------------------------


def figure_heterogeneity(
    writer: FigureWriter,
    runs: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> None:
    rows = [row for row in runs if str(row.get("experiment")) == "heterogeneity"]
    if not rows:
        return
    profile_of = {str(row["instance_id"]): str(row.get("correlation_profile") or "")
                  for row in manifest}
    spearman_of = {str(row["instance_id"]): _number(row, "spearman_b_effect")
                   for row in manifest}
    by_profile: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[profile_of.get(str(row["instance_id"]), "")].append(row)
    order = [name for name in ("homogeneous", "independent", "positive_corr", "negative_corr")
             if name in by_profile]
    if not order:
        return
    budgets = sorted({float(row["budget_ratio"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})

    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.2), gridspec_kw={"wspace": 0.34})
    axis = axes[0]
    width = 0.8 / max(1, len(methods))
    for index, method in enumerate(methods):
        centres, medians = [], []
        for position, profile in enumerate(order):
            values = [
                _number(row, "kappa")
                for row in by_profile[profile]
                if str(row["method"]) == method
            ]
            values = [value for value in values if value is not None]
            centres.append(position + (index - len(methods) / 2 + 0.5) * width)
            medians.append(float(np.median(values)) if values else np.nan)
        axis.bar(centres, medians, width=width * 0.9,
                 color=METHOD_COLORS.get(method, "#8fb3d9"), label=short(method))
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([PROFILE_LABELS.get(name, name) for name in order], rotation=12)
    axis.set_ylabel("defended margin  $\\kappa$")
    axis.set_xlabel("$b_v$--$m_v$ correlation profile")
    axis.set_title("a  Stability across cost/effectiveness coupling", loc="left")
    axis.legend(loc="upper left", handlelength=1.2)

    axis = axes[1]
    labels, values = [], []
    for profile in order:
        sample = next(
            (spearman_of[str(row["instance_id"])] for row in by_profile[profile]
             if spearman_of.get(str(row["instance_id"])) is not None),
            None,
        )
        labels.append(PROFILE_LABELS.get(profile, profile))
        values.append(sample if sample is not None else float("nan"))
    axis.bar(range(len(labels)), values, color="#1b4f8a", width=0.55)
    axis.axhline(0.0, color="#444444", linewidth=0.7)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=12)
    axis.set_ylabel("realised Spearman $\\rho(b_v, m_v)$")
    axis.set_xlabel("profile")
    axis.set_title("b  Measured, not asserted, correlation", loc="left")
    axis.set_ylim(-1.15, 1.15)
    writer.save(figure, "fig7")


# ---------------------------------------------------------------------------
# figure 8: greedy vs optimum
# ---------------------------------------------------------------------------


def figure_greedy_gap(
    writer: FigureWriter,
    table_rows: Sequence[Mapping[str, Any]],
) -> None:
    values = [
        _number(row, "greedy_gap")
        for row in table_rows
        if _number(row, "greedy_gap") is not None
    ]
    values = [value for value in values if value is not None]
    if not values:
        return
    figure, axis = plt.subplots(figsize=(SINGLE_WIDTH + 1.6, 3.0))
    axis.hist(values, bins=min(30, max(6, len(values) // 3)), color="#c1560b", alpha=0.85)
    axis.axvline(float(np.mean(values)), color="#1b4f8a", linewidth=1.0,
                 label=f"mean {np.mean(values):.3f}")
    axis.axvline(float(np.median(values)), color="#2e8b57", linewidth=1.0, linestyle="--",
                 label=f"median {np.median(values):.3f}")
    exact_hits = sum(1 for value in values if value <= 1e-9)
    axis.set_xlabel("MPCF-Greedy relative gap to the certified optimum")
    axis.set_ylabel("instance-budget cells")
    axis.set_title(
        f"Greedy attains the optimum on {exact_hits}/{len(values)} cells", loc="left"
    )
    axis.legend(loc="upper right", handlelength=1.4)
    writer.save(figure, "fig8")


# ---------------------------------------------------------------------------
# figure 9: role composition
# ---------------------------------------------------------------------------


def figure_role_composition(writer: FigureWriter, runs: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row
        for row in runs
        if str(row.get("experiment")) == "role" and _number(row, "relative_gap") is not None
    ]
    if not rows:
        return
    compositions = sorted({str(row["composition"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    if len(compositions) < 2 or not methods:
        return
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 3.3))
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        centres, medians, lows, highs = [], [], [], []
        for position, composition in enumerate(compositions):
            values = [
                float(row["relative_gap"])
                for row in rows
                if str(row["method"]) == method and str(row["composition"]) == composition
            ]
            if not values:
                continue
            centres.append(position + (index - len(methods) / 2 + 0.5) * width)
            medians.append(float(np.median(values)))
            lows.append(float(np.percentile(values, 25)))
            highs.append(float(np.percentile(values, 75)))
        axis.bar(
            centres, medians, width=width * 0.9,
            color=METHOD_COLORS.get(method, "#8fb3d9"), label=short(method),
            yerr=[np.subtract(medians, lows), np.subtract(highs, medians)],
            error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#555555"},
        )
    axis.set_xticks(range(len(compositions)))
    axis.set_xticklabels(compositions, rotation=0)
    axis.set_xlabel("role composition  S-C-L-E")
    axis.set_ylabel("median relative gap (IQR)")
    axis.set_title(
        "Role composition: 45 bases, cluster unit = base_id", loc="left"
    )
    axis.legend(
        ncol=min(len(methods), 4),
        loc="upper left",
        columnspacing=1.0,
        handlelength=1.2,
    )
    writer.save(figure, "fig9")


def figure_seed_convergence(writer: FigureWriter, output: Path) -> None:
    """Supplementary figure: estimate stability against the seed count.

    The point of the panel is robustness, so the figure plots the mean relative
    gap against the number of random realizations, with the 2.5-97.5% band taken
    across *all* subsets of that size rather than one accumulation order.
    """

    rows = _read(output / "statistics" / "seed_convergence.csv")
    if not rows:
        return
    by_method: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for row in rows:
        k = row.get("n_seeds")
        estimate = _number(row, "estimate_mean")
        low = _number(row, "band_low")
        high = _number(row, "band_high")
        if k is None or estimate is None:
            continue
        by_method[str(row["method"])].append(
            (int(k), estimate, low if low is not None else estimate,
             high if high is not None else estimate)
        )
    if not by_method:
        return
    keep = [m for m in ("MPCF-Exact", "MPCF-CG", "MPCF-Greedy",
                        "PathFrequencyProtect", "InitialPathCutProtect",
                        "BetweennessProtect", "BPDReference-Protect",
                        "OfficialGND-Protect") if m in by_method]
    keep = keep or sorted(by_method)

    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 3.6))
    for method in keep:
        series = sorted(by_method[method])
        ks = [item[0] for item in series]
        means = [item[1] for item in series]
        lows = [item[2] for item in series]
        highs = [item[3] for item in series]
        colour = METHOD_COLORS.get(method, "#7b7b7b")
        axis.plot(ks, means, marker="o", markersize=3, linewidth=1.1,
                  color=colour, label=short(method))
        axis.fill_between(ks, lows, highs, color=colour, alpha=0.10, linewidth=0)
    axis.axvline(5, color="#b03030", linewidth=0.9, linestyle=":")
    axis.text(5, axis.get_ylim()[1], " original 5 seeds", va="top", ha="left",
              fontsize=7.5, color="#b03030")
    axis.set_xlabel("number of random graph realizations $n_{\\mathrm{seed}}$")
    axis.set_ylabel("mean relative gap")
    axis.set_title(
        "Seed-count convergence: mean over all $\\binom{10}{k}$ seed subsets "
        "(band: 2.5-97.5%)",
        loc="left",
    )
    axis.set_xticks(sorted({item[0] for series in by_method.values() for item in series}))
    axis.legend(loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.5)
    writer.save(figure, "figS1")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    runs = _read(output / "results" / "runs_long.csv")
    if not runs:
        raise SystemExit(f"No run rows under {output}")
    manifest = _read(output / "metadata" / "graph_manifest.csv")
    greedy_table = _read(output / "tables" / "table_heterogeneity_greedy_gap.csv")
    if not greedy_table:
        greedy_table = _read(output / "tables" / "table_greedy_gap.csv")

    writer = FigureWriter(output / "figures")
    figure_study_design(writer, manifest)
    figure_main_gap(writer, runs)
    figure_gap_by_budget(writer, runs)
    figure_scaling(writer, runs)
    figure_heterogeneity(writer, runs, manifest)
    figure_greedy_gap(writer, greedy_table)
    figure_role_composition(writer, runs)
    figure_seed_convergence(writer, output)
    print(f"[rev-plots] wrote {len(writer.written)} figures: {', '.join(writer.written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
