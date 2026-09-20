#!/usr/bin/env python3
"""Turn ``runs_long.csv`` into every statistics table the revision requires.

Reads only raw solver output.  Writes, under ``<output>/statistics``:

``statistics_graph_level.csv``
    Friedman omnibus per analysis panel with its df and n.
``pairwise_graph_level.csv``
    Pre-specified pairwise Wilcoxon signed-rank tests against ``MPCF-Exact``
    with raw and Holm-adjusted p, rank-biserial effect and a clustered CI.
``statistics_by_budget.csv``
    The same tests repeated inside each budget separately (n = 45 each).
``role_repeated_statistics.csv``
    Role-composition analysis clustered on ``base_id``.
``bootstrap_summary.csv``
    Estimate and clustered 95% CI per method and metric.
``zero_budget_audit.csv``
    Descriptive results for all cells and for the effective-budget subset.

and under ``<output>/methods`` / ``<output>/tables`` the variant registry and
the paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmcd_f.rev import robustness as RB  # noqa: E402
from rmcd_f.rev import stats as S  # noqa: E402
from rmcd_f.rev.panels import DEFAULT_SEEDS, MAIN_TIERS  # noqa: E402
from rmcd_f.rev.seedext import (  # noqa: E402
    SEED_EXTENSION_METHODS,
    SEED_EXTENSION_SEEDS,
)
from rmcd_f.rev.ids import config_hash  # noqa: E402
from rmcd_f.rev.registry import (  # noqa: E402
    REGISTRY_COLUMNS,
    comparison_variant_names,
    registry_rows,
    validate_against_paper_claims,
)
from rmcd_f.rev.schema import coerce_row, write_csv  # noqa: E402

PRIMARY_METRIC = "relative_gap"
SECONDARY_METRIC = "kappa"
REFERENCE = "MPCF-Exact"
ANALYSIS_PANEL = "graph_level_over_budgets"
BY_BUDGET_PREFIX = "graph_level_budget_"


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read back a CSV this script just wrote, for the paper-facing table."""

    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def _numeric(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "NA":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def graph_level_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    budgets: Sequence[float],
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Primary (budget-averaged) and secondary (per-budget) analyses."""

    omnibus: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []

    primary = S.build_panel(
        rows, name=ANALYSIS_PANEL, metric=PRIMARY_METRIC, methods=methods, unit="graph"
    )
    omnibus.append(S.friedman_row(primary))
    primary_pairs = S.pairwise_rows(
        primary, reference=REFERENCE, resamples=resamples, seed=seed
    )
    pairwise.extend(S.apply_holm(primary_pairs, family_key="panel"))
    bootstrap.extend(
        S.bootstrap_summary_rows(
            primary, metric_label=PRIMARY_METRIC, resamples=resamples, seed=seed
        )
    )

    panels = S.build_panels_by_budget(
        rows,
        metric=PRIMARY_METRIC,
        methods=methods,
        budgets=budgets,
        unit="graph",
    )
    for label, panel in sorted(panels.items()):
        panel = replace(panel, name=f"{BY_BUDGET_PREFIX}{label}")
        omnibus.append(S.friedman_row(panel))
        budget_pairs = S.pairwise_rows(
            panel, reference=REFERENCE, resamples=resamples, seed=seed
        )
        pairwise.extend(S.apply_holm(budget_pairs, family_key="panel"))
        bootstrap.extend(
            S.bootstrap_summary_rows(
                panel,
                metric_label=PRIMARY_METRIC,
                resamples=resamples,
                seed=seed,
            )
        )
        budget_bootstrap = S.bootstrap_summary_rows(
            panel, metric_label=SECONDARY_METRIC, resamples=resamples, seed=seed
        )
        bootstrap.extend(budget_bootstrap)

    kappa_panel = S.build_panel(
        rows, name=ANALYSIS_PANEL, metric=SECONDARY_METRIC, methods=methods, unit="graph"
    )
    bootstrap.extend(
        S.bootstrap_summary_rows(
            kappa_panel, metric_label=SECONDARY_METRIC, resamples=resamples, seed=seed
        )
    )
    return omnibus, pairwise, bootstrap


def role_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Base-ID clustered analysis of the role-composition panel.

    The unit is one ``base_id``; all six compositions and all three budgets of
    a base enter or leave the resample together, so the 810 records are never
    treated as 810 independent observations.
    """

    panel = S.build_panel(
        rows, name="role_base_id_over_compositions", metric=PRIMARY_METRIC,
        methods=methods, unit="base",
    )
    output = [S.friedman_row(panel)]
    output.extend(
        S.apply_holm(
            S.pairwise_rows(panel, reference=REFERENCE, resamples=resamples, seed=seed),
            family_key="panel",
        )
    )
    for row in output:
        row["clustering_unit"] = "base_id"
        row["repeated_measures"] = "composition x budget"
    return output


def zero_budget_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Descriptive split between all cells and effective-budget cells.

    The zero-effective set is restricted to the instances actually present in
    ``rows``: the cell ledger covers every experiment, and counting cells from
    panels that are not being audited would report a number that does not match
    the records on either side of the split.
    """

    instance_ids = {str(row["instance_id"]) for row in rows}
    zero_keys = {
        (str(cell["instance_id"]), round(float(cell["budget_ratio"]), 10))
        for cell in cells
        if int(cell.get("zero_effective_budget") or 0) == 1
        and str(cell["instance_id"]) in instance_ids
    }
    all_rows = [row for row in rows if str(row.get("method")) in methods]
    effective_rows = [
        row
        for row in all_rows
        if (str(row["instance_id"]), round(float(row["budget_ratio"]), 10)) not in zero_keys
    ]
    output = S.descriptive_rows(
        all_rows, panel="all_cells", methods=methods
    ) + S.descriptive_rows(
        effective_rows, panel="nonzero_effective_budget", methods=methods
    )
    for row in output:
        row["zero_effective_cells_excluded"] = (
            len(zero_keys) if row["panel"] == "nonzero_effective_budget" else 0
        )
        row["all_cells_total"] = len(all_rows)
        row["effective_cells_total"] = len(effective_rows)
    return output


def scaling_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``N x budget x method`` runtime and gap aggregate from raw runs.

    A run that hit the time limit records the wall clock at which the process was
    stopped, not the time the solver would have needed.  Those rows are real
    measurements of elapsed time, but they are **right-censored** as measurements
    of solver speed, so this table keeps them apart instead of averaging the two
    kinds together:

    * ``runtime_selection_median_s``        every run (censored values included)
    * ``runtime_selection_median_completed_s``   runs that finished on their own
    * ``runtime_selection_censored_count``  how many were stopped

    ``solved_count`` counts runs whose **own certificate closed**, which is a
    property of the method, not of its answers: ``MPCF-Greedy`` never closes one
    by construction, so its zero says nothing about solution quality.  For that,
    read ``at_known_optimum_count`` (results equal to a certified optimum found
    elsewhere) and ``empty_selection_count`` (runs that returned no protection at
    all because the first selection round did not finish).
    """

    grouped: dict[tuple[int, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["N"]), float(row["budget_ratio"]), str(row["method"]))].append(row)
    output: list[dict[str, Any]] = []
    for (size, budget, method), values in sorted(grouped.items()):
        selection = [
            value
            for value in (_numeric(row, "runtime_selection_s") for row in values)
            if value is not None
        ]
        total = [
            value
            for value in (_numeric(row, "runtime_total_s") for row in values)
            if value is not None
        ]
        gaps = [
            value
            for value in (_numeric(row, "relative_gap") for row in values)
            if value is not None
        ]
        rows_with_opt = [row for row in values if _numeric(row, "kappa_opt") is not None]
        solved = [
            row
            for row in values
            if str(row.get("certificate_mode")) in {"solver_closed", "cg_closed"}
        ]
        completed = [row for row in values if str(row.get("status")) != "time_limit"]
        censored = [row for row in values if str(row.get("status")) == "time_limit"]
        completed_selection = [
            value
            for value in (_numeric(row, "runtime_selection_s") for row in completed)
            if value is not None
        ]
        # A result that matches a certified optimum found by another method.  For
        # Greedy this is the only quality signal available, since it proves
        # nothing itself.
        at_known_optimum = [
            row
            for row in values
            if _numeric(row, "kappa") is not None
            and _numeric(row, "kappa_opt") is not None
            and abs(_numeric(row, "kappa") - _numeric(row, "kappa_opt")) <= 1e-9
        ]
        empty_selection = [
            row for row in values if (_numeric(row, "selected_count") or 0.0) == 0.0
        ]
        output.append(
            {
                "N": size,
                "budget_ratio": budget,
                "method": method,
                "n_instances": len(values),
                "time_limit_s": _numeric(values[0], "time_limit_s"),
                "solved_count": len(solved),
                "solved_rate": len(solved) / len(values) if values else "",
                "completed_count": len(completed),
                "completed_rate": len(completed) / len(values) if values else "",
                "time_limit_count": len(censored),
                "instances_with_optimum": len(rows_with_opt),
                "gap_coverage_rate": len(rows_with_opt) / len(values) if values else "",
                "at_known_optimum_count": len(at_known_optimum),
                "empty_selection_count": len(empty_selection),
                "runtime_selection_median_s": _stat(selection, np.median),
                "runtime_selection_median_completed_s": _stat(completed_selection, np.median),
                "runtime_selection_max_s": _stat(selection, np.max),
                "runtime_total_median_s": _stat(total, np.median),
                "runtime_total_max_s": _stat(total, np.max),
                # Defined only where a certified optimum exists, so this is the
                # median over the covered subset -- see scaling_gap_bounds_table
                # for the interval over every instance.
                "relative_gap_median": _stat(gaps, np.median),
                "relative_gap_max": _stat(gaps, np.max),
                "relative_gap_mean": _stat(gaps, np.mean),
                "bb_nodes_median": _stat(
                    [
                        value
                        for value in (_numeric(row, "bb_nodes") for row in values)
                        if value is not None
                    ],
                    np.median,
                ),
                "final_gap_max": _stat(
                    [
                        value
                        for value in (_numeric(row, "final_gap") for row in values)
                        if value is not None
                    ],
                    np.max,
                ),
                "status_counts": json.dumps(
                    _counts(str(row.get("status")) for row in values), sort_keys=True
                ),
            }
        )
    return output


def _stat(values: Sequence[float], function) -> float | str:
    return float(function(values)) if len(values) else ""


def _counts(values) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for value in values:
        output[str(value)] += 1
    return dict(output)


def main_table(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> list[dict[str, Any]]:
    """Main paper table: method x budget descriptive results."""

    return S.descriptive_rows(
        [row for row in rows if str(row.get("method")) in methods],
        panel="main",
        methods=methods,
    )


def heterogeneity_table(
    rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Joint ``b_v`` x ``m_v`` results by correlation profile and budget."""

    profile_by_instance = {
        str(row["instance_id"]): row for row in manifest
    }
    grouped: dict[tuple[str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        spec = profile_by_instance.get(str(row["instance_id"]), {})
        profile = str(spec.get("correlation_profile") or "unknown")
        grouped[(profile, float(row["budget_ratio"]), str(row["method"]))].append(row)
    output: list[dict[str, Any]] = []
    for (profile, budget, method), values in sorted(grouped.items()):
        spec = profile_by_instance.get(str(values[0]["instance_id"]), {})
        gaps = [
            value
            for value in (_numeric(row, "relative_gap") for row in values)
            if value is not None
        ]
        output.append(
            {
                "correlation_profile": profile,
                "b_profile": spec.get("b_profile", ""),
                "budget_ratio": budget,
                "method": method,
                "n_instances": len(values),
                "spearman_b_effect_mean": _numeric(spec, "spearman_b_effect"),
                "sum_b_mean": _numeric(spec, "total_protection_cost"),
                "kappa_mean": _stat(
                    [
                        value
                        for value in (_numeric(row, "kappa") for row in values)
                        if value is not None
                    ],
                    np.mean,
                ),
                "relative_gap_mean": _stat(gaps, np.mean),
                "relative_gap_median": _stat(gaps, np.median),
                "relative_gap_max": _stat(gaps, np.max),
                "runtime_selection_median_s": _stat(
                    [
                        value
                        for value in (_numeric(row, "runtime_selection_s") for row in values)
                        if value is not None
                    ],
                    np.median,
                ),
            }
        )
    return output


def greedy_gap_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Greedy gap to the certified optimum, per instance and budget."""

    by_cell: dict[tuple[str, float], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_cell[(str(row["instance_id"]), round(float(row["budget_ratio"]), 10))][
            str(row["method"])
        ] = row
    output: list[dict[str, Any]] = []
    for (instance, budget), methods in sorted(by_cell.items()):
        exact = methods.get("MPCF-Exact")
        greedy = methods.get("MPCF-Greedy")
        if exact is None or greedy is None:
            continue
        optimum = _numeric(exact, "kappa_opt")
        greedy_kappa = _numeric(greedy, "kappa")
        output.append(
            {
                "instance_id": instance,
                "graph_id": exact["graph_id"],
                "budget_ratio": budget,
                "kappa_exact": _numeric(exact, "kappa"),
                "kappa_greedy": greedy_kappa,
                "kappa_opt": optimum,
                "greedy_gap": (
                    (optimum - greedy_kappa) / optimum
                    if optimum and greedy_kappa is not None and optimum > 0
                    else ""
                ),
                "exact_certificate_mode": exact.get("certificate_mode"),
                "greedy_status": greedy.get("status"),
                "greedy_runtime_s": _numeric(greedy, "runtime_selection_s"),
                "exact_runtime_s": _numeric(exact, "runtime_selection_s"),
            }
        )
    return output


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=S.DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    parser.add_argument("--reference", default=REFERENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output).resolve()
    runs_path = root / "results" / "runs_long.csv"
    if not runs_path.exists():
        raise SystemExit(f"Missing {runs_path}")
    rows = _read_rows(runs_path)
    cells_path = root / "results" / "instance_budget_cells.csv"
    cells = _read_rows(cells_path) if cells_path.exists() else []
    manifest_path = root / "metadata" / "graph_manifest.csv"
    manifest = _read_rows(manifest_path) if manifest_path.exists() else []

    statistics_dir = root / "statistics"
    tables_dir = root / "tables"
    methods_dir = root / "methods"
    for directory in (statistics_dir, tables_dir, methods_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparison = comparison_variant_names()
    main_rows = [row for row in rows if str(row.get("experiment")) == "main"]
    role_rows = [row for row in rows if str(row.get("experiment")) == "role"]
    scaling_rows = [row for row in rows if str(row.get("experiment")) == "scaling"]
    heterogeneity_rows = [
        row for row in rows if str(row.get("experiment")) == "heterogeneity"
    ]

    main_methods = _complete_methods(main_rows, comparison + ("MPCF-Exact", "MPCF-CG"))
    budgets = sorted({float(row["budget_ratio"]) for row in main_rows}) or [0.02, 0.05, 0.10]

    omnibus: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    if main_rows:
        omnibus, pairwise, bootstrap = graph_level_statistics(
            main_rows,
            methods=main_methods,
            budgets=budgets,
            resamples=args.resamples,
            seed=args.seed,
        )
        write_csv(
            statistics_dir / "statistics_graph_level.csv",
            [row for row in omnibus if row.get("panel") == ANALYSIS_PANEL],
        )
        write_csv(
            statistics_dir / "pairwise_graph_level.csv",
            [row for row in pairwise if row.get("panel") == ANALYSIS_PANEL],
        )
        write_csv(
            statistics_dir / "statistics_by_budget.csv",
            [
                row
                for row in omnibus + pairwise
                if str(row.get("panel", "")).startswith(BY_BUDGET_PREFIX)
            ],
        )
        write_csv(
            statistics_dir / "bootstrap_summary.csv", bootstrap
        )
        write_csv(
            statistics_dir / "pairwise_all_panels.csv", pairwise
        )

    if role_rows:
        role_methods = _complete_methods(role_rows, comparison + ("MPCF-Exact", "MPCF-CG"))
        write_csv(
            statistics_dir / "role_repeated_statistics.csv",
            role_statistics(
                role_rows,
                methods=role_methods,
                resamples=args.resamples,
                seed=args.seed,
            ),
        )
        write_csv(tables_dir / "table_role.csv", main_table(role_rows, role_methods))

    if main_rows:
        write_csv(
            statistics_dir / "zero_budget_audit.csv",
            zero_budget_audit(main_rows, methods=main_methods, cells=cells),
        )

    seedext_rows = [row for row in rows if str(row.get("experiment")) == "seedext"]
    if seedext_rows:
        pooled = [row for row in main_rows if str(row.get("method")) in SEED_EXTENSION_METHODS]
        pooled = pooled + seedext_rows
        methods = [m for m in SEED_EXTENSION_METHODS
                   if any(str(r.get("method")) == m for r in pooled)]
        write_csv(
            statistics_dir / "seed_effect_stability.csv",
            RB.effect_stability_rows(
                pooled,
                methods=methods,
                original_seeds=DEFAULT_SEEDS,
                extension_seeds=SEED_EXTENSION_SEEDS,
            ),
        )
        write_csv(
            statistics_dir / "seed_paired_intervals.csv",
            RB.paired_interval_rows(
                pooled,
                methods=methods,
                original_seeds=DEFAULT_SEEDS,
                extension_seeds=SEED_EXTENSION_SEEDS,
                resamples=args.resamples,
                seed=args.seed,
            ),
        )
        all_seeds = tuple(DEFAULT_SEEDS) + tuple(SEED_EXTENSION_SEEDS)
        write_csv(
            statistics_dir / "seed_convergence.csv",
            RB.convergence_rows(pooled, methods=methods, seeds=all_seeds),
        )
        write_csv(
            statistics_dir / "seed_stratified_bootstrap.csv",
            RB.stratified_bootstrap_rows(
                pooled,
                methods=methods,
                resamples=args.resamples,
                seed=args.seed,
            ),
        )
        stability = read_csv_rows(statistics_dir / "seed_effect_stability.csv")
        write_csv(tables_dir / "table_seed_robustness.csv", stability)
        (statistics_dir / "seed_robustness_manifest.json").write_text(
            json.dumps(
                RB.manifest_provenance(
                    original_seeds=DEFAULT_SEEDS,
                    extension_seeds=SEED_EXTENSION_SEEDS,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if scaling_rows:
        write_csv(tables_dir / "table_scaling.csv", scaling_table(scaling_rows))
        write_csv(
            tables_dir / "table_scaling_gap_bounds.csv",
            S.scaling_gap_bounds(scaling_rows),
        )

    if heterogeneity_rows:
        write_csv(
            tables_dir / "table_heterogeneity.csv",
            heterogeneity_table(heterogeneity_rows, manifest),
        )
        write_csv(
            tables_dir / "table_heterogeneity_greedy_gap.csv",
            greedy_gap_rows(heterogeneity_rows),
        )

    if main_rows:
        write_csv(tables_dir / "table_main.csv", main_table(main_rows, main_methods))
        write_csv(
            tables_dir / "table_statistics.csv",
            [row for row in pairwise if row.get("panel") == ANALYSIS_PANEL],
        )
        write_csv(tables_dir / "table_greedy_gap.csv", greedy_gap_rows(main_rows))

    coverage: dict[str, int] = defaultdict(int)
    for row in rows:
        coverage[str(row.get("method"))] += 1
    registry = registry_rows(coverage, include_conversions=True)
    write_csv(methods_dir / "method_variant_registry.csv", registry, columns=REGISTRY_COLUMNS)
    write_variant_coverage(methods_dir, rows, registry)
    (root / "methods" / "variant_reconciliation.json").write_text(
        json.dumps(validate_against_paper_claims(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_experiment_splits(root, rows)
    write_solver_logs(root, rows)
    (root / "statistics" / "statistics_manifest.json").write_text(
        json.dumps(
            {
                "primary_metric": (
                    "relative gap g = (kappa_opt - kappa)/kappa_opt, averaged over the "
                    "three budgets inside each frozen graph"
                ),
                "primary_unit": "graph_id (45 independent graphs)",
                "repeated_measure": "budget_ratio (3 levels, averaged before testing)",
                "omnibus_test": "Friedman, one block per graph",
                "pairwise_test": "Wilcoxon signed-rank vs MPCF-Exact on graph-level pairs",
                "multiple_comparison": "Holm within each analysis panel",
                "effect_size": "matched-pairs rank-biserial",
                "interval": f"clustered percentile bootstrap, {args.resamples} resamples",
                "bootstrap_seed": args.seed,
                "role_clustering": "base_id (composition x budget resampled together)",
                "reference_method": args.reference,
                "n_main_graphs": len({str(row["graph_id"]) for row in main_rows}),
                "n_main_records": len(main_rows),
                "n_role_bases": len({str(row["base_id"]) for row in role_rows}),
                "n_role_records": len(role_rows),
                "seed_extension_seeds": list(SEED_EXTENSION_SEEDS),
                "seed_extension_graphs": len({str(r["graph_id"]) for r in seedext_rows}),
                "seed_extension_methods": list(SEED_EXTENSION_METHODS),
                "primary_family_size": len(
                    [row for row in pairwise if row.get("panel") == ANALYSIS_PANEL]
                ),
                "config_hash": config_hash({"methods": main_methods, "budgets": budgets}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[rev-statistics] graphs={len({str(r['graph_id']) for r in main_rows})} "
        f"methods={len(main_methods)} primary_tests="
        f"{len([r for r in pairwise if r.get('panel') == ANALYSIS_PANEL])}"
    )
    return 0


def _complete_methods(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> list[str]:
    """Methods whose rows are complete on the panel's blocks.

    A method with missing cells would silently shrink the Friedman matrix, so
    only fully covered methods enter the omnibus and the pairwise family.
    """

    blocks = {str(row["graph_id"]) for row in rows}
    if not blocks:
        return []
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        coverage[str(row["method"])].add(str(row["graph_id"]))
    complete = [
        method
        for method in candidates
        if coverage.get(method) and coverage[method] >= blocks
    ]
    return complete


def write_variant_coverage(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    registry: Sequence[Mapping[str, Any]],
) -> None:
    """Per-variant coverage so a missing baseline can never vanish silently."""

    observed: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"records": 0, "instances": set(), "graphs": set(), "ok": 0, "failed": 0}
    )
    for row in rows:
        entry = observed[str(row.get("method"))]
        entry["records"] += 1
        entry["instances"].add(str(row.get("instance_id")))
        entry["graphs"].add(str(row.get("graph_id")))
        if str(row.get("status")) in {"error"}:
            entry["failed"] += 1
        else:
            entry["ok"] += 1
    output: list[dict[str, Any]] = []
    for item in registry:
        name = str(item["variant_name"])
        entry = observed.get(name)
        output.append(
            {
                "variant_id": item["variant_id"],
                "variant_name": name,
                "base_method": item["base_method"],
                "method_class": item["method_class"],
                "predeclared_panel": item["predeclared_panel"],
                "in_comparison_set": item["in_comparison_set"],
                "dependency_environment": item["dependency_environment"],
                "records": entry["records"] if entry else 0,
                "instances_covered": len(entry["instances"]) if entry else 0,
                "graphs_covered": len(entry["graphs"]) if entry else 0,
                "successful_runs": entry["ok"] if entry else 0,
                "failed_runs": entry["failed"] if entry else 0,
                "evaluated": int(bool(entry and entry["ok"])),
                "exclusion_reason": item.get("exclusion_reason", ""),
            }
        )
    write_csv(directory / "variant_coverage.csv", output)


def write_experiment_splits(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """One results file per experiment, as the archive layout specifies."""

    names = {
        "main": "main_results.csv",
        "role": "role_results.csv",
        "scaling": "scaling_runs.csv",
        "heterogeneity": "heterogeneity_runs.csv",
        "capacity": "capacity_runs.csv",
        "public": "public_runs.csv",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("experiment"))].append(row)
    for experiment, subset in grouped.items():
        write_csv(root / "results" / names.get(experiment, f"{experiment}_runs.csv"), subset)


#: Solver-specific columns kept in each solver log, mirroring the fields the
#: revision protocol requires a reader to be able to audit.
SOLVER_LOG_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "exact": (
        "instance_id", "graph_id", "base_id", "topology", "N", "seed",
        "budget_ratio", "budget_abs", "method", "variant", "status",
        "incumbent", "best_bound", "final_gap", "bb_nodes", "kappa",
        "replay_kappa", "certificate_mode", "selected_count", "actual_cost",
        "runtime_selection_s", "runtime_evaluation_s", "runtime_total_s",
        "time_limit_s", "config_hash", "code_commit",
    ),
    "cg": (
        "instance_id", "graph_id", "base_id", "topology", "N", "seed",
        "budget_ratio", "budget_abs", "method", "variant", "status",
        "cg_L", "cg_U", "cg_iterations", "cg_cuts", "kappa", "replay_kappa",
        "certificate_mode", "final_gap", "bb_nodes", "actual_cost",
        "runtime_selection_s", "runtime_evaluation_s", "runtime_total_s",
        "time_limit_s", "config_hash", "code_commit",
    ),
    "greedy": (
        "instance_id", "graph_id", "base_id", "topology", "N", "seed",
        "budget_ratio", "budget_abs", "method", "variant", "status",
        "incumbent", "kappa", "kappa_opt", "relative_gap", "actual_cost",
        "selected_count", "cg_iterations", "certificate_mode",
        "runtime_selection_s", "runtime_evaluation_s", "runtime_total_s",
        "time_limit_s", "config_hash", "code_commit",
    ),
}


def write_solver_logs(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Split the unified rows into per-solver logs under ``logs/``."""

    mapping = {
        "exact": ("MPCF-Exact",),
        "cg": ("MPCF-CG",),
        "greedy": ("MPCF-Greedy",),
    }
    for folder, methods in mapping.items():
        subset = [row for row in rows if str(row.get("method")) in methods]
        if not subset:
            continue
        write_csv(
            root / "logs" / folder / "solver_log.csv",
            subset,
            columns=SOLVER_LOG_COLUMNS[folder],
        )
    others = [
        row
        for row in rows
        if str(row.get("method")) not in {"MPCF-Exact", "MPCF-CG", "MPCF-Greedy"}
    ]
    if others:
        write_csv(root / "logs" / "baselines" / "solver_log.csv", others)


if __name__ == "__main__":
    raise SystemExit(main())
