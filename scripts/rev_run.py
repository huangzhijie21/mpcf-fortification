#!/usr/bin/env python3
"""Run the MPCF revision experiments and emit the unified ``runs_long.csv``.

The runner only produces raw result rows plus the run metadata.  Every
statistic, table and figure is produced later, from those CSVs, by
``scripts/rev_statistics.py`` and ``scripts/rev_plots.py`` -- no figure is ever
drawn from a live solver object.

Examples
--------
    python scripts/rev_run.py --experiment main        --output results --workers 24
    python scripts/rev_run.py --experiment scaling     --output results --workers 24 --time-limit 3600
    python scripts/rev_run.py --experiment heterogeneity --output results --workers 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmcd_f.rev import panels as panels_module  # noqa: E402
from rmcd_f.rev.env import observe_environment, write_environment  # noqa: E402
from rmcd_f.rev.harness import (  # noqa: E402
    RANKING_METHODS,
    certify_cells,
    run_instance,
)
from rmcd_f.rev.ids import code_commit as resolve_commit  # noqa: E402
from rmcd_f.rev.ids import config_hash  # noqa: E402
from rmcd_f.rev.panels import InstanceSpec  # noqa: E402
from rmcd_f.rev.seedext import (  # noqa: E402
    SEED_EXTENSION_METHODS,
    SEED_EXTENSION_SEEDS,
    build_seed_extension_panel,
)
from rmcd_f.rev.registry import (  # noqa: E402
    EXPECTED_COMPARISON_VARIANTS,
    MAIN_PANEL_EXCLUDED,
    comparison_variant_names,
    cost_conversion_variants,
)
from rmcd_f.rev.schema import RUN_COLUMNS, RunRecord, write_csv, write_runs_long  # noqa: E402

EXPERIMENTS = ("main", "scaling", "role", "heterogeneity", "seedext")

MPCF_METHODS: tuple[str, ...] = ("MPCF-Exact", "MPCF-CG", "MPCF-Greedy")

#: Default method panel per experiment.
DEFAULT_METHODS: Mapping[str, tuple[str, ...]] = {
    "main": tuple(comparison_variant_names()) + ("MPCF-Exact", "MPCF-CG"),
    "scaling": MPCF_METHODS,
    "role": ("MPCF-Exact", "MPCF-Greedy") + tuple(RANKING_METHODS),
    "heterogeneity": ("MPCF-Exact", "MPCF-Greedy"),
    "seedext": SEED_EXTENSION_METHODS,
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    print(f"[{_stamp()}] [rev-run] {message}", flush=True)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# official sequences
# ---------------------------------------------------------------------------


def load_official_sequences(
    roots: Sequence[Path],
) -> tuple[dict[tuple[str, str], tuple[tuple[Any, ...], float, str]], list[Path]]:
    """Load frozen upstream dismantling rankings, keyed by (fingerprint, method)."""

    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("official_sequences.csv")))
            files.extend(sorted(root.rglob("official_sequences_complete.csv")))
        else:
            raise SystemExit(f"Official sequence root does not exist: {root}")
    from rmcd_f.official_review import OFFICIAL_METHOD_SPECS

    store: dict[tuple[str, str], tuple[tuple[Any, ...], float, str]] = {}
    for path in dict.fromkeys(files):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "SUCCESS":
                    continue
                fingerprint = (row.get("graph_fingerprint") or "").strip()
                method = (row.get("method") or "").strip()
                if not fingerprint or method not in OFFICIAL_METHOD_SPECS:
                    continue
                try:
                    sequence = tuple(json.loads(row["sequence_json"]))
                except (KeyError, TypeError, ValueError):
                    raise SystemExit(f"Malformed sequence_json in {path}")
                try:
                    runtime = float(row.get("runtime_seconds") or 0.0)
                except ValueError:
                    runtime = 0.0
                key = (fingerprint, method)
                existing = store.get(key)
                if existing is not None and existing[0] != sequence:
                    raise SystemExit(f"Conflicting official rankings for {key}.")
                store[key] = (sequence, runtime, str(path))
    return store, sorted(dict.fromkeys(files))


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def _worker(payload: Mapping[str, Any]) -> tuple[str, int]:
    """Run one frozen instance and write its shard. Must stay importable."""

    from rmcd_f.rev.harness import certify_cells as _certify
    from rmcd_f.rev.harness import run_instance as _run

    spec = InstanceSpec(**payload["spec"])
    records, cells = _run(
        spec,
        payload["methods"],
        official_sequences=payload["official"],
        exact_time_limit=payload["time_limits"]["exact"],
        cg_time_limit=payload["time_limits"]["cg"],
        greedy_time_limit=payload["time_limits"]["greedy"],
        cg_max_iterations=payload["cg_max_iterations"],
        commit=payload["commit"],
        config_hash_value=payload["config_hash"],
    )
    cells = _certify(records, cells)
    shard = Path(payload["shard_dir"]) / f"{_safe(spec.instance_id)}.csv"
    write_csv(shard, [record.to_row() for record in records], columns=list(RUN_COLUMNS))
    cell_path = Path(payload["shard_dir"]) / f"{_safe(spec.instance_id)}.cells.json"
    # Atomic: shards of one panel run concurrently and each may re-read the
    # whole shard tree to build runs_long.csv, so a reader must never observe a
    # half-written ledger.
    temporary = cell_path.with_name(f"{cell_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                key: {
                    "kappa_opt": cell.kappa_opt,
                    "certifying_methods": list(cell.certifying_methods),
                    "certificate_conflict": cell.certificate_conflict,
                    "undefended": cell.undefended,
                    "budget_abs": cell.budget_abs,
                    "budget_abs_floor": cell.budget_abs_floor,
                    "zero_effective_budget": cell.zero_effective_budget,
                }
                for key, cell in cells.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, cell_path)
    return spec.instance_id, len(records)


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


# ---------------------------------------------------------------------------
# panel construction
# ---------------------------------------------------------------------------


def build_panel(
    experiment: str,
    output: Path,
    args: argparse.Namespace,
) -> list[InstanceSpec]:
    if experiment == "main":
        return panels_module.build_main_panel(
            output,
            seeds=args.seeds or panels_module.DEFAULT_SEEDS,
            topologies=args.topologies,
            budget_ratios=args.budget_fractions,
            capacity_levels=args.capacity_levels,
            cost_levels=args.cost_levels,
            fortification_multiplier=args.fortification_multiplier,
        )
    if experiment == "scaling":
        return panels_module.build_scaling_panel(
            output,
            sizes=args.scaling_sizes,
            seeds=args.scaling_seeds,
            topologies=args.topologies,
            budget_ratios=args.budget_fractions,
            capacity_levels=args.capacity_levels,
            cost_levels=args.cost_levels,
            fortification_multiplier=args.fortification_multiplier,
        )
    if experiment == "role":
        return panels_module.build_role_panel(
            output,
            seeds=args.seeds or panels_module.DEFAULT_SEEDS,
            topologies=args.topologies,
            budget_ratios=args.budget_fractions,
            capacity_levels=args.capacity_levels,
            cost_levels=args.cost_levels,
            fortification_multiplier=args.fortification_multiplier,
        )
    if experiment == "seedext":
        return build_seed_extension_panel(
            output,
            seeds=args.seeds or SEED_EXTENSION_SEEDS,
            topologies=args.topologies,
            budget_ratios=args.budget_fractions,
            capacity_levels=args.capacity_levels,
            cost_levels=args.cost_levels,
            fortification_multiplier=args.fortification_multiplier,
        )
    if experiment == "heterogeneity":
        return panels_module.build_heterogeneity_panel(
            output,
            seeds=args.seeds or panels_module.DEFAULT_SEEDS,
            topologies=args.topologies,
            budget_ratios=args.budget_fractions,
            capacity_levels=args.capacity_levels,
            attack_cost_levels=args.cost_levels,
            protection_cost_levels=args.protection_cost_levels,
            effect_levels=args.effect_levels,
            fortification_multiplier=args.fortification_multiplier,
        )
    raise SystemExit(f"Unknown experiment {experiment!r}.")


def resolve_methods(
    experiment: str,
    requested: str | None,
    available_official: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the requested panel and report which upstream methods are missing.

    Only methods that are *transfers of an upstream dismantling ranking* depend
    on a frozen sequence file.  ``BPDReference-Protect`` also ends in
    ``-Protect`` but is an in-package baseline, so name shape alone must never
    decide availability.
    """

    from rmcd_f.official_review import OFFICIAL_METHOD_SPECS

    if requested:
        methods = [token.strip() for token in requested.split(",") if token.strip()]
    else:
        methods = list(DEFAULT_METHODS[experiment])
    resolved: list[str] = []
    missing: list[str] = []
    for method in methods:
        source = method[: -len("-Protect")] if method.endswith("-Protect") else None
        if (
            source is not None
            and source in OFFICIAL_METHOD_SPECS
            and method not in available_official
        ):
            # A missing upstream ranking must not silently delete the variant:
            # it is reported in the coverage table instead.
            missing.append(method)
            continue
        resolved.append(method)
    return tuple(dict.fromkeys(resolved)), tuple(dict.fromkeys(missing))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=(*EXPERIMENTS, "all"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--methods")
    parser.add_argument("--official-sequence-roots", default="")
    parser.add_argument(
        "--seeds",
        type=lambda value: tuple(int(v) for v in value.split(",") if v.strip()),
        default=None,
        help=(
            "Generator seeds. Defaults to the frozen benchmark seeds for main/role/"
            "heterogeneity and to the prespecified extension seeds for seedext, so a "
            "seed-extension run can never silently reuse the original graphs."
        ),
    )
    parser.add_argument("--topologies", type=lambda value: tuple(v.strip() for v in value.split(",") if v.strip()), default=panels_module.TOPOLOGIES)
    parser.add_argument("--budget-fractions", type=lambda value: tuple(float(v) for v in value.split(",") if v.strip()), default=panels_module.DEFAULT_BUDGET_RATIOS)
    parser.add_argument("--capacity-levels", type=lambda value: tuple(int(v) for v in value.split(",") if v.strip()), default=panels_module.DEFAULT_CAPACITY_LEVELS)
    parser.add_argument("--cost-levels", type=lambda value: tuple(int(v) for v in value.split(",") if v.strip()), default=panels_module.DEFAULT_COST_LEVELS)
    parser.add_argument("--protection-cost-levels", type=lambda value: tuple(int(v) for v in value.split(",") if v.strip()), default=panels_module.DEFAULT_PROTECTION_COST_LEVELS)
    parser.add_argument("--effect-levels", type=lambda value: tuple(float(v) for v in value.split(",") if v.strip()), default=panels_module.DEFAULT_EFFECT_LEVELS)
    parser.add_argument("--scaling-sizes", type=lambda value: tuple(int(v) for v in value.split(",") if v.strip()), default=panels_module.SCALING_SIZES)
    parser.add_argument("--scaling-seeds", type=lambda value: tuple(int(v) for v in value.split(",") if v.strip()), default=(11, 22, 33))
    parser.add_argument("--fortification-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help=(
            "One unified wall-clock limit applied to MPCF-Exact, MPCF-CG and "
            "MPCF-Greedy so that the scaling comparison is fair."
        ),
    )
    parser.add_argument("--cg-max-iterations", type=int, default=10_000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Merge the existing shards into runs_long.csv without solving "
            "anything. Run this once after all concurrent shards have stopped "
            "so the published file is not built from a partially written tree."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--shard-name",
        default="",
        help=(
            "Sub-directory under <output>/shards for this pass. Every pass writes "
            "its own shards and the final runs_long.csv concatenates all of them, "
            "so a later pass (for example the upstream baselines) can add methods "
            "to an existing panel without re-solving the earlier ones."
        ),
    )
    parser.add_argument("--include-conversions", action="store_true", help="Add the heterogeneous-cost conversion variants.")
    parser.add_argument("--limit-instances", type=int, default=0, help="Smoke-test hook: solve only the first N instances.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = tuple(Path(item.strip()) for item in args.official_sequence_roots.split(",") if item.strip())
    official, sequence_files = load_official_sequences(roots)
    available_official = {method for _, method in official}
    official_names = {
        f"{method}-Protect" for method in available_official
    }
    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)

    commit = resolve_commit(ROOT)
    configuration = {
        "experiments": list(experiments),
        "seeds": list(args.seeds) if args.seeds else None,
        "resolved_seeds": {
            name: list(
                (args.seeds or (SEED_EXTENSION_SEEDS if name == "seedext"
                                else panels_module.DEFAULT_SEEDS))
            )
            for name in experiments
        },
        "topologies": list(args.topologies),
        "budget_fractions": list(args.budget_fractions),
        "capacity_levels": list(args.capacity_levels),
        "cost_levels": list(args.cost_levels),
        "protection_cost_levels": list(args.protection_cost_levels),
        "effect_levels": list(args.effect_levels),
        "scaling_sizes": list(args.scaling_sizes),
        "scaling_seeds": list(args.scaling_seeds),
        "fortification_multiplier": args.fortification_multiplier,
        "time_limit_s": args.time_limit,
        "cg_max_iterations": args.cg_max_iterations,
        "include_conversions": args.include_conversions,
        "code_commit": commit,
    }
    config_digest = config_hash(configuration)
    time_limits = {
        "exact": args.time_limit,
        "cg": args.time_limit,
        "greedy": args.time_limit,
    }
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "experiment_config.json").write_text(
        json.dumps({**configuration, "config_hash": config_digest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        import yaml  # type: ignore

        (metadata / "experiment_config.yaml").write_text(
            yaml.safe_dump({**configuration, "config_hash": config_digest}, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass
    shard_tag = f"{args.experiment}_{args.shard_name or 'default'}"
    environment = observe_environment(
        root=ROOT,
        time_limit_s=args.time_limit,
        mip_gap=0.0,
        solver_seed=0,
        graph_seeds=list(args.seeds or panels_module.DEFAULT_SEEDS),
        threads=args.workers,
        extra={
            "config_hash": config_digest,
            "experiments": list(experiments),
            "experiment": args.experiment,
            "shard": args.shard_name or "",
            "runtime_definition": "selection, evaluation and total are recorded per run row",
        },
    )
    write_environment(metadata / "environment.json", environment)
    # One environment file per pass: the hardware-uniformity gate uses these to
    # prove that no experiment pooled runtimes from different CPU models.
    write_environment(metadata / f"environment_{shard_tag}.json", environment)

    shard_dir = output / "shards" / (args.shard_name or args.experiment)
    shard_dir.mkdir(parents=True, exist_ok=True)
    all_specs: list[InstanceSpec] = []
    summary: list[dict[str, Any]] = []
    for experiment in experiments:
        specs = build_panel(experiment, output, args)
        if args.limit_instances:
            specs = specs[: args.limit_instances]
        methods, missing = resolve_methods(experiment, args.methods, official_names)
        if args.include_conversions and experiment in {"main", "heterogeneity"}:
            methods = tuple(
                dict.fromkeys(
                    list(methods)
                    + [row.variant_name for row in cost_conversion_variants()]
                )
            )
        _log(
            f"{experiment}: {len(specs)} frozen instances; {len(methods)} methods; "
            f"budgets={list(args.budget_fractions)}"
        )
        if missing:
            _log(
                f"{experiment}: {len(missing)} upstream transfer variants have no "
                f"frozen sequence yet and are reported as missing: "
                f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}"
            )
        all_specs.extend(specs)
        summary.append(
            {
                "experiment": experiment,
                "instances": len(specs),
                "methods": len(methods),
                "budget_ratios": _json(list(args.budget_fractions)),
                "expected_runs": len(specs) * len(methods) * len(args.budget_fractions),
                "method_list": _json(list(methods)),
                "missing_upstream_variants": _json(list(missing)),
                "missing_upstream_count": len(missing),
            }
        )
        if args.prepare_only:
            continue
        if not args.collect_only:
            _solve(
                specs,
                methods,
                official,
                shard_dir,
                args,
                commit,
                config_digest,
                time_limits,
            )

    write_csv(metadata / "panel_summary.csv", summary)
    manifest_rows = [spec.as_manifest_row() for spec in all_specs]
    # ``graph_manifest.csv`` is the archive-level inventory of frozen instances
    # and must accumulate across passes.  Writing only this pass's rows would
    # leave the published archive describing whichever experiment ran last.
    # The per-experiment manifests below stay pass-specific.
    write_csv(
        metadata / "graph_manifest.csv",
        _merge_manifest(metadata / "graph_manifest.csv", manifest_rows),
    )
    # A manifest the upstream baseline matrix can consume unchanged.  It is
    # written per experiment as well as in aggregate, because each panel run
    # only knows its own instances and would otherwise overwrite the previous
    # panel's manifest.
    manifest_columns = [
        "instance_id",
        "graph_id",
        "base_id",
        "topology",
        "seed",
        "scale_tier",
        "N",
        "num_edges",
        "task_path_count",
        "instance_file",
        "graph_fingerprint",
        "experiment",
    ]
    baseline_manifest = [
        {
            "instance_id": spec.instance_id,
            "graph_id": spec.graph_id,
            "base_id": spec.base_id,
            "topology": spec.topology,
            "seed": spec.seed,
            "scale_tier": spec.scale_tier,
            "N": spec.node_count,
            "num_edges": spec.edge_count,
            "task_path_count": spec.task_path_count,
            "instance_file": spec.instance_file,
            "graph_fingerprint": spec.graph_fingerprint,
            "experiment": spec.experiment,
        }
        for spec in all_specs
    ]
    write_csv(metadata / "instance_manifest.csv", baseline_manifest, columns=manifest_columns)
    for experiment in experiments:
        subset = [
            row for row in baseline_manifest if row["experiment"] == experiment
        ]
        if subset:
            write_csv(
                metadata / f"instance_manifest_{experiment}.csv",
                subset,
                columns=manifest_columns,
            )
    if args.prepare_only:
        _log(f"prepared {len(all_specs)} frozen instances under {output}")
        return 0

    shards_root = output / "shards"
    records = _collect(shards_root)
    if not records:
        _log("no run records produced")
        return 1
    cells = _load_cells(shards_root)
    _apply_optima(records, cells)
    written = write_runs_long(output / "results" / "runs_long.csv", records)
    _log(f"wrote {written} run rows to {output / 'results' / 'runs_long.csv'}")
    write_csv(
        output / "results" / "instance_budget_cells.csv",
        [_cell_row(key, value) for key, value in sorted(cells.items())],
    )
    _log(
        f"variants: {EXPECTED_COMPARISON_VARIANTS} comparison + {len(MAIN_PANEL_EXCLUDED)} "
        f"panel-excluded upstream methods; official sequence files: {len(sequence_files)}"
    )
    return 0


def _cell_row(key: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    instance, _, ratio = key.rpartition("|")
    return {
        "instance_id": instance,
        "budget_ratio": float(ratio) if ratio else "",
        "budget_abs": cell.get("budget_abs"),
        "budget_abs_floor": cell.get("budget_abs_floor"),
        "zero_effective_budget": cell.get("zero_effective_budget"),
        "undefended_kappa": cell.get("undefended"),
        "kappa_opt": cell.get("kappa_opt"),
        "certifying_methods": ",".join(cell.get("certifying_methods") or []),
        "certificate_conflict": int(cell.get("certificate_conflict") or 0),
    }


def _solve(
    specs: Sequence[InstanceSpec],
    methods: Sequence[str],
    official: Mapping[tuple[str, str], tuple[Sequence[Any], float, str]],
    shard_dir: Path,
    args: argparse.Namespace,
    commit: str,
    config_digest: str,
    time_limits: Mapping[str, float],
) -> None:
    pending = [spec for spec in specs if not _shard_done(shard_dir, spec)]
    if args.resume and len(pending) != len(specs):
        _log(f"resume: {len(specs) - len(pending)} instances already solved")
    if not pending:
        return
    payloads = [
        {
            "spec": _spec_payload(spec),
            "methods": list(methods),
            "official": {
                key: [list(value[0]), value[1], value[2]]
                for key, value in official.items()
            },
            "time_limits": dict(time_limits),
            "cg_max_iterations": args.cg_max_iterations,
            "commit": commit,
            "config_hash": config_digest,
            "shard_dir": str(shard_dir),
        }
        for spec in pending
    ]
    if args.workers == 1:
        for index, payload in enumerate(payloads, start=1):
            _worker(payload)
            _log(f"solved {index}/{len(payloads)}")
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as exc:  # keep going; the shard records the failure
                _log(f"instance failed: {type(exc).__name__}: {exc}")
            if index % 5 == 0 or index == len(payloads):
                _log(f"solved {index}/{len(payloads)}")


def _spec_payload(spec: InstanceSpec) -> dict[str, Any]:
    payload = dict(spec.__dict__)
    payload["extra"] = dict(spec.extra)
    return payload


def _shard_done(shard_dir: Path, spec: InstanceSpec) -> bool:
    return (shard_dir / f"{_safe(spec.instance_id)}.csv").exists()


def _collect(root: Path) -> list[dict[str, Any]]:
    """Read every shard under ``root`` so multiple method passes accumulate.

    Shards of one panel may still be running while another finishes and merges,
    so unreadable or partially written files are skipped rather than aborting
    the whole merge.  The final collection after all shards stop is the
    authoritative one.
    """

    from rmcd_f.rev.schema import coerce_row

    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                records.extend(coerce_row(row) for row in csv.DictReader(handle))
        except (OSError, UnicodeDecodeError) as exc:
            _log(f"skipping unreadable shard {path.name}: {type(exc).__name__}")
    return records


def _merge_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Union a pass's frozen instances into the accumulating archive manifest."""

    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for entry in csv.DictReader(handle):
                    merged[str(entry.get("instance_id"))] = dict(entry)
        except (OSError, csv.Error):
            pass
    for row in rows:
        merged[str(row.get("instance_id"))] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _load_cells(root: Path) -> dict[str, Mapping[str, Any]]:
    """Merge the per-cell ledgers of every pass.

    A later pass that only adds baselines recomputes the cell ledger without an
    exact solver, so its ``kappa_opt`` is ``None``; merging must never let that
    overwrite a certified value found by an earlier pass.  A ledger being
    rewritten by a concurrent shard is skipped, not fatal.
    """

    cells: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.cells.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            existing = cells.get(key)
            if existing is None:
                cells[key] = dict(value)
                continue
            merged = dict(existing)
            if existing.get("kappa_opt") is None and value.get("kappa_opt") is not None:
                merged["kappa_opt"] = value["kappa_opt"]
            names = list(
                dict.fromkeys(
                    list(existing.get("certifying_methods") or [])
                    + list(value.get("certifying_methods") or [])
                )
            )
            merged["certifying_methods"] = names
            merged["certificate_conflict"] = int(
                existing.get("certificate_conflict") or 0
            ) or int(value.get("certificate_conflict") or 0)
            for field in ("undefended", "budget_abs", "budget_abs_floor", "zero_effective_budget"):
                if merged.get(field) is None:
                    merged[field] = value.get(field)
            cells[key] = merged
    return cells


def _apply_optima(records: list[dict[str, Any]], cells: Mapping[str, Mapping[str, Any]]) -> None:
    """Fill ``kappa_opt`` and the cell-level audit fields on every run row."""

    filled = 0
    for row in records:
        key = f"{row['instance_id']}|{float(row['budget_ratio']):.10g}"
        cell = cells.get(key)
        if not cell:
            continue
        row["undefended_kappa"] = cell.get("undefended")
        row["budget_abs_floor"] = cell.get("budget_abs_floor")
        row["zero_effective_budget"] = cell.get("zero_effective_budget")
        row["certifying_methods"] = ",".join(cell.get("certifying_methods") or [])
        row["certificate_conflict"] = int(cell.get("certificate_conflict") or 0)
        if cell.get("kappa_opt") is not None:
            row["kappa_opt"] = cell["kappa_opt"]
            filled += 1
    _log(f"certified optimum available for {filled}/{len(records)} run rows")


if __name__ == "__main__":
    raise SystemExit(main())
