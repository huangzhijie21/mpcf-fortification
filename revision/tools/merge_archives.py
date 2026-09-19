#!/usr/bin/env python3
"""Merge the per-server archives of the distributed MPCF revision run.

Each server produced its own ``results/`` tree.  Panels are disjoint across
servers by construction (one experiment is always produced on one CPU model),
so merging is a concatenation with duplicate protection, not an aggregation.

What is merged
--------------
``results/runs_long.csv``            deduplicated on the run identity
``results/instance_budget_cells.csv`` deduplicated on (instance, budget)
``metadata/graph_manifest.csv``      union of frozen instances
``metadata/environment_*.json``      every pass, kept for the hardware gate

The merged archive is then handed to ``rev_statistics.py`` / ``rev_plots.py`` /
``rev_gates.py`` unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rmcd_f.rev.schema import RUN_COLUMNS, coerce_row, write_csv  # noqa: E402

RUN_IDENTITY = ("experiment", "instance_id", "budget_ratio", "method", "variant")
CELL_IDENTITY = ("instance_id", "budget_ratio")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def _key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple:
    payload = []
    for field in fields:
        value = row.get(field)
        if field == "budget_ratio" and value is not None:
            value = round(float(value), 10)
        payload.append(value)
    return tuple(payload)


def merge_runs(inputs: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: dict[tuple, dict[str, Any]] = {}
    duplicates = 0
    conflicts = 0
    for root in inputs:
        for row in _read(root / "results" / "runs_long.csv"):
            key = _key(row, RUN_IDENTITY)
            existing = merged.get(key)
            if existing is None:
                merged[key] = row
                continue
            duplicates += 1
            # Only numeric results may conflict; provenance fields may differ
            # legitimately between passes.
            for field in ("kappa", "kappa_opt", "actual_cost", "selected_count"):
                left, right = existing.get(field), row.get(field)
                if left is None or right is None:
                    continue
                try:
                    if abs(float(left) - float(right)) > 1e-9:
                        conflicts += 1
                        break
                except (TypeError, ValueError):
                    conflicts += 1
                    break
    ordered = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("experiment")),
            str(row.get("instance_id")),
            float(row.get("budget_ratio") or 0.0),
            str(row.get("method")),
        ),
    )
    return ordered, {"duplicates": duplicates, "conflicts": conflicts}


def merge_cells(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple, dict[str, Any]] = {}
    for root in inputs:
        for row in _read(root / "results" / "instance_budget_cells.csv"):
            key = _key(row, CELL_IDENTITY)
            existing = merged.get(key)
            if existing is None:
                merged[key] = row
                continue
            # Never let a later pass blank out a certified optimum.
            if existing.get("kappa_opt") in (None, "", "NA") and row.get("kappa_opt") not in (
                None,
                "",
                "NA",
            ):
                merged[key] = row
    return sorted(merged.values(), key=lambda row: str(row.get("instance_id")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated result roots downloaded from each server.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    inputs = [Path(item.strip()) for item in args.inputs.split(",") if item.strip()]
    missing = [path for path in inputs if not (path / "results" / "runs_long.csv").exists()]
    if missing:
        raise SystemExit(f"Missing runs_long.csv under: {[str(p) for p in missing]}")

    output = Path(args.output)
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)

    rows, stats = merge_runs(inputs)
    write_csv(
        output / "results" / "runs_long.csv", rows, columns=list(RUN_COLUMNS)
    )
    cells = merge_cells(inputs)
    if cells:
        write_csv(output / "results" / "instance_budget_cells.csv", cells)

    # Union of frozen instances, checked for fingerprint agreement.
    manifests: dict[str, dict[str, Any]] = {}
    disagreed: list[str] = []
    for root in inputs:
        for row in _read(root / "metadata" / "graph_manifest.csv"):
            key = str(row.get("instance_id"))
            existing = manifests.get(key)
            if existing is None:
                manifests[key] = row
            elif str(existing.get("graph_fingerprint")) != str(row.get("graph_fingerprint")):
                disagreed.append(key)
    if disagreed:
        raise SystemExit(
            "Frozen instance fingerprints disagree across servers for: "
            f"{disagreed[:5]} -- refusing to merge."
        )

    # Keep only instances that actually produced runs.
    #
    # A superseded pass can leave manifest rows behind for instances that were
    # frozen and then never solved -- specifically, the first seed-extension
    # attempt froze 45 graphs on the *main* seeds before the seed wiring was
    # corrected.  Those rows have no runs and are not part of the published
    # panel, so carrying them would over-report the frozen-instance count.
    # They are dropped and named in the summary rather than removed silently.
    run_instances = {str(row.get("instance_id")) for row in rows}
    orphans = sorted(key for key in manifests if key not in run_instances)
    for key in orphans:
        manifests.pop(key, None)
    if orphans:
        print(
            f"[merge] dropped {len(orphans)} manifest rows with no runs "
            f"(superseded freezes): {orphans[:3]}{' ...' if len(orphans) > 3 else ''}",
            flush=True,
        )

    write_csv(
        output / "metadata" / "graph_manifest.csv",
        sorted(manifests.values(), key=lambda row: str(row.get("instance_id"))),
    )

    # Every environment file, renamed by its source server, for the gate.
    #
    # Only environments whose experiment actually contributed rows from that
    # server are kept: a server that ran a panel in an earlier, superseded
    # attempt still has its environment file on disk, and carrying it over
    # would make the hardware-uniformity gate report a CPU model mix that the
    # published rows never had.
    kept_env = 0
    for root in inputs:
        contributed = {
            str(row.get("experiment")) for row in _read(root / "results" / "runs_long.csv")
        }
        for path in sorted((root / "metadata").glob("*environment*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            experiment = str((payload.get("extra") or {}).get("experiment") or "")
            if contributed and experiment and experiment not in contributed:
                continue
            shutil.copyfile(path, output / "metadata" / f"{root.name}__{path.name}")
            kept_env += 1

    summary = {
        "inputs": [str(path) for path in inputs],
        "run_rows": len(rows),
        "cell_rows": len(cells),
        "instances": len(manifests),
        "environment_files": kept_env,
        "duplicate_rows": stats["duplicates"],
        "conflicting_rows": stats["conflicts"],
        "manifest_rows_dropped_no_runs": len(orphans),
        "dropped_instance_ids": orphans,
        "experiments": sorted({str(row.get("experiment")) for row in rows}),
        "methods": sorted({str(row.get("method")) for row in rows}),
        "revisions": sorted({str(row.get("code_commit")) for row in rows}),
    }
    (output / "metadata" / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
