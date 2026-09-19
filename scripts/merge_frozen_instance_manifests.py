#!/usr/bin/env python3
"""Merge frozen instance manifests while rechecking graph identity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from rmcd_f.cli import load_graph
from rmcd_f.official_review import graph_fingerprint


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(raw: str, manifest: Path) -> Path:
    supplied = Path(raw)
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.extend((manifest.parent / supplied, manifest.parent / "instances" / supplied.name))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise SystemExit(f"Cannot resolve {raw!r} from {manifest}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = tuple(Path(item.strip()).resolve() for item in args.manifests.split(",") if item.strip())
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Manifest does not exist: {path}")
        for row in _read(path):
            instance = _resolve(row.get("instance_file", ""), path)
            graph = load_graph(instance)
            observed = graph_fingerprint(graph)
            expected = row.get("graph_fingerprint", "")
            if observed != expected:
                raise SystemExit(f"Fingerprint mismatch for {instance}.")
            if observed in seen:
                raise SystemExit(f"Duplicate frozen fingerprint {observed}.")
            seen.add(observed)
            rows.append({**row, "instance_file": str(instance)})
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} rows; observed {len(rows)}.")
    rows.sort(key=lambda row: (str(row.get("scale_tier", "")), str(row.get("topology", "")), int(row.get("seed", 0))))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} frozen instances into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
