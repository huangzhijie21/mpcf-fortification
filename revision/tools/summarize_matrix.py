#!/usr/bin/env python3
"""Summarise an official review-matrix run: per-method coverage and failures."""
from __future__ import annotations

import csv
import collections
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/root/mpcf_rev/results/official_all_matrix/matrix_task_status.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
print(f"total task rows: {len(rows)}")

by_method: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
by_status: collections.Counter = collections.Counter()
errors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
for row in rows:
    method = row.get("method", "")
    status = row.get("status", "")
    by_method[method][status] += 1
    by_status[status] += 1
    if status != "SUCCESS":
        errors[method][(row.get("error") or row.get("termination_reason") or "")[:58]] += 1

print("\n=== overall ===")
for status, count in by_status.most_common():
    print(f"  {status:18s} {count}")

print("\n=== per method (SUCCESS/total) ===")
complete, partial, absent = [], [], []
for method in sorted(by_method):
    counter = by_method[method]
    total = sum(counter.values())
    success = counter.get("SUCCESS", 0)
    tag = "OK " if success == total else ("PART" if success else "NONE")
    print(f"  [{tag}] {method:42s} {success:3d}/{total:<3d} " +
          " ".join(f"{k}={v}" for k, v in sorted(counter.items()) if k != "SUCCESS"))
    (complete if success == total else partial if success else absent).append(method)

print(f"\ncomplete={len(complete)} partial={len(partial)} absent={len(absent)}")

print("\n=== failure reasons (absent/partial methods) ===")
for method in absent + partial:
    for reason, count in errors[method].most_common(2):
        if reason:
            print(f"  {method:42s} x{count:<3d} {reason}")
