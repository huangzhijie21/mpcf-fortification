"""The unified run-level record written to ``runs_long.csv``.

Every solver in every experiment funnels into this one schema.  Fields that a
given method does not produce are written as the literal ``NA`` rather than
being dropped, so that one file can be read by the statistics, plotting and
auditing layers without per-method special cases.

The last four fields (``bb_nodes`` onwards) are the solver-logging block the
reviewers asked for: an Exact row must carry its incumbent, bound, gap,
branch-and-bound node count and an independent replay of the fixed protection
set; a cut-generation row must carry its ``L``/``U`` interval, iteration count
and cut count; a greedy row must carry its objective, realised cost, selection
count, iteration count and runtime.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, fields
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NA = "NA"

RUN_COLUMNS: tuple[str, ...] = (
    # --- identity -------------------------------------------------------
    "experiment",
    "instance_id",
    "graph_id",
    "base_id",
    "topology",
    "N",
    "num_edges",
    "seed",
    "composition",
    "budget_ratio",
    "budget_abs",
    "method",
    "variant",
    # --- objective ------------------------------------------------------
    "kappa",
    "kappa_opt",
    "relative_gap",
    "actual_cost",
    "selected_count",
    "selected_nodes",
    # --- runtime --------------------------------------------------------
    "runtime_selection_s",
    "runtime_evaluation_s",
    "runtime_total_s",
    "status",
    # --- solver logging -------------------------------------------------
    "incumbent",
    "best_bound",
    "final_gap",
    "bb_nodes",
    "replay_kappa",
    "certificate_mode",
    "cg_L",
    "cg_U",
    "cg_iterations",
    "cg_cuts",
    "time_limit_s",
    # --- provenance -----------------------------------------------------
    "config_hash",
    "code_commit",
)

#: Columns that must contain a finite number whenever the status is ``optimal``
#: for the corresponding solver family.  Used by :mod:`rmcd_f.rev.gates`.
REQUIRED_BY_CERTIFICATE_MODE: Mapping[str, tuple[str, ...]] = {
    "solver_closed": ("incumbent", "best_bound", "final_gap", "bb_nodes", "replay_kappa"),
    "grid_closed": ("incumbent", "best_bound", "final_gap", "bb_nodes", "replay_kappa"),
    "cg_closed": ("cg_L", "cg_U", "cg_iterations", "cg_cuts", "replay_kappa"),
    "open": (),
}

CERTIFICATE_MODES: tuple[str, ...] = (
    "solver_closed",
    "grid_closed",
    "cg_closed",
    "open",
)


def _finite(value: Any) -> Any:
    """Return ``value`` when it is finite, else ``NA``."""

    if value is None or value is True or value is False:
        return NA if value is None else int(value)
    if isinstance(value, str):
        return value if value else NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value if value != "" else NA
    if not math.isfinite(number):
        return NA
    if number.is_integer() and abs(number) < 2**53:
        return int(number)
    return number


def _selected_nodes(nodes: Iterable[Any] | None) -> str:
    if nodes is None:
        return NA
    if isinstance(nodes, str):
        # Already-serialised column read back from a shard CSV.
        return nodes if nodes else NA
    payload = sorted(str(node) for node in nodes)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


@dataclass
class RunRecord:
    """One ``(instance, budget, method, variant)`` result row."""

    experiment: str
    instance_id: str
    graph_id: str
    base_id: str
    topology: str
    N: int
    num_edges: int
    seed: int
    method: str
    variant: str
    budget_ratio: float
    budget_abs: float

    composition: str | None = None
    kappa: float | None = None
    kappa_opt: float | None = None
    actual_cost: float | None = None
    selected_count: int | None = None
    selected_nodes: Sequence[Any] | None = None
    runtime_selection_s: float | None = None
    runtime_evaluation_s: float | None = None
    runtime_total_s: float | None = None
    status: str = "unknown"

    incumbent: float | None = None
    best_bound: float | None = None
    final_gap: float | None = None
    bb_nodes: int | None = None
    replay_kappa: float | None = None
    certificate_mode: str = "open"
    cg_L: float | None = None
    cg_U: float | None = None
    cg_iterations: int | None = None
    cg_cuts: int | None = None
    time_limit_s: float | None = None

    config_hash: str = ""
    code_commit: str = ""

    def relative_gap(self) -> float | None:
        """``(kappa_opt - kappa) / kappa_opt`` when an optimum is certified."""

        if self.kappa_opt in (None, NA) or self.kappa in (None, NA):
            return None
        try:
            optimum = float(self.kappa_opt)
            value = float(self.kappa)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(optimum) or not math.isfinite(value) or optimum <= 0.0:
            return None
        return (optimum - value) / optimum

    def to_row(self) -> dict[str, Any]:
        payload = {key: getattr(self, key) for key in RUN_COLUMNS if key != "relative_gap"}
        # ``relative_gap`` is a column but not a constructor field.
        payload["relative_gap"] = self.relative_gap()
        payload["selected_nodes"] = _selected_nodes(self.selected_nodes)
        return {key: _finite(payload.get(key)) for key in RUN_COLUMNS}


def write_runs_long(path: Path, records: Iterable[RunRecord | Mapping[str, Any]]) -> int:
    """Write ``runs_long.csv`` with a fixed header and ``NA`` padding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(record.to_row() if isinstance(record, RunRecord) else _mapping_row(record))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, NA) for key in RUN_COLUMNS})
    temporary.replace(path)
    return len(rows)


def _mapping_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: row.get(key, NA) for key in RUN_COLUMNS}
    if payload.get("relative_gap") in (NA, None, ""):
        try:
            optimum = float(row.get("kappa_opt"))
            value = float(row.get("kappa"))
            if math.isfinite(optimum) and math.isfinite(value) and optimum > 0.0:
                payload["relative_gap"] = (optimum - value) / optimum
        except (TypeError, ValueError):
            pass
    payload["selected_nodes"] = _selected_nodes(row.get("selected_nodes"))
    return {key: _finite(payload.get(key)) for key in RUN_COLUMNS}


NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "N", "num_edges", "seed", "budget_ratio", "budget_abs", "kappa", "kappa_opt",
        "relative_gap", "actual_cost", "selected_count", "runtime_selection_s",
        "runtime_evaluation_s", "runtime_total_s", "incumbent", "best_bound",
        "final_gap", "bb_nodes", "replay_kappa", "cg_L", "cg_U", "cg_iterations",
        "cg_cuts", "time_limit_s",
    }
)


def coerce_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one raw CSV mapping into typed values, mapping ``NA`` to ``None``."""

    row: dict[str, Any] = {}
    for key, value in entry.items():
        if value is None or value == NA or value == "":
            row[key] = None
        elif key in NUMERIC_COLUMNS:
            try:
                number = float(value)
                row[key] = int(number) if number.is_integer() else number
            except (TypeError, ValueError):
                row[key] = value
        else:
            row[key] = value
    if row.get("selected_nodes"):
        try:
            row["selected_nodes"] = json.loads(row["selected_nodes"])
        except (TypeError, ValueError):
            pass
    return row


def read_runs_long(path: Path) -> list[dict[str, Any]]:
    """Read ``runs_long.csv`` back into typed rows (``NA`` -> ``None``)."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [coerce_row(entry) for entry in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    """Write a CSV with an explicit, deterministic column order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite(row.get(key)) for key in columns})
    temporary.replace(path)
