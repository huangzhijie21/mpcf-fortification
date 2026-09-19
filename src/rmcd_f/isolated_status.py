"""Machine-readable status ledger for isolated official baseline runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


FIELDNAMES = (
    "method",
    "status",
    "exit_code",
    "started_utc",
    "finished_utc",
    "elapsed_seconds",
    "timeout_seconds",
    "termination_reason",
    "method_output",
    "method_log",
)


def classify_exit_code(exit_code: int) -> tuple[str, str]:
    """Map a process exit code to an auditable terminal status."""

    if exit_code == 0:
        return "SUCCESS", "completed"
    if exit_code == 124:
        return "TIMEOUT", "timeout"
    if exit_code >= 128:
        return "SIGNAL", f"signal_{exit_code - 128}"
    return "FAILED", f"exit_code_{exit_code}"


def initialize_status_file(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
    return target


def append_status(
    path: str | Path,
    *,
    method: str,
    exit_code: int,
    started_utc: str,
    finished_utc: str,
    elapsed_seconds: float,
    timeout_seconds: float,
    method_output: str,
    method_log: str,
) -> dict[str, str | int | float]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Isolated status file was not initialized: {target}"
        )
    status, reason = classify_exit_code(exit_code)
    row: dict[str, str | int | float] = {
        "method": method,
        "status": status,
        "exit_code": exit_code,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout_seconds,
        "termination_reason": reason,
        "method_output": method_output,
        "method_log": method_log,
    }
    with target.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writerow(row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--path", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--path", required=True)
    record.add_argument("--method", required=True)
    record.add_argument("--exit-code", required=True, type=int)
    record.add_argument("--started-utc", required=True)
    record.add_argument("--finished-utc", required=True)
    record.add_argument("--elapsed-seconds", required=True, type=float)
    record.add_argument("--timeout-seconds", required=True, type=float)
    record.add_argument("--method-output", required=True)
    record.add_argument("--method-log", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        initialize_status_file(args.path)
        return 0
    append_status(
        args.path,
        method=args.method,
        exit_code=args.exit_code,
        started_utc=args.started_utc,
        finished_utc=args.finished_utc,
        elapsed_seconds=args.elapsed_seconds,
        timeout_seconds=args.timeout_seconds,
        method_output=args.method_output,
        method_log=args.method_log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
