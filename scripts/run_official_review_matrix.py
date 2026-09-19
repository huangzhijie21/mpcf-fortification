#!/usr/bin/env python3
"""Run official baselines as isolated method-by-instance tasks.

The upstream review implementations mix compiled executables, TensorFlow 1.x,
graph-tool, and PyG.  A whole-method process therefore has two undesirable
properties: one pathological graph blocks every later instance, and mutable
module/process state can leak between instances.  This runner gives every
method-instance pair its own process, timeout, output directory, and log while
serializing only method families that share build artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rmcd_f.cli import load_graph
from rmcd_f.official_review import (
    FINDER_LEGACY_ADAPTER_VERSION,
    GND_FAMILY_ADAPTER_VERSION,
    OFFICIAL_ADAPTER_VERSION,
    OFFICIAL_METHOD_SPECS,
    expand_official_methods,
    graph_fingerprint as adapter_graph_fingerprint,
)


@dataclass(frozen=True)
class Task:
    method: str
    instance: dict[str, str]
    manifest: Path
    output: Path
    log: Path


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _progress(message: str) -> None:
    print(f"[{_timestamp()}] [official-matrix] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _safe_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    ).strip("_")
    return token or "instance"


def _resource_group(method: str) -> str:
    """Return a lock group only when upstream tasks share mutable files."""

    if method.startswith("OfficialCI_L"):
        return "ci"
    if method in {"OfficialGND", "OfficialGNDR"}:
        return "gnd"
    if method.startswith("OfficialEI_S"):
        return "ei"
    if method in {"OfficialMinSum", "OfficialMinSumR"}:
        return "minsum"
    if method in {"OfficialGDM", "OfficialGDMR", "OfficialCoreGDM"}:
        return "gdm"
    if method == "OfficialEGND":
        return "egnd"
    if method == "OfficialFINDER_R":
        return "finder"
    if method == "OfficialCoreHD":
        return "corehd"
    if method in {"OfficialEigenvectorStatic", "OfficialEigenvectorDynamic"}:
        return "eigenvector"
    if method in {
        "OfficialNetworkEntanglementSmallR",
        "OfficialNetworkEntanglementMidR",
        "OfficialNetworkEntanglementLargeR",
    }:
        return "network_entanglement_reinsertion"
    if method == "OfficialVertexEntanglementR":
        return "vertex_entanglement_reinsertion"
    return ""


def _resolve_python(requested: str) -> Path:
    candidate = Path(requested).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    discovered = shutil.which(requested)
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError(f"Python executable is unavailable: {requested}")


def _python_for(method: str, args: argparse.Namespace) -> Path:
    if method == "OfficialFINDER_R":
        return _resolve_python(args.finder_python)
    if OFFICIAL_METHOD_SPECS[method].environment == "gdm":
        return _resolve_python(args.gdm_python)
    return _resolve_python(args.dismantling_python)


def _timeout_for(method: str, args: argparse.Namespace) -> int:
    if method == "OfficialFINDER_R":
        return args.finder_timeout_seconds
    if OFFICIAL_METHOD_SPECS[method].environment == "gdm":
        return args.gdm_timeout_seconds
    return args.general_timeout_seconds


def _target_environment(
    python: Path,
    project_root: Path,
    method: str,
    native_threads: int,
) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(project_root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    prefix = python.parent.parent
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get(
        "PATH", ""
    )
    library = prefix / "lib"
    environment["CONDA_PREFIX"] = str(prefix)
    environment["LD_LIBRARY_PATH"] = str(library) + (
        os.pathsep + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    environment.setdefault("PYTHONHASHSEED", "0")
    for variable in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = str(native_threads)
    if method == "OfficialFINDER_R":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    return environment


def _command(
    task: Task,
    python: Path,
    args: argparse.Namespace,
    project_root: Path,
) -> list[str]:
    if task.method == "OfficialFINDER_R":
        script = project_root / "scripts" / "export_finder_legacy.py"
        return [
            str(python),
            str(script),
            "--instance-manifest",
            str(task.manifest),
            "--review-path",
            str(args.review_path),
            "--stop-condition",
            str(args.stop_condition),
            "--output",
            str(task.output),
        ]
    script = project_root / "scripts" / "run_official_review_sequences.py"
    return [
        str(python),
        str(script),
        "--instance-manifest",
        str(task.manifest),
        "--methods",
        task.method,
        "--review-path",
        str(args.review_path),
        "--stop-condition",
        str(args.stop_condition),
        "--brute-force-max-n",
        str(args.brute_force_max_n),
        "--output",
        str(task.output),
    ]


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    else:
        process.terminate()
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()


def _completed_row(task: Task) -> dict[str, str] | None:
    path = task.output / "official_sequences.csv"
    if not path.is_file():
        return None
    rows = _read_csv(path)
    if len(rows) != 1:
        return None
    row = rows[0]
    instance_file = Path(task.instance.get("instance_file", ""))
    if not instance_file.is_file():
        return None
    expected_adapter_fingerprint = adapter_graph_fingerprint(
        load_graph(instance_file)
    )
    expected_projection_fingerprint = task.instance.get(
        "projection_fingerprint", ""
    )
    expected_adapter_version = (
        FINDER_LEGACY_ADAPTER_VERSION
        if task.method == "OfficialFINDER_R"
        else OFFICIAL_ADAPTER_VERSION
    )
    row_projection_fingerprint = row.get("projection_fingerprint", "")
    row_instance_file = Path(row.get("instance_file", ""))
    if (
        row.get("method") != task.method
        or row.get("graph_fingerprint")
        != expected_adapter_fingerprint
        or not row_instance_file.is_file()
        or row_instance_file.resolve() != instance_file.resolve()
        or (
            expected_projection_fingerprint
            and row_projection_fingerprint
            and row_projection_fingerprint != expected_projection_fingerprint
        )
        or (
            expected_projection_fingerprint
            and not row_projection_fingerprint
            and task.method != "OfficialFINDER_R"
        )
        or row.get("adapter_version") != expected_adapter_version
    ):
        return None
    if task.method in {"OfficialGND", "OfficialGNDR"} and row.get(
        "gnd_family_adapter_version"
    ) != GND_FAMILY_ADAPTER_VERSION:
        return None
    return row


def _run_task(
    task: Task,
    args: argparse.Namespace,
    project_root: Path,
    semaphores: dict[str, threading.BoundedSemaphore],
) -> dict[str, Any]:
    previous = _completed_row(task) if args.resume else None
    if previous and previous.get("status") in {"SUCCESS", "NOT_APPLICABLE"}:
        return {
            "method": task.method,
            "graph_fingerprint": task.instance.get("graph_fingerprint", ""),
            "topology": task.instance.get("topology", ""),
            "seed": task.instance.get("seed", ""),
            "status": previous["status"],
            "termination_reason": "resumed_audited_task",
            "row_status": previous["status"],
            "process_exit_code": 0,
            "elapsed_seconds": 0.0,
            "queue_wait_seconds": 0.0,
            "process_runtime_seconds": 0.0,
            "total_elapsed_seconds": 0.0,
            "native_threads": args.native_threads,
            "method_output": str(task.output),
            "method_log": str(task.log),
        }

    python = _python_for(task.method, args)
    timeout_seconds = _timeout_for(task.method, args)
    resource_group = _resource_group(task.method)
    semaphore = semaphores.get(resource_group)
    started_utc = _timestamp()
    total_started = time.monotonic()
    task.log.parent.mkdir(parents=True, exist_ok=True)
    task.output.mkdir(parents=True, exist_ok=True)
    command = _command(task, python, args, project_root)
    process_exit_code: int | str = ""
    termination_reason = "completed"
    timed_out = False

    queue_started = time.monotonic()
    if semaphore is not None:
        semaphore.acquire()
    acquired = time.monotonic()
    acquired_utc = _timestamp()
    process_started = acquired
    process_finished = acquired
    try:
        with task.log.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                f"[{started_utc}] START {task.method} "
                f"{task.instance.get('topology', '')}/"
                f"seed{task.instance.get('seed', '')}\n"
            )
            log_handle.write("COMMAND " + " ".join(command) + "\n")
            log_handle.flush()
            popen_kwargs: dict[str, Any] = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            process_started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=_target_environment(
                    python,
                    project_root,
                    task.method,
                    args.native_threads,
                ),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
            try:
                process_exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_reason = "task_timeout"
                _terminate_process_tree(process)
                process_exit_code = 124
            process_finished = time.monotonic()
    finally:
        if semaphore is not None:
            semaphore.release()

    finished = time.monotonic()
    queue_wait = acquired - queue_started
    process_runtime = process_finished - process_started
    elapsed = finished - total_started
    sequence_row = _completed_row(task)
    if timed_out:
        status = "TIMEOUT"
    elif sequence_row is not None and sequence_row.get("status") in {
        "SUCCESS",
        "NOT_APPLICABLE",
    }:
        status = sequence_row["status"]
    else:
        status = "FAILED"
        if process_exit_code != 0:
            termination_reason = f"process_exit_{process_exit_code}"
        elif sequence_row is None:
            termination_reason = "missing_single_sequence_row"
        else:
            termination_reason = "adapter_reported_failure"
    return {
        "method": task.method,
        "graph_fingerprint": task.instance.get("graph_fingerprint", ""),
        "topology": task.instance.get("topology", ""),
        "seed": task.instance.get("seed", ""),
        "dependency_environment": OFFICIAL_METHOD_SPECS[task.method].environment,
        "resource_group": resource_group,
        "python_executable": str(python),
        "timeout_seconds": timeout_seconds,
        "native_threads": args.native_threads,
        "started_utc": started_utc,
        "acquired_utc": acquired_utc,
        "finished_utc": _timestamp(),
        "elapsed_seconds": elapsed,
        "queue_wait_seconds": queue_wait,
        "process_runtime_seconds": process_runtime,
        "total_elapsed_seconds": elapsed,
        "process_exit_code": process_exit_code,
        "status": status,
        "termination_reason": termination_reason,
        "row_status": sequence_row.get("status", "") if sequence_row else "",
        "error_type": sequence_row.get("error_type", "") if sequence_row else "",
        "error": sequence_row.get("error", "") if sequence_row else "",
        "task_manifest": str(task.manifest),
        "method_output": str(task.output),
        "method_log": str(task.log),
    }


def _prepare_tasks(
    manifest: Path,
    methods: tuple[str, ...],
    output: Path,
    log_root: Path,
) -> list[Task]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        instances = list(reader)
        fields = list(reader.fieldnames or ())
    if not instances or "instance_file" not in fields:
        raise SystemExit("Instance manifest is empty or lacks instance_file.")

    tasks: list[Task] = []
    for instance in instances:
        instance = dict(instance)
        instance_path = Path(instance["instance_file"]).expanduser()
        if not instance_path.is_absolute():
            instance_path = manifest.parent / instance_path
        instance["instance_file"] = str(instance_path.resolve())
        slug = _safe_token(
            f"{instance.get('topology', 'graph')}_seed{instance.get('seed', '')}_"
            f"{instance.get('graph_fingerprint', '')[:10]}"
        )
        for method in methods:
            task_manifest = output / "_task_manifests" / method / f"{slug}.csv"
            _write_csv(task_manifest, [instance])
            tasks.append(
                Task(
                    method=method,
                    instance=instance,
                    manifest=task_manifest,
                    output=output / "tasks" / method / slug,
                    log=log_root / method / f"{slug}.log",
                )
            )
    return tasks


def _reuse_timeout_statuses(
    tasks: list[Task],
    args: argparse.Namespace,
    status_path: Path,
    instance_manifest: Path,
) -> tuple[list[dict[str, Any]], list[Task]]:
    """Reuse only audited timeouts from an identical preregistered matrix."""

    if not (args.resume and args.reuse_timeouts and status_path.is_file()):
        return [], tasks
    matrix_manifest = status_path.parent / "matrix_manifest.json"
    if not matrix_manifest.is_file():
        return [], tasks
    try:
        metadata = json.loads(matrix_manifest.read_text(encoding="utf-8"))
        previous_manifest = Path(str(metadata["instance_manifest"])).resolve()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return [], tasks
    if (
        metadata.get("adapter_version") != OFFICIAL_ADAPTER_VERSION
        or previous_manifest != instance_manifest.resolve()
    ):
        return [], tasks

    previous = {
        (row.get("method", ""), row.get("graph_fingerprint", "")): row
        for row in _read_csv(status_path)
        if row.get("status") == "TIMEOUT"
    }
    reused: list[dict[str, Any]] = []
    pending: list[Task] = []
    for task in tasks:
        row = previous.get(
            (task.method, task.instance.get("graph_fingerprint", ""))
        )
        try:
            same_timeout = (
                row is not None
                and int(float(row.get("timeout_seconds", "")))
                == _timeout_for(task.method, args)
            )
        except (TypeError, ValueError):
            same_timeout = False
        if not same_timeout:
            pending.append(task)
            continue
        retained = dict(row)
        retained["termination_reason"] = "resumed_preregistered_timeout"
        retained["resumed_timeout"] = 1
        reused.append(retained)
    return reused, pending


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-manifest", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--review-path", default="external/review-main")
    parser.add_argument("--dismantling-python", required=True)
    parser.add_argument("--gdm-python", required=True)
    parser.add_argument("--finder-python", required=True)
    parser.add_argument("--stop-condition", type=int, default=1)
    parser.add_argument("--brute-force-max-n", type=int, default=18)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--general-timeout-seconds", type=int, default=900)
    parser.add_argument("--gdm-timeout-seconds", type=int, default=1800)
    parser.add_argument("--finder-timeout-seconds", type=int, default=1800)
    parser.add_argument("--gdm-slots", type=int, default=1)
    parser.add_argument("--finder-slots", type=int, default=1)
    parser.add_argument("--eigenvector-slots", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-timeouts",
        action="store_true",
        help=(
            "With --resume, retain prior TIMEOUT rows only when the adapter, "
            "frozen manifest, graph fingerprint, method, and timeout agree."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.workers,
        args.gdm_slots,
        args.finder_slots,
        args.eigenvector_slots,
        args.native_threads,
    ) < 1:
        raise SystemExit("Worker and resource-slot counts must be positive.")
    if min(
        args.general_timeout_seconds,
        args.gdm_timeout_seconds,
        args.finder_timeout_seconds,
    ) < 1:
        raise SystemExit("Timeouts must be positive.")
    methods = expand_official_methods(args.methods)
    manifest = Path(args.instance_manifest).resolve()
    output = Path(args.output).resolve()
    log_root = (
        Path(args.log_root).resolve()
        if args.log_root
        else output / "logs"
    )
    project_root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    for marker in (
        "_RUNNING",
        "_SUCCESS",
        "_PARTIAL",
        "_INCOMPLETE",
        "_ZERO_SUCCESS",
    ):
        path = output / marker
        if path.exists():
            path.unlink()
    (output / "_RUNNING").write_text("", encoding="utf-8")

    tasks = _prepare_tasks(manifest, methods, output, log_root)
    status_path = output / "matrix_task_status.csv"
    reused_statuses, pending_tasks = _reuse_timeout_statuses(
        tasks,
        args,
        status_path,
        manifest,
    )
    semaphores: dict[str, threading.BoundedSemaphore] = {}
    for task in tasks:
        group = _resource_group(task.method)
        if not group or group in semaphores:
            continue
        slots = (
            args.gdm_slots
            if group == "gdm"
            else args.finder_slots
            if group == "finder"
            else args.eigenvector_slots
            if group == "eigenvector"
            else 1
        )
        semaphores[group] = threading.BoundedSemaphore(slots)

    (output / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": _timestamp(),
                "adapter_version": OFFICIAL_ADAPTER_VERSION,
                "instance_manifest": str(manifest),
                "methods": list(methods),
                "task_count": len(tasks),
                "workers": args.workers,
                "gdm_slots": args.gdm_slots,
                "finder_slots": args.finder_slots,
                "eigenvector_slots": args.eigenvector_slots,
                "native_threads": args.native_threads,
                "general_timeout_seconds": args.general_timeout_seconds,
                "gdm_timeout_seconds": args.gdm_timeout_seconds,
                "finder_timeout_seconds": args.finder_timeout_seconds,
                "reuse_timeouts": bool(args.reuse_timeouts),
                "reused_timeout_count": len(reused_statuses),
                "resource_groups": {
                    method: _resource_group(method) for method in methods
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _progress(
        f"starting {len(tasks)} isolated tasks with {args.workers} workers; "
        f"reused_timeouts={len(reused_statuses)}"
    )
    statuses: list[dict[str, Any]] = list(reused_statuses)
    statuses.sort(
        key=lambda item: (
            item.get("method", ""),
            item.get("topology", ""),
            item.get("seed", ""),
        )
    )
    _write_csv(status_path, statuses)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_task,
                task,
                args,
                project_root,
                semaphores,
            ): task
            for task in pending_tasks
        }
        for completed, future in enumerate(
            as_completed(futures), start=len(reused_statuses) + 1
        ):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # Preserve every orchestration failure.
                row = {
                    "method": task.method,
                    "graph_fingerprint": task.instance.get(
                        "graph_fingerprint", ""
                    ),
                    "topology": task.instance.get("topology", ""),
                    "seed": task.instance.get("seed", ""),
                    "status": "FAILED",
                    "termination_reason": "orchestrator_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "method_output": str(task.output),
                    "method_log": str(task.log),
                }
            statuses.append(row)
            statuses.sort(
                key=lambda item: (
                    item.get("method", ""),
                    item.get("topology", ""),
                    item.get("seed", ""),
                )
            )
            _write_csv(status_path, statuses)
            _progress(
                f"{completed}/{len(tasks)} {row['status']} "
                f"{row['method']} {row.get('topology', '')}/"
                f"seed{row.get('seed', '')}"
            )

    (output / "_RUNNING").unlink(missing_ok=True)
    failures = [row for row in statuses if row["status"] not in {
        "SUCCESS",
        "NOT_APPLICABLE",
    }]
    not_applicable = [
        row for row in statuses if row["status"] == "NOT_APPLICABLE"
    ]
    success_count = len(statuses) - len(failures) - len(not_applicable)
    if success_count == 0:
        (output / "_ZERO_SUCCESS").write_text("", encoding="utf-8")
    marker = (
        "_INCOMPLETE"
        if failures
        else "_PARTIAL"
        if not_applicable
        else "_SUCCESS"
    )
    (output / marker).write_text("", encoding="utf-8")
    _progress(
        f"finished: success={success_count}, "
        f"not_applicable={len(not_applicable)}, failed={len(failures)}"
    )
    return 4 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
