#!/usr/bin/env python3
"""Generate canonical official NetworkDismantling removal sequences.

Run this script inside the dependency environment required by each official
method group.  Its CSV output is environment-neutral and can be merged later
by ``run_critical_set_study.py`` for one common ``Omega_R < K`` evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rmcd_f.cli import load_graph
from rmcd_f.official_review import (
    OFFICIAL_ADAPTER_VERSION,
    GND_FAMILY_ADAPTER_VERSION,
    OFFICIAL_METHOD_SPECS,
    OfficialMethodNotApplicableError,
    REVIEW_REPOSITORY_URL,
    expand_official_methods,
    graph_fingerprint,
    review_source_identity,
    run_official_review_sequence,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _progress(message: str) -> None:
    print(f"[{_timestamp()}] [official-review] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _method_catalog(selected: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "selected_for_run": int(method in selected),
            "method_group": spec.group,
            "dependency_environment": spec.environment,
            "upstream_module": spec.module,
            "upstream_function": spec.function,
            "adapter_kind": spec.kind,
            "dependency_method": spec.dependency or "",
            "required_modules": ",".join(spec.required_modules),
            "experimental_upstream": int(spec.experimental_upstream),
            "derived_output": int(spec.derived_output),
            "effective_reinsertion": int(
                method.endswith("R") or method == "OfficialFINDER_R"
            ),
            "upstream_metadata_conflict": int(
                method == "OfficialFINDER_R"
            ),
            "small_scale_only": int(spec.kind == "bruteforce_target"),
            "official_repository": REVIEW_REPOSITORY_URL,
            "adapter_version": OFFICIAL_ADAPTER_VERSION,
        }
        for method, spec in OFFICIAL_METHOD_SPECS.items()
    ]


def _resolve_instance(raw: str, manifest: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    if candidate.is_file():
        return candidate.resolve()
    relative = (manifest.parent / candidate).resolve()
    if relative.is_file():
        return relative
    raise FileNotFoundError(f"Instance file from manifest was not found: {raw}")


def _sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status_row(
    instance: dict[str, str],
    *,
    instance_file: Path,
    fingerprint: str,
    method: str,
    status: str,
    sequence: tuple[Any, ...] = (),
    runtime_seconds: float | str = "",
    review_root: str = "",
    projection: str = "simple_undirected_physical_equipment_projection",
    sequence_semantics: str = "official_structural_removal_order",
    source_identity_kind: str = "",
    source_identity: str = "",
    compatibility_patches: tuple[str, ...] = (),
    projection_fingerprint: str = "",
    sequence_sha256: str = "",
    dependency_method: str = "",
    dependency_sequence_sha256: str = "",
    stop_condition: int = 1,
    verified_final_lcc_size: int | str = "",
    brute_force_max_n: int = 18,
    applicability_code: str = "",
    applicability_detail: str = "",
    error_type: str = "",
    error: str = "",
) -> dict[str, Any]:
    spec = OFFICIAL_METHOD_SPECS[method]
    if spec.kind == "heuristic_static":
        sequence_semantics = "upstream_static_sorter_python_active_lcc_order"
    elif spec.kind == "heuristic_dynamic":
        sequence_semantics = "upstream_sorter_active_graph_dynamic_order"
    effective_reinsertion = (
        method.endswith("R") or method == "OfficialFINDER_R"
    )
    determinism_status = (
        "seeded_numpy_state_restored"
        if method == "OfficialRandomStatic"
        else "upstream_randomness_not_fully_controlled"
        if method
        in {
            "OfficialEGND",
            "OfficialGDM",
            "OfficialGDMR",
            "OfficialCoreGDM",
            "OfficialFINDER_R",
        }
        else "deterministic_or_upstream_not_declared"
    )
    effective_parameters = {
        "stop_condition": stop_condition,
        "algorithm_seed": instance.get("seed", ""),
        "brute_force_max_n": brute_force_max_n,
        "effective_reinsertion": effective_reinsertion,
        "uses_upstream_function_defaults": not spec.kind.startswith("heuristic_"),
        "sequence_lcc_target_verified": bool(
            status == "SUCCESS" and verified_final_lcc_size != ""
        ),
        "applicability_code": applicability_code,
        "applicability_detail": applicability_detail,
    }
    if spec.kind == "heuristic_static":
        effective_parameters.update(
            {
                "uses_upstream_sorter_defaults": True,
                "adapter_execution": "initial_scores_then_python_active_lcc",
                "removed_nodes_excluded_from_lcc": True,
                "tie_break": "stable_initial_vertex_order",
            }
        )
    elif spec.kind == "heuristic_dynamic":
        effective_parameters.update(
            {
                "uses_upstream_sorter_defaults": True,
                "adapter_execution": "rescore_surviving_graph_view_each_step",
                "removed_nodes_excluded_from_rescoring": True,
                "removed_nodes_excluded_from_lcc": True,
                "tie_break": "first_active_vertex_order",
            }
        )
    return {
        "study_id": instance.get("study_id", ""),
        "topology": instance.get("topology", ""),
        "seed": instance.get("seed", ""),
        "instance_file": str(instance_file),
        "graph_fingerprint": fingerprint,
        "projection_fingerprint": projection_fingerprint,
        "method": method,
        "method_group": spec.group,
        "dependency_environment": spec.environment,
        "upstream_module": spec.module,
        "upstream_function": spec.function,
        "adapter_kind": spec.kind,
        "experimental_upstream": int(spec.experimental_upstream),
        "derived_output": int(spec.derived_output),
        "effective_reinsertion": int(effective_reinsertion),
        "upstream_metadata_conflict": int(method == "OfficialFINDER_R"),
        "status": status,
        "sequence_json": json.dumps(sequence, ensure_ascii=False),
        "sequence_length": len(sequence),
        "runtime_seconds": runtime_seconds,
        "projection": projection,
        "sequence_semantics": sequence_semantics,
        "source_identity_kind": source_identity_kind,
        "source_identity": source_identity,
        "compatibility_patches_json": json.dumps(compatibility_patches),
        "adapter_version": OFFICIAL_ADAPTER_VERSION,
        "gnd_family_adapter_version": (
            GND_FAMILY_ADAPTER_VERSION
            if method in {"OfficialGND", "OfficialGNDR"}
            else ""
        ),
        "sequence_sha256": sequence_sha256,
        "dependency_method": dependency_method,
        "dependency_sequence_sha256": dependency_sequence_sha256,
        "stop_condition": stop_condition,
        "verified_final_lcc_size": verified_final_lcc_size,
        "brute_force_max_n": brute_force_max_n,
        "algorithm_seed": instance.get("seed", ""),
        "determinism_status": determinism_status,
        "effective_parameters_json": json.dumps(
            effective_parameters,
            sort_keys=True,
        ),
        "review_root": review_root,
        "review_repository": REVIEW_REPOSITORY_URL,
        "applicability_code": applicability_code,
        "applicability_detail": applicability_detail,
        "error_type": error_type,
        "error": error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-manifest", required=True)
    parser.add_argument(
        "--methods",
        default="core",
        help=(
            "Comma-separated review method names or groups: derived, "
            "core, ml, finder, heuristic, stable, experimental, "
            "all_scalable, all."
        ),
    )
    parser.add_argument("--review-path")
    parser.add_argument("--review-zip")
    parser.add_argument("--extraction-root")
    parser.add_argument("--stop-condition", type=int, default=1)
    parser.add_argument(
        "--brute-force-max-n",
        type=int,
        default=18,
        help=(
            "Safety ceiling for the official exhaustive Brute Force method; "
            "ignored by other methods."
        ),
    )
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.instance_manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Instance manifest does not exist: {manifest_path}")
    if args.stop_condition < 1:
        raise SystemExit("--stop-condition must be at least 1")
    if args.brute_force_max_n < 4:
        raise SystemExit("--brute-force-max-n must be at least 4")
    if args.limit_instances is not None and args.limit_instances <= 0:
        raise SystemExit("--limit-instances must be positive")
    try:
        methods = expand_official_methods(args.methods)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    instances = _read_csv(manifest_path)
    if args.limit_instances is not None:
        instances = instances[: args.limit_instances]
    if not instances:
        raise SystemExit("Instance manifest contains no rows.")

    archive = Path(args.review_zip).resolve() if args.review_zip else None
    review_root, source_identity_kind, source_identity = review_source_identity(
        review_path=args.review_path,
        zip_path=args.review_zip,
        extraction_root=args.extraction_root,
        progress=_progress,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for marker_name in ("_SUCCESS", "_INCOMPLETE"):
        marker_path = output / marker_name
        if marker_path.is_file():
            marker_path.unlink()
    _write_csv(
        output / "official_method_catalog.csv",
        _method_catalog(methods),
    )
    sequence_path = output / "official_sequences.csv"
    manifest_output = output / "official_sequence_manifest.json"
    previous: dict[tuple[str, str], dict[str, str]] = {}
    if args.resume and sequence_path.is_file():
        if not manifest_output.is_file():
            raise SystemExit(
                "--resume requires the previous official_sequence_manifest.json"
            )
        previous_manifest = json.loads(
            manifest_output.read_text(encoding="utf-8")
        )
        current_identity = {
            "instance_manifest_sha256": _sha256(manifest_path),
            "adapter_version": OFFICIAL_ADAPTER_VERSION,
            "stop_condition": args.stop_condition,
            "brute_force_max_n": args.brute_force_max_n,
            "source_identity_kind": source_identity_kind,
            "source_identity": source_identity,
        }
        mismatches = {
            key: (previous_manifest.get(key), value)
            for key, value in current_identity.items()
            if previous_manifest.get(key) != value
        }
        if mismatches:
            raise SystemExit(
                "--resume configuration/source identity does not match the "
                f"previous run: {mismatches}"
            )
        for row in _read_csv(sequence_path):
            if row.get("status") != "SUCCESS":
                continue
            expected_seed = row.get("seed", "")
            reusable = (
                row.get("adapter_version") == OFFICIAL_ADAPTER_VERSION
                and row.get("stop_condition") == str(args.stop_condition)
                and row.get("brute_force_max_n")
                == str(args.brute_force_max_n)
                and row.get("algorithm_seed") == expected_seed
                and row.get("source_identity_kind")
                == source_identity_kind
                and row.get("source_identity") == source_identity
            )
            if reusable:
                previous[(row["graph_fingerprint"], row["method"])] = row
    run_manifest = {
        "created_utc": _timestamp(),
        "instance_manifest": str(manifest_path),
        "instance_manifest_sha256": _sha256(manifest_path),
        "methods": methods,
        "adapter_version": OFFICIAL_ADAPTER_VERSION,
        "gnd_family_adapter_version": GND_FAMILY_ADAPTER_VERSION,
        "stop_condition": args.stop_condition,
        "brute_force_max_n": args.brute_force_max_n,
        "review_repository": REVIEW_REPOSITORY_URL,
        "review_path": args.review_path,
        "resolved_review_root": str(review_root),
        "source_identity_kind": source_identity_kind,
        "source_identity": source_identity,
        "review_zip": str(archive) if archive is not None else "",
        "review_zip_sha256": _sha256(archive),
        "extraction_root": args.extraction_root,
        "sequence_semantics": (
            "method-specific sequence semantics are recorded in each status row; "
            "no RMCD threshold is applied in this stage"
        ),
    }
    _write_json(manifest_output, run_manifest)

    rows: list[dict[str, Any]] = []
    total = len(instances) * len(methods)
    completed = 0
    for instance in instances:
        instance_file = _resolve_instance(instance["instance_file"], manifest_path)
        graph = load_graph(instance_file)
        fingerprint = graph_fingerprint(graph)
        from rmcd_f.provenance import official_projection_fingerprint
        projection_fingerprint = official_projection_fingerprint(graph)
        for method in methods:
            completed += 1
            cached = previous.get((fingerprint, method))
            if cached is not None:
                rows.append(dict(cached))
                _progress(f"resume {completed}/{total}: {method} on {instance_file.name}")
                continue
            _progress(f"run {completed}/{total}: {method} on {instance_file.name}")
            try:
                result = run_official_review_sequence(
                    graph,
                    method,
                    review_path=args.review_path,
                    zip_path=args.review_zip,
                    extraction_root=args.extraction_root,
                    stop_condition=args.stop_condition,
                    brute_force_max_nodes=args.brute_force_max_n,
                    seed=int(instance.get("seed", 0) or 0),
                    progress=_progress,
                )
                rows.append(
                    _status_row(
                        instance,
                        instance_file=instance_file,
                        fingerprint=fingerprint,
                        method=method,
                        status="SUCCESS",
                        sequence=result.sequence,
                        runtime_seconds=result.runtime_seconds,
                        review_root=result.review_root,
                        projection=result.projection,
                        sequence_semantics=result.sequence_semantics,
                        source_identity_kind=result.source_identity_kind,
                        source_identity=result.source_identity,
                        compatibility_patches=result.compatibility_patches,
                        projection_fingerprint=result.projection_fingerprint,
                        sequence_sha256=result.sequence_sha256,
                        dependency_method=result.dependency_method,
                        dependency_sequence_sha256=(
                            result.dependency_sequence_sha256
                        ),
                        stop_condition=result.stop_condition,
                        verified_final_lcc_size=(
                            result.verified_final_lcc_size
                        ),
                        brute_force_max_n=args.brute_force_max_n,
                    )
                )
            except OfficialMethodNotApplicableError as exc:
                rows.append(
                    _status_row(
                        instance,
                        instance_file=instance_file,
                        fingerprint=fingerprint,
                        method=method,
                        status="NOT_APPLICABLE",
                        review_root=str(review_root),
                        source_identity_kind=source_identity_kind,
                        source_identity=source_identity,
                        projection_fingerprint=projection_fingerprint,
                        stop_condition=args.stop_condition,
                        brute_force_max_n=args.brute_force_max_n,
                        applicability_code=exc.code,
                        applicability_detail=exc.detail,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )
                _progress(
                    f"not applicable: {method} on {instance_file.name}: "
                    f"{exc.code}: {exc.detail}"
                )
            except Exception as exc:
                rows.append(
                    _status_row(
                        instance,
                        instance_file=instance_file,
                        fingerprint=fingerprint,
                        method=method,
                        status="FAILED",
                        review_root=str(review_root),
                        source_identity_kind=source_identity_kind,
                        source_identity=source_identity,
                        projection_fingerprint=projection_fingerprint,
                        stop_condition=args.stop_condition,
                        brute_force_max_n=args.brute_force_max_n,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )
                _progress(
                    f"failed {method} on {instance_file.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
            _write_csv(sequence_path, rows)

    rows.sort(
        key=lambda row: (
            row["topology"],
            int(row["seed"]) if str(row["seed"]).isdigit() else str(row["seed"]),
            row["method"],
        )
    )
    _write_csv(sequence_path, rows)
    _write_csv(
        output / "official_method_status.csv",
        (
            {key: value for key, value in row.items() if key != "sequence_json"}
            for row in rows
        ),
    )
    failures = [row for row in rows if row["status"] == "FAILED"]
    not_applicable = [
        row for row in rows if row["status"] == "NOT_APPLICABLE"
    ]
    for marker_name in ("_SUCCESS", "_PARTIAL", "_INCOMPLETE"):
        marker_path = output / marker_name
        if marker_path.exists():
            marker_path.unlink()
    marker = (
        "_INCOMPLETE"
        if failures
        else "_PARTIAL"
        if not_applicable
        else "_SUCCESS"
    )
    (output / marker).write_text("", encoding="utf-8")
    _progress(
        f"finished with {len(rows) - len(failures) - len(not_applicable)} "
        f"success, {len(not_applicable)} not applicable, and "
        f"{len(failures)} failed at {output}"
    )
    return 4 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
