#!/usr/bin/env python
"""Export OfficialFINDER_R sequences from the legacy FINDER environment.

This script calls the bundled upstream FINDER class and pretrained checkpoint
directly.  It intentionally avoids importing the NetworkDismantling package,
whose outer wrapper adds a graph-tool dependency that FINDER itself does not
need.  The adapter is syntax-compatible with Python 3.7 and does not import
rmcd_f or the main experiment environment.
"""

from __future__ import print_function

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path


METHOD = "OfficialFINDER_R"
FINDER_REQUIRED_SOURCE_FILES = (
    "PrepareBatchGraph.cpp",
    "PrepareBatchGraph.h",
    "config.cpp",
    "config.h",
    "decrease_strategy.cpp",
    "disjoint_set.cpp",
    "disjoint_set.h",
    "graph.cpp",
    "graph.h",
    "graph_struct.cpp",
    "graph_struct.h",
    "graph_utils.cpp",
    "graph_utils.h",
    "i_env.h",
    "msg_pass.cpp",
    "msg_pass.h",
    "mvc_env.cpp",
    "mvc_env.h",
    "nstep_replay_mem.cpp",
    "nstep_replay_mem.h",
    "nstep_replay_mem_prioritized.cpp",
    "nstep_replay_mem_prioritized.h",
    "utils.cpp",
    "utils.h",
)
ADAPTER_VERSION = "legacy-finder-direct-1.5"
FINDER_MODEL_NAME = "nrange_30_50_iter_78000.ckpt"
FINDER_EXTENSION_MODULES = (
    "PrepareBatchGraph",
    "graph",
    "mvc_env",
    "utils",
    "nstep_replay_mem",
    "nstep_replay_mem_prioritized",
    "graph_struct",
    "FINDER",
)
REPOSITORY = "https://github.com/NetworkDismantling/review"
SOURCE_IDENTITY_EXCLUDED_RELATIVE = frozenset(
    (
        "network_dismantling/decycler/COPYING",
        "network_dismantling/decycler/Makefile",
        "network_dismantling/decycler/gnp.py",
        "network_dismantling/EGND/config.h",
        "network_dismantling/EGND/config_r.h",
        "network_dismantling/FINDER_ND/FINDER.c",
        "network_dismantling/FINDER_ND/PrepareBatchGraph.cpp",
        "network_dismantling/FINDER_ND/graph.cpp",
        "network_dismantling/FINDER_ND/graph_struct.cpp",
        "network_dismantling/FINDER_ND/mvc_env.cpp",
        "network_dismantling/FINDER_ND/nstep_replay_mem.cpp",
        (
            "network_dismantling/FINDER_ND/"
            "nstep_replay_mem_prioritized.cpp"
        ),
        "network_dismantling/FINDER_ND/utils.cpp",
        "network_dismantling/GDM/reinsert.py",
    )
)
SOURCE_IDENTITY_GENERATED_NAMES = frozenset(
    (
        "CI",
        "CMakeCache.txt",
        "EnsembleGND",
        "EnsembleGNDR",
        "GND",
        "coreHD",
        "decycler",
        "dismantler",
        "exploimmun",
        "gmon.out",
        "reinsertion",
        "reverse-greedy",
    )
)
SOURCE_IDENTITY_GENERATED_PARTS = frozenset(
    (
        ".git",
        ".idea",
        "CMakeFiles",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "out",
    )
)
SOURCE_IDENTITY_GENERATED_SUFFIXES = frozenset(
    (".a", ".dll", ".dylib", ".exe", ".o", ".pyc", ".so")
)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_generated_source_artifact(path, root):
    relative = path.relative_to(root)
    relative_text = relative.as_posix()
    if relative_text in SOURCE_IDENTITY_EXCLUDED_RELATIVE:
        return True
    if path.name in SOURCE_IDENTITY_GENERATED_NAMES:
        return True
    if path.suffix in SOURCE_IDENTITY_GENERATED_SUFFIXES:
        return True
    return any(
        part in SOURCE_IDENTITY_GENERATED_PARTS
        or part.endswith(".egg-info")
        for part in relative.parts
    )


def _source_hash(root):
    digest = hashlib.sha256()
    package = root / "network_dismantling"
    for path in sorted(package.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or _is_generated_source_artifact(path, root):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _graph_fingerprint(payload):
    records = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(records, list) or not isinstance(raw_edges, list):
        raise RuntimeError("Instance JSON requires list fields nodes and edges.")

    def token(node):
        return (type(node).__qualname__, repr(node))

    ordered = sorted(records, key=lambda record: _stable_key(record["id"]))
    metadata = payload.get("metadata", {})
    if str(metadata.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        nodes = [
            {
                "id": token(record["id"]),
                "task_stage": str(record.get("task_stage", "")),
                "capacity": int(record["capacity"]),
                "attack_cost": str(float(record["attack_cost"])),
                "protect_cost": str(float(record["protect_cost"])),
                "removable": bool(record.get("removable", True)),
            }
            for record in ordered
        ]
        edges = sorted(
            (
                (token(edge[0]), token(edge[1]))
                for edge in raw_edges
            ),
            key=repr,
        )
        encoded = json.dumps(
            {
                "directed": True,
                "nodes": nodes,
                "edges": edges,
                "source_nodes": [
                    token(node) for node in metadata.get("source_nodes", [])
                ],
                "target_nodes": [
                    token(node) for node in metadata.get("target_nodes", [])
                ],
                "input_semantics": metadata.get("input_semantics"),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    nodes = [
        {
            "id": token(record["id"]),
            "role": str(record["role"]),
            "capacity": int(record["capacity"]),
            "attack_cost": str(float(record["attack_cost"])),
        }
        for record in ordered
    ]
    edges = sorted(
        (
            (token(edge[0]), token(edge[1]))
            for edge in raw_edges
        ),
        key=repr,
    )
    encoded = json.dumps(
        {"directed": True, "nodes": nodes, "edges": edges},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _projection_fingerprint(payload):
    records = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(records, list) or not isinstance(raw_edges, list):
        raise RuntimeError("Instance JSON requires list fields nodes and edges.")

    def token(node):
        type_name = "{}.{}".format(
            type(node).__module__, type(node).__qualname__
        )
        return (type_name, repr(node))

    nodes = [
        token(record["id"])
        for record in sorted(
            records, key=lambda record: _stable_key(record["id"])
        )
    ]
    edges = set()
    for left, right in raw_edges:
        if left == right:
            continue
        edges.add(tuple(sorted((token(left), token(right)), key=repr)))
    encoded = json.dumps(
        {"directed": False, "nodes": nodes, "edges": sorted(edges, key=repr)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_instance(raw, manifest):
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    candidate = (manifest.parent / path).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("Instance file not found: {}".format(raw))


def _stable_key(node):
    return (repr(node), "{}.{}".format(type(node).__module__, type(node).__name__))


def _load_projection(path, fingerprint):
    import networkx as nx

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    nodes = [record["id"] for record in payload["nodes"]]
    if len(nodes) != len(set(map(repr, nodes))):
        raise RuntimeError("Instance contains duplicate node identities.")
    nodes = sorted(nodes, key=_stable_key)
    node_to_static = {node: index for index, node in enumerate(nodes)}

    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.graph["filename"] = fingerprint

    edges = set()
    for left, right in payload["edges"]:
        if left == right:
            continue
        first, second = sorted((node_to_static[left], node_to_static[right]))
        edges.add((first, second))
    graph.add_edges_from(sorted(edges))
    return graph, dict((index, node) for node, index in node_to_static.items())


def _ensure_builds(root):
    finder = root / "network_dismantling" / "FINDER_ND"
    source_directory = finder / "src" / "lib"
    missing = tuple(
        name
        for name in FINDER_REQUIRED_SOURCE_FILES
        if not (source_directory / name).is_file()
    )
    if missing:
        raise RuntimeError(
            "Complete FINDER_ND/src/lib is required; missing: {0}.".format(
                ", ".join(missing)
            )
        )
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    missing_extensions = tuple(
        name
        for name in FINDER_EXTENSION_MODULES
        if not (finder / (name + extension_suffix)).is_file()
    )
    if missing_extensions:
        # Upstream FINDER_ND is a standalone extension directory.  The review
        # integration adds __init__.py, but its setup.py still declares bare
        # module names.  Cython 0.29 then treats the source as package code and
        # does not load the adjacent same-name .pxd declarations.  Build from
        # a clean standalone staging tree, matching the upstream layout, and
        # copy back only the ABI-specific extension modules.
        with tempfile.TemporaryDirectory(
            prefix="rmcd-finder-build-"
        ) as temporary:
            build_root = Path(temporary) / "FINDER_ND"
            build_root.mkdir(parents=True)
            shutil.copy2(
                str(finder / "setup.py"),
                str(build_root / "setup.py"),
            )
            for pattern in ("*.pyx", "*.pxd"):
                for source in finder.glob(pattern):
                    shutil.copy2(str(source), str(build_root / source.name))
            shutil.copytree(
                str(finder / "src"),
                str(build_root / "src"),
            )
            subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "build_ext",
                    "--inplace",
                    "--force",
                ],
                cwd=str(build_root),
                check=True,
            )
            missing_from_build = tuple(
                name
                for name in FINDER_EXTENSION_MODULES
                if not (build_root / (name + extension_suffix)).is_file()
            )
            if missing_from_build:
                raise RuntimeError(
                    (
                        "Standalone FINDER build did not produce modules: "
                        "{0}."
                    ).format(", ".join(missing_from_build))
                )
            for name in FINDER_EXTENSION_MODULES:
                generated = build_root / (name + extension_suffix)
                shutil.copy2(str(generated), str(finder / generated.name))

    still_missing = tuple(
        name
        for name in FINDER_EXTENSION_MODULES
        if not (finder / (name + extension_suffix)).is_file()
    )
    if still_missing:
        raise RuntimeError(
            "FINDER extension build did not produce modules for this Python "
            "ABI: {0}.".format(", ".join(still_missing))
        )


def _coerce_rows(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = value.replace(",", " ").split()
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def _static_ids(run):
    if isinstance(run, dict):
        payload = _coerce_rows(run.get("removals"))
    elif hasattr(run, "columns") and "removals" in run.columns:
        payload = []
        for value in _coerce_rows(run["removals"]):
            candidate = _coerce_rows(value)
            if candidate:
                payload = candidate
                break
    else:
        payload = _coerce_rows(run)
    if not payload:
        return ()
    scalar = all(isinstance(item, (int, float, str)) for item in payload)
    raw_ids = payload if scalar else []
    if not scalar:
        for position, row in enumerate(payload, 1):
            if hasattr(row, "tolist"):
                row = row.tolist()
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                raise RuntimeError(
                    "Malformed removal row {}: {!r}".format(position, row)
                )
            raw_ids.append(row[1])
    ordered = []
    seen = set()
    for position, raw in enumerate(raw_ids, 1):
        if isinstance(raw, bool):
            raise RuntimeError("Boolean node ID at position {}".format(position))
        value = float(raw)
        if not value.is_integer():
            raise RuntimeError("Non-integral node ID {!r}".format(raw))
        static_id = int(value)
        if static_id < 0 or static_id in seen:
            raise RuntimeError(
                "Invalid/repeated node ID {} at position {}".format(
                    static_id, position
                )
            )
        ordered.append(static_id)
        seen.add(static_id)
    return tuple(ordered)


def _run(graph, review_root, stop_condition, fingerprint):
    del stop_condition, fingerprint
    finder_root = review_root / "network_dismantling" / "FINDER_ND"
    finder_path = str(finder_root)
    if finder_path not in sys.path:
        sys.path.insert(0, finder_path)

    from FINDER import FINDER

    # FINDER uses TensorFlow 1.x's process-global default graph.  Without a
    # reset, sequential instances append variables and checkpoint restoration
    # fails on the second graph (for example Variable_14/Variable_100).
    try:
        import tensorflow as tensorflow_module
    except ImportError:
        tensorflow_module = None
    if tensorflow_module is not None:
        reset_default_graph = getattr(
            tensorflow_module,
            "reset_default_graph",
            None,
        )
        if reset_default_graph is None:
            compat = getattr(tensorflow_module, "compat", None)
            reset_default_graph = getattr(
                getattr(compat, "v1", None),
                "reset_default_graph",
                None,
            )
        if reset_default_graph is None:
            raise RuntimeError(
                "FINDER TensorFlow runtime does not expose reset_default_graph."
            )
        reset_default_graph()

    model_path = finder_root / "models" / FINDER_MODEL_NAME
    if not (model_path.with_suffix(model_path.suffix + ".index")).is_file():
        raise FileNotFoundError(
            "Official FINDER checkpoint is incomplete: {0}".format(model_path)
        )

    dqn = FINDER()
    try:
        dqn.LoadModel(str(model_path))
        solution, _ = dqn.EvaluateRealData(g=graph.copy(), stepRatio=0.01)
        # This exactly follows the review wrapper defaults.  strategyID=0
        # completes the official ordering without an additional heuristic.
        solution, _, _ = dqn.EvaluateSol(
            g=graph.copy(),
            solution=solution,
            strategyID=0,
            reInsertStep=0.001,
        )
        return solution
    finally:
        session = getattr(dqn, "session", None)
        if session is not None:
            session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-manifest", required=True)
    parser.add_argument("--review-path", required=True)
    parser.add_argument("--stop-condition", type=int, default=1)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    manifest = Path(args.instance_manifest).resolve()
    review_root = Path(args.review_path).resolve()
    if not (review_root / "network_dismantling" / "__init__.py").is_file():
        raise SystemExit("Invalid review source: {}".format(review_root))
    if args.stop_condition < 1:
        raise SystemExit("--stop-condition must be at least 1")
    sys.path.insert(0, str(review_root))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for marker in ("_SUCCESS", "_INCOMPLETE"):
        path = output / marker
        if path.is_file():
            path.unlink()

    source_identity = _source_hash(review_root)
    manifest_payload = {
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "instance_manifest": str(manifest),
        "instance_manifest_sha256": _file_hash(manifest),
        "method": METHOD,
        "adapter_version": ADAPTER_VERSION,
        "stop_condition": args.stop_condition,
        "review_repository": REPOSITORY,
        "resolved_review_root": str(review_root),
        "source_identity_kind": "source_tree_sha256",
        "source_identity": source_identity,
        "effective_reinsertion": True,
        "upstream_metadata_conflict": True,
    }
    (output / "official_sequence_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "official_method_catalog.csv",
        [
            {
                "method": METHOD,
                "selected_for_run": 1,
                "method_group": "finder",
                "dependency_environment": "finder",
                "upstream_module": "network_dismantling.FINDER_ND.FINDER",
                "upstream_function": "FINDER.EvaluateRealData",
                "adapter_kind": "legacy_direct_official_export",
                "experimental_upstream": 0,
                "derived_output": 0,
                "effective_reinsertion": 1,
                "upstream_metadata_conflict": 1,
                "official_repository": REPOSITORY,
                "adapter_version": ADAPTER_VERSION,
            }
        ],
    )
    _ensure_builds(review_root)
    instances = _read_csv(manifest)
    if args.limit_instances:
        instances = instances[: args.limit_instances]
    rows = []
    for instance in instances:
        instance_file = _resolve_instance(instance["instance_file"], manifest)
        fingerprint = instance["graph_fingerprint"]
        started = time.time()
        try:
            with instance_file.open("r", encoding="utf-8") as handle:
                instance_payload = json.load(handle)
            actual_fingerprint = _graph_fingerprint(instance_payload)
            if actual_fingerprint != fingerprint:
                raise RuntimeError(
                    "Instance graph fingerprint mismatch: expected {0}, "
                    "computed {1}.".format(fingerprint, actual_fingerprint)
                )
            graph, mapping = _load_projection(instance_file, fingerprint)
            projection_fingerprint = _projection_fingerprint(instance_payload)
            run = _run(
                graph,
                review_root,
                min(args.stop_condition, max(1, int(graph.number_of_nodes()))),
                fingerprint,
            )
            ids = _static_ids(run)
            unknown = [value for value in ids if value not in mapping]
            if unknown:
                raise RuntimeError(
                    "Out-of-range static node IDs: {!r}".format(unknown)
                )
            sequence = [mapping[value] for value in ids]
            if not sequence:
                raise RuntimeError("FINDER returned an empty removal sequence.")
            status = "SUCCESS"
            error_type = ""
            error = ""
        except Exception as exc:
            sequence = []
            status = "FAILED"
            error_type = type(exc).__name__
            error = str(exc)
        rows.append(
            {
                "study_id": instance.get("study_id", ""),
                "topology": instance.get("topology", ""),
                "seed": instance.get("seed", ""),
                "instance_file": str(instance_file),
                "graph_fingerprint": fingerprint,
                "projection_fingerprint": projection_fingerprint,
                "method": METHOD,
                "method_group": "finder",
                "dependency_environment": "finder",
                "upstream_module": "network_dismantling.FINDER_ND.FINDER",
                "upstream_function": "FINDER.EvaluateRealData",
                "adapter_kind": "legacy_direct_official_export",
                "experimental_upstream": 0,
                "derived_output": 0,
                "effective_reinsertion": 1,
                "upstream_metadata_conflict": 1,
                "status": status,
                "sequence_json": json.dumps(sequence, ensure_ascii=False),
                "sequence_length": len(sequence),
                "runtime_seconds": time.time() - started,
                "projection": (
                    "simple_undirected_physical_equipment_projection"
                ),
                "sequence_semantics": "official_structural_removal_order",
                "source_identity_kind": "source_tree_sha256",
                "source_identity": source_identity,
                "compatibility_patches_json": json.dumps(
                    [
                        "direct_networkx_export_without_graph_tool_wrapper",
                        "standalone_upstream_cython_build_staging",
                        "tensorflow_default_graph_reset_per_instance",
                    ]
                ),
                "adapter_version": ADAPTER_VERSION,
                "stop_condition": args.stop_condition,
                "brute_force_max_n": 18,
                "algorithm_seed": instance.get("seed", ""),
                "determinism_status": "upstream_randomness_not_fully_controlled",
                "effective_parameters_json": json.dumps(
                    {
                        "effective_reinsertion": True,
                        "uses_upstream_function_defaults": True,
                    },
                    sort_keys=True,
                ),
                "review_root": str(review_root),
                "review_repository": REPOSITORY,
                "error_type": error_type,
                "error": error,
            }
        )
        _write_csv(output / "official_sequences.csv", rows)

    failures = [row for row in rows if row["status"] != "SUCCESS"]
    _write_csv(
        output / "official_method_status.csv",
        [
            dict(
                (key, value)
                for key, value in row.items()
                if key != "sequence_json"
            )
            for row in rows
        ],
    )
    marker = "_INCOMPLETE" if failures else "_SUCCESS"
    (output / marker).write_text("", encoding="utf-8")
    return 4 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
