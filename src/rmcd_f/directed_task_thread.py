"""Source-gated directed task-thread inputs for MPCF.

This module is an input adapter, not a new optimization model.  It converts one
source-backed task thread into the same path-closed system consumed by PathCut,
MPCF-Exact, and MPCF-CG.  Fixed S-C-L-E graphs remain supported separately as
the original four-stage special case.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from .model import NodeId, stable_node_key, stable_nodes
from .operational_motif import (
    OperationalMotifSystem,
    build_directed_task_path_system,
)


DIRECTED_TASK_THREAD_MAPPING_VERSION_V1 = "mpcf-directed-task-thread-mapping-v1"
DIRECTED_TASK_THREAD_MAPPING_VERSION = "mpcf-directed-task-thread-mapping-v2"
SUPPORTED_DIRECTED_TASK_THREAD_MAPPING_VERSIONS = frozenset(
    {DIRECTED_TASK_THREAD_MAPPING_VERSION_V1, DIRECTED_TASK_THREAD_MAPPING_VERSION}
)
ALLOWED_NODE_KINDS = frozenset({"implemented_component", "task_boundary"})
ALLOWED_COST_POLICIES = frozenset({"source_backed", "unit_cost_sensitivity"})


class DirectedTaskThreadError(ValueError):
    """Raised when a proposed task-thread mapping is not auditable."""


@dataclass(frozen=True)
class DirectedTaskThreadSemanticAudit:
    dataset_id: str
    source_family: str
    mapping_version: str
    schema_version: str
    mapping_sha256: str
    task_id: str
    node_count: int
    edge_count: int
    implemented_component_count: int
    boundary_node_count: int
    source_file_count: int
    source_file_sha256_json: str
    node_evidence_count: int
    edge_evidence_count: int
    endpoint_evidence_count: int
    declared_path_evidence_count: int
    enumerated_task_path_count: int
    declared_legal_path_count: int
    matched_task_path_count: int
    task_mode_id: str
    operating_condition: str
    declared_path_family_sha256: str
    enumerated_path_family_sha256: str
    unique_declared_path_ids: bool
    unique_declared_path_sequences: bool
    one_task_mode: bool
    one_operating_condition: bool
    declared_paths_structurally_valid: bool
    legal_paths_source_backed: bool
    legal_path_family_exact: bool
    undeclared_enumerated_path_count: int
    declared_missing_path_count: int
    statistically_independent_testbed: bool
    unique_component_instances: bool
    boundary_nodes_fixed: bool
    boundary_nodes_exact: bool
    boundary_orientation_valid: bool
    endpoints_explicit_and_sourced: bool
    directed_interfaces_source_backed: bool
    no_inferred_interfaces: bool
    model_input_basis_complete: bool
    scope_declared: bool
    graph_mapping_identity: bool
    cost_policy: str
    cost_policy_consistent: bool
    cost_claim_scope: str
    passed: bool
    reason: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_source_path(source_root: Path, relative: str) -> Path:
    """Resolve one declared source without allowing it to escape its root."""

    candidate = Path(relative)
    if (
        not relative.strip()
        or candidate.is_absolute()
        or "\\" in relative
        or ".." in candidate.parts
    ):
        raise DirectedTaskThreadError(
            f"Source path must stay inside source_root: {relative!r}."
        )
    root = source_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DirectedTaskThreadError(
            f"Source path must stay inside source_root: {relative!r}."
        ) from exc
    return resolved


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectedTaskThreadError(f"{label} must be a JSON object.")
    return value


def _records(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise DirectedTaskThreadError(f"{label} must be a JSON list.")
    return [
        _mapping(record, label=f"{label}[{index}]")
        for index, record in enumerate(value)
    ]


def _load_mapping(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    mapping_path = Path(path)
    try:
        root = _mapping(
            json.loads(mapping_path.read_text(encoding="utf-8")),
            label="directed task-thread mapping",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectedTaskThreadError(
            f"Could not load mapping {mapping_path}: {exc}"
        ) from exc
    if root.get("schema_version") not in SUPPORTED_DIRECTED_TASK_THREAD_MAPPING_VERSIONS:
        raise DirectedTaskThreadError(
            f"Unsupported directed task-thread schema {root.get('schema_version')!r}."
        )
    return mapping_path, root


def _evidence_count(
    records: list[Mapping[str, Any]],
    *,
    label: str,
    source_paths: frozenset[str],
) -> int:
    count = 0
    for index, record in enumerate(records):
        evidence = _records(record.get("evidence"), label=f"{label}[{index}].evidence")
        if not evidence:
            raise DirectedTaskThreadError(f"{label}[{index}] has no source evidence.")
        for evidence_index, item in enumerate(evidence):
            source_path = item.get("source_path")
            locator = item.get("locator")
            if source_path not in source_paths:
                raise DirectedTaskThreadError(
                    f"{label}[{index}].evidence[{evidence_index}] references "
                    "an undeclared source file."
                )
            if not isinstance(locator, str) or not locator.strip():
                raise DirectedTaskThreadError(
                    f"{label}[{index}].evidence[{evidence_index}] requires a locator."
                )
            count += 1
    return count


def _path_family_audit(
    root: Mapping[str, Any],
    system: OperationalMotifSystem,
    *,
    source_paths: frozenset[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare source-declared legal paths with every path enabled by the graph."""

    schema_version = str(root.get("schema_version", ""))
    raw_paths = root.get("declared_legal_paths")
    if schema_version == DIRECTED_TASK_THREAD_MAPPING_VERSION_V1:
        declared: list[Mapping[str, Any]] = []
    else:
        declared = _records(raw_paths, label="declared_legal_paths")
        if not declared:
            raise DirectedTaskThreadError(
                "Schema v2 requires at least one declared legal task path."
            )

    task = _mapping(root.get("task"), label="task")
    sources = set(task.get("source_nodes", ()))
    targets = set(task.get("target_nodes", ()))
    graph_nodes = set(system.graph.nodes)
    graph_edges = set(system.graph.edges)
    path_ids: list[str] = []
    sequences: list[tuple[NodeId, ...]] = []
    modes: list[str] = []
    conditions: list[str] = []
    structural_flags: list[bool] = []
    declared_rows: list[dict[str, Any]] = []
    evidence_count = 0
    if declared:
        evidence_count = _evidence_count(
            declared,
            label="declared_legal_paths",
            source_paths=source_paths,
        )
    for index, record in enumerate(declared):
        path_id = record.get("path_id")
        task_mode = record.get("task_mode_id")
        operating_condition = record.get("operating_condition")
        ordered = record.get("ordered_node_ids")
        if not isinstance(path_id, str) or not path_id.strip():
            raise DirectedTaskThreadError(
                f"declared_legal_paths[{index}].path_id must be nonempty."
            )
        if not isinstance(task_mode, str) or not task_mode.strip():
            raise DirectedTaskThreadError(
                f"declared_legal_paths[{index}].task_mode_id must be nonempty."
            )
        if not isinstance(operating_condition, str) or not operating_condition.strip():
            raise DirectedTaskThreadError(
                "declared_legal_paths"
                f"[{index}].operating_condition must be nonempty."
            )
        if not isinstance(ordered, list) or len(ordered) < 2:
            raise DirectedTaskThreadError(
                f"declared_legal_paths[{index}].ordered_node_ids must contain "
                "at least two nodes."
            )
        sequence = tuple(ordered)
        known_nodes = all(node in graph_nodes for node in sequence)
        simple_path = len(sequence) == len(set(sequence))
        correct_boundaries = sequence[0] in sources and sequence[-1] in targets
        edges_exist = known_nodes and all(
            (left, right) in graph_edges
            for left, right in zip(sequence, sequence[1:])
        )
        structurally_valid = bool(
            known_nodes and simple_path and correct_boundaries and edges_exist
        )
        path_ids.append(path_id)
        sequences.append(sequence)
        modes.append(task_mode)
        conditions.append(operating_condition)
        structural_flags.append(structurally_valid)
        declared_rows.append(
            {
                "path_status": "DECLARED",
                "path_id": path_id,
                "task_mode_id": task_mode,
                "operating_condition": operating_condition,
                "ordered_node_ids_json": json.dumps(
                    list(sequence), ensure_ascii=True
                ),
                "path_length_nodes": len(sequence),
                "structurally_valid": int(structurally_valid),
                "source_evidence_count": len(record.get("evidence", ())),
            }
        )

    enumerated_sequences = {
        tuple(motif.ordered_nodes) for motif in system.motifs
    }
    declared_sequences = set(sequences)
    undeclared = enumerated_sequences - declared_sequences
    missing = declared_sequences - enumerated_sequences
    unique_ids = bool(path_ids) and len(path_ids) == len(set(path_ids))
    unique_sequences = bool(sequences) and len(sequences) == len(declared_sequences)
    one_mode = bool(modes) and len(set(modes)) == 1
    one_condition = bool(conditions) and len(set(conditions)) == 1
    structurally_valid = bool(structural_flags) and all(structural_flags)
    source_backed = bool(declared) and evidence_count >= len(declared)
    exact = bool(
        unique_ids
        and unique_sequences
        and one_mode
        and one_condition
        and structurally_valid
        and source_backed
        and not undeclared
        and not missing
    )

    rows: list[dict[str, Any]] = []
    for row, sequence in zip(declared_rows, sequences):
        status = (
            "DECLARED_MATCHED"
            if sequence in enumerated_sequences
            else "DECLARED_MISSING_FROM_GRAPH"
        )
        rows.append({**row, "path_status": status})
    for sequence in sorted(
        undeclared,
        key=lambda path: tuple(stable_node_key(node) for node in path),
    ):
        rows.append(
            {
                "path_status": "UNDECLARED_ENUMERATED",
                "path_id": "",
                "task_mode_id": "",
                "operating_condition": "",
                "ordered_node_ids_json": json.dumps(
                    list(sequence), ensure_ascii=True
                ),
                "path_length_nodes": len(sequence),
                "structurally_valid": 1,
                "source_evidence_count": 0,
            }
        )
    def family_sha256(paths: set[tuple[NodeId, ...]]) -> str:
        ordered_paths = sorted(
            paths,
            key=lambda path: tuple(stable_node_key(node) for node in path),
        )
        payload = json.dumps(
            [list(path) for path in ordered_paths],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return (
        {
            "declared_path_evidence_count": evidence_count,
            "enumerated_task_path_count": len(enumerated_sequences),
            "declared_legal_path_count": len(declared),
            "matched_task_path_count": len(
                declared_sequences & enumerated_sequences
            ),
            "task_mode_id": modes[0] if one_mode else "",
            "operating_condition": conditions[0] if one_condition else "",
            "declared_path_family_sha256": family_sha256(declared_sequences),
            "enumerated_path_family_sha256": family_sha256(enumerated_sequences),
            "unique_declared_path_ids": unique_ids,
            "unique_declared_path_sequences": unique_sequences,
            "one_task_mode": one_mode,
            "one_operating_condition": one_condition,
            "declared_paths_structurally_valid": structurally_valid,
            "legal_paths_source_backed": source_backed,
            "legal_path_family_exact": exact,
            "undeclared_enumerated_path_count": len(undeclared),
            "declared_missing_path_count": len(missing),
        },
        rows,
    )


def directed_task_path_family_rows(
    system: OperationalMotifSystem,
    mapping_path: str | Path,
) -> list[dict[str, Any]]:
    """Return the auditable declared/enumerated path-family comparison rows."""

    _, root = _load_mapping(mapping_path)
    source_files = _records(root.get("source_files"), label="source_files")
    source_paths = frozenset(
        str(record.get("path")) for record in source_files if record.get("path")
    )
    _, rows = _path_family_audit(root, system, source_paths=source_paths)
    return rows


def build_directed_task_thread_system(
    mapping_path: str | Path,
) -> OperationalMotifSystem:
    """Build the exact directed graph and complete task-path family in a mapping."""

    mapping_file, root = _load_mapping(mapping_path)
    nodes = _records(root.get("nodes"), label="nodes")
    edges = _records(root.get("edges"), label="edges")
    task = _mapping(root.get("task"), label="task")
    sources = task.get("source_nodes")
    targets = task.get("target_nodes")
    if not isinstance(sources, list) or not sources:
        raise DirectedTaskThreadError("task.source_nodes must be a nonempty list.")
    if not isinstance(targets, list) or not targets:
        raise DirectedTaskThreadError("task.target_nodes must be a nonempty list.")
    if len(sources) != len(set(sources)):
        raise DirectedTaskThreadError("task.source_nodes contains duplicates.")
    if len(targets) != len(set(targets)):
        raise DirectedTaskThreadError("task.target_nodes contains duplicates.")

    graph = nx.DiGraph()
    for index, record in enumerate(nodes):
        required = {
            "id",
            "component_instance_id",
            "node_kind",
            "task_stage",
            "capacity",
            "attack_cost",
            "protect_cost",
            "removable",
        }
        missing = sorted(required - set(record))
        if missing:
            raise DirectedTaskThreadError(
                f"nodes[{index}] is missing fields: {missing}."
            )
        graph.add_node(
            record["id"],
            component_instance_id=record["component_instance_id"],
            node_kind=record["node_kind"],
            task_stage=record["task_stage"],
            capacity=record["capacity"],
            attack_cost=record["attack_cost"],
            protect_cost=record["protect_cost"],
            removable=record["removable"],
            public_label=record.get("public_label", record["id"]),
        )
    for index, record in enumerate(edges):
        if {"source", "target"} - set(record):
            raise DirectedTaskThreadError(
                f"edges[{index}] is missing source or target."
            )
        if record["source"] not in graph or record["target"] not in graph:
            raise DirectedTaskThreadError(
                f"edges[{index}] references an unknown endpoint."
            )
        graph.add_edge(record["source"], record["target"])
    graph.graph.update(
        {
            "dataset_id": root.get("dataset_id", ""),
            "mapping_version": root.get("mapping_version", ""),
            "source_family": root.get("source_family", ""),
            "task_id": task.get("task_id", ""),
            "mapping_sha256": _sha256(mapping_file),
            "scope_statement": root.get("scope_statement", ""),
            "input_semantics": root.get("schema_version", ""),
        }
    )
    try:
        return build_directed_task_path_system(
            graph,
            source_nodes=sources,
            target_nodes=targets,
            library_version=(
                "source-backed-directed-task-thread-v2"
                if root.get("schema_version") == DIRECTED_TASK_THREAD_MAPPING_VERSION
                else "source-backed-directed-task-thread-v1"
            ),
        )
    except (TypeError, ValueError, nx.NetworkXException) as exc:
        raise DirectedTaskThreadError(
            f"Declared task thread violates MPCF path semantics: {exc}"
        ) from exc


def audit_directed_task_thread_mapping(
    system: OperationalMotifSystem,
    mapping_path: str | Path,
    source_root: str | Path,
) -> DirectedTaskThreadSemanticAudit:
    """Verify source identity, task boundaries, interfaces, costs, and node units."""

    mapping_file, root = _load_mapping(mapping_path)
    source_directory = Path(source_root)
    required = {
        "dataset_id",
        "source_family",
        "mapping_version",
        "statistically_independent_testbed",
        "independence_basis",
        "scope_statement",
        "mapping_policy",
        "source_files",
        "task",
        "nodes",
        "edges",
    }
    missing = sorted(required - set(root))
    if missing:
        raise DirectedTaskThreadError(f"Mapping is missing fields: {missing}.")
    dataset_id = root["dataset_id"]
    source_family = root["source_family"]
    mapping_version = root["mapping_version"]
    if not all(
        isinstance(value, str) and value.strip()
        for value in (dataset_id, source_family, mapping_version)
    ):
        raise DirectedTaskThreadError(
            "dataset_id, source_family and mapping_version must be nonempty strings."
        )

    source_files = _records(root["source_files"], label="source_files")
    if not source_files:
        raise DirectedTaskThreadError("At least one frozen source file is required.")
    declared_paths: set[str] = set()
    source_hashes: list[str] = []
    for index, record in enumerate(source_files):
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not relative or relative in declared_paths:
            raise DirectedTaskThreadError(
                f"source_files[{index}].path is missing or duplicate."
            )
        if not isinstance(expected, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected
        ):
            raise DirectedTaskThreadError(f"source_files[{index}].sha256 is invalid.")
        source = _safe_source_path(source_directory, relative)
        if not source.is_file():
            raise DirectedTaskThreadError(f"Required source file is missing: {source}.")
        observed = _sha256(source)
        if observed.lower() != expected.lower():
            raise DirectedTaskThreadError(
                f"Source checksum mismatch for {relative}: "
                f"expected={expected}; observed={observed}."
            )
        declared_paths.add(relative)
        source_hashes.append(observed.lower())
    source_paths = frozenset(declared_paths)

    nodes = _records(root["nodes"], label="nodes")
    edges = _records(root["edges"], label="edges")
    task = _mapping(root["task"], label="task")
    task_id = task.get("task_id")
    completion = task.get("completion_statement")
    if not isinstance(task_id, str) or not task_id.strip():
        raise DirectedTaskThreadError("task.task_id must be a nonempty string.")
    if not isinstance(completion, str) or not completion.strip():
        raise DirectedTaskThreadError(
            "task.completion_statement must be a nonempty statement."
        )
    endpoint_evidence = _evidence_count(
        [task], label="task", source_paths=source_paths
    )
    node_evidence = _evidence_count(
        nodes, label="nodes", source_paths=source_paths
    )
    edge_evidence = _evidence_count(
        edges, label="edges", source_paths=source_paths
    )
    path_family, _ = _path_family_audit(
        root,
        system,
        source_paths=source_paths,
    )

    policy = _mapping(root["mapping_policy"], label="mapping_policy")
    no_inferred = policy.get("no_inferred_interfaces") is True
    node_unit_ok = policy.get("node_unit") == "implemented_component_with_boundary_terminals"
    edge_semantics = policy.get("edge_semantics")
    if not isinstance(edge_semantics, str) or not edge_semantics.strip():
        raise DirectedTaskThreadError("mapping_policy.edge_semantics must be explicit.")
    cost_policy = policy.get("cost_policy")
    if cost_policy not in ALLOWED_COST_POLICIES:
        raise DirectedTaskThreadError(
            f"mapping_policy.cost_policy must be one of {sorted(ALLOWED_COST_POLICIES)}."
        )
    cost_claim_scope = (
        "source-backed heterogeneous attack and protection costs"
        if cost_policy == "source_backed"
        else "unit-cost structural sensitivity only"
    )

    mapped_nodes: dict[NodeId, Mapping[str, Any]] = {}
    component_ids: list[str] = []
    boundary_count = 0
    component_count = 0
    boundary_fixed = True
    bases_complete = True
    observed_attack_costs: list[float] = []
    observed_protect_costs: list[float] = []
    for index, record in enumerate(nodes):
        required_node = {
            "id",
            "component_instance_id",
            "node_kind",
            "task_stage",
            "capacity",
            "attack_cost",
            "protect_cost",
            "removable",
            "task_stage_basis",
            "attack_cost_basis",
            "protect_cost_basis",
            "evidence",
        }
        missing_node = sorted(required_node - set(record))
        if missing_node:
            raise DirectedTaskThreadError(
                f"nodes[{index}] is missing fields: {missing_node}."
            )
        node = record["id"]
        if node in mapped_nodes:
            raise DirectedTaskThreadError(f"Duplicate mapped node ID {node!r}.")
        kind = record["node_kind"]
        if kind not in ALLOWED_NODE_KINDS:
            raise DirectedTaskThreadError(
                f"nodes[{index}].node_kind must be one of {sorted(ALLOWED_NODE_KINDS)}."
            )
        instance = record["component_instance_id"]
        if not isinstance(instance, str) or not instance.strip():
            raise DirectedTaskThreadError(
                f"nodes[{index}].component_instance_id is invalid."
            )
        for field in ("task_stage_basis", "attack_cost_basis", "protect_cost_basis"):
            if not isinstance(record[field], str) or not record[field].strip():
                bases_complete = False
        if kind == "implemented_component":
            component_count += 1
            component_ids.append(instance)
            observed_attack_costs.append(float(record["attack_cost"]))
            observed_protect_costs.append(float(record["protect_cost"]))
        else:
            boundary_count += 1
            boundary_fixed = boundary_fixed and not bool(record["removable"])
        mapped_nodes[node] = record
    unique_components = len(component_ids) == len(set(component_ids))
    cost_policy_consistent = (
        cost_policy == "source_backed"
        or (
            bool(observed_attack_costs)
            and all(isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in observed_attack_costs)
            and all(isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in observed_protect_costs)
        )
    )

    mapped_edges: set[tuple[NodeId, NodeId]] = set()
    for index, record in enumerate(edges):
        if {"source", "target", "evidence"} - set(record):
            raise DirectedTaskThreadError(f"edges[{index}] is incomplete.")
        edge = (record["source"], record["target"])
        if edge in mapped_edges:
            raise DirectedTaskThreadError(f"Duplicate mapped edge {edge!r}.")
        mapped_edges.add(edge)

    raw_sources = task.get("source_nodes")
    raw_targets = task.get("target_nodes")
    sources = tuple(raw_sources) if isinstance(raw_sources, list) else ()
    targets = tuple(raw_targets) if isinstance(raw_targets, list) else ()
    endpoint_set = set(sources) | set(targets)
    boundary_nodes = {
        node
        for node, record in mapped_nodes.items()
        if record["node_kind"] == "task_boundary"
    }
    endpoints_explicit = (
        bool(sources)
        and bool(targets)
        and len(sources) == len(set(sources))
        and len(targets) == len(set(targets))
        and not (set(sources) & set(targets))
        and endpoint_set <= set(mapped_nodes)
        and all(mapped_nodes[node]["node_kind"] == "task_boundary" for node in endpoint_set)
        and endpoint_evidence >= 1
    )
    boundary_exact = endpoints_explicit and boundary_nodes == endpoint_set
    boundary_orientation = (
        boundary_exact
        and all(system.graph.in_degree(node) == 0 for node in sources)
        and all(system.graph.out_degree(node) == 0 for node in targets)
    )

    graph_nodes = set(system.graph.nodes)
    attributes_match = set(mapped_nodes) == graph_nodes
    if attributes_match:
        for node, record in mapped_nodes.items():
            data = system.graph.nodes[node]
            if not (
                data.get("component_instance_id") == record["component_instance_id"]
                and data.get("node_kind") == record["node_kind"]
                and data.get("task_stage") == record["task_stage"]
                and int(data["capacity"]) == int(record["capacity"])
                and isclose(float(data["attack_cost"]), float(record["attack_cost"]), abs_tol=1e-12)
                and isclose(float(data["protect_cost"]), float(record["protect_cost"]), abs_tol=1e-12)
                and bool(data.get("removable", True)) == bool(record["removable"])
            ):
                attributes_match = False
                break
    graph_identity = attributes_match and mapped_edges == set(system.graph.edges)
    independent = root["statistically_independent_testbed"] is True
    independence_basis = root["independence_basis"]
    scope_statement = root["scope_statement"]
    if not isinstance(independence_basis, str) or not independence_basis.strip():
        raise DirectedTaskThreadError("independence_basis must be explicit.")
    scope_declared = isinstance(scope_statement, str) and bool(scope_statement.strip())
    directed_backed = bool(edges) and edge_evidence >= len(edges)
    conditions = (
        independent,
        node_unit_ok,
        unique_components,
        boundary_fixed,
        boundary_exact,
        boundary_orientation,
        endpoints_explicit,
        directed_backed,
        no_inferred,
        bases_complete,
        cost_policy_consistent,
        scope_declared,
        graph_identity,
        path_family["unique_declared_path_ids"],
        path_family["unique_declared_path_sequences"],
        path_family["one_task_mode"],
        path_family["one_operating_condition"],
        path_family["declared_paths_structurally_valid"],
        path_family["legal_paths_source_backed"],
        path_family["legal_path_family_exact"],
    )
    labels = (
        "statistical independence",
        "implemented-component node unit with fixed task boundaries",
        "unique component instances",
        "non-removable task-boundary nodes",
        "task-boundary nodes exactly equal declared endpoints",
        "source boundaries have no incoming edge and target boundaries no outgoing edge",
        "explicit source-backed task endpoints",
        "source-backed directed interfaces",
        "no inferred interfaces",
        "complete task-stage and cost bases",
        "declared cost policy consistent with numeric costs",
        "declared study scope",
        "graph/mapping identity",
        "unique declared legal-path IDs",
        "unique declared legal-path sequences",
        "one bounded task mode",
        "one bounded operating condition",
        "structurally valid declared legal paths",
        "source-backed legal paths",
        "declared legal-path family exactly equals graph-enumerated family",
    )
    failures = [label for label, passed in zip(labels, conditions) if not passed]
    return DirectedTaskThreadSemanticAudit(
        dataset_id=str(dataset_id),
        source_family=str(source_family),
        mapping_version=str(mapping_version),
        schema_version=str(root.get("schema_version", "")),
        mapping_sha256=_sha256(mapping_file),
        task_id=task_id,
        node_count=system.graph.number_of_nodes(),
        edge_count=system.graph.number_of_edges(),
        implemented_component_count=component_count,
        boundary_node_count=boundary_count,
        source_file_count=len(source_files),
        source_file_sha256_json=json.dumps(sorted(source_hashes)),
        node_evidence_count=node_evidence,
        edge_evidence_count=edge_evidence,
        endpoint_evidence_count=endpoint_evidence,
        declared_path_evidence_count=path_family["declared_path_evidence_count"],
        enumerated_task_path_count=path_family["enumerated_task_path_count"],
        declared_legal_path_count=path_family["declared_legal_path_count"],
        matched_task_path_count=path_family["matched_task_path_count"],
        task_mode_id=path_family["task_mode_id"],
        operating_condition=path_family["operating_condition"],
        declared_path_family_sha256=path_family[
            "declared_path_family_sha256"
        ],
        enumerated_path_family_sha256=path_family[
            "enumerated_path_family_sha256"
        ],
        unique_declared_path_ids=path_family["unique_declared_path_ids"],
        unique_declared_path_sequences=path_family["unique_declared_path_sequences"],
        one_task_mode=path_family["one_task_mode"],
        one_operating_condition=path_family["one_operating_condition"],
        declared_paths_structurally_valid=path_family[
            "declared_paths_structurally_valid"
        ],
        legal_paths_source_backed=path_family["legal_paths_source_backed"],
        legal_path_family_exact=path_family["legal_path_family_exact"],
        undeclared_enumerated_path_count=path_family[
            "undeclared_enumerated_path_count"
        ],
        declared_missing_path_count=path_family["declared_missing_path_count"],
        statistically_independent_testbed=independent,
        unique_component_instances=unique_components,
        boundary_nodes_fixed=boundary_fixed,
        boundary_nodes_exact=boundary_exact,
        boundary_orientation_valid=boundary_orientation,
        endpoints_explicit_and_sourced=endpoints_explicit,
        directed_interfaces_source_backed=directed_backed,
        no_inferred_interfaces=no_inferred,
        model_input_basis_complete=bases_complete,
        scope_declared=scope_declared,
        graph_mapping_identity=graph_identity,
        cost_policy=str(cost_policy),
        cost_policy_consistent=cost_policy_consistent,
        cost_claim_scope=cost_claim_scope,
        passed=all(conditions),
        reason="all directed task-thread semantic gates passed" if all(conditions) else "; ".join(failures),
    )


def directed_task_thread_json(system: OperationalMotifSystem) -> dict[str, Any]:
    """Serialize a generic path-closed system without legacy role validation."""

    return {
        "nodes": [
            {"id": node, **dict(system.graph.nodes[node])}
            for node in stable_nodes(system.graph.nodes)
        ],
        "edges": [
            [left, right]
            for left, right in sorted(
                system.graph.edges,
                key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1])),
            )
        ],
        "metadata": {
            **dict(system.graph.graph),
            "source_nodes": list(system.source_nodes),
            "target_nodes": list(system.target_nodes),
            "library_version": system.library_version,
        },
    }


def save_directed_task_thread_system(
    system: OperationalMotifSystem, path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            directed_task_thread_json(system),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
