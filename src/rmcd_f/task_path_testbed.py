"""Source and nontriviality gates for real MPCF task-path testbeds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import isclose
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from .model import ROLES, NodeId, stable_node_key, stable_nodes, validate_graph
from .operational_motif import (
    OperationalMotifSystem,
    audit_path_closed_motif_system,
    build_operational_motif_system,
    solve_exact_motif_cover,
    solve_path_closed_motif_cut,
    task_path_endpoints,
)


TASK_PATH_TESTBED_MAPPING_VERSION = "mpcf-real-testbed-mapping-v1"


class TaskPathTestbedError(ValueError):
    """Raised when a claimed real testbed lacks auditable source semantics."""


@dataclass(frozen=True)
class TaskPathSemanticAudit:
    dataset_id: str
    source_family: str
    mapping_version: str
    mapping_sha256: str
    node_count: int
    edge_count: int
    source_file_count: int
    source_file_sha256_json: str
    node_evidence_count: int
    edge_evidence_count: int
    unique_physical_entity_count: int
    statistically_independent_testbed: bool
    physical_equipment_only: bool
    role_definitions_complete: bool
    directed_interfaces_source_backed: bool
    no_inferred_interfaces: bool
    model_input_basis_complete: bool
    scope_declared: bool
    graph_mapping_identity: bool
    passed: bool
    reason: str


@dataclass(frozen=True)
class TaskPathNontrivialityAudit:
    path_closed: bool
    task_path_count: int
    endpoint_pair_count: int
    alternative_endpoint_pair_count: int
    internally_disjoint_endpoint_pair_count: int
    shared_internal_node_count: int
    maximum_internal_path_incidence: int
    minimum_cut_cost: float | None
    minimum_cut_size: int
    discovered_optimal_cutset_count: int
    first_cut_exclusion_audit_complete: bool
    passed: bool
    reason: str
    optimal_cutsets: tuple[frozenset[NodeId], ...]


@dataclass(frozen=True)
class TaskPathEndpointAudit:
    source_node: NodeId
    target_node: NodeId
    path_count: int
    has_alternative_paths: bool
    has_internally_disjoint_alternatives: bool
    unique_internal_node_count: int
    shared_internal_node_count: int
    minimum_internal_nodes_per_path: int
    maximum_internal_nodes_per_path: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskPathTestbedError(f"{label} must be a JSON object.")
    return value


def _records(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TaskPathTestbedError(f"{label} must be a JSON list.")
    return [_mapping(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


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
            raise TaskPathTestbedError(f"{label}[{index}] has no source evidence.")
        for evidence_index, item in enumerate(evidence):
            source_path = item.get("source_path")
            locator = item.get("locator")
            if source_path not in source_paths:
                raise TaskPathTestbedError(
                    f"{label}[{index}].evidence[{evidence_index}] references an undeclared source."
                )
            if not isinstance(locator, str) or not locator.strip():
                raise TaskPathTestbedError(
                    f"{label}[{index}].evidence[{evidence_index}] requires a locator."
                )
            count += 1
    return count


def build_task_path_testbed_graph(mapping_path: str | Path) -> nx.DiGraph:
    """Build the exact MPCF graph declared by one audited mapping."""

    mapping_file = Path(mapping_path)
    try:
        root = _mapping(
            json.loads(mapping_file.read_text(encoding="utf-8")),
            label="task-path mapping",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskPathTestbedError(f"Could not load mapping {mapping_file}: {exc}") from exc
    graph = nx.DiGraph()
    for index, record in enumerate(_records(root.get("nodes"), label="nodes")):
        required = {
            "id",
            "role",
            "capacity",
            "attack_cost",
            "protect_cost",
            "removable",
        }
        missing = sorted(required - set(record))
        if missing:
            raise TaskPathTestbedError(f"nodes[{index}] is missing fields: {missing}.")
        graph.add_node(
            record["id"],
            role=record["role"],
            capacity=record["capacity"],
            attack_cost=record["attack_cost"],
            protect_cost=record["protect_cost"],
            removable=record["removable"],
            physical_entity_id=record.get("physical_entity_id", ""),
            public_label=record.get("public_label", record["id"]),
        )
    for index, record in enumerate(_records(root.get("edges"), label="edges")):
        if {"source", "target"} - set(record):
            raise TaskPathTestbedError(f"edges[{index}] is missing source or target.")
        graph.add_edge(record["source"], record["target"])
    graph.graph.update(
        {
            "dataset_id": root.get("dataset_id", ""),
            "mapping_version": root.get("mapping_version", ""),
            "source_family": root.get("source_family", ""),
            "mapping_sha256": _sha256(mapping_file),
            "scope_statement": root.get("scope_statement", ""),
        }
    )
    try:
        return validate_graph(graph)
    except (TypeError, ValueError) as exc:
        raise TaskPathTestbedError(
            f"Declared real-testbed graph violates MPCF model semantics: {exc}"
        ) from exc


def audit_task_path_mapping(
    graph: nx.Graph,
    mapping_path: str | Path,
    source_root: str | Path,
) -> TaskPathSemanticAudit:
    """Verify that a real testbed mapping is explicit, physical and sourced."""

    equipment = validate_graph(graph)
    mapping_file = Path(mapping_path)
    source_directory = Path(source_root)
    try:
        root = _mapping(
            json.loads(mapping_file.read_text(encoding="utf-8")),
            label="task-path mapping",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskPathTestbedError(f"Could not load mapping {mapping_file}: {exc}") from exc
    required = {
        "schema_version",
        "dataset_id",
        "source_family",
        "mapping_version",
        "statistically_independent_testbed",
        "independence_basis",
        "scope_statement",
        "mapping_policy",
        "source_files",
        "nodes",
        "edges",
    }
    missing = sorted(required - set(root))
    if missing:
        raise TaskPathTestbedError(f"Task-path mapping is missing fields: {missing}.")
    if root["schema_version"] != TASK_PATH_TESTBED_MAPPING_VERSION:
        raise TaskPathTestbedError(
            f"Unsupported task-path mapping schema {root['schema_version']!r}."
        )
    dataset_id = root["dataset_id"]
    source_family = root["source_family"]
    mapping_version = root["mapping_version"]
    if not all(isinstance(value, str) and value.strip() for value in (dataset_id, source_family, mapping_version)):
        raise TaskPathTestbedError("dataset_id, source_family and mapping_version must be nonempty strings.")
    source_files = _records(root["source_files"], label="source_files")
    if not source_files:
        raise TaskPathTestbedError("At least one frozen source file is required.")
    declared_paths: set[str] = set()
    source_hashes: list[str] = []
    for index, record in enumerate(source_files):
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not relative or relative in declared_paths:
            raise TaskPathTestbedError(f"source_files[{index}].path is missing or duplicate.")
        if not isinstance(expected, str) or len(expected) != 64:
            raise TaskPathTestbedError(f"source_files[{index}].sha256 is invalid.")
        source = source_directory / relative
        if not source.is_file():
            raise TaskPathTestbedError(f"Required source file is missing: {source}.")
        observed = _sha256(source)
        if observed.lower() != expected.lower():
            raise TaskPathTestbedError(
                f"Source checksum mismatch for {relative}: expected={expected}; observed={observed}."
            )
        declared_paths.add(relative)
        source_hashes.append(observed.lower())
    source_paths = frozenset(declared_paths)
    policy = _mapping(root["mapping_policy"], label="mapping_policy")
    role_definitions = _mapping(policy.get("role_definitions"), label="mapping_policy.role_definitions")
    role_complete = set(role_definitions) == set(ROLES) and all(
        isinstance(role_definitions[role], str) and role_definitions[role].strip()
        for role in ROLES
    )
    physical_only = policy.get("node_unit") == "physical_equipment"
    no_inferred = policy.get("no_inferred_interfaces") is True
    edge_semantics = policy.get("edge_semantics")
    if not isinstance(edge_semantics, str) or not edge_semantics.strip():
        raise TaskPathTestbedError("mapping_policy.edge_semantics must be explicit.")
    independence_basis = root["independence_basis"]
    scope_statement = root["scope_statement"]
    if not isinstance(independence_basis, str) or not independence_basis.strip():
        raise TaskPathTestbedError("independence_basis must be a nonempty statement.")
    scope_declared = isinstance(scope_statement, str) and bool(scope_statement.strip())
    nodes = _records(root["nodes"], label="nodes")
    edges = _records(root["edges"], label="edges")
    node_evidence = _evidence_count(nodes, label="nodes", source_paths=source_paths)
    edge_evidence = _evidence_count(edges, label="edges", source_paths=source_paths)
    mapped_nodes: dict[NodeId, str] = {}
    physical_entities: list[str] = []
    for index, record in enumerate(nodes):
        required_node = {
            "id",
            "physical_entity_id",
            "role",
            "capacity",
            "attack_cost",
            "protect_cost",
            "removable",
            "role_basis",
            "capacity_basis",
            "attack_cost_basis",
            "protect_cost_basis",
            "evidence",
        }
        missing_node = sorted(required_node - set(record))
        if missing_node:
            raise TaskPathTestbedError(f"nodes[{index}] is missing fields: {missing_node}.")
        node = record["id"]
        physical_entity = record["physical_entity_id"]
        if node in mapped_nodes:
            raise TaskPathTestbedError(f"Duplicate mapped node ID {node!r}.")
        if not isinstance(physical_entity, str) or not physical_entity.strip():
            raise TaskPathTestbedError(f"nodes[{index}].physical_entity_id is invalid.")
        for basis in (
            "role_basis",
            "capacity_basis",
            "attack_cost_basis",
            "protect_cost_basis",
        ):
            if not isinstance(record[basis], str) or not record[basis].strip():
                raise TaskPathTestbedError(f"nodes[{index}].{basis} must be explicit.")
        mapped_nodes[node] = str(record["role"])
        physical_entities.append(physical_entity)
    if len(set(physical_entities)) != len(physical_entities):
        raise TaskPathTestbedError(
            "One physical entity is split into multiple task-path nodes without an allowed aggregation rule."
        )
    mapped_edges: set[tuple[NodeId, NodeId]] = set()
    for index, record in enumerate(edges):
        if {"source", "target", "evidence"} - set(record):
            raise TaskPathTestbedError(f"edges[{index}] is incomplete.")
        edge = (record["source"], record["target"])
        if edge in mapped_edges:
            raise TaskPathTestbedError(f"Duplicate mapped edge {edge!r}.")
        mapped_edges.add(edge)
    graph_nodes = {node: str(equipment.nodes[node]["role"]) for node in equipment.nodes}
    graph_edges = set(equipment.edges)
    by_id = {record["id"]: record for record in nodes}
    attributes_match = set(by_id) == set(equipment.nodes)
    if attributes_match:
        for node, record in by_id.items():
            attributes = equipment.nodes[node]
            if not (
                attributes["role"] == record["role"]
                and int(attributes["capacity"]) == int(record["capacity"])
                and isclose(
                    float(attributes["attack_cost"]),
                    float(record["attack_cost"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and isclose(
                    float(attributes["protect_cost"]),
                    float(record["protect_cost"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and bool(attributes.get("removable", True)) == bool(record["removable"])
            ):
                attributes_match = False
                break
    graph_identity = (
        mapped_nodes == graph_nodes
        and mapped_edges == graph_edges
        and attributes_match
    )
    independent = root["statistically_independent_testbed"] is True
    directed_backed = edge_evidence >= len(edges) and bool(edges)
    model_input_basis_complete = all(
        all(
            isinstance(record[field], str) and bool(record[field].strip())
            for field in (
                "role_basis",
                "capacity_basis",
                "attack_cost_basis",
                "protect_cost_basis",
            )
        )
        for record in nodes
    )
    conditions = (
        independent,
        physical_only,
        role_complete,
        directed_backed,
        no_inferred,
        model_input_basis_complete,
        scope_declared,
        graph_identity,
    )
    failures = [
        label
        for label, passed in zip(
            (
                "statistical independence",
                "physical-equipment node unit",
                "complete role definitions",
                "source-backed directed interfaces",
                "no inferred interfaces",
                "complete model-input bases",
                "declared study scope",
                "graph/mapping identity",
            ),
            conditions,
        )
        if not passed
    ]
    return TaskPathSemanticAudit(
        dataset_id=str(dataset_id),
        source_family=str(source_family),
        mapping_version=str(mapping_version),
        mapping_sha256=_sha256(mapping_file),
        node_count=equipment.number_of_nodes(),
        edge_count=equipment.number_of_edges(),
        source_file_count=len(source_files),
        source_file_sha256_json=json.dumps(sorted(source_hashes)),
        node_evidence_count=node_evidence,
        edge_evidence_count=edge_evidence,
        unique_physical_entity_count=len(set(physical_entities)),
        statistically_independent_testbed=independent,
        physical_equipment_only=physical_only,
        role_definitions_complete=role_complete,
        directed_interfaces_source_backed=directed_backed,
        no_inferred_interfaces=no_inferred,
        model_input_basis_complete=model_input_basis_complete,
        scope_declared=scope_declared,
        graph_mapping_identity=graph_identity,
        passed=all(conditions),
        reason="all semantic evidence gates passed" if all(conditions) else "; ".join(failures),
    )


def _internally_disjoint_pair_exists(paths: list[tuple[NodeId, ...]]) -> bool:
    return any(
        set(left[1:-1]).isdisjoint(right[1:-1])
        for left, right in combinations(paths, 2)
    )


def task_path_endpoint_audits(
    graph: nx.Graph | OperationalMotifSystem,
) -> tuple[TaskPathEndpointAudit, ...]:
    """Summarize substitute-path evidence for every source-target pair."""

    system = (
        graph
        if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    groups: dict[tuple[NodeId, NodeId], list[tuple[NodeId, ...]]] = defaultdict(list)
    for motif in system.motifs:
        path = motif.ordered_nodes
        groups[(path[0], path[-1])].append(path)

    rows: list[TaskPathEndpointAudit] = []
    for (source, target), paths in sorted(
        groups.items(),
        key=lambda item: (
            stable_node_key(item[0][0]),
            stable_node_key(item[0][1]),
        ),
    ):
        incidence: Counter[NodeId] = Counter()
        internal_sizes: list[int] = []
        for path in paths:
            internal = path[1:-1]
            incidence.update(internal)
            internal_sizes.append(len(internal))
        rows.append(
            TaskPathEndpointAudit(
                source_node=source,
                target_node=target,
                path_count=len(paths),
                has_alternative_paths=len(paths) >= 2,
                has_internally_disjoint_alternatives=(
                    _internally_disjoint_pair_exists(paths)
                ),
                unique_internal_node_count=len(incidence),
                shared_internal_node_count=sum(
                    count >= 2 for count in incidence.values()
                ),
                minimum_internal_nodes_per_path=min(internal_sizes, default=0),
                maximum_internal_nodes_per_path=max(internal_sizes, default=0),
            )
        )
    return tuple(rows)


def _internally_disjoint_path_count(
    system: OperationalMotifSystem,
    source: NodeId,
    target: NodeId,
) -> int:
    """Return exact internally node-disjoint path count with fixed endpoints."""

    auxiliary = nx.DiGraph()
    removable = frozenset(system.removable_nodes)
    path_limit = max(1, len(system.graph.nodes) + 1)
    for node in stable_nodes(system.graph.nodes):
        node_in = ("__task_disjoint_in__", node)
        node_out = ("__task_disjoint_out__", node)
        internal_capacity = (
            path_limit
            if node in {source, target} or node not in removable
            else 1
        )
        auxiliary.add_edge(node_in, node_out, capacity=internal_capacity)
    for left, right in system.graph.edges:
        auxiliary.add_edge(
            ("__task_disjoint_out__", left),
            ("__task_disjoint_in__", right),
            capacity=path_limit,
        )
    return int(
        nx.maximum_flow_value(
            auxiliary,
            ("__task_disjoint_out__", source),
            ("__task_disjoint_in__", target),
            capacity="capacity",
        )
    )


def _distinct_optimal_cutsets(
    system: OperationalMotifSystem,
    *,
    time_limit_per_solve: float,
) -> tuple[tuple[frozenset[NodeId], ...], bool, float | None]:
    first = solve_path_closed_motif_cut(system)
    if not first.optimal or first.objective is None:
        return (), False, None
    optimum = float(first.objective)
    cutsets = {first.selected_set}
    complete = True
    for node in stable_nodes(first.selected_set):
        alternate = solve_exact_motif_cover(
            system,
            force_excluded=(node,),
            objective_upper_bound=optimum,
            time_limit=time_limit_per_solve,
        )
        if alternate.optimal and alternate.objective is not None:
            if isclose(alternate.objective, optimum, rel_tol=0.0, abs_tol=1e-7):
                cutsets.add(alternate.selected_set)
        elif alternate.status != "INFEASIBLE":
            complete = False
    ordered = tuple(
        sorted(
            cutsets,
            key=lambda cut: (len(cut), tuple(stable_node_key(node) for node in stable_nodes(cut))),
        )
    )
    return ordered, complete, optimum


def audit_task_path_nontriviality(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    min_task_paths: int = 4,
    min_disjoint_endpoint_pairs: int = 1,
    min_cut_size: int = 2,
    min_distinct_optimal_cutsets: int = 1,
    cutset_time_limit: float = 60.0,
) -> TaskPathNontrivialityAudit:
    """Audit whether a mapped testbed has nontrivial substitute task paths."""

    if min_task_paths < 1 or min_disjoint_endpoint_pairs < 1:
        raise ValueError("Path and disjoint-pair thresholds must be positive.")
    if min_cut_size < 1 or min_distinct_optimal_cutsets < 1:
        raise ValueError("Cut thresholds must be positive.")
    if cutset_time_limit <= 0.0:
        raise ValueError("cutset_time_limit must be positive.")
    system = graph if isinstance(graph, OperationalMotifSystem) else build_operational_motif_system(graph)
    closure = audit_path_closed_motif_system(system)
    groups: dict[tuple[NodeId, NodeId], list[tuple[NodeId, ...]]] = defaultdict(list)
    internal_incidence: Counter[NodeId] = Counter()
    for motif in system.motifs:
        path = motif.ordered_nodes
        groups[(path[0], path[-1])].append(path)
        internal_incidence.update(path[1:-1])
    alternative_pairs = sum(len(paths) >= 2 for paths in groups.values())
    disjoint_pairs = sum(
        _internally_disjoint_path_count(system, source, target) >= 2
        for source, target in groups
    )
    shared_internal = sum(count >= 2 for count in internal_incidence.values())
    max_incidence = max(internal_incidence.values(), default=0)
    cutsets, cut_search_complete, minimum_cut_cost = _distinct_optimal_cutsets(
        system,
        time_limit_per_solve=cutset_time_limit,
    )
    first_cut_size = len(cutsets[0]) if cutsets else 0
    conditions = (
        closure.path_closed,
        len(system.motifs) >= min_task_paths,
        disjoint_pairs >= min_disjoint_endpoint_pairs,
        first_cut_size >= min_cut_size,
        len(cutsets) >= min_distinct_optimal_cutsets,
    )
    labels = (
        "complete task-path family",
        f"at least {min_task_paths} task paths",
        f"at least {min_disjoint_endpoint_pairs} endpoint pair(s) with internally disjoint alternatives",
        f"minimum cut cardinality at least {min_cut_size}",
        f"at least {min_distinct_optimal_cutsets} distinct optimal cutsets",
    )
    failures = [label for label, passed in zip(labels, conditions) if not passed]
    return TaskPathNontrivialityAudit(
        path_closed=closure.path_closed,
        task_path_count=len(system.motifs),
        endpoint_pair_count=len(groups),
        alternative_endpoint_pair_count=alternative_pairs,
        internally_disjoint_endpoint_pair_count=disjoint_pairs,
        shared_internal_node_count=shared_internal,
        maximum_internal_path_incidence=max_incidence,
        minimum_cut_cost=minimum_cut_cost,
        minimum_cut_size=first_cut_size,
        discovered_optimal_cutset_count=len(cutsets),
        first_cut_exclusion_audit_complete=cut_search_complete,
        passed=all(conditions),
        reason="all nontriviality gates passed" if all(conditions) else "; ".join(failures),
        optimal_cutsets=cutsets,
    )
