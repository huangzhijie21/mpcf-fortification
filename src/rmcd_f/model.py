"""Frozen data model and validation for RMCD-F equipment networks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite
from numbers import Integral, Real
from typing import Any, Hashable, Iterable, Mapping, TypeAlias

import networkx as nx


NodeId: TypeAlias = Hashable

ROLES: tuple[str, ...] = ("S", "C", "L", "E")
ROLE_INDEX: dict[str, int] = {role: index for index, role in enumerate(ROLES)}
LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {("S", "C"), ("C", "L"), ("L", "E")}
)
MAX_EXACT_CAPACITY_SUM = 2**53 - 2


class GraphValidationError(ValueError):
    """Raised when an input graph violates the frozen four-role semantics."""


class SolveStatus(str, Enum):
    """Status labels that do not conflate feasibility with optimality."""

    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    TIME_LIMIT = "TIME_LIMIT"
    LIMIT_REACHED = "LIMIT_REACHED"
    DUAL_CLOSED_GAP = "DUAL_CLOSED_GAP"
    INFEASIBLE = "INFEASIBLE"
    UNBREAKABLE = "UNBREAKABLE"
    UNBOUNDED = "UNBOUNDED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SolverInfo:
    """Solver metadata preserved for every mathematical-programming result."""

    status: SolveStatus
    optimal: bool
    scipy_status: int | None = None
    message: str = ""
    objective: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    mip_gap: float | None = None
    runtime_seconds: float = 0.0


def stable_node_key(node: NodeId) -> tuple[str, str]:
    """Return a deterministic key without requiring mutually comparable IDs."""

    return (repr(node), f"{type(node).__module__}.{type(node).__qualname__}")


def stable_nodes(nodes: Iterable[NodeId]) -> tuple[NodeId, ...]:
    """Sort arbitrary hashable node identifiers deterministically by ``repr``."""

    return tuple(sorted(nodes, key=stable_node_key))


def _require_hashable(node: Any) -> None:
    try:
        hash(node)
    except TypeError as exc:
        raise GraphValidationError(f"Node ID is not hashable: {node!r}") from exc


def _require_capacity(node: NodeId, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GraphValidationError(
            f"Node {node!r} capacity must be a non-negative integer; got {value!r}."
        )
    capacity = int(value)
    if capacity < 0:
        raise GraphValidationError(
            f"Node {node!r} capacity must be non-negative; got {capacity}."
        )
    return capacity


def _require_positive_cost(node: NodeId, name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GraphValidationError(
            f"Node {node!r} {name} must be a finite positive number; got {value!r}."
        )
    cost = float(value)
    if not isfinite(cost) or cost <= 0.0:
        raise GraphValidationError(
            f"Node {node!r} {name} must be finite and positive; got {value!r}."
        )
    return cost


def _node_attributes(node: NodeId, data: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        name
        for name in ("role", "capacity", "attack_cost", "protect_cost")
        if name not in data
    ]
    if missing:
        raise GraphValidationError(
            f"Node {node!r} is missing required attributes: {', '.join(missing)}."
        )

    role = data["role"]
    if role not in ROLES:
        raise GraphValidationError(
            f"Node {node!r} role must be exactly one of {ROLES}; got {role!r}."
        )

    normalized = dict(data)
    normalized["role"] = role
    normalized["capacity"] = _require_capacity(node, data["capacity"])
    normalized["attack_cost"] = _require_positive_cost(
        node, "attack_cost", data["attack_cost"]
    )
    normalized["protect_cost"] = _require_positive_cost(
        node, "protect_cost", data["protect_cost"]
    )
    return normalized


def validate_graph(graph: nx.Graph) -> nx.DiGraph:
    """Validate and normalize an equipment graph.

    Parallel duplicates are merged, while self-loops, same-role edges, skipped
    layers, and reverse edges raise :class:`GraphValidationError`.
    """

    if not isinstance(graph, nx.Graph):
        raise GraphValidationError("Input must be a NetworkX graph.")
    if not graph.is_directed():
        raise GraphValidationError("RMCD-F requires a directed equipment graph.")
    if graph.number_of_nodes() == 0:
        raise GraphValidationError("Equipment graph must contain at least one node.")

    normalized = nx.DiGraph()
    for node, data in sorted(graph.nodes(data=True), key=lambda item: stable_node_key(item[0])):
        _require_hashable(node)
        normalized.add_node(node, **_node_attributes(node, data))

    present_roles = {data["role"] for _, data in normalized.nodes(data=True)}
    missing_roles = [role for role in ROLES if role not in present_roles]
    if missing_roles:
        raise GraphValidationError(
            "Equipment graph must contain all four roles; missing: "
            + ", ".join(missing_roles)
            + "."
        )

    total_capacity = sum(int(data["capacity"]) for _, data in normalized.nodes(data=True))
    if total_capacity > MAX_EXACT_CAPACITY_SUM:
        raise GraphValidationError(
            "Total equipment capacity exceeds the exact double-precision MILP "
            f"limit {MAX_EXACT_CAPACITY_SUM}; scale capacity units before solving."
        )
    attack_costs = [
        float(data["attack_cost"]) for _, data in normalized.nodes(data=True)
    ]
    protect_costs = [
        float(data["protect_cost"]) for _, data in normalized.nodes(data=True)
    ]
    try:
        total_attack_cost = fsum(attack_costs)
        total_protect_cost = fsum(protect_costs)
        fortification_big_m = fsum((total_attack_cost, min(attack_costs)))
    except OverflowError as exc:
        raise GraphValidationError(
            "Aggregate attack/protection costs overflow double precision; rescale costs."
        ) from exc
    if not all(
        isfinite(value)
        for value in (total_attack_cost, total_protect_cost, fortification_big_m)
    ):
        raise GraphValidationError(
            "Aggregate attack/protection costs must remain finite; rescale costs."
        )

    raw_edges = sorted(
        graph.edges(), key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1]))
    )
    for source, target in raw_edges:
        if source == target:
            raise GraphValidationError(f"Self-loop is not allowed: {source!r} -> {target!r}.")
        source_role = normalized.nodes[source]["role"]
        target_role = normalized.nodes[target]["role"]
        if (source_role, target_role) not in LEGAL_TRANSITIONS:
            raise GraphValidationError(
                "Illegal role edge "
                f"{source!r}({source_role}) -> {target!r}({target_role}); "
                "only S->C, C->L, and L->E are allowed."
            )
        normalized.add_edge(source, target)

    normalized.graph.update(graph.graph)
    normalized.graph["rmcd_roles"] = ROLES
    return normalized


def ensure_node_subset(
    graph: nx.DiGraph, nodes: Iterable[NodeId], *, label: str
) -> frozenset[NodeId]:
    """Validate a node subset and return it as a frozenset."""

    result = frozenset(nodes)
    unknown = stable_nodes(node for node in result if node not in graph)
    if unknown:
        raise ValueError(f"Unknown {label} nodes: {unknown!r}.")
    return result


def attack_cost(graph: nx.DiGraph, nodes: Iterable[NodeId]) -> float:
    """Return the exogenous attack cost of a node set."""

    return float(sum(float(graph.nodes[node]["attack_cost"]) for node in nodes))


def protection_cost(graph: nx.DiGraph, nodes: Iterable[NodeId]) -> float:
    """Return the exogenous protection cost of a node set."""

    return float(sum(float(graph.nodes[node]["protect_cost"]) for node in nodes))
