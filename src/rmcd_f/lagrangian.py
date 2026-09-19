"""Lagrangian role-cut bounds for RMCD without overstating optimality."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Iterable, Sequence

import networkx as nx

from .flow import FlowNode, SINK, SOURCE, motif_capacity
from .model import (
    NodeId,
    attack_cost,
    ensure_node_subset,
    stable_node_key,
    stable_nodes,
    validate_graph,
)


@dataclass(frozen=True)
class CutBlockerResult:
    """Exact minimum-cost capacity removal inside one fixed role cut."""

    cut_nodes: tuple[NodeId, ...]
    attack_set: frozenset[NodeId]
    attack_cost: float | None
    original_cut_capacity: int
    residual_cut_capacity: int | None
    feasible: bool


@dataclass(frozen=True)
class LagrangianCutBound:
    """One valid lower bound and its weighted minimum-cut witness."""

    lambda_value: float
    lower_bound: float
    weighted_cut_value: float
    cut_nodes: tuple[NodeId, ...]
    node_weights: tuple[tuple[NodeId, float], ...]
    u_inf_cut: float
    candidate_upper: CutBlockerResult


@dataclass(frozen=True)
class LagrangianSearchResult:
    """A finite scan of valid bounds; it is not a maximization certificate."""

    bounds: tuple[LagrangianCutBound, ...]
    best_lower_bound: float
    best_upper_bound: float | None
    best_attack_set: frozenset[NodeId]
    numerically_closed: bool
    certified_optimal: bool
    exact_lambda_maximized: bool = False


def _validate_k(graph: nx.DiGraph, K: int) -> int:
    if isinstance(K, bool) or not isinstance(K, Integral):
        raise ValueError("K must be an integer.")
    threshold = int(K)
    initial_capacity = motif_capacity(graph).value
    if not 1 <= threshold <= initial_capacity:
        raise ValueError(
            f"K must satisfy 1 <= K <= Omega_0={initial_capacity}; got {threshold}."
        )
    return threshold


def solve_cut_blocker(
    graph: nx.Graph,
    cut_nodes: Iterable[NodeId],
    K: int,
    *,
    forbidden_attack: Iterable[NodeId] = (),
) -> CutBlockerResult:
    """Solve the integer capacity-cover problem on a fixed node cut by DP."""

    equipment = validate_graph(graph)
    threshold = _validate_k(equipment, K)
    cut = stable_nodes(ensure_node_subset(equipment, cut_nodes, label="cut"))
    forbidden = ensure_node_subset(
        equipment, forbidden_attack, label="forbidden_attack"
    )
    total_capacity = sum(int(equipment.nodes[node]["capacity"]) for node in cut)
    required_removal = max(0, total_capacity - (threshold - 1))
    if required_removal == 0:
        return CutBlockerResult(
            cut, frozenset(), 0.0, total_capacity, total_capacity, True
        )

    # State key is removed capacity capped at the requirement. Values are
    # minimum cost and a deterministic node tuple attaining that cost.
    states: dict[int, tuple[float, tuple[NodeId, ...]]] = {0: (0.0, ())}
    tolerance = 1e-12
    for node in cut:
        if node in forbidden:
            continue
        capacity = int(equipment.nodes[node]["capacity"])
        if capacity <= 0:
            continue
        cost = float(equipment.nodes[node]["attack_cost"])
        updated = dict(states)
        for removed, (current_cost, selected) in states.items():
            new_removed = min(required_removal, removed + capacity)
            candidate = (current_cost + cost, (*selected, node))
            incumbent = updated.get(new_removed)
            if incumbent is None or candidate[0] < incumbent[0] - tolerance:
                updated[new_removed] = candidate
            elif incumbent is not None and abs(candidate[0] - incumbent[0]) <= tolerance:
                candidate_key = tuple(map(repr, candidate[1]))
                incumbent_key = tuple(map(repr, incumbent[1]))
                if candidate_key < incumbent_key:
                    updated[new_removed] = candidate
        states = updated

    if required_removal not in states:
        return CutBlockerResult(cut, frozenset(), None, total_capacity, None, False)
    cost, selected = states[required_removal]
    selected_set = frozenset(selected)
    removed_capacity = sum(
        int(equipment.nodes[node]["capacity"]) for node in selected_set
    )
    return CutBlockerResult(
        cut,
        selected_set,
        float(cost),
        total_capacity,
        total_capacity - removed_capacity,
        True,
    )


def lagrangian_cut_lower_bound(
    graph: nx.Graph,
    K: int,
    lambda_value: float,
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
) -> LagrangianCutBound:
    """Compute one valid Lagrangian lower bound using its own cut infinity."""

    equipment = validate_graph(graph)
    if not isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda_value must be finite and non-negative.")
    threshold = _validate_k(equipment, K)
    protected_set = ensure_node_subset(equipment, protected, label="protected")
    excluded_set = ensure_node_subset(equipment, excluded, label="excluded")
    forbidden = protected_set | excluded_set

    weights: dict[NodeId, float] = {}
    for node in stable_nodes(equipment.nodes):
        capacity_penalty = float(lambda_value) * float(
            equipment.nodes[node]["capacity"]
        )
        if node in forbidden:
            weights[node] = capacity_penalty
        else:
            weights[node] = min(
                float(equipment.nodes[node]["attack_cost"]), capacity_penalty
            )
    u_inf_cut = 1.0 + sum(weights.values())

    weighted = nx.DiGraph()
    weighted.add_nodes_from((SOURCE, SINK))
    node_in: dict[NodeId, FlowNode] = {}
    node_out: dict[NodeId, FlowNode] = {}
    for node in stable_nodes(equipment.nodes):
        incoming = FlowNode("in", node)
        outgoing = FlowNode("out", node)
        node_in[node] = incoming
        node_out[node] = outgoing
        weighted.add_edge(incoming, outgoing, capacity=weights[node])
        role = equipment.nodes[node]["role"]
        if role == "S":
            weighted.add_edge(SOURCE, incoming, capacity=u_inf_cut)
        elif role == "E":
            weighted.add_edge(outgoing, SINK, capacity=u_inf_cut)
    for source, target in sorted(
        equipment.edges,
        key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1])),
    ):
        weighted.add_edge(node_out[source], node_in[target], capacity=u_inf_cut)

    cut_value, (source_side, sink_side) = nx.minimum_cut(
        weighted, SOURCE, SINK, capacity="capacity"
    )
    cut_nodes = stable_nodes(
        node
        for node in equipment
        if node_in[node] in source_side and node_out[node] in sink_side
    )
    lower_bound = -float(lambda_value) * float(threshold - 1) + float(cut_value)
    candidate = solve_cut_blocker(
        equipment, cut_nodes, threshold, forbidden_attack=forbidden
    )
    return LagrangianCutBound(
        float(lambda_value),
        lower_bound,
        float(cut_value),
        cut_nodes,
        tuple((node, weights[node]) for node in stable_nodes(weights)),
        u_inf_cut,
        candidate,
    )


def scan_lagrangian_bounds(
    graph: nx.Graph,
    K: int,
    lambdas: Sequence[float],
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
    tolerance: float = 1e-8,
) -> LagrangianSearchResult:
    """Scan caller-specified multipliers and retain honest LB/UB metadata."""

    if not lambdas:
        raise ValueError("At least one Lagrange multiplier is required.")
    bounds = tuple(
        lagrangian_cut_lower_bound(
            graph,
            K,
            value,
            protected=protected,
            excluded=excluded,
        )
        for value in lambdas
    )
    best_lower = max(bound.lower_bound for bound in bounds)
    feasible_candidates = [
        bound.candidate_upper
        for bound in bounds
        if bound.candidate_upper.feasible
        and bound.candidate_upper.attack_cost is not None
    ]
    if feasible_candidates:
        best_candidate = min(
            feasible_candidates,
            key=lambda candidate: (
                float(candidate.attack_cost),
                tuple(map(repr, stable_nodes(candidate.attack_set))),
            ),
        )
        best_upper = float(best_candidate.attack_cost)
        best_attack = best_candidate.attack_set
    else:
        best_upper = None
        best_attack = frozenset()
    numerically_closed = (
        best_upper is not None and abs(best_upper - best_lower) <= tolerance
    )
    # A finite floating-point lambda scan is not an exact maximization of the
    # Lagrangian dual. It can close the numerical gap but cannot by itself
    # certify global optimality.
    certified = False
    return LagrangianSearchResult(
        bounds,
        best_lower,
        best_upper,
        best_attack,
        numerically_closed,
        certified,
        False,
    )
