"""Auditable utilities for the predeclared MPCF supplementary experiments."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import floor, isclose, lcm
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .flow import motif_capacity
from .model import NodeId, attack_cost, protection_cost, stable_node_key, stable_nodes, validate_graph
from .operational_motif import (
    OperationalMotifSystem,
    build_path_closed_system,
    solve_path_closed_motif_cut,
)
from .protection_baselines import select_budget_feasible_ranking


SCORE_METHODS: tuple[str, ...] = (
    "degree",
    "betweenness",
    "task_filtered_betweenness",
    "directed_path_frequency",
)


@dataclass(frozen=True)
class ScoreSelection:
    method: str
    policy: str
    protection_sequence: tuple[NodeId, ...]
    protection_set: frozenset[NodeId]
    protection_cost: float
    budget: float
    score_sum: float
    candidate_count: int
    prefix_length: int | None


@dataclass(frozen=True)
class BruteForceCapacityAttack:
    threshold_k: int
    initial_capacity: int
    attack_set: frozenset[NodeId]
    attack_cost: float | None
    residual_capacity: int | None
    candidate_subset_count: int
    capacity_evaluation_count: int
    complete: bool


def _system(graph: nx.Graph | OperationalMotifSystem) -> OperationalMotifSystem:
    return graph if isinstance(graph, OperationalMotifSystem) else build_path_closed_system(graph)


def task_relevant_nodes(
    graph: nx.Graph | OperationalMotifSystem,
) -> frozenset[NodeId]:
    """Return nodes occurring in at least one complete directed task path."""

    system = _system(graph)
    return frozenset().union(*(motif.removable_nodes for motif in system.motifs))


def protection_scores(
    graph: nx.Graph | OperationalMotifSystem,
    method: str,
) -> dict[NodeId, float]:
    """Compute one frozen score without consulting a protection outcome."""

    if method not in SCORE_METHODS:
        raise ValueError(f"method must be one of {SCORE_METHODS}; got {method!r}.")
    system = _system(graph)
    nodes = tuple(system.removable_nodes)
    projection = nx.Graph(system.graph.to_undirected())
    projection.remove_edges_from(nx.selfloop_edges(projection))
    if method == "degree":
        return {node: float(projection.degree(node)) for node in nodes}
    if method == "betweenness":
        centrality = nx.betweenness_centrality(projection, normalized=True)
        return {node: float(centrality.get(node, 0.0)) for node in nodes}
    if method == "task_filtered_betweenness":
        relevant = task_relevant_nodes(system)
        filtered = projection.subgraph(relevant).copy()
        centrality = nx.betweenness_centrality(filtered, normalized=True)
        return {node: float(centrality.get(node, 0.0)) for node in stable_nodes(relevant)}
    return {
        node: float(sum(node in motif.removable_nodes for motif in system.motifs))
        for node in nodes
    }


def rank_scores(scores: Mapping[NodeId, float]) -> tuple[NodeId, ...]:
    return tuple(
        sorted(scores, key=lambda node: (-float(scores[node]), stable_node_key(node)))
    )


def _normalized_utilities(
    scores: Mapping[NodeId, float], ranking: Sequence[NodeId]
) -> dict[NodeId, float]:
    maximum = max((float(scores[node]) for node in ranking), default=0.0)
    if maximum > 0.0:
        return {node: float(scores[node]) / maximum for node in ranking}
    count = max(1, len(ranking))
    return {node: (count - index) / count for index, node in enumerate(ranking)}


def select_score_policy(
    graph: nx.Graph | OperationalMotifSystem,
    scores: Mapping[NodeId, float],
    budget: float,
    *,
    method: str,
    policy: str,
    prefix_multiplier: int = 3,
) -> ScoreSelection:
    """Apply a fixed budget policy to one precomputed score vector.

    Policies are ``raw`` (score ranking plus affordable scan), ``per_cost``
    (normalized score divided by protection cost), ``prefix_knapsack``
    (exact 0-1 knapsack over a predeclared top-ranked prefix), and
    ``full_knapsack`` (the same exact score/cost knapsack over every eligible
    node).  The full policy is useful for checking that a prefix restriction,
    rather than the frozen structural score itself, is not driving a result.
    """

    if policy not in {"raw", "per_cost", "prefix_knapsack", "full_knapsack"}:
        raise ValueError(
            "policy must be raw, per_cost, prefix_knapsack, or full_knapsack."
        )
    if prefix_multiplier < 1:
        raise ValueError("prefix_multiplier must be positive.")
    system = _system(graph)
    ranking = rank_scores(scores)
    utilities = _normalized_utilities(scores, ranking)
    if policy == "raw":
        selection = select_budget_feasible_ranking(
            system, ranking, budget, method=method, source_method=method
        )
        chosen_sequence = selection.protection_sequence
        prefix_length = None
    elif policy == "per_cost":
        ratio_ranking = tuple(
            sorted(
                ranking,
                key=lambda node: (
                    -utilities[node] / float(system.graph.nodes[node]["protect_cost"]),
                    -utilities[node],
                    stable_node_key(node),
                ),
            )
        )
        selection = select_budget_feasible_ranking(
            system, ratio_ranking, budget, method=method, source_method=method
        )
        chosen_sequence = selection.protection_sequence
        prefix_length = None
    elif policy == "prefix_knapsack":
        minimum_cost = min(
            (float(system.graph.nodes[node]["protect_cost"]) for node in ranking),
            default=1.0,
        )
        maximum_cardinality = max(0, floor(float(budget) / minimum_cost + 1e-9))
        prefix_length = min(
            len(ranking), max(1, prefix_multiplier * maximum_cardinality)
        )
        prefix = ranking[:prefix_length]
        chosen = _knapsack_select(system, prefix, utilities, budget)
        chosen_sequence = tuple(node for node in ranking if node in chosen)
    else:
        prefix_length = len(ranking)
        chosen = _knapsack_select(system, ranking, utilities, budget)
        chosen_sequence = tuple(node for node in ranking if node in chosen)
    chosen_set = frozenset(chosen_sequence)
    return ScoreSelection(
        method=method,
        policy=policy,
        protection_sequence=chosen_sequence,
        protection_set=chosen_set,
        protection_cost=protection_cost(system.graph, chosen_set),
        budget=float(budget),
        score_sum=sum(utilities[node] for node in chosen_set),
        candidate_count=len(ranking),
        prefix_length=prefix_length,
    )


def _knapsack_select(
    system: OperationalMotifSystem,
    candidates: Sequence[NodeId],
    utilities: Mapping[NodeId, float],
    budget: float,
) -> frozenset[NodeId]:
    fractions = [
        Fraction(str(float(system.graph.nodes[node]["protect_cost"]))).limit_denominator(1000)
        for node in candidates
    ]
    scale = 1
    for value in fractions:
        scale = lcm(scale, value.denominator)
    integer_costs = [value.numerator * (scale // value.denominator) for value in fractions]
    capacity = floor(float(budget) * scale + 1e-9)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, (node, cost) in enumerate(zip(candidates, integer_costs)):
        updates = dict(states)
        for spent, (utility, selected) in states.items():
            new_spent = spent + cost
            if new_spent > capacity:
                continue
            candidate = (utility + float(utilities[node]), selected + (index,))
            incumbent = updates.get(new_spent)
            if (
                incumbent is None
                or candidate[0] > incumbent[0] + 1e-12
                or isclose(candidate[0], incumbent[0], abs_tol=1e-12)
                and candidate[1] < incumbent[1]
            ):
                updates[new_spent] = candidate
        states = updates
    best_spent, (_best_utility, indices) = min(
        states.items(),
        key=lambda item: (-item[1][0], item[0], item[1][1]),
    )
    _ = best_spent
    return frozenset(candidates[index] for index in indices)


def brute_force_capacity_attack(
    graph: nx.Graph,
    threshold_k: int,
    *,
    max_nodes: int = 16,
) -> BruteForceCapacityAttack:
    """Enumerate every attack subset on a tiny capacity-threshold instance."""

    equipment = validate_graph(graph)
    initial = motif_capacity(equipment).value
    if isinstance(threshold_k, bool) or not 1 <= int(threshold_k) <= initial:
        raise ValueError("threshold_k must lie between 1 and initial capacity.")
    nodes = tuple(
        node
        for node in stable_nodes(equipment.nodes)
        if bool(equipment.nodes[node].get("removable", True))
    )
    if len(nodes) > max_nodes:
        raise ValueError(f"Capacity brute force is limited to {max_nodes} nodes.")
    best_set: frozenset[NodeId] | None = None
    best_cost: float | None = None
    best_residual: int | None = None
    candidates = 0
    evaluations = 0
    for size in range(len(nodes) + 1):
        for raw in combinations(nodes, size):
            candidates += 1
            candidate = frozenset(raw)
            cost = attack_cost(equipment, candidate)
            if best_cost is not None and cost > best_cost + 1e-9:
                continue
            residual = motif_capacity(equipment, candidate).value
            evaluations += 1
            if residual > int(threshold_k) - 1:
                continue
            if (
                best_cost is None
                or cost < best_cost - 1e-9
                or isclose(cost, best_cost, rel_tol=0.0, abs_tol=1e-9)
                and tuple(stable_node_key(node) for node in stable_nodes(candidate))
                < tuple(stable_node_key(node) for node in stable_nodes(best_set or ()))
            ):
                best_set = candidate
                best_cost = cost
                best_residual = residual
    return BruteForceCapacityAttack(
        threshold_k=int(threshold_k),
        initial_capacity=initial,
        attack_set=best_set or frozenset(),
        attack_cost=best_cost,
        residual_capacity=best_residual,
        candidate_subset_count=candidates,
        capacity_evaluation_count=evaluations,
        complete=True,
    )


def alternative_minimum_path_cut(
    graph: nx.Graph | OperationalMotifSystem,
    reference_cut: Iterable[NodeId] | None = None,
) -> tuple[frozenset[NodeId], frozenset[NodeId] | None, float]:
    """Return one minimum cut and, when it exists, a distinct equal-cost cut."""

    system = _system(graph)
    reference = solve_path_closed_motif_cut(system)
    if not reference.optimal or reference.objective is None:
        raise RuntimeError("Reference PathCut failed.")
    first = frozenset(reference_cut) if reference_cut is not None else reference.selected_set
    optimum = float(reference.objective)
    base_costs = {
        node: float(system.graph.nodes[node]["attack_cost"])
        for node in system.removable_nodes
    }
    big_m = sum(base_costs.values()) + max(base_costs.values()) + 1.0
    for node in stable_nodes(first):
        costs = dict(base_costs)
        costs[node] = big_m
        candidate = solve_path_closed_motif_cut(system, costs=costs)
        if (
            candidate.optimal
            and candidate.objective is not None
            and isclose(float(candidate.objective), optimum, rel_tol=0.0, abs_tol=1e-8)
            and candidate.selected_set != first
        ):
            return first, candidate.selected_set, optimum
    return first, None, optimum


def exchange_groups(
    sets: Sequence[frozenset[NodeId]],
) -> tuple[frozenset[NodeId], ...]:
    """Group nodes connected by observed one-for-one optimal-set exchanges."""

    graph = nx.Graph()
    graph.add_nodes_from(frozenset().union(*sets) if sets else ())
    for left, right in combinations(sets, 2):
        removed = left - right
        added = right - left
        if len(removed) == len(added) == 1:
            graph.add_edge(next(iter(removed)), next(iter(added)))
    groups = [
        frozenset(component)
        for component in nx.connected_components(graph)
        if len(component) > 1
    ]
    return tuple(sorted(groups, key=lambda group: tuple(stable_node_key(n) for n in stable_nodes(group))))
