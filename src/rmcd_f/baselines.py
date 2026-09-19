"""Transparent baseline sequences for the RMCD critical-set study.

These are deliberately small, dependency-free reference methods.  They are
not labelled as official implementations of published dismantling packages.
Every sequence is evaluated against the same role-motif capacity threshold as
RMCD, which prevents a role-blind baseline from being credited for a target it
did not actually breach.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Literal, Sequence

import networkx as nx

from .flow import motif_capacity
from .model import NodeId, attack_cost, stable_node_key, stable_nodes, validate_graph


BaselineMethod = Literal[
    "random",
    "degree",
    "degree-structural",
    "degree-per-cost",
    "betweenness",
    "betweenness-structural",
    "betweenness-per-cost",
    "kcore",
    "kcore-structural",
    "kcore-per-cost",
    "motif-greedy",
    "motif-greedy-per-cost",
]
BASELINE_METHODS: tuple[str, ...] = (
    "random",
    "degree",
    "degree-structural",
    "degree-per-cost",
    "betweenness",
    "betweenness-structural",
    "betweenness-per-cost",
    "kcore",
    "kcore-structural",
    "kcore-per-cost",
    "motif-greedy",
    "motif-greedy-per-cost",
)


_LEGACY_PER_COST_ALIASES = {
    "degree": "degree-per-cost",
    "betweenness": "betweenness-per-cost",
    "kcore": "kcore-per-cost",
    "motif-greedy": "motif-greedy-per-cost",
}


@dataclass(frozen=True)
class BaselineCriticalSetResult:
    method: str
    threshold: int
    initial_capacity: int
    selected_order: tuple[NodeId, ...]
    critical_set: frozenset[NodeId]
    critical_cost: float | None
    residual_capacity: int
    breached: bool
    runtime_seconds: float
    objective_awareness: str = "cost-blind"


def baseline_objective_awareness(method: str) -> str:
    """Describe whether a baseline ranking directly uses attack costs."""

    canonical = _LEGACY_PER_COST_ALIASES.get(method, method)
    return "cost-aware" if canonical.endswith("-per-cost") else "cost-blind"


def _score_order(
    graph: nx.DiGraph, scores: dict[NodeId, float]
) -> tuple[NodeId, ...]:
    return tuple(
        sorted(
            scores,
            key=lambda node: (
                -float(scores[node]),
                stable_node_key(node),
            ),
        )
    )


def _static_order(
    graph: nx.DiGraph, method: str, seed: int
) -> tuple[NodeId, ...]:
    candidates = tuple(
        node for node in stable_nodes(graph.nodes) if graph.nodes[node]["capacity"] > 0
    )
    canonical = _LEGACY_PER_COST_ALIASES.get(method, method)
    if canonical == "random":
        shuffled = list(candidates)
        random.Random(seed).shuffle(shuffled)
        return tuple(shuffled)

    undirected = graph.to_undirected()
    if canonical.startswith("degree-"):
        raw = {node: float(undirected.degree(node)) for node in candidates}
    elif canonical.startswith("betweenness-"):
        values = nx.betweenness_centrality(undirected, normalized=True)
        raw = {node: float(values[node]) for node in candidates}
    elif canonical.startswith("kcore-"):
        values = nx.core_number(undirected)
        raw = {node: float(values[node]) for node in candidates}
    else:
        raise ValueError(f"Unknown static baseline method: {method!r}.")

    scores = raw
    if canonical.endswith("-per-cost"):
        scores = {
            node: raw[node] / float(graph.nodes[node]["attack_cost"])
            for node in candidates
        }
    return _score_order(graph, scores)


def _motif_greedy_order(
    graph: nx.DiGraph, K: int
) -> tuple[tuple[NodeId, ...], int]:
    order, residuals = _motif_greedy_trace(graph, K)
    return order, residuals[-1]


def _motif_greedy_trace(
    graph: nx.DiGraph, K: int
) -> tuple[tuple[NodeId, ...], tuple[int, ...]]:
    attacked: set[NodeId] = set()
    order: list[NodeId] = []
    current = motif_capacity(graph).value
    residuals = [current]
    candidates = {
        node for node in graph if int(graph.nodes[node]["capacity"]) > 0
    }
    while current >= K and candidates:
        evaluated: list[tuple[float, int, float, tuple[str, str], NodeId, int]] = []
        for node in candidates:
            residual = motif_capacity(graph, attacked | {node}).value
            gain = current - residual
            cost = float(graph.nodes[node]["attack_cost"])
            evaluated.append(
                (
                    -(gain / cost),
                    -gain,
                    cost,
                    stable_node_key(node),
                    node,
                    residual,
                )
            )
        _ratio, _gain, _cost, _key, selected, residual = min(evaluated)
        attacked.add(selected)
        candidates.remove(selected)
        order.append(selected)
        current = residual
        residuals.append(current)
    return tuple(order), tuple(residuals)


def evaluate_baseline_critical_sets(
    graph: nx.Graph,
    thresholds: Sequence[int],
    method: BaselineMethod,
    *,
    seed: int = 0,
) -> tuple[BaselineCriticalSetResult, ...]:
    """Evaluate several thresholds from one shared ranking/capacity trace."""

    if method not in BASELINE_METHODS:
        raise ValueError(f"method must be one of {BASELINE_METHODS}; got {method!r}.")
    equipment = validate_graph(graph)
    initial = motif_capacity(equipment).value
    requested = tuple(int(value) for value in thresholds)
    if not requested:
        raise ValueError("thresholds must contain at least one value.")
    if any(not 1 <= value <= initial for value in requested):
        raise ValueError(
            f"Every K must satisfy 1 <= K <= Omega_0={initial}; got {requested}."
        )

    started = perf_counter()
    minimum = min(requested)
    canonical = _LEGACY_PER_COST_ALIASES.get(method, method)
    if canonical == "motif-greedy-per-cost":
        ranking, residuals = _motif_greedy_trace(equipment, minimum)
    else:
        ranking = _static_order(equipment, method, seed)
        attacked: list[NodeId] = []
        trace = [initial]
        for node in ranking:
            if trace[-1] < minimum:
                break
            attacked.append(node)
            trace.append(motif_capacity(equipment, attacked).value)
        ranking = tuple(attacked)
        residuals = tuple(trace)
    runtime = perf_counter() - started

    results: list[BaselineCriticalSetResult] = []
    for threshold in requested:
        prefix_length = next(
            (index for index, residual in enumerate(residuals) if residual < threshold),
            len(residuals) - 1,
        )
        selected_order = tuple(ranking[:prefix_length])
        residual = residuals[prefix_length]
        breached = residual < threshold
        critical_set = frozenset(selected_order)
        results.append(
            BaselineCriticalSetResult(
                method=method,
                threshold=threshold,
                initial_capacity=initial,
                selected_order=selected_order,
                critical_set=critical_set,
                critical_cost=(
                    attack_cost(equipment, critical_set) if breached else None
                ),
                residual_capacity=residual,
                breached=breached,
                runtime_seconds=runtime,
                objective_awareness=baseline_objective_awareness(method),
            )
        )
    return tuple(results)


def evaluate_baseline_critical_set(
    graph: nx.Graph,
    K: int,
    method: BaselineMethod,
    *,
    seed: int = 0,
) -> BaselineCriticalSetResult:
    """Return the first baseline prefix that breaches the common threshold."""

    return evaluate_baseline_critical_sets(graph, (K,), method, seed=seed)[0]


def evaluate_sequence_critical_set(
    graph: nx.Graph,
    K: int,
    method: str,
    sequence: Iterable[NodeId],
) -> BaselineCriticalSetResult:
    """Evaluate an externally generated removal sequence at the RMCD threshold.

    The external method is credited only with the first prefix that is
    independently verified to satisfy ``Omega_R(G-D) < K``. Unknown nodes and
    duplicate removals are rejected instead of being silently repaired.
    """

    equipment = validate_graph(graph)
    initial = motif_capacity(equipment).value
    if not 1 <= int(K) <= initial:
        raise ValueError(f"K must satisfy 1 <= K <= Omega_0={initial}; got {K}.")

    ordered: list[NodeId] = []
    seen: set[NodeId] = set()
    for position, node in enumerate(sequence, start=1):
        if node not in equipment:
            raise ValueError(
                f"{method} sequence contains unknown node {node!r} "
                f"at position {position}."
            )
        if node in seen:
            raise ValueError(
                f"{method} sequence repeats node {node!r} at position {position}."
            )
        if int(equipment.nodes[node]["capacity"]) <= 0:
            raise ValueError(
                f"{method} sequence contains non-attackable node {node!r}."
            )
        ordered.append(node)
        seen.add(node)

    started = perf_counter()
    selected: list[NodeId] = []
    residual = initial
    for node in ordered:
        if residual < K:
            break
        selected.append(node)
        residual = motif_capacity(equipment, selected).value

    breached = residual < K
    critical_set = frozenset(selected)
    runtime = perf_counter() - started
    return BaselineCriticalSetResult(
        method=method,
        threshold=int(K),
        initial_capacity=initial,
        selected_order=tuple(selected),
        critical_set=critical_set,
        critical_cost=attack_cost(equipment, critical_set) if breached else None,
        residual_capacity=residual,
        breached=breached,
        runtime_seconds=runtime,
        objective_awareness="cost-blind",
    )
