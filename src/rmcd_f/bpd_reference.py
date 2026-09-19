"""Transparent reference implementation of classical network BPD.

The Nature Reviews Physics comparison table includes belief-propagation-guided
decimation (BPD), but the accompanying ``NetworkDismantling/review`` source
tree does not register that method.  This module implements equations (2)-(4)
and the decycling/tree-breaking procedure of Mugisha and Zhou, Phys. Rev. E 94,
012305 (2016).  It is labelled ``BPDReference`` throughout the experiments and
is never represented as an upstream review implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, isfinite, log
import random
from statistics import median
from time import perf_counter
from typing import Mapping

import networkx as nx

from .model import NodeId, stable_node_key, stable_nodes, validate_graph
from .operational_motif import build_path_closed_system


BPD_REFERENCE_VERSION = "mugisha-zhou-2016-v1"


@dataclass(frozen=True)
class BPDReferenceStep:
    step: int
    selected_node: NodeId
    phase: str
    deletion_probability: float | None
    lcc_size_before: int
    lcc_size_after: int
    core_size_before: int
    bp_rounds: int
    converged: bool


@dataclass(frozen=True)
class BPDReferenceResult:
    method: str
    sequence: tuple[NodeId, ...]
    residual_lcc_size: int
    decycling_size: int
    tree_breaker_size: int
    runtime_seconds: float
    x: float
    decimation_fraction: float
    bp_rounds: int
    damping: float
    tolerance: float
    stop_condition: int
    trace: tuple[BPDReferenceStep, ...]
    version: str = BPD_REFERENCE_VERSION


def _largest_component_size(graph: nx.Graph) -> int:
    return max((len(component) for component in nx.connected_components(graph)), default=0)


def _attackable_nodes(graph: nx.DiGraph) -> frozenset[NodeId]:
    return frozenset(
        node
        for node, data in graph.nodes(data=True)
        if bool(data.get("removable", True)) and int(data.get("capacity", 1)) > 0
    )


def _normalized_costs(
    graph: nx.DiGraph,
    candidates: frozenset[NodeId],
    costs: Mapping[NodeId, float] | None,
) -> dict[NodeId, float]:
    raw = {
        node: float(costs[node] if costs is not None else graph.nodes[node]["attack_cost"])
        for node in candidates
    }
    if any(not isfinite(value) or value <= 0 for value in raw.values()):
        raise ValueError("BPD node costs must be finite and positive.")
    scale = float(median(raw.values())) if raw else 1.0
    normalized = {node: value / scale for node, value in raw.items()}
    # Non-removable nodes still participate in cavity messages.  Assigning a
    # large finite deletion cost keeps them available as structural supports
    # without ever admitting them to the ranked decimation candidates.
    protected_cost = max((value for value in normalized.values()), default=1.0) * 1e6
    return {
        node: normalized.get(node, protected_cost)
        for node in graph.nodes
    }


def _safe_exp(log_value: float) -> float:
    return exp(max(-700.0, min(700.0, log_value)))


def _bp_deletion_probabilities(
    graph: nx.Graph,
    costs: Mapping[NodeId, float],
    *,
    x: float,
    rounds: int,
    damping: float,
    tolerance: float,
    seed: int,
) -> tuple[dict[NodeId, float], bool]:
    """Iterate the cavity probabilities used by classical BPD."""

    directed_edges = tuple(
        (left, right)
        for left in stable_nodes(graph.nodes)
        for right in stable_nodes(graph.neighbors(left))
    )
    rng = random.Random(seed)
    q0: dict[tuple[NodeId, NodeId], float] = {}
    qi: dict[tuple[NodeId, NodeId], float] = {}
    for edge in directed_edges:
        jitter = rng.uniform(-1e-3, 1e-3)
        q0[edge] = 0.20 + jitter
        qi[edge] = 0.40 - jitter

    converged = False
    for _round in range(rounds):
        next_q0: dict[tuple[NodeId, NodeId], float] = {}
        next_qi: dict[tuple[NodeId, NodeId], float] = {}
        max_delta = 0.0
        for node, excluded in directed_edges:
            neighbors = tuple(
                other for other in graph.neighbors(node) if other != excluded
            )
            log_product = 0.0
            ratio_sum = 0.0
            for other in neighbors:
                denominator = max(1e-300, q0[(other, node)] + qi[(other, node)])
                log_product += log(denominator)
                ratio_sum += (1.0 - q0[(other, node)]) / denominator
            weighted_product = _safe_exp(x * costs[node] + log_product)
            normalization = 1.0 + weighted_product * (1.0 + ratio_sum)
            raw_q0 = 1.0 / normalization
            raw_qi = weighted_product / normalization
            edge = (node, excluded)
            value_q0 = (1.0 - damping) * q0[edge] + damping * raw_q0
            value_qi = (1.0 - damping) * qi[edge] + damping * raw_qi
            next_q0[edge] = value_q0
            next_qi[edge] = value_qi
            max_delta = max(
                max_delta,
                abs(value_q0 - q0[edge]),
                abs(value_qi - qi[edge]),
            )
        q0, qi = next_q0, next_qi
        if max_delta <= tolerance:
            converged = True
            break

    probabilities: dict[NodeId, float] = {}
    for node in stable_nodes(graph.nodes):
        log_product = 0.0
        ratio_sum = 0.0
        for other in graph.neighbors(node):
            denominator = max(1e-300, q0[(other, node)] + qi[(other, node)])
            log_product += log(denominator)
            ratio_sum += (1.0 - q0[(other, node)]) / denominator
        weighted_product = _safe_exp(x * costs[node] + log_product)
        probabilities[node] = 1.0 / (
            1.0 + weighted_product * (1.0 + ratio_sum)
        )
    return probabilities, converged


def _tree_breaker_node(
    forest: nx.Graph,
    component: frozenset[NodeId],
    candidates: frozenset[NodeId],
    costs: Mapping[NodeId, float],
) -> NodeId:
    available = tuple(node for node in component if node in candidates)
    if not available:
        raise RuntimeError("A large tree component contains no removable node.")
    subgraph = forest.subgraph(component)
    evaluated: list[tuple[int, float, tuple[str, str], NodeId]] = []
    for node in available:
        residual = subgraph.copy()
        residual.remove_node(node)
        largest = _largest_component_size(residual)
        evaluated.append((largest, costs[node], stable_node_key(node), node))
    return min(evaluated)[-1]


def bpd_reference_sequence(
    graph: nx.Graph,
    *,
    costs: Mapping[NodeId, float] | None = None,
    x: float = 12.0,
    decimation_fraction: float = 0.01,
    bp_rounds: int = 50,
    damping: float = 0.5,
    tolerance: float = 1e-8,
    stop_condition: int = 1,
    seed: int = 0,
) -> BPDReferenceResult:
    """Generate a classical BPD structural dismantling sequence.

    The decycling phase repeatedly scores the residual 2-core.  Once a forest
    remains, a deterministic centroid-style tree breaker is used until the LCC
    target is met.  In agreement with Table 1 of Artime et al. (2024), no
    reinsertion phase is applied to this canonical BPD reference.
    """

    if x <= 0 or not 0.0 < decimation_fraction <= 1.0:
        raise ValueError("x and decimation_fraction must be positive.")
    if bp_rounds < 1 or not 0.0 < damping <= 1.0 or tolerance <= 0:
        raise ValueError("Invalid BPD iteration parameters.")
    if stop_condition < 1:
        raise ValueError("stop_condition must be at least one.")
    if str(graph.graph.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        equipment = build_path_closed_system(graph).graph
    else:
        equipment = validate_graph(graph)
    attackable = _attackable_nodes(equipment)
    normalized_cost = _normalized_costs(equipment, attackable, costs)
    projection = nx.Graph(equipment.to_undirected())
    projection.remove_edges_from(nx.selfloop_edges(projection))
    initial_n = projection.number_of_nodes()
    working = projection.copy()
    started = perf_counter()
    sequence: list[NodeId] = []
    trace: list[BPDReferenceStep] = []
    decycling_size = 0

    while working.number_of_edges() and not nx.is_forest(working):
        core = nx.k_core(working, k=2)
        core_candidates = frozenset(core.nodes) & attackable
        if not core_candidates:
            raise RuntimeError("Residual cyclic 2-core contains no removable node.")
        probabilities, converged = _bp_deletion_probabilities(
            core,
            normalized_cost,
            x=x,
            rounds=bp_rounds,
            damping=damping,
            tolerance=tolerance,
            seed=seed + len(sequence),
        )
        batch = max(1, int(ceil(decimation_fraction * initial_n)))
        ranked = sorted(
            core_candidates,
            key=lambda node: (
                -probabilities[node],
                normalized_cost[node],
                stable_node_key(node),
            ),
        )
        for node in ranked[:batch]:
            if node not in working:
                continue
            before = _largest_component_size(working)
            core_size = core.number_of_nodes()
            working.remove_node(node)
            sequence.append(node)
            decycling_size += 1
            trace.append(
                BPDReferenceStep(
                    step=len(sequence),
                    selected_node=node,
                    phase="decycling",
                    deletion_probability=float(probabilities[node]),
                    lcc_size_before=before,
                    lcc_size_after=_largest_component_size(working),
                    core_size_before=core_size,
                    bp_rounds=bp_rounds,
                    converged=converged,
                )
            )
            if nx.is_forest(working):
                break

    while _largest_component_size(working) > stop_condition:
        components = sorted(
            (frozenset(component) for component in nx.connected_components(working)),
            key=lambda component: (-len(component), tuple(stable_node_key(n) for n in stable_nodes(component))),
        )
        component = components[0]
        node = _tree_breaker_node(working, component, attackable, normalized_cost)
        before = len(component)
        working.remove_node(node)
        sequence.append(node)
        trace.append(
            BPDReferenceStep(
                step=len(sequence),
                selected_node=node,
                phase="tree_breaking",
                deletion_probability=None,
                lcc_size_before=before,
                lcc_size_after=_largest_component_size(working),
                core_size_before=0,
                bp_rounds=0,
                converged=True,
            )
        )

    return BPDReferenceResult(
        method="BPDReference",
        sequence=tuple(sequence),
        residual_lcc_size=_largest_component_size(working),
        decycling_size=decycling_size,
        tree_breaker_size=len(sequence) - decycling_size,
        runtime_seconds=perf_counter() - started,
        x=float(x),
        decimation_fraction=float(decimation_fraction),
        bp_rounds=bp_rounds,
        damping=float(damping),
        tolerance=float(tolerance),
        stop_condition=stop_condition,
        trace=tuple(trace),
    )
