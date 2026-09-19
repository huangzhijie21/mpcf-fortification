"""Budget-feasible protection baselines for task-path fortification studies."""

from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter
from typing import Any, Iterable, Sequence

import networkx as nx

from .bpd_reference import bpd_reference_sequence
from .model import NodeId, protection_cost, stable_node_key, stable_nodes
from .operational_motif import (
    OperationalMotifSystem,
    build_path_closed_system,
    solve_path_closed_motif_cut,
)


LOCAL_PROTECTION_METHODS: tuple[str, ...] = (
    "NoProtection",
    "RandomProtect",
    "DegreeProtect",
    "BetweennessProtect",
    "KCoreProtect",
    "PageRankProtect",
    "PathFrequencyProtect",
    "InitialPathCutProtect",
    "BPDReference-Protect",
)


@dataclass(frozen=True)
class ProtectionSelectionResult:
    method: str
    source_method: str
    protection_sequence: tuple[NodeId, ...]
    protection_set: frozenset[NodeId]
    protection_cost: float
    budget: float
    ranking_length: int
    skipped_unaffordable: int
    skipped_unknown: int
    runtime_seconds: float
    provenance: str


def _system(graph: nx.Graph | OperationalMotifSystem) -> OperationalMotifSystem:
    return (
        graph
        if isinstance(graph, OperationalMotifSystem)
        else build_path_closed_system(graph)
    )


def protection_method_name(source_method: str) -> str:
    """Return an unambiguous name for a dismantling-ranking transfer baseline."""

    return f"{source_method}-Protect"


def coerce_node_sequence(
    sequence: Sequence[Any], graph: nx.Graph | OperationalMotifSystem
) -> tuple[NodeId, ...]:
    """Map JSON-decoded sequence IDs back to frozen equipment node IDs."""

    system = _system(graph)
    removable = frozenset(system.removable_nodes)
    by_text: dict[str, list[NodeId]] = {}
    for node in system.removable_nodes:
        by_text.setdefault(str(node), []).append(node)
    normalized: list[NodeId] = []
    for raw in sequence:
        node: NodeId | None = raw if raw in removable else None
        if node is None:
            matches = by_text.get(str(raw), ())
            if len(matches) == 1:
                node = matches[0]
        if node is not None and node not in normalized:
            normalized.append(node)
    return tuple(normalized)


def select_budget_feasible_ranking(
    graph: nx.Graph | OperationalMotifSystem,
    sequence: Sequence[Any],
    budget: float,
    *,
    method: str,
    source_method: str | None = None,
    provenance: str = "",
    runtime_seconds: float = 0.0,
) -> ProtectionSelectionResult:
    """Scan a fixed ranking once and retain every currently affordable node.

    Under heterogeneous protection costs, stopping at the first unaffordable
    node would discard later affordable ranked nodes.  The preregistered rule
    therefore preserves order while skipping only nodes that do not fit the
    remaining budget.  No task-path outcome is consulted during this scan.
    """

    system = _system(graph)
    if budget < 0.0:
        raise ValueError("budget must be non-negative.")
    normalized = coerce_node_sequence(sequence, system)
    skipped_unknown = len(tuple(dict.fromkeys(sequence))) - len(normalized)
    selected: list[NodeId] = []
    spent = 0.0
    skipped_unaffordable = 0
    for node in normalized:
        cost = float(system.graph.nodes[node]["protect_cost"])
        if spent + cost <= float(budget) + 1e-9:
            selected.append(node)
            spent += cost
        else:
            skipped_unaffordable += 1
    chosen = frozenset(selected)
    return ProtectionSelectionResult(
        method=method,
        source_method=source_method or method,
        protection_sequence=tuple(selected),
        protection_set=chosen,
        protection_cost=protection_cost(system.graph, chosen),
        budget=float(budget),
        ranking_length=len(normalized),
        skipped_unaffordable=skipped_unaffordable,
        skipped_unknown=max(0, skipped_unknown),
        runtime_seconds=float(runtime_seconds),
        provenance=provenance,
    )


def _rank_scores(scores: dict[NodeId, float]) -> tuple[NodeId, ...]:
    return tuple(
        sorted(
            scores,
            key=lambda node: (-scores[node], stable_node_key(node)),
        )
    )


def local_protection_sequence(
    graph: nx.Graph | OperationalMotifSystem,
    method: str,
    *,
    seed: int = 0,
) -> tuple[tuple[NodeId, ...], float, str]:
    """Generate a preregistered local baseline ranking without outcome access."""

    system = _system(graph)
    if method not in LOCAL_PROTECTION_METHODS:
        raise ValueError(f"Unknown local protection method {method!r}.")
    started = perf_counter()
    nodes = list(system.removable_nodes)
    if method == "NoProtection":
        sequence: tuple[NodeId, ...] = ()
        provenance = "empty protection set"
    elif method == "RandomProtect":
        rng = random.Random(seed)
        rng.shuffle(nodes)
        sequence = tuple(nodes)
        provenance = "deterministic random permutation on frozen node IDs"
    else:
        projection = nx.Graph(system.graph.to_undirected())
        projection.remove_edges_from(nx.selfloop_edges(projection))
        if method == "DegreeProtect":
            sequence = _rank_scores(
                {node: float(projection.degree(node)) for node in nodes}
            )
            provenance = "static undirected degree"
        elif method == "BetweennessProtect":
            centrality = nx.betweenness_centrality(projection, normalized=True)
            sequence = _rank_scores(
                {node: float(centrality.get(node, 0.0)) for node in nodes}
            )
            provenance = "static undirected betweenness centrality"
        elif method == "KCoreProtect":
            core = nx.core_number(projection) if projection.number_of_edges() else {
                node: 0 for node in projection
            }
            sequence = _rank_scores(
                {node: float(core.get(node, 0)) for node in nodes}
            )
            provenance = "static undirected k-core number"
        elif method == "PageRankProtect":
            pagerank = nx.pagerank(projection)
            sequence = _rank_scores(
                {node: float(pagerank.get(node, 0.0)) for node in nodes}
            )
            provenance = "static undirected PageRank"
        elif method == "PathFrequencyProtect":
            counts = {
                node: float(sum(node in motif.removable_nodes for motif in system.motifs))
                for node in nodes
            }
            sequence = _rank_scores(counts)
            provenance = "number of complete task paths containing each node"
        elif method == "InitialPathCutProtect":
            cut = solve_path_closed_motif_cut(system)
            if not cut.optimal:
                raise RuntimeError("InitialPathCutProtect requires an optimal PathCut.")
            path_counts = {
                node: float(sum(node in motif.removable_nodes for motif in system.motifs))
                for node in cut.selected_set
            }
            sequence = _rank_scores(path_counts)
            provenance = "single undefended minimum task-path cut; no adaptive reselection"
        elif method == "BPDReference-Protect":
            bpd = bpd_reference_sequence(system.graph, seed=seed, stop_condition=1)
            sequence = bpd.sequence
            provenance = "transparent Mugisha-Zhou BPD reference ranking"
        else:  # pragma: no cover - protected by the registry check.
            raise AssertionError(method)
    return sequence, perf_counter() - started, provenance


def select_local_protection(
    graph: nx.Graph | OperationalMotifSystem,
    method: str,
    budget: float,
    *,
    seed: int = 0,
) -> ProtectionSelectionResult:
    sequence, runtime, provenance = local_protection_sequence(
        graph, method, seed=seed
    )
    return select_budget_feasible_ranking(
        graph,
        sequence,
        budget,
        method=method,
        source_method=method,
        provenance=provenance,
        runtime_seconds=runtime,
    )
