"""Operational-motif critical-set models and OM-BPD.

The module deliberately solves one narrow problem.  A fixed library is used
to enumerate typed, directed S-C-L-E operational-motif instances.  A physical
equipment node disables every motif instance that contains it.  The objective
is therefore the minimum-cost set of equipment nodes that intersects every
motif instance.

OM-BPD is a Min-Sum belief-propagation-guided decimation heuristic for this
factor graph.  It never evaluates a task-performance curve while selecting
nodes.  Exact MILP and LP routines are provided only as small-instance
certificates and lower bounds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, vstack

from .flow import enumerate_role_motifs
from .model import NodeId, stable_node_key, stable_nodes, validate_graph


OPERATIONAL_MOTIF_LIBRARY_VERSION = "scle-path-v1"
OM_BPD_VERSION = "1.1"
OM_BPD_CERTIFIED_VERSION = "1.0"
MAX_AUDITED_TASK_PATHS = 100_000


@dataclass(frozen=True)
class OperationalMotifInstance:
    """One matched typed, directed operational motif."""

    motif_id: int
    template: str
    ordered_nodes: tuple[NodeId, ...]
    removable_nodes: frozenset[NodeId]


@dataclass(frozen=True)
class OperationalMotifSystem:
    """Frozen factor graph used by every compared node sequence."""

    graph: nx.DiGraph
    motifs: tuple[OperationalMotifInstance, ...]
    factors: tuple[frozenset[NodeId], ...]
    removable_nodes: tuple[NodeId, ...]
    library_version: str = OPERATIONAL_MOTIF_LIBRARY_VERSION
    source_nodes: tuple[NodeId, ...] = ()
    target_nodes: tuple[NodeId, ...] = ()


@dataclass(frozen=True)
class PathClosureAudit:
    """Whether a catalog is exactly the complete directed task-path family."""

    path_closed: bool
    observed_motif_count: int
    expected_path_count: int
    missing_path_count: int
    extra_motif_count: int
    removable_factor_mismatch_count: int
    classification: str
    reason: str


@dataclass(frozen=True)
class OMBPDStep:
    """One auditable decimation decision."""

    step: int
    selected_node: NodeId
    selected_score: float
    selection_reason: str
    active_factor_count_before: int
    active_factor_count_after: int
    candidate_count: int
    bp_iterations: int
    converged: bool
    max_message_delta: float


@dataclass(frozen=True)
class OMBPDRepairStep:
    """One exact solve in the bounded exchange neighbourhood."""

    round: int
    cost_before: float
    cost_after: float
    improvement: float
    selected_count_before: int
    selected_count_after: int
    retained_required: int
    dropped_nodes: tuple[NodeId, ...]
    added_nodes: tuple[NodeId, ...]
    solver_status: str
    solver_optimal: bool
    runtime_seconds: float


@dataclass(frozen=True)
class OMBPDResult:
    """OM-BPD solution and its forward/reinsertion audit."""

    method: str
    raw_sequence: tuple[NodeId, ...]
    raw_set: frozenset[NodeId]
    critical_sequence: tuple[NodeId, ...]
    critical_set: frozenset[NodeId]
    raw_cost: float
    critical_cost: float
    initial_factor_count: int
    residual_factor_count: int
    inclusion_minimal: bool
    reinsertion_used: bool
    trace: tuple[OMBPDStep, ...]
    runtime_seconds: float
    damping: float
    tolerance: float
    max_bp_iterations: int
    oscillation_average_window: int
    repair_applied: bool = False
    repair_status: str = "DISABLED"
    repair_drop_limit: int = 0
    repair_max_rounds: int = 0
    repair_trace: tuple[OMBPDRepairStep, ...] = ()
    version: str = OM_BPD_VERSION


@dataclass(frozen=True)
class MotifCoverCandidate:
    """One feasible upper-bound candidate considered by OM-BPD-C."""

    source: str
    sequence: tuple[NodeId, ...]
    selected_set: frozenset[NodeId]
    objective: float
    residual_factor_count: int
    motif_position: int | None = None


@dataclass(frozen=True)
class OMBPDCertifiedResult:
    """Anytime OM-BPD result with explicit lower/upper-bound accounting."""

    method: str
    critical_sequence: tuple[NodeId, ...]
    critical_set: frozenset[NodeId]
    critical_cost: float
    residual_factor_count: int
    inclusion_minimal: bool
    heuristic_incumbent_source: str
    heuristic_incumbent_cost: float
    incumbent_source: str
    candidates: tuple[MotifCoverCandidate, ...]
    bpd_result: OMBPDResult
    motif_position_runtime_seconds: float
    lp_status: str
    lp_objective: float | None
    lp_runtime_seconds: float
    lp_certificate_valid: bool
    lp_validation_reason: str
    effective_lower_bound: float | None
    lp_solution_integral: bool
    lp_fractional_variable_count: int
    lp_max_fractionality: float
    lp_max_constraint_violation: float
    lp_objective_recompute_error: float | None
    exact_closure_attempted: bool
    exact_status: str
    exact_objective: float | None
    exact_lower_bound: float | None
    exact_runtime_seconds: float
    certified_optimal: bool
    bound_consistent: bool
    certificate_source: str
    termination_reason: str
    bound_absolute_tolerance: float
    bound_relative_tolerance: float
    absolute_gap: float | None
    relative_gap: float | None
    heuristic_selection_runtime_seconds: float
    certificate_runtime_seconds: float
    runtime_seconds: float
    version: str = OM_BPD_CERTIFIED_VERSION


@dataclass(frozen=True)
class MotifCoverSolveResult:
    """Exact or LP solution metadata without overstating optimality."""

    method: str
    status: str
    optimal: bool
    selected_set: frozenset[NodeId]
    objective: float | None
    lower_bound: float | None
    relative_gap: float | None
    residual_factor_count: int
    runtime_seconds: float
    message: str
    solution_values: tuple[tuple[NodeId, float], ...] = ()


@dataclass(frozen=True)
class OptimalFamilyResult:
    """Small-instance audit of non-unique optimal critical sets."""

    optimum_cost: float
    reference_set: frozenset[NodeId]
    must_nodes: frozenset[NodeId]
    possible_nodes: frozenset[NodeId]
    interchangeable_nodes: frozenset[NodeId]
    exact: bool
    runtime_seconds: float


def build_operational_motif_system(graph: nx.Graph) -> OperationalMotifSystem:
    """Match the preregistered S-C-L-E motif on a validated equipment graph."""

    equipment = validate_graph(graph)
    removable = tuple(
        node
        for node in stable_nodes(equipment.nodes)
        if bool(equipment.nodes[node].get("removable", True))
    )
    removable_set = frozenset(removable)
    motifs: list[OperationalMotifInstance] = []
    for motif_id, ordered in enumerate(enumerate_role_motifs(equipment)):
        candidates = frozenset(node for node in ordered if node in removable_set)
        if not candidates:
            raise ValueError(
                "An operational motif has no removable physical equipment node: "
                f"{ordered!r}."
            )
        motifs.append(
            OperationalMotifInstance(
                motif_id=motif_id,
                template="S-C-L-E",
                ordered_nodes=tuple(ordered),
                removable_nodes=candidates,
            )
        )
    factors = tuple(item.removable_nodes for item in motifs)
    return OperationalMotifSystem(
        graph=equipment,
        motifs=tuple(motifs),
        factors=factors,
        removable_nodes=removable,
        source_nodes=tuple(
            node
            for node in stable_nodes(equipment.nodes)
            if equipment.nodes[node]["role"] == "S"
        ),
        target_nodes=tuple(
            node
            for node in stable_nodes(equipment.nodes)
            if equipment.nodes[node]["role"] == "E"
        ),
    )


def build_path_closed_system(graph: nx.Graph) -> OperationalMotifSystem:
    """Build the path-closed system declared by one frozen graph.

    Legacy equipment graphs retain the preregistered S-C-L-E matcher.  A
    generic directed-task-thread graph is accepted only when its serialized
    metadata explicitly declares the audited v2 input semantics and fixed
    source/target boundaries.  This keeps input adaptation separate from the
    MPCF objective and solvers.
    """

    input_semantics = str(graph.graph.get("input_semantics", ""))
    if input_semantics.startswith("mpcf-directed-task-thread-mapping-v"):
        sources = graph.graph.get("source_nodes")
        targets = graph.graph.get("target_nodes")
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                "A serialized directed task thread requires metadata.source_nodes."
            )
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                "A serialized directed task thread requires metadata.target_nodes."
            )
        return build_directed_task_path_system(
            nx.DiGraph(graph),
            source_nodes=sources,
            target_nodes=targets,
            library_version=str(
                graph.graph.get("library_version", "directed-task-thread-v2")
            ),
        )
    return build_operational_motif_system(graph)


def build_directed_task_path_system(
    graph: nx.DiGraph,
    *,
    source_nodes: Iterable[NodeId],
    target_nodes: Iterable[NodeId],
    library_version: str = "directed-task-thread-v1",
    max_paths: int = MAX_AUDITED_TASK_PATHS,
) -> OperationalMotifSystem:
    """Build a path-closed system without imposing four fixed role labels.

    This adapter changes only the input representation.  MPCF still maximizes
    the exact minimum weighted vertex cut after finite node-cost uplift.
    Nodes require positive ``attack_cost`` and ``protect_cost`` attributes;
    ``removable`` defaults to true.  Optional ``role`` or ``task_stage`` labels
    remain descriptive and never determine path membership.
    """

    if not isinstance(graph, nx.DiGraph) or not graph.is_directed():
        raise ValueError("A directed task thread requires a NetworkX DiGraph.")
    if graph.number_of_nodes() == 0:
        raise ValueError("A directed task thread cannot be empty.")
    normalized = nx.DiGraph()
    for node in stable_nodes(graph.nodes):
        data = dict(graph.nodes[node])
        for field in ("attack_cost", "protect_cost"):
            try:
                value = float(data[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Node {node!r} requires a finite positive {field}."
                ) from exc
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Node {node!r} requires a finite positive {field}."
                )
            data[field] = value
        data["capacity"] = int(data.get("capacity", 1))
        if data["capacity"] < 0:
            raise ValueError(f"Node {node!r} capacity must be non-negative.")
        data["removable"] = bool(data.get("removable", True))
        data.setdefault("task_stage", str(data.get("role", "component")))
        normalized.add_node(node, **data)
    for left, right in graph.edges:
        if left == right:
            raise ValueError(f"Self-loop is not allowed: {left!r} -> {right!r}.")
        normalized.add_edge(left, right, **dict(graph.edges[left, right]))
    normalized.graph.update(graph.graph)

    sources = tuple(stable_nodes(source_nodes))
    targets = tuple(stable_nodes(target_nodes))
    provisional = OperationalMotifSystem(
        graph=normalized,
        motifs=(),
        factors=(),
        removable_nodes=tuple(
            node
            for node in stable_nodes(normalized.nodes)
            if bool(normalized.nodes[node].get("removable", True))
        ),
        library_version=library_version,
        source_nodes=sources,
        target_nodes=targets,
    )
    sources, targets = task_path_endpoints(provisional)
    paths = enumerate_directed_task_paths(
        normalized, sources, targets, max_paths=max_paths
    )
    if not paths:
        raise ValueError("The declared task boundary contains no directed task path.")
    removable = frozenset(provisional.removable_nodes)
    motifs: list[OperationalMotifInstance] = []
    for motif_id, path in enumerate(paths):
        factor = frozenset(node for node in path if node in removable)
        if not factor:
            raise ValueError(
                "A declared task path has no removable component: " f"{path!r}."
            )
        motifs.append(
            OperationalMotifInstance(
                motif_id=motif_id,
                template="DIRECTED_TASK_THREAD",
                ordered_nodes=path,
                removable_nodes=factor,
            )
        )
    return OperationalMotifSystem(
        graph=normalized,
        motifs=tuple(motifs),
        factors=tuple(item.removable_nodes for item in motifs),
        removable_nodes=provisional.removable_nodes,
        library_version=library_version,
        source_nodes=sources,
        target_nodes=targets,
    )


def task_path_endpoints(
    system: OperationalMotifSystem,
) -> tuple[tuple[NodeId, ...], tuple[NodeId, ...]]:
    """Return explicit task boundaries, with legacy S/E roles as fallback."""

    sources = system.source_nodes or tuple(
        node
        for node in stable_nodes(system.graph.nodes)
        if system.graph.nodes[node].get("role") == "S"
    )
    targets = system.target_nodes or tuple(
        node
        for node in stable_nodes(system.graph.nodes)
        if system.graph.nodes[node].get("role") == "E"
    )
    if not sources or not targets:
        raise ValueError("A path-closed task system requires explicit source and target nodes.")
    if set(sources) & set(targets):
        raise ValueError("Task source and target node sets must be disjoint.")
    unknown = (set(sources) | set(targets)) - set(system.graph.nodes)
    if unknown:
        raise ValueError(f"Task endpoints contain unknown nodes: {stable_nodes(unknown)!r}.")
    return tuple(stable_nodes(sources)), tuple(stable_nodes(targets))


def enumerate_directed_task_paths(
    graph: nx.DiGraph,
    source_nodes: Iterable[NodeId],
    target_nodes: Iterable[NodeId],
    *,
    max_paths: int = MAX_AUDITED_TASK_PATHS,
) -> tuple[tuple[NodeId, ...], ...]:
    """Enumerate a bounded complete family of directed source-target paths."""

    if max_paths < 1:
        raise ValueError("max_paths must be positive.")
    paths: set[tuple[NodeId, ...]] = set()
    for source in stable_nodes(source_nodes):
        for target in stable_nodes(target_nodes):
            if source == target or not nx.has_path(graph, source, target):
                continue
            for path in nx.all_simple_paths(graph, source, target):
                paths.add(tuple(path))
                if len(paths) > max_paths:
                    raise ValueError(
                        "The complete directed task-path family exceeds the audited "
                        f"limit of {max_paths}; narrow the declared task boundary."
                    )
    return tuple(
        sorted(
            paths,
            key=lambda path: tuple(stable_node_key(node) for node in path),
        )
    )


def audit_path_closed_motif_system(
    graph: nx.Graph | OperationalMotifSystem,
) -> PathClosureAudit:
    """Audit the exact structural condition required by the path-cut oracle.

    A catalog is path closed only when it contains every directed path between
    the declared task boundaries exactly once, contains no additional path,
    and each factor contains precisely the removable nodes on that path.  The
    legacy S-C-L-E system is the four-stage special case.
    """

    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    sources, targets = task_path_endpoints(system)
    expected_paths = enumerate_directed_task_paths(system.graph, sources, targets)
    observed_paths = tuple(motif.ordered_nodes for motif in system.motifs)
    expected_counter = Counter(expected_paths)
    observed_counter = Counter(observed_paths)
    missing = sum((expected_counter - observed_counter).values())
    extra = sum((observed_counter - expected_counter).values())
    removable = frozenset(system.removable_nodes)
    factor_mismatches = sum(
        motif.removable_nodes
        != frozenset(node for node in motif.ordered_nodes if node in removable)
        for motif in system.motifs
    )
    path_closed = missing == 0 and extra == 0 and factor_mismatches == 0
    if path_closed:
        classification = "PATH_CLOSED_WEIGHTED_VERTEX_CUT"
        reason = (
            "Catalog equals the complete directed task-path family with exact "
            "removable-node factors."
        )
    else:
        classification = "GENERAL_MOTIF_COVER"
        reason = (
            f"missing_paths={missing}; extra_motifs={extra}; "
            f"factor_mismatches={factor_mismatches}."
        )
    return PathClosureAudit(
        path_closed=path_closed,
        observed_motif_count=len(observed_paths),
        expected_path_count=len(expected_paths),
        missing_path_count=missing,
        extra_motif_count=extra,
        removable_factor_mismatch_count=factor_mismatches,
        classification=classification,
        reason=reason,
    )


def _factor_key(factor: frozenset[NodeId]) -> tuple[int, tuple[tuple[str, str], ...]]:
    return len(factor), tuple(stable_node_key(node) for node in stable_nodes(factor))


def reduce_factor_family(
    factors: Iterable[Iterable[NodeId]],
) -> tuple[frozenset[NodeId], ...]:
    """Remove duplicates and constraints dominated by a strict subset."""

    unique = {frozenset(factor) for factor in factors}
    if any(not factor for factor in unique):
        raise ValueError("An empty motif factor is unbreakable by equipment removal.")
    ordered = sorted(unique, key=_factor_key)
    minimal: list[frozenset[NodeId]] = []
    for factor in ordered:
        if any(previous <= factor for previous in minimal):
            continue
        minimal.append(factor)
    return tuple(minimal)


def residual_factors(
    factors: Iterable[Iterable[NodeId]],
    selected: Iterable[NodeId],
) -> tuple[frozenset[NodeId], ...]:
    """Return motif constraints not yet intersected by ``selected``."""

    chosen = frozenset(selected)
    return tuple(
        factor_set
        for factor_set in (frozenset(factor) for factor in factors)
        if factor_set.isdisjoint(chosen)
    )


def covers_all_factors(
    factors: Iterable[Iterable[NodeId]], selected: Iterable[NodeId]
) -> bool:
    chosen = frozenset(selected)
    return all(not frozenset(factor).isdisjoint(chosen) for factor in factors)


def is_inclusion_minimal_cover(
    factors: Iterable[Iterable[NodeId]], selected: Iterable[NodeId]
) -> bool:
    chosen = frozenset(selected)
    if not covers_all_factors(factors, chosen):
        return False
    return all(not covers_all_factors(factors, chosen - {node}) for node in chosen)


def reverse_reinsert(
    factors: Iterable[Iterable[NodeId]], sequence: Sequence[NodeId]
) -> tuple[NodeId, ...]:
    """Greedily remove redundant selected nodes in reverse decimation order."""

    original = tuple(dict.fromkeys(sequence))
    kept = set(original)
    frozen_factors = tuple(frozenset(factor) for factor in factors)
    for node in reversed(original):
        candidate = kept - {node}
        if covers_all_factors(frozen_factors, candidate):
            kept.remove(node)
    return tuple(node for node in original if node in kept)


def _costs(
    system: OperationalMotifSystem,
    costs: Mapping[NodeId, float] | None,
) -> dict[NodeId, float]:
    values = {
        node: float(
            costs[node]
            if costs is not None
            else system.graph.nodes[node]["attack_cost"]
        )
        for node in system.removable_nodes
    }
    bad = [node for node, value in values.items() if not isfinite(value) or value <= 0]
    if bad:
        raise ValueError(f"Removal costs must be finite and positive: {bad!r}.")
    return values


def _cover_cost(nodes: Iterable[NodeId], costs: Mapping[NodeId, float]) -> float:
    return float(sum(float(costs[node]) for node in nodes))


def _ordered_cover_subset(
    factors: Sequence[frozenset[NodeId]],
    selected: Iterable[NodeId],
    costs: Mapping[NodeId, float],
    *,
    preferred_order: Sequence[NodeId] = (),
) -> tuple[NodeId, ...]:
    """Order and prune a feasible cover by residual gain per unit cost."""

    remaining = set(selected)
    if not covers_all_factors(factors, remaining):
        raise ValueError("Candidate nodes do not cover every operational motif.")
    active = tuple(factors)
    preferred_rank = {node: rank for rank, node in enumerate(preferred_order)}
    ordered: list[NodeId] = []
    while active:
        eligible = [
            node for node in remaining if any(node in factor for factor in active)
        ]
        if not eligible:
            raise RuntimeError("A feasible candidate lost coverage while being ordered.")
        node = min(
            eligible,
            key=lambda candidate: (
                -sum(candidate in factor for factor in active) / costs[candidate],
                preferred_rank.get(candidate, len(preferred_rank)),
                costs[candidate],
                stable_node_key(candidate),
            ),
        )
        ordered.append(node)
        remaining.remove(node)
        active = tuple(factor for factor in active if node not in factor)
    return reverse_reinsert(factors, tuple(ordered))


def _cover_candidate(
    factors: Sequence[frozenset[NodeId]],
    selected: Iterable[NodeId],
    costs: Mapping[NodeId, float],
    *,
    source: str,
    motif_position: int | None = None,
    preferred_order: Sequence[NodeId] = (),
) -> MotifCoverCandidate:
    sequence = _ordered_cover_subset(
        factors,
        selected,
        costs,
        preferred_order=preferred_order,
    )
    chosen = frozenset(sequence)
    return MotifCoverCandidate(
        source=source,
        sequence=sequence,
        selected_set=chosen,
        objective=_cover_cost(chosen, costs),
        residual_factor_count=len(residual_factors(factors, chosen)),
        motif_position=motif_position,
    )


def motif_position_cover_candidates(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    costs: Mapping[NodeId, float] | None = None,
) -> tuple[MotifCoverCandidate, ...]:
    """Return feasible covers induced by mandatory positions of the motif template.

    The construction uses matched motif positions, not node identifiers.  A
    position is admitted only when its removable nodes hit every original
    motif, so it is an audited upper bound rather than an assumed role rule.
    """

    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    frozen = tuple(system.factors)
    cost = _costs(system, costs)
    if not frozen:
        return (
            MotifCoverCandidate(
                source="empty_motif_system",
                sequence=(),
                selected_set=frozenset(),
                objective=0.0,
                residual_factor_count=0,
                motif_position=None,
            ),
        )
    max_positions = max(len(motif.ordered_nodes) for motif in system.motifs)
    removable = frozenset(system.removable_nodes)
    output: list[MotifCoverCandidate] = []
    for position in range(max_positions):
        if any(position >= len(motif.ordered_nodes) for motif in system.motifs):
            continue
        selected = {
            motif.ordered_nodes[position]
            for motif in system.motifs
            if motif.ordered_nodes[position] in removable
        }
        if not covers_all_factors(frozen, selected):
            continue
        roles = {
            str(system.graph.nodes[node].get("role", "")) for node in selected
        }
        role_label = next(iter(roles)) if len(roles) == 1 else "mixed"
        output.append(
            _cover_candidate(
                frozen,
                selected,
                cost,
                source=f"motif_position_{position}_{role_label}",
                motif_position=position,
            )
        )
    return tuple(output)


def motif_position_cover_sequence(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    costs: Mapping[NodeId, float] | None = None,
) -> tuple[NodeId, ...]:
    """Return the least-cost audited mandatory-position cover."""

    candidates = motif_position_cover_candidates(graph, costs=costs)
    if not candidates:
        raise RuntimeError("No motif position induces a feasible removable-node cover.")
    best = min(
        candidates,
        key=lambda candidate: (
            candidate.objective,
            len(candidate.selected_set),
            candidate.motif_position if candidate.motif_position is not None else -1,
            tuple(stable_node_key(node) for node in candidate.sequence),
        ),
    )
    return best.sequence


def _run_min_sum(
    active: tuple[frozenset[NodeId], ...],
    costs: Mapping[NodeId, float],
    *,
    damping: float,
    tolerance: float,
    max_iterations: int,
    average_window: int,
) -> tuple[dict[NodeId, float], int, bool, float]:
    factor_messages = {
        (factor_index, node): 0.0
        for factor_index, factor in enumerate(active)
        for node in factor
    }
    incidence: dict[NodeId, list[int]] = {}
    for factor_index, factor in enumerate(active):
        for node in factor:
            incidence.setdefault(node, []).append(factor_index)
    history: list[dict[tuple[int, NodeId], float]] = []
    converged = False
    max_delta = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        variable_messages: dict[tuple[int, NodeId], float] = {}
        for node, factor_indices in incidence.items():
            total = float(costs[node]) + sum(
                factor_messages[(index, node)] for index in factor_indices
            )
            for index in factor_indices:
                variable_messages[(index, node)] = total - factor_messages[(index, node)]

        updated: dict[tuple[int, NodeId], float] = {}
        max_delta = 0.0
        for factor_index, factor in enumerate(active):
            for node in factor:
                incoming = [
                    variable_messages[(factor_index, other)]
                    for other in factor
                    if other != node
                ]
                if not incoming:
                    raise RuntimeError("Singleton constraints must be reduced before BP.")
                raw = -max(0.0, min(incoming))
                old = factor_messages[(factor_index, node)]
                value = (1.0 - damping) * old + damping * raw
                updated[(factor_index, node)] = value
                max_delta = max(max_delta, abs(value - old))
        factor_messages = updated
        history.append(dict(updated))
        if len(history) > average_window:
            history.pop(0)
        if max_delta <= tolerance:
            converged = True
            break

    if not converged and history:
        factor_messages = {
            key: float(sum(snapshot[key] for snapshot in history) / len(history))
            for key in factor_messages
        }
    beliefs = {
        node: float(costs[node])
        + sum(factor_messages[(index, node)] for index in indices)
        for node, indices in incidence.items()
    }
    return beliefs, iteration, converged, max_delta


def solve_om_bpd(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    factors: Iterable[Iterable[NodeId]] | None = None,
    costs: Mapping[NodeId, float] | None = None,
    damping: float = 0.5,
    tolerance: float = 1e-6,
    max_bp_iterations: int = 200,
    oscillation_average_window: int = 20,
    reinsertion: bool = True,
    bounded_exchange_repair: bool = False,
    repair_drop_limit: int = 3,
    repair_max_rounds: int = 3,
    repair_max_nodes: int = 32,
    repair_time_limit: float = 5.0,
    method: str = "OM-BPD",
) -> OMBPDResult:
    """Solve operational-motif cover by Min-Sum guided decimation."""

    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must satisfy 0 < damping <= 1.")
    if tolerance <= 0.0 or max_bp_iterations < 1:
        raise ValueError("tolerance and max_bp_iterations must be positive.")
    if oscillation_average_window < 1:
        raise ValueError("oscillation_average_window must be positive.")
    if bounded_exchange_repair:
        if repair_drop_limit < 1 or repair_max_rounds < 1:
            raise ValueError("repair_drop_limit and repair_max_rounds must be positive.")
        if repair_max_nodes < 1 or repair_time_limit <= 0.0:
            raise ValueError("repair_max_nodes and repair_time_limit must be positive.")
    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    original = tuple(
        frozenset(factor) for factor in (system.factors if factors is None else factors)
    )
    candidate_nodes = frozenset(system.removable_nodes)
    unknown = sorted(
        {node for factor in original for node in factor if node not in candidate_nodes},
        key=stable_node_key,
    )
    if unknown:
        raise ValueError(f"Motif factors contain non-removable nodes: {unknown!r}.")
    cost = _costs(system, costs)
    started = perf_counter()
    active = reduce_factor_family(original) if original else ()
    sequence: list[NodeId] = []
    trace: list[OMBPDStep] = []

    while active:
        active = reduce_factor_family(active)
        singleton_nodes = stable_nodes(
            next(iter(factor)) for factor in active if len(factor) == 1
        )
        if singleton_nodes:
            selected = min(
                singleton_nodes,
                key=lambda node: (cost[node], stable_node_key(node)),
            )
            score = -float("inf")
            iterations = 0
            converged = True
            max_delta = 0.0
            reason = "forced_singleton"
            candidate_count = len({node for factor in active for node in factor})
        else:
            beliefs, iterations, converged, max_delta = _run_min_sum(
                active,
                cost,
                damping=damping,
                tolerance=tolerance,
                max_iterations=max_bp_iterations,
                average_window=oscillation_average_window,
            )
            selected = min(
                beliefs,
                key=lambda node: (
                    beliefs[node],
                    cost[node],
                    stable_node_key(node),
                ),
            )
            score = beliefs[selected]
            reason = "min_sum_belief"
            candidate_count = len(beliefs)
        before = len(active)
        sequence.append(selected)
        active = tuple(factor for factor in active if selected not in factor)
        trace.append(
            OMBPDStep(
                step=len(sequence),
                selected_node=selected,
                selected_score=float(score),
                selection_reason=reason,
                active_factor_count_before=before,
                active_factor_count_after=len(active),
                candidate_count=candidate_count,
                bp_iterations=iterations,
                converged=converged,
                max_message_delta=float(max_delta),
            )
        )

    raw_sequence = tuple(sequence)
    critical_sequence = (
        reverse_reinsert(original, raw_sequence) if reinsertion else raw_sequence
    )
    repair_trace: tuple[OMBPDRepairStep, ...] = ()
    repair_status = "DISABLED"
    if bounded_exchange_repair:
        critical_sequence, repair_trace, repair_status = _bounded_exchange_repair(
            system,
            original,
            critical_sequence,
            cost,
            drop_limit=repair_drop_limit,
            max_rounds=repair_max_rounds,
            max_nodes=repair_max_nodes,
            time_limit=repair_time_limit,
        )
    critical_set = frozenset(critical_sequence)
    remaining = residual_factors(original, critical_set)
    return OMBPDResult(
        method=method,
        raw_sequence=raw_sequence,
        raw_set=frozenset(raw_sequence),
        critical_sequence=critical_sequence,
        critical_set=critical_set,
        raw_cost=_cover_cost(raw_sequence, cost),
        critical_cost=_cover_cost(critical_sequence, cost),
        initial_factor_count=len(original),
        residual_factor_count=len(remaining),
        inclusion_minimal=is_inclusion_minimal_cover(original, critical_set),
        reinsertion_used=reinsertion,
        trace=tuple(trace),
        runtime_seconds=perf_counter() - started,
        damping=damping,
        tolerance=tolerance,
        max_bp_iterations=max_bp_iterations,
        oscillation_average_window=oscillation_average_window,
        repair_applied=any(step.improvement > 1e-9 for step in repair_trace),
        repair_status=repair_status,
        repair_drop_limit=(repair_drop_limit if bounded_exchange_repair else 0),
        repair_max_rounds=(repair_max_rounds if bounded_exchange_repair else 0),
        repair_trace=repair_trace,
    )


def pairwise_projection_factors(
    factors: Iterable[Iterable[NodeId]],
) -> tuple[frozenset[NodeId], ...]:
    """Return the 2-section edge constraints used by the projection ablation."""

    pairs: set[frozenset[NodeId]] = set()
    for factor in factors:
        nodes = stable_nodes(set(factor))
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                pairs.add(frozenset((left, right)))
    return tuple(sorted(pairs, key=_factor_key))


def motif_degree_sequence(
    system: OperationalMotifSystem,
    *,
    costs: Mapping[NodeId, float] | None = None,
    reinsertion: bool = True,
) -> tuple[NodeId, ...]:
    """Static factor-degree ranking ablation, normalized by removal cost."""

    cost = _costs(system, costs)
    counts = {node: 0 for node in system.removable_nodes}
    for factor in system.factors:
        for node in factor:
            counts[node] += 1
    ranking = tuple(
        sorted(
            (node for node, count in counts.items() if count),
            key=lambda node: (
                -(counts[node] / cost[node]),
                -counts[node],
                cost[node],
                stable_node_key(node),
            ),
        )
    )
    selected: list[NodeId] = []
    for node in ranking:
        selected.append(node)
        if covers_all_factors(system.factors, selected):
            break
    sequence = tuple(selected)
    return reverse_reinsert(system.factors, sequence) if reinsertion else sequence


def _incidence_matrix(
    system: OperationalMotifSystem,
    factors: Sequence[frozenset[NodeId]],
) -> tuple[tuple[NodeId, ...], csr_matrix]:
    nodes = system.removable_nodes
    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    columns: list[int] = []
    for row, factor in enumerate(factors):
        for node in factor:
            rows.append(row)
            columns.append(node_index[node])
    data = np.ones(len(rows), dtype=float)
    return nodes, csr_matrix((data, (rows, columns)), shape=(len(factors), len(nodes)))


def solve_path_closed_motif_cut(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    costs: Mapping[NodeId, float] | None = None,
) -> MotifCoverSolveResult:
    """Solve a complete path-family motif cover by weighted vertex cut.

    The reduction is valid only when the audited catalog is exactly all
    directed paths between the declared task boundaries. Each component node
    is split into an in/out pair;
    its split arc has the removal cost, while topology arcs have a finite
    big-M capacity strictly larger than the cost of removing every removable
    node. A minimum super-source/super-sink cut is therefore a minimum-cost
    equipment set intersecting every operational path.
    """

    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    audit = audit_path_closed_motif_system(system)
    started = perf_counter()
    if not audit.path_closed:
        return MotifCoverSolveResult(
            method="OM-PathCutExact",
            status="NOT_APPLICABLE",
            optimal=False,
            selected_set=frozenset(),
            objective=None,
            lower_bound=None,
            relative_gap=None,
            residual_factor_count=len(system.factors),
            runtime_seconds=perf_counter() - started,
            message=audit.reason,
        )

    cost = _costs(system, costs)
    if not system.factors:
        return MotifCoverSolveResult(
            method="OM-PathCutExact",
            status="OPTIMAL",
            optimal=True,
            selected_set=frozenset(),
            objective=0.0,
            lower_bound=0.0,
            relative_gap=0.0,
            residual_factor_count=0,
            runtime_seconds=perf_counter() - started,
            message="The path-closed motif family is empty.",
        )

    total_removal_cost = fsum(cost.values())
    big_m = fsum((total_removal_cost, min(cost.values())))
    if not isfinite(big_m) or big_m <= total_removal_cost:
        raise ValueError(
            "Removal costs cannot form a finite path-cut big-M; rescale costs."
        )

    auxiliary = nx.DiGraph()
    source = ("__rmcd_path_cut_source__",)
    sink = ("__rmcd_path_cut_sink__",)
    auxiliary.add_node(source)
    auxiliary.add_node(sink)
    task_sources, task_targets = task_path_endpoints(system)
    task_source_set = frozenset(task_sources)
    task_target_set = frozenset(task_targets)
    removable = frozenset(system.removable_nodes)
    for node in stable_nodes(system.graph.nodes):
        node_in = ("__rmcd_path_cut_in__", node)
        node_out = ("__rmcd_path_cut_out__", node)
        auxiliary.add_edge(
            node_in,
            node_out,
            capacity=cost[node] if node in removable else big_m,
            arc_kind="equipment" if node in removable else "fixed_equipment",
            equipment_node=node,
        )
        if node in task_source_set:
            auxiliary.add_edge(
                source, node_in, capacity=big_m, arc_kind="topology"
            )
        if node in task_target_set:
            auxiliary.add_edge(
                node_out, sink, capacity=big_m, arc_kind="topology"
            )
    for left, right in sorted(
        system.graph.edges,
        key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1])),
    ):
        auxiliary.add_edge(
            ("__rmcd_path_cut_out__", left),
            ("__rmcd_path_cut_in__", right),
            capacity=big_m,
            arc_kind="topology",
        )

    cut_value, partition = nx.minimum_cut(
        auxiliary,
        source,
        sink,
        capacity="capacity",
        flow_func=nx.algorithms.flow.edmonds_karp,
    )
    source_side, sink_side = partition
    selected = {
        node
        for node in removable
        if ("__rmcd_path_cut_in__", node) in source_side
        and ("__rmcd_path_cut_out__", node) in sink_side
    }
    if float(cut_value) >= big_m - 1e-8:
        raise RuntimeError(
            "The path-cut crossed a fixed topology arc; the catalog is not "
            "finitely dismantlable under the declared removable nodes."
        )
    ordered = _ordered_cover_subset(system.factors, selected, cost)
    chosen = frozenset(ordered)
    residual = len(residual_factors(system.factors, chosen))
    objective = _cover_cost(chosen, cost)
    tolerance = 1e-8 * max(1.0, abs(float(cut_value)), abs(objective))
    if residual:
        raise RuntimeError("The path-cut certificate does not hit every motif.")
    if not isclose(objective, float(cut_value), rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            "The recovered equipment cut does not match the minimum-cut value: "
            f"set_cost={objective}; cut_value={cut_value}."
        )
    return MotifCoverSolveResult(
        method="OM-PathCutExact",
        status="OPTIMAL",
        optimal=True,
        selected_set=chosen,
        objective=objective,
        lower_bound=objective,
        relative_gap=0.0,
        residual_factor_count=0,
        runtime_seconds=perf_counter() - started,
        message=(
            "Certified by node-splitting weighted minimum cut on the complete "
            "directed task-path family."
        ),
    )


def _milp_status_name(status: int) -> str:
    return {
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "ERROR",
    }.get(int(status), "ERROR")


def solve_exact_motif_cover(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    factors: Iterable[Iterable[NodeId]] | None = None,
    costs: Mapping[NodeId, float] | None = None,
    force_selected: Iterable[NodeId] = (),
    force_excluded: Iterable[NodeId] = (),
    objective_upper_bound: float | None = None,
    time_limit: float | None = None,
) -> MotifCoverSolveResult:
    """Solve the weighted motif cover MILP and preserve solver status."""

    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    raw = system.factors if factors is None else tuple(factors)
    frozen = reduce_factor_family(raw) if raw else ()
    cost = _costs(system, costs)
    selected = frozenset(force_selected)
    excluded = frozenset(force_excluded)
    if selected & excluded:
        raise ValueError("force_selected and force_excluded must be disjoint.")
    unknown = (selected | excluded) - frozenset(system.removable_nodes)
    if unknown:
        raise ValueError(f"Unknown forced nodes: {stable_nodes(unknown)!r}.")
    started = perf_counter()
    if not frozen:
        chosen = selected
        return MotifCoverSolveResult(
            "OM-Exact", "OPTIMAL", True, chosen, _cover_cost(chosen, cost),
            _cover_cost(chosen, cost), 0.0, 0, perf_counter() - started,
            "No operational motif constraints.",
        )
    nodes, matrix = _incidence_matrix(system, frozen)
    lower = np.zeros(len(nodes), dtype=float)
    upper = np.ones(len(nodes), dtype=float)
    index = {node: position for position, node in enumerate(nodes)}
    for node in selected:
        lower[index[node]] = 1.0
    for node in excluded:
        upper[index[node]] = 0.0
    objective = np.array([cost[node] for node in nodes], dtype=float)
    constraint_matrix = matrix
    constraint_lower = np.ones(len(frozen), dtype=float)
    constraint_upper = np.full(len(frozen), np.inf, dtype=float)
    if objective_upper_bound is not None:
        if not isfinite(objective_upper_bound) or objective_upper_bound < 0.0:
            raise ValueError("objective_upper_bound must be finite and nonnegative.")
        constraint_matrix = vstack(
            (matrix, csr_matrix(objective.reshape(1, -1))), format="csr"
        )
        constraint_lower = np.concatenate((constraint_lower, [-np.inf]))
        constraint_upper = np.concatenate(
            (constraint_upper, [float(objective_upper_bound) + 1e-9])
        )
    options: dict[str, Any] = {"presolve": True, "mip_rel_gap": 0.0}
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive when supplied.")
        options["time_limit"] = float(time_limit)
    result = milp(
        c=objective,
        integrality=np.ones(len(nodes), dtype=int),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            constraint_matrix, constraint_lower, constraint_upper
        ),
        options=options,
    )
    status_name = _milp_status_name(int(result.status))
    chosen = frozenset(
        node for node, value in zip(nodes, result.x if result.x is not None else [])
        if float(value) >= 0.5
    )
    residual = len(residual_factors(frozen, chosen))
    objective = _cover_cost(chosen, cost) if chosen and residual == 0 else (
        0.0 if not chosen and residual == 0 else None
    )
    lower_bound = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    return MotifCoverSolveResult(
        method="OM-Exact",
        status=status_name,
        optimal=result.status == 0 and residual == 0,
        selected_set=chosen,
        objective=objective,
        lower_bound=(float(lower_bound) if lower_bound is not None else None),
        relative_gap=(float(gap) if gap is not None else None),
        residual_factor_count=residual,
        runtime_seconds=perf_counter() - started,
        message=str(result.message),
    )


def _solve_bounded_exchange_neighbourhood(
    system: OperationalMotifSystem,
    factors: Sequence[frozenset[NodeId]],
    costs: Mapping[NodeId, float],
    incumbent: frozenset[NodeId],
    *,
    drop_limit: int,
    time_limit: float,
) -> tuple[MotifCoverSolveResult, int]:
    """Optimize one exact neighbourhood while retaining most incumbent nodes."""

    frozen = reduce_factor_family(factors) if factors else ()
    started = perf_counter()
    retained_required = max(1, len(incumbent) - drop_limit)
    nodes, matrix = _incidence_matrix(system, frozen)
    index = {node: position for position, node in enumerate(nodes)}
    unknown = incumbent - frozenset(nodes)
    if unknown:
        raise ValueError(f"Unknown incumbent nodes: {stable_nodes(unknown)!r}.")

    retention_row = np.zeros(len(nodes), dtype=float)
    for node in incumbent:
        retention_row[index[node]] = 1.0
    constraint_matrix = vstack(
        (matrix, csr_matrix(retention_row.reshape(1, -1))), format="csr"
    )
    lower = np.concatenate(
        (np.ones(len(frozen), dtype=float), np.array([retained_required], dtype=float))
    )
    upper = np.full(len(frozen) + 1, np.inf, dtype=float)
    result = milp(
        c=np.array([costs[node] for node in nodes], dtype=float),
        integrality=np.ones(len(nodes), dtype=int),
        bounds=Bounds(np.zeros(len(nodes)), np.ones(len(nodes))),
        constraints=LinearConstraint(constraint_matrix, lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": time_limit},
    )
    chosen = frozenset(
        node
        for node, value in zip(nodes, result.x if result.x is not None else [])
        if float(value) >= 0.5
    )
    residual = len(residual_factors(frozen, chosen))
    objective = _cover_cost(chosen, costs) if residual == 0 else None
    lower_bound = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    return (
        MotifCoverSolveResult(
            method="OM-BPD-R-neighbourhood",
            status=_milp_status_name(int(result.status)),
            optimal=result.status == 0 and residual == 0,
            selected_set=chosen,
            objective=objective,
            lower_bound=(float(lower_bound) if lower_bound is not None else None),
            relative_gap=(float(gap) if gap is not None else None),
            residual_factor_count=residual,
            runtime_seconds=perf_counter() - started,
            message=str(result.message),
        ),
        retained_required,
    )


def _bounded_exchange_repair(
    system: OperationalMotifSystem,
    factors: Sequence[frozenset[NodeId]],
    sequence: Sequence[NodeId],
    costs: Mapping[NodeId, float],
    *,
    drop_limit: int,
    max_rounds: int,
    max_nodes: int,
    time_limit: float,
) -> tuple[tuple[NodeId, ...], tuple[OMBPDRepairStep, ...], str]:
    """Apply monotone local exact exchanges without claiming global optimality."""

    current_sequence = tuple(dict.fromkeys(sequence))
    current_set = frozenset(current_sequence)
    if not factors:
        return current_sequence, (), "SKIPPED_NO_FACTORS"
    if len(system.removable_nodes) > max_nodes:
        return current_sequence, (), "SKIPPED_NODE_LIMIT"
    if len(current_set) < 2:
        return current_sequence, (), "SKIPPED_SMALL_SET"

    trace: list[OMBPDRepairStep] = []
    improved_once = False
    status = "NO_IMPROVEMENT"
    for round_index in range(1, max_rounds + 1):
        cost_before = _cover_cost(current_set, costs)
        solved, retained_required = _solve_bounded_exchange_neighbourhood(
            system,
            factors,
            costs,
            current_set,
            drop_limit=min(drop_limit, len(current_set) - 1),
            time_limit=time_limit,
        )
        candidate_set = solved.selected_set if solved.optimal else current_set
        candidate_sequence = tuple(
            node for node in current_sequence if node in candidate_set
        ) + tuple(stable_nodes(candidate_set - current_set))
        if solved.optimal:
            candidate_sequence = reverse_reinsert(factors, candidate_sequence)
            candidate_set = frozenset(candidate_sequence)
        cost_after = _cover_cost(candidate_set, costs)
        improvement = cost_before - cost_after if solved.optimal else 0.0
        trace.append(
            OMBPDRepairStep(
                round=round_index,
                cost_before=cost_before,
                cost_after=cost_after,
                improvement=improvement,
                selected_count_before=len(current_set),
                selected_count_after=len(candidate_set),
                retained_required=retained_required,
                dropped_nodes=tuple(stable_nodes(current_set - candidate_set)),
                added_nodes=tuple(stable_nodes(candidate_set - current_set)),
                solver_status=solved.status,
                solver_optimal=solved.optimal,
                runtime_seconds=solved.runtime_seconds,
            )
        )
        if not solved.optimal:
            status = (
                "SOLVER_LIMIT_AFTER_IMPROVEMENT"
                if improved_once
                else f"SOLVER_{solved.status}"
            )
            break
        if improvement <= 1e-9:
            status = "IMPROVED_LOCAL_OPTIMUM" if improved_once else "NO_IMPROVEMENT"
            break
        current_sequence = candidate_sequence
        current_set = candidate_set
        improved_once = True
    else:
        status = "IMPROVED_ROUND_LIMIT" if improved_once else "NO_IMPROVEMENT"
    return current_sequence, tuple(trace), status


def solve_motif_cover_lp(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    factors: Iterable[Iterable[NodeId]] | None = None,
    costs: Mapping[NodeId, float] | None = None,
) -> MotifCoverSolveResult:
    """Return the standard fractional set-cover lower bound."""

    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    raw = system.factors if factors is None else tuple(factors)
    frozen = reduce_factor_family(raw) if raw else ()
    cost = _costs(system, costs)
    started = perf_counter()
    if not frozen:
        return MotifCoverSolveResult(
            "OM-LP", "OPTIMAL", True, frozenset(), 0.0, 0.0, 0.0, 0,
            perf_counter() - started, "No operational motif constraints.",
        )
    nodes, matrix = _incidence_matrix(system, frozen)
    result = linprog(
        c=np.array([cost[node] for node in nodes], dtype=float),
        A_ub=-matrix,
        b_ub=-np.ones(len(frozen)),
        bounds=[(0.0, 1.0)] * len(nodes),
        method="highs",
    )
    objective = float(result.fun) if result.success else None
    solution_values = tuple(
        (node, float(value))
        for node, value in zip(nodes, result.x if result.x is not None else [])
    )
    return MotifCoverSolveResult(
        method="OM-LP",
        status="OPTIMAL" if result.success else "ERROR",
        optimal=bool(result.success),
        selected_set=frozenset(),
        objective=objective,
        lower_bound=objective,
        relative_gap=0.0 if result.success else None,
        residual_factor_count=len(frozen),
        runtime_seconds=perf_counter() - started,
        message=str(result.message),
        solution_values=solution_values,
    )


def _lp_cover_candidate(
    factors: Sequence[frozenset[NodeId]],
    costs: Mapping[NodeId, float],
    values: Mapping[NodeId, float],
    *,
    integrality_tolerance: float,
) -> tuple[MotifCoverCandidate, bool, int]:
    fractional_count = sum(
        integrality_tolerance < value < 1.0 - integrality_tolerance
        for value in values.values()
    )
    integral = fractional_count == 0
    if integral:
        selected = {
            node for node, value in values.items() if value >= 1.0 - integrality_tolerance
        }
        source = "lp_integral_primal"
    else:
        rank = max((len(factor) for factor in factors), default=1)
        threshold = 1.0 / rank - integrality_tolerance
        selected = {node for node, value in values.items() if value >= threshold}
        source = f"lp_rank_{rank}_threshold"
    active = residual_factors(factors, selected)
    remaining = set(values) - selected
    while active:
        eligible = [
            node for node in remaining if any(node in factor for factor in active)
        ]
        if not eligible:
            raise RuntimeError("LP-guided rounding could not complete a motif cover.")
        node = min(
            eligible,
            key=lambda candidate: (
                -sum(candidate in factor for factor in active) / costs[candidate],
                -values[candidate],
                costs[candidate],
                stable_node_key(candidate),
            ),
        )
        selected.add(node)
        remaining.remove(node)
        active = residual_factors(active, (node,))
    return (
        _cover_candidate(
            factors,
            selected,
            costs,
            source=source,
        ),
        integral,
        fractional_count,
    )


def _lp_certificate_diagnostics(
    system: OperationalMotifSystem,
    factors: Sequence[frozenset[NodeId]],
    costs: Mapping[NodeId, float],
    solved: MotifCoverSolveResult,
    *,
    integrality_tolerance: float,
    feasibility_tolerance: float,
    bound_absolute_tolerance: float,
    bound_relative_tolerance: float,
) -> tuple[bool, str, bool, int, float, float, float | None]:
    """Validate an LP primal before its objective is accepted as a lower bound."""

    values = dict(solved.solution_values)
    if not solved.optimal:
        return False, "LP_NOT_OPTIMAL", False, 0, 0.0, 0.0, None
    if solved.objective is None or not isfinite(solved.objective):
        return False, "LP_OBJECTIVE_INVALID", False, 0, 0.0, 0.0, None
    if not factors:
        tolerance = bound_absolute_tolerance + bound_relative_tolerance * abs(
            solved.objective
        )
        valid = abs(solved.objective) <= tolerance
        return (
            valid,
            "VALID_EMPTY_SYSTEM" if valid else "LP_EMPTY_OBJECTIVE_NONZERO",
            True,
            0,
            0.0,
            0.0,
            abs(solved.objective),
        )

    expected = frozenset(system.removable_nodes)
    observed = frozenset(values)
    if observed != expected:
        return False, "LP_VARIABLE_SET_MISMATCH", False, 0, 0.0, 0.0, None
    if any(not isfinite(value) for value in values.values()):
        return False, "LP_PRIMAL_NONFINITE", False, 0, 0.0, 0.0, None

    bound_violation = max(
        (
            max(0.0, -value, value - 1.0)
            for value in values.values()
        ),
        default=0.0,
    )
    fractional_count = sum(
        integrality_tolerance < value < 1.0 - integrality_tolerance
        for value in values.values()
    )
    integral = fractional_count == 0
    max_fractionality = max(
        (min(abs(value), abs(1.0 - value)) for value in values.values()),
        default=0.0,
    )
    constraint_violation = max(
        (
            max(0.0, 1.0 - sum(values[node] for node in factor))
            for factor in factors
        ),
        default=0.0,
    )
    max_violation = max(bound_violation, constraint_violation)
    recomputed = sum(costs[node] * values[node] for node in system.removable_nodes)
    objective_error = abs(recomputed - solved.objective)
    objective_tolerance = (
        bound_absolute_tolerance
        + bound_relative_tolerance
        * max(abs(recomputed), abs(solved.objective))
    )
    if max_violation > feasibility_tolerance:
        return (
            False,
            "LP_PRIMAL_INFEASIBLE",
            integral,
            fractional_count,
            max_fractionality,
            max_violation,
            objective_error,
        )
    if objective_error > objective_tolerance:
        return (
            False,
            "LP_OBJECTIVE_RECOMPUTE_MISMATCH",
            integral,
            fractional_count,
            max_fractionality,
            max_violation,
            objective_error,
        )
    return (
        True,
        "VALID",
        integral,
        fractional_count,
        max_fractionality,
        max_violation,
        objective_error,
    )


def _assess_optimality_bounds(
    upper_bound: float,
    lower_bound: float | None,
    *,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
    tolerance: float | None = None,
) -> tuple[bool, bool, float | None, float | None, str]:
    """Assess a minimization certificate without masking inconsistent bounds."""

    if tolerance is not None:
        absolute_tolerance = tolerance
        relative_tolerance = 0.0
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Bound tolerances must be non-negative.")
    if not isfinite(upper_bound):
        return False, False, None, None, "UPPER_BOUND_INVALID"
    if lower_bound is None or not isfinite(lower_bound):
        return False, True, None, None, "NO_VALID_LOWER_BOUND"
    scale_tolerance = absolute_tolerance + relative_tolerance * max(
        abs(upper_bound), abs(lower_bound)
    )
    if lower_bound > upper_bound + scale_tolerance:
        return False, False, None, None, "BOUND_INCONSISTENT"
    gap = max(0.0, upper_bound - lower_bound)
    denominator = max(abs(upper_bound), absolute_tolerance)
    relative_gap = gap / denominator if denominator > 0.0 else 0.0
    if gap <= scale_tolerance:
        return True, True, 0.0, 0.0, "BOUND_CLOSED"
    return False, True, gap, relative_gap, "OPEN_GAP"


def solve_om_bpd_certified(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    factors: Iterable[Iterable[NodeId]] | None = None,
    costs: Mapping[NodeId, float] | None = None,
    use_lp_candidate: bool = True,
    exact_closure: bool = True,
    exact_time_limit: float | None = 60.0,
    integrality_tolerance: float = 1e-8,
    feasibility_tolerance: float = 1e-8,
    bound_absolute_tolerance: float = 1e-9,
    bound_relative_tolerance: float = 1e-9,
    method: str = "OM-BPD-C",
) -> OMBPDCertifiedResult:
    """Solve motif cover with OM-BPD candidates and explicit certification.

    The method is anytime: OM-BPD, mandatory motif-position covers, and an
    LP-guided cover provide feasible upper bounds.  Equality with the LP lower
    bound proves optimality.  Only an unresolved gap triggers the bounded MILP
    closure.  A timed-out closure is reported as an open gap, never as optimal.
    """

    if integrality_tolerance <= 0.0:
        raise ValueError("integrality_tolerance must be positive.")
    if feasibility_tolerance <= 0.0:
        raise ValueError("feasibility_tolerance must be positive.")
    if bound_absolute_tolerance < 0.0 or bound_relative_tolerance < 0.0:
        raise ValueError("Bound tolerances must be non-negative.")
    if exact_closure and exact_time_limit is not None and exact_time_limit <= 0.0:
        raise ValueError("exact_time_limit must be positive when supplied.")
    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    original = tuple(
        frozenset(factor) for factor in (system.factors if factors is None else factors)
    )
    unknown = {
        node for factor in original for node in factor
        if node not in frozenset(system.removable_nodes)
    }
    if unknown:
        raise ValueError(
            f"Motif factors contain non-removable nodes: {stable_nodes(unknown)!r}."
        )
    cost = _costs(system, costs)
    started = perf_counter()
    bpd = solve_om_bpd(
        system,
        factors=original,
        costs=cost,
        method="OM-BPD",
    )
    candidates: list[MotifCoverCandidate] = [
        _cover_candidate(
            original,
            bpd.critical_set,
            cost,
            source="om_bpd",
            preferred_order=bpd.raw_sequence,
        )
    ]
    position_started = perf_counter()
    if factors is None:
        candidates.extend(motif_position_cover_candidates(system, costs=cost))
    position_runtime = perf_counter() - position_started

    lp = solve_motif_cover_lp(system, factors=original, costs=cost)
    lp_values = dict(lp.solution_values)
    (
        lp_certificate_valid,
        lp_validation_reason,
        lp_integral,
        fractional_count,
        lp_max_fractionality,
        lp_max_constraint_violation,
        lp_objective_recompute_error,
    ) = _lp_certificate_diagnostics(
        system,
        original,
        cost,
        lp,
        integrality_tolerance=integrality_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        bound_absolute_tolerance=bound_absolute_tolerance,
        bound_relative_tolerance=bound_relative_tolerance,
    )
    if lp_certificate_valid and (lp_values or not original):
        lp_candidate, lp_integral, fractional_count = _lp_cover_candidate(
            original,
            cost,
            lp_values,
            integrality_tolerance=integrality_tolerance,
        )
        if use_lp_candidate:
            candidates.append(lp_candidate)

    priority = {"om_bpd": 0, "lp_integral_primal": 20, "milp_incumbent": 30}

    def candidate_key(candidate: MotifCoverCandidate) -> tuple[Any, ...]:
        return (
            candidate.objective,
            len(candidate.selected_set),
            priority.get(candidate.source, 10),
            candidate.source,
            tuple(stable_node_key(node) for node in candidate.sequence),
        )

    heuristic_candidates = [
        candidate
        for candidate in candidates
        if candidate.source == "om_bpd"
        or candidate.source.startswith("motif_position_")
        or candidate.source == "empty_motif_system"
    ]
    heuristic_incumbent = min(heuristic_candidates, key=candidate_key)
    incumbent = min(candidates, key=candidate_key)
    raw_lower = lp.objective if lp_certificate_valid else None
    effective_lower = raw_lower
    (
        certified,
        bound_consistent,
        gap,
        relative_gap,
        termination_reason,
    ) = _assess_optimality_bounds(
        incumbent.objective,
        effective_lower,
        absolute_tolerance=bound_absolute_tolerance,
        relative_tolerance=bound_relative_tolerance,
    )
    certificate_source = (
        "LP_RELAXATION_MATCH"
        if certified
        else (
            f"LP_INVALID:{lp_validation_reason}"
            if not lp_certificate_valid
            else termination_reason
        )
    )
    exact_attempted = False
    exact_status = "NOT_NEEDED" if certified else "DISABLED"
    exact_objective: float | None = None
    exact_lower: float | None = None
    exact_runtime = 0.0

    if not certified and exact_closure:
        exact_attempted = True
        exact = solve_exact_motif_cover(
            system,
            factors=original,
            costs=cost,
            objective_upper_bound=incumbent.objective,
            time_limit=exact_time_limit,
        )
        exact_status = exact.status
        exact_objective = exact.objective
        exact_lower = exact.lower_bound
        exact_runtime = exact.runtime_seconds
        if exact.objective is not None and exact.residual_factor_count == 0:
            exact_candidate = _cover_candidate(
                original,
                exact.selected_set,
                cost,
                source="milp_incumbent",
                preferred_order=bpd.raw_sequence,
            )
            candidates.append(exact_candidate)
            incumbent = min(candidates, key=candidate_key)
        if exact.optimal and exact.objective is not None:
            effective_lower = exact.objective
        elif exact.lower_bound is not None and isfinite(exact.lower_bound):
            effective_lower = max(
                value for value in (effective_lower, exact.lower_bound)
                if value is not None
            )
        (
            certified,
            bound_consistent,
            gap,
            relative_gap,
            termination_reason,
        ) = _assess_optimality_bounds(
            incumbent.objective,
            effective_lower,
            absolute_tolerance=bound_absolute_tolerance,
            relative_tolerance=bound_relative_tolerance,
        )
        if certified:
            certificate_source = (
                "MILP_OPTIMAL" if exact.optimal else "MILP_BOUND_MATCH"
            )
        else:
            certificate_source = termination_reason
    return OMBPDCertifiedResult(
        method=method,
        critical_sequence=incumbent.sequence,
        critical_set=incumbent.selected_set,
        critical_cost=incumbent.objective,
        residual_factor_count=incumbent.residual_factor_count,
        inclusion_minimal=is_inclusion_minimal_cover(
            original, incumbent.selected_set
        ),
        heuristic_incumbent_source=heuristic_incumbent.source,
        heuristic_incumbent_cost=heuristic_incumbent.objective,
        incumbent_source=incumbent.source,
        candidates=tuple(candidates),
        bpd_result=bpd,
        motif_position_runtime_seconds=position_runtime,
        lp_status=lp.status,
        lp_objective=lp.objective,
        lp_runtime_seconds=lp.runtime_seconds,
        lp_certificate_valid=lp_certificate_valid,
        lp_validation_reason=lp_validation_reason,
        effective_lower_bound=effective_lower,
        lp_solution_integral=lp_integral,
        lp_fractional_variable_count=fractional_count,
        lp_max_fractionality=lp_max_fractionality,
        lp_max_constraint_violation=lp_max_constraint_violation,
        lp_objective_recompute_error=lp_objective_recompute_error,
        exact_closure_attempted=exact_attempted,
        exact_status=exact_status,
        exact_objective=exact_objective,
        exact_lower_bound=exact_lower,
        exact_runtime_seconds=exact_runtime,
        certified_optimal=certified,
        bound_consistent=bound_consistent,
        certificate_source=certificate_source,
        termination_reason=termination_reason,
        bound_absolute_tolerance=bound_absolute_tolerance,
        bound_relative_tolerance=bound_relative_tolerance,
        absolute_gap=gap,
        relative_gap=relative_gap,
        heuristic_selection_runtime_seconds=(
            bpd.runtime_seconds + position_runtime
        ),
        certificate_runtime_seconds=lp.runtime_seconds + exact_runtime,
        runtime_seconds=perf_counter() - started,
    )


def solve_om_bpd_position(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    costs: Mapping[NodeId, float] | None = None,
    integrality_tolerance: float = 1e-8,
    method: str = "OM-BPD-P",
) -> OMBPDCertifiedResult:
    """Return the scalable BP-plus-position incumbent with an LP audit only."""

    return solve_om_bpd_certified(
        graph,
        costs=costs,
        use_lp_candidate=False,
        exact_closure=False,
        integrality_tolerance=integrality_tolerance,
        method=method,
    )


def analyze_optimal_motif_family(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    costs: Mapping[NodeId, float] | None = None,
    time_limit_per_solve: float | None = None,
    tolerance: float = 1e-8,
) -> OptimalFamilyResult:
    """Classify must/possible nodes by exact inclusion and exclusion solves."""

    system = (
        graph if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    started = perf_counter()
    reference = solve_exact_motif_cover(
        system, costs=costs, time_limit=time_limit_per_solve
    )
    if not reference.optimal or reference.objective is None:
        raise RuntimeError("Optimal-family analysis requires an optimal reference MILP.")
    optimum = reference.objective
    must: set[NodeId] = set()
    possible: set[NodeId] = set()
    for node in system.removable_nodes:
        without = solve_exact_motif_cover(
            system,
            costs=costs,
            force_excluded=(node,),
            time_limit=time_limit_per_solve,
        )
        if without.status == "INFEASIBLE" or (
            without.optimal
            and without.objective is not None
            and without.objective > optimum + tolerance
        ):
            must.add(node)
        with_node = solve_exact_motif_cover(
            system,
            costs=costs,
            force_selected=(node,),
            time_limit=time_limit_per_solve,
        )
        if (
            with_node.optimal
            and with_node.objective is not None
            and abs(with_node.objective - optimum) <= tolerance
        ):
            possible.add(node)
    return OptimalFamilyResult(
        optimum_cost=optimum,
        reference_set=reference.selected_set,
        must_nodes=frozenset(must),
        possible_nodes=frozenset(possible),
        interchangeable_nodes=frozenset(possible - must),
        exact=True,
        runtime_seconds=perf_counter() - started,
    )
