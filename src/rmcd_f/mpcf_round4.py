"""Round-4 audits for MPCF without changing its primary objective.

The module contains three deliberately separate diagnostics:

* deterministic budget saturation of a primary-optimal defense;
* a certified tertiary tie-break at fixed PathCut value and realized cost;
* bounded enumeration of minimum and one-quantum-near directed task cuts.

None of these routines is used by ``MPCF-Exact`` itself.  They only audit
representative choice, alternative capacity responses, and cut competition.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .model import NodeId, SolveStatus, protection_cost, stable_node_key, stable_nodes
from .operational_motif import (
    OperationalMotifSystem,
    audit_path_closed_motif_system,
    build_operational_motif_system,
)
from .task_path_fortification import (
    _build_node_split_arcs,
    evaluate_task_path_fortification,
    fortified_attack_costs,
    mpcf_objective_cost_quantum,
)
from .mpcf_capacity_evaluation import evaluate_role_motif_capacity_thresholds


@dataclass(frozen=True)
class MPCFTertiaryIteration:
    iteration: int
    master_upper_bound: float
    oracle_value: float | None
    path_margin: float
    protection_set: frozenset[NodeId]
    protection_cost: float
    added_path_cut: bool
    added_capacity_attack: bool
    master_runtime_seconds: float
    path_oracle_runtime_seconds: float
    capacity_oracle_runtime_seconds: float


@dataclass(frozen=True)
class MPCFCapacityTiebreakResult:
    protection_set: frozenset[NodeId]
    protection_cost: float
    primary_margin: float
    remaining_fraction: float
    capacity_attack_cost: float | None
    lower_bound: float | None
    upper_bound: float | None
    optimal: bool
    status: SolveStatus
    path_cut_count: int
    capacity_attack_count: int
    iterations: tuple[MPCFTertiaryIteration, ...]
    runtime_seconds: float


@dataclass(frozen=True)
class NearCutFamilyResult:
    optimum: float
    cost_quantum: float | None
    maximum_cost: float
    cuts: tuple[frozenset[NodeId], ...]
    enumeration_complete: bool
    enumeration_limit: int
    mandatory_nodes: frozenset[NodeId]
    possible_nodes: frozenset[NodeId]
    mean_pairwise_jaccard: float
    runtime_seconds: float


@dataclass(frozen=True)
class _TertiaryMaster:
    protection_set: frozenset[NodeId]
    theta: float | None
    upper_bound: float | None
    status: SolveStatus
    optimal: bool
    runtime_seconds: float
    message: str


def _system(graph: nx.Graph | OperationalMotifSystem) -> OperationalMotifSystem:
    system = (
        graph
        if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )
    audit = audit_path_closed_motif_system(system)
    if not audit.path_closed:
        raise ValueError("Round-4 MPCF audits require a path-closed task system.")
    return system


def _uplifts(
    system: OperationalMotifSystem,
    multiplier: float,
    uplifts: Mapping[NodeId, float] | None,
) -> dict[NodeId, float]:
    result = {
        node: float(
            uplifts[node]
            if uplifts is not None and node in uplifts
            else multiplier * float(system.graph.nodes[node]["attack_cost"])
        )
        for node in system.removable_nodes
    }
    if any(not isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("Fortification uplifts must be finite and non-negative.")
    return result


def saturate_primary_optimal_defense(
    graph: nx.Graph | OperationalMotifSystem,
    protection_set: Iterable[NodeId],
    budget: float,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
) -> tuple[frozenset[NodeId], tuple[NodeId, ...]]:
    """Return an inclusion-maximal deterministic superset within ``budget``.

    Additional nodes are ordered by exact PathCut marginal gain per cost.  A
    superset cannot reduce the primary margin because fortification only raises
    attack costs.  The routine therefore changes only the representative, not
    the attained primary optimum.
    """

    system = _system(graph)
    selected = set(protection_set)
    unknown = selected - set(system.removable_nodes)
    if unknown:
        raise ValueError(f"Unknown protected nodes: {stable_nodes(unknown)!r}.")
    delta = _uplifts(system, fortification_multiplier, uplifts)
    current = evaluate_task_path_fortification(
        system,
        selected,
        fortification_multiplier=fortification_multiplier,
        uplifts=delta,
    )
    if not current.optimal or current.objective is None:
        raise RuntimeError("Initial PathCut replay failed during saturation.")
    sequence: list[NodeId] = []
    while True:
        spent = protection_cost(system.graph, selected)
        affordable = [
            node
            for node in system.removable_nodes
            if node not in selected
            and spent + float(system.graph.nodes[node]["protect_cost"])
            <= float(budget) + 1e-9
        ]
        if not affordable:
            break
        evaluated: list[tuple[float, float, int, tuple[str, str], NodeId, Any]] = []
        for node in affordable:
            oracle = evaluate_task_path_fortification(
                system,
                (*selected, node),
                fortification_multiplier=fortification_multiplier,
                uplifts=delta,
            )
            if not oracle.optimal or oracle.objective is None:
                raise RuntimeError("Candidate PathCut replay failed during saturation.")
            gain = float(oracle.objective) - float(current.objective)
            cost = float(system.graph.nodes[node]["protect_cost"])
            evaluated.append(
                (
                    gain / cost,
                    gain,
                    int(node in current.selected_set),
                    stable_node_key(node),
                    node,
                    oracle,
                )
            )
        _score, _gain, _in_cut, _key, chosen, current = min(
            evaluated,
            key=lambda item: (-item[0], -item[1], -item[2], item[3]),
        )
        selected.add(chosen)
        sequence.append(chosen)
    return frozenset(selected), tuple(sequence)


def truncate_ordered_selection_to_cost(
    graph: nx.Graph | OperationalMotifSystem,
    ordered_nodes: Iterable[NodeId],
    cost_ceiling: float,
) -> frozenset[NodeId]:
    """Reapply an unchanged ranking at a common realized-cost ceiling."""

    system = _system(graph)
    selected: set[NodeId] = set()
    for node in ordered_nodes:
        if node not in system.removable_nodes or node in selected:
            continue
        candidate = protection_cost(system.graph, (*selected, node))
        if candidate <= float(cost_ceiling) + 1e-9:
            selected.add(node)
    return frozenset(selected)


def _solver_status(result: Any) -> tuple[SolveStatus, bool]:
    if int(result.status) == 0 and getattr(result, "x", None) is not None:
        return SolveStatus.OPTIMAL, True
    if int(result.status) == 1:
        return SolveStatus.TIME_LIMIT, False
    if int(result.status) == 2:
        return SolveStatus.INFEASIBLE, False
    if int(result.status) == 3:
        return SolveStatus.UNBOUNDED, False
    return SolveStatus.ERROR, False


def _solve_tertiary_master(
    system: OperationalMotifSystem,
    target_cost: float,
    primary_margin: float,
    delta: Mapping[NodeId, float],
    path_cuts: Sequence[frozenset[NodeId]],
    capacity_attacks: Sequence[frozenset[NodeId]],
    *,
    time_limit: float | None,
    tolerance: float,
) -> _TertiaryMaster:
    nodes = tuple(system.removable_nodes)
    index = {node: position for position, node in enumerate(nodes)}
    theta_index = len(nodes)
    variable_count = theta_index + 1
    objective = np.zeros(variable_count, dtype=float)
    objective[theta_index] = -1.0
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[: len(nodes)] = 1
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    maximum_attack_cost = fsum(
        float(system.graph.nodes[node]["attack_cost"]) + delta[node]
        for node in nodes
    )
    upper[theta_index] = maximum_attack_cost
    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    cost_row = {
        index[node]: float(system.graph.nodes[node]["protect_cost"])
        for node in nodes
    }
    rows.append(cost_row)
    lbs.append(target_cost - tolerance)
    ubs.append(target_cost + tolerance)
    for cut in path_cuts:
        rows.append({index[node]: delta[node] for node in cut})
        lbs.append(
            primary_margin
            - fsum(float(system.graph.nodes[node]["attack_cost"]) for node in cut)
            - tolerance
        )
        ubs.append(np.inf)
    for attack in capacity_attacks:
        row = {theta_index: 1.0}
        for node in attack:
            row[index[node]] = -delta[node]
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(
            fsum(float(system.graph.nodes[node]["attack_cost"]) for node in attack)
        )
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for column_index, value in row.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), variable_count),
    ).tocsr()
    options: dict[str, object] = {"presolve": True, "mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    started = perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.array(lbs), np.array(ubs)),
        options=options,
    )
    runtime = perf_counter() - started
    status, optimal = _solver_status(result)
    vector = getattr(result, "x", None)
    chosen = frozenset(
        node
        for node in nodes
        if vector is not None and float(vector[index[node]]) >= 0.5
    )
    theta = None if vector is None else float(vector[theta_index])
    dual = getattr(result, "mip_dual_bound", None)
    bound = None if dual is None or not isfinite(float(dual)) else -float(dual)
    return _TertiaryMaster(
        chosen,
        theta,
        bound,
        status,
        optimal,
        runtime,
        str(result.message),
    )


def _capacity_point(
    system: OperationalMotifSystem,
    protection_set: Iterable[NodeId],
    remaining_fraction: float,
    *,
    fortification_multiplier: float,
    uplifts: Mapping[NodeId, float],
    time_limit: float | None,
):
    evaluation = evaluate_role_motif_capacity_thresholds(
        system.graph,
        protection_set,
        (remaining_fraction,),
        fortification_multiplier=fortification_multiplier,
        uplifts=uplifts,
        solver="rmcd-exact",
        time_limit=time_limit,
        independent_verification=True,
    )
    point = evaluation.points[0]
    if not point.global_optimum_certified or point.attack_cost is None:
        raise RuntimeError(
            "The tertiary capacity oracle did not return a finite certified attack."
        )
    return point


def solve_mpcf_capacity_tiebreak(
    graph: nx.Graph | OperationalMotifSystem,
    target_protection_cost: float,
    primary_margin: float,
    remaining_fraction: float,
    *,
    initial_protection_set: Iterable[NodeId] = (),
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    max_iterations: int = 64,
    time_limit_per_solve: float | None = 300.0,
    tolerance: float = 1e-7,
) -> MPCFCapacityTiebreakResult:
    """Maximize capacity-threshold attack cost within one MPCF optimum tier.

    The master fixes realized protection cost, enforces all discovered task
    cuts at ``primary_margin``, and maximizes the minimum discovered capacity
    attack cost.  Exact PathCut and exact capacity-response oracles separate
    the two constraint families.  Closure therefore certifies the tertiary
    representative without altering the MPCF primary objective.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    system = _system(graph)
    delta = _uplifts(system, fortification_multiplier, uplifts)
    initial = frozenset(initial_protection_set)
    if abs(protection_cost(system.graph, initial) - target_protection_cost) > tolerance:
        raise ValueError("initial_protection_set does not have target_protection_cost.")
    initial_path = evaluate_task_path_fortification(
        system,
        initial,
        fortification_multiplier=fortification_multiplier,
        uplifts=delta,
    )
    if (
        not initial_path.optimal
        or initial_path.objective is None
        or float(initial_path.objective) < primary_margin - tolerance
    ):
        raise ValueError("Initial tertiary witness does not attain the primary margin.")
    initial_capacity = _capacity_point(
        system,
        initial,
        remaining_fraction,
        fortification_multiplier=fortification_multiplier,
        uplifts=delta,
        time_limit=time_limit_per_solve,
    )
    path_cuts = [initial_path.selected_set]
    capacity_attacks = [initial_capacity.attack_set]
    incumbent = initial
    lower_bound = float(initial_capacity.attack_cost)
    upper_bound: float | None = None
    trace: list[MPCFTertiaryIteration] = []
    started = perf_counter()
    for iteration in range(1, max_iterations + 1):
        master = _solve_tertiary_master(
            system,
            float(target_protection_cost),
            float(primary_margin),
            delta,
            path_cuts,
            capacity_attacks,
            time_limit=time_limit_per_solve,
            tolerance=tolerance,
        )
        if master.theta is None or not master.optimal:
            return MPCFCapacityTiebreakResult(
                incumbent,
                protection_cost(system.graph, incumbent),
                primary_margin,
                remaining_fraction,
                lower_bound,
                lower_bound,
                master.upper_bound,
                False,
                master.status,
                len(path_cuts),
                len(capacity_attacks),
                tuple(trace),
                perf_counter() - started,
            )
        upper_bound = master.theta
        path = evaluate_task_path_fortification(
            system,
            master.protection_set,
            fortification_multiplier=fortification_multiplier,
            uplifts=delta,
        )
        if not path.optimal or path.objective is None:
            raise RuntimeError("Tertiary PathCut separation failed.")
        path_violation = float(path.objective) < primary_margin - tolerance
        capacity_value: float | None = None
        capacity_runtime = 0.0
        capacity_violation = False
        if path_violation:
            if path.selected_set in path_cuts:
                raise RuntimeError("Duplicate violated PathCut prevents tertiary closure.")
            path_cuts.append(path.selected_set)
        else:
            capacity_started = perf_counter()
            point = _capacity_point(
                system,
                master.protection_set,
                remaining_fraction,
                fortification_multiplier=fortification_multiplier,
                uplifts=delta,
                time_limit=time_limit_per_solve,
            )
            capacity_runtime = perf_counter() - capacity_started
            capacity_value = float(point.attack_cost)
            if capacity_value > lower_bound + tolerance:
                lower_bound = capacity_value
                incumbent = master.protection_set
            capacity_violation = capacity_value < master.theta - tolerance
            if capacity_violation:
                if point.attack_set in capacity_attacks:
                    raise RuntimeError(
                        "Duplicate violated capacity attack prevents tertiary closure."
                    )
                capacity_attacks.append(point.attack_set)
        trace.append(
            MPCFTertiaryIteration(
                iteration,
                master.theta,
                capacity_value,
                float(path.objective),
                master.protection_set,
                protection_cost(system.graph, master.protection_set),
                path_violation,
                capacity_violation,
                master.runtime_seconds,
                path.runtime_seconds,
                capacity_runtime,
            )
        )
        if not path_violation and not capacity_violation:
            return MPCFCapacityTiebreakResult(
                master.protection_set,
                protection_cost(system.graph, master.protection_set),
                primary_margin,
                remaining_fraction,
                capacity_value,
                capacity_value,
                master.theta,
                True,
                SolveStatus.OPTIMAL,
                len(path_cuts),
                len(capacity_attacks),
                tuple(trace),
                perf_counter() - started,
            )
    return MPCFCapacityTiebreakResult(
        incumbent,
        protection_cost(system.graph, incumbent),
        primary_margin,
        remaining_fraction,
        lower_bound,
        lower_bound,
        upper_bound,
        False,
        SolveStatus.LIMIT_REACHED,
        len(path_cuts),
        len(capacity_attacks),
        tuple(trace),
        perf_counter() - started,
    )


def enumerate_near_task_cuts(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    max_excess_quanta: int = 1,
    enumeration_limit: int = 500,
    time_limit_per_solve: float | None = 60.0,
) -> NearCutFamilyResult:
    """Enumerate distinct directed task cuts up to a lattice-relative bound."""

    if max_excess_quanta < 0 or enumeration_limit < 1:
        raise ValueError("Near-cut limits must be non-negative and non-zero.")
    system = _system(graph)
    base_oracle = evaluate_task_path_fortification(system, ())
    if not base_oracle.optimal or base_oracle.objective is None:
        raise RuntimeError("Base PathCut failed before near-cut enumeration.")
    optimum = float(base_oracle.objective)
    quantum_record = mpcf_objective_cost_quantum(system)
    quantum = quantum_record.quantum if quantum_record.valid else None
    if max_excess_quanta and quantum is None:
        raise ValueError("A valid objective lattice is required for near-cut enumeration.")
    maximum_cost = optimum + max_excess_quanta * float(quantum or 0.0)
    zero_delta = {node: 0.0 for node in system.removable_nodes}
    source, sink, arcs, _big_m = _build_node_split_arcs(system, zero_delta)
    vertices = sorted(
        {source, sink, *(arc.tail for arc in arcs), *(arc.head for arc in arcs)},
        key=repr,
    )
    z_index = {vertex: index for index, vertex in enumerate(vertices)}
    nodes = tuple(system.removable_nodes)
    x_offset = len(vertices)
    x_index = {node: x_offset + index for index, node in enumerate(nodes)}
    variable_count = len(vertices) + len(nodes)
    objective = np.zeros(variable_count, dtype=float)
    for node in nodes:
        objective[x_index[node]] = float(system.graph.nodes[node]["attack_cost"])
    integrality = np.ones(variable_count, dtype=np.uint8)
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    lower[z_index[source]] = upper[z_index[source]] = 1.0
    lower[z_index[sink]] = upper[z_index[sink]] = 0.0
    fixed_rows: list[dict[int, float]] = []
    fixed_lbs: list[float] = []
    fixed_ubs: list[float] = []
    for arc in arcs:
        if arc.uplift_node is None:
            fixed_rows.append({z_index[arc.tail]: 1.0, z_index[arc.head]: -1.0})
            fixed_lbs.append(-np.inf)
            fixed_ubs.append(0.0)
            continue
        x = x_index[arc.uplift_node]
        tail = z_index[arc.tail]
        head = z_index[arc.head]
        fixed_rows.extend(
            (
                {tail: 1.0, head: -1.0, x: -1.0},
                {x: 1.0, tail: -1.0},
                {x: 1.0, head: 1.0},
            )
        )
        fixed_lbs.extend((-np.inf, -np.inf, -np.inf))
        fixed_ubs.extend((0.0, 0.0, 1.0))
    fixed_rows.append(
        {
            x_index[node]: float(system.graph.nodes[node]["attack_cost"])
            for node in nodes
        }
    )
    fixed_lbs.append(-np.inf)
    fixed_ubs.append(maximum_cost + 1e-9)
    cuts: list[frozenset[NodeId]] = []
    complete = False
    started = perf_counter()
    while len(cuts) < enumeration_limit:
        rows = list(fixed_rows)
        lbs = list(fixed_lbs)
        ubs = list(fixed_ubs)
        for cut in cuts:
            rows.append(
                {
                    x_index[node]: (1.0 if node in cut else -1.0)
                    for node in nodes
                }
            )
            lbs.append(-np.inf)
            ubs.append(float(len(cut) - 1))
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        for row_index, row in enumerate(rows):
            for column_index, value in row.items():
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(value)
        matrix = coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(rows), variable_count),
        ).tocsr()
        options: dict[str, object] = {"presolve": True, "mip_rel_gap": 0.0}
        if time_limit_per_solve is not None:
            options["time_limit"] = float(time_limit_per_solve)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, np.array(lbs), np.array(ubs)),
            options=options,
        )
        status, optimal = _solver_status(result)
        if status == SolveStatus.INFEASIBLE:
            complete = True
            break
        if not optimal or result.x is None:
            break
        cut = frozenset(
            node for node in nodes if float(result.x[x_index[node]]) >= 0.5
        )
        if not cut or cut in cuts:
            break
        cuts.append(cut)
    mandatory = set(cuts[0]) if cuts else set()
    possible: set[NodeId] = set()
    for cut in cuts:
        mandatory.intersection_update(cut)
        possible.update(cut)
    similarities: list[float] = []
    for left_index, left in enumerate(cuts):
        for right in cuts[left_index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    return NearCutFamilyResult(
        optimum,
        quantum,
        maximum_cost,
        tuple(cuts),
        complete,
        enumeration_limit,
        frozenset(mandatory),
        frozenset(possible),
        fmean(similarities) if similarities else 1.0,
        perf_counter() - started,
    )
