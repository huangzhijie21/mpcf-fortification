"""Parametric cut decomposition for Role-Motif Capacity Dismantling.

The implementation keeps the raw Lagrangian bound and the primal attack bound
separate.  Optimality requires either direct bound equality or an exact
rational objective-lattice closure against a verified feasible attack cost.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from math import gcd
from numbers import Integral
from time import perf_counter
from typing import Iterable, Literal

import networkx as nx

from .flow import FlowNode, SINK, SOURCE, motif_capacity
from .model import (
    NodeId,
    SolveStatus,
    SolverInfo,
    ensure_node_subset,
    stable_node_key,
    stable_nodes,
    validate_graph,
)
from .rmcd import RMCDResult, solve_rmcd_exact


@dataclass(frozen=True)
class PCDLine:
    """One affine Lagrangian configuration ``intercept + slope * lambda``."""

    intercept: Fraction
    slope: int
    cut_nodes: tuple[NodeId, ...]
    attacked_on_cut: frozenset[NodeId]

    def value(self, lambda_value: Fraction) -> Fraction:
        return self.intercept + self.slope * lambda_value


@dataclass(frozen=True)
class PCDIteration:
    """One exact-rational weighted-cut oracle call."""

    oracle_call: int
    lambda_value: Fraction
    dual_value: Fraction
    master_upper_before: Fraction | None
    cut_nodes: tuple[NodeId, ...]
    candidate_attack: frozenset[NodeId]
    candidate_cost: Fraction | None
    incumbent_lower: Fraction
    incumbent_upper: Fraction
    new_line_count: int


@dataclass(frozen=True)
class PCDResult:
    """A bounded PCD solve, optionally completed by the exact MILP oracle."""

    rmcd: RMCDResult
    initial_capacity: int
    lower_bound: float | None
    upper_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    dual_master_upper: float | None
    dual_search_closed: bool
    certified_by_pcd: bool
    certification_source: str
    exact_fallback_used: bool
    homogeneous_closed_form: bool
    objective_cost_quantum: Fraction | None
    lattice_closed: bool
    termination_reason: str
    trace: tuple[PCDIteration, ...]
    line_pool: tuple[PCDLine, ...]
    cut_pool: tuple[tuple[NodeId, ...], ...]

    @property
    def optimal(self) -> bool:
        return self.rmcd.optimal


@dataclass(frozen=True)
class _OracleResult:
    lambda_value: Fraction
    dual_value: Fraction
    cut_nodes: tuple[NodeId, ...]
    lines: tuple[PCDLine, ...]


def _fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid rational costs.")
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    return Fraction(str(float(value)))


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right) if left and right else 0


def _objective_cost_quantum(
    graph: nx.DiGraph, attackable: Iterable[NodeId]
) -> Fraction | None:
    """Return the greatest rational quantum shared by all attack costs.

    Every feasible RMCD objective is a subset sum of the attack costs.  If all
    costs are integer multiples of ``q``, the unknown optimum lies on the
    lattice ``q * Z``.  This permits a rigorous bound closure when the next
    lattice point above the lower bound equals a verified incumbent.
    """

    costs = tuple(_fraction(graph.nodes[node]["attack_cost"]) for node in attackable)
    if not costs:
        return None
    common_denominator = 1
    for cost in costs:
        common_denominator = _lcm(common_denominator, cost.denominator)
    integer_costs = tuple(
        cost.numerator * (common_denominator // cost.denominator) for cost in costs
    )
    common_numerator = 0
    for value in integer_costs:
        common_numerator = gcd(common_numerator, abs(value))
    if common_numerator == 0:
        return None
    return Fraction(common_numerator, common_denominator)


def _lattice_ceiling(value: Fraction, quantum: Fraction | None) -> Fraction | None:
    if quantum is None or quantum <= 0:
        return None
    scaled = value / quantum
    multiple = -(-scaled.numerator // scaled.denominator)
    return multiple * quantum


def _validate_threshold(graph: nx.DiGraph, K: int) -> tuple[int, int]:
    if isinstance(K, bool) or not isinstance(K, Integral):
        raise ValueError("K must be an integer.")
    threshold = int(K)
    initial_capacity = motif_capacity(graph).value
    if not 1 <= threshold <= initial_capacity:
        raise ValueError(
            f"K must satisfy 1 <= K <= Omega_0={initial_capacity}; got {threshold}."
        )
    return threshold, initial_capacity


def _weighted_cut_oracle(
    graph: nx.DiGraph,
    K: int,
    lambda_value: Fraction,
    forbidden: frozenset[NodeId],
) -> _OracleResult:
    """Evaluate the exact-rational Lagrangian cut oracle."""

    costs = {node: _fraction(graph.nodes[node]["attack_cost"]) for node in graph}
    rational_weights: dict[NodeId, Fraction] = {}
    for node in stable_nodes(graph.nodes):
        capacity_weight = lambda_value * int(graph.nodes[node]["capacity"])
        rational_weights[node] = (
            capacity_weight
            if node in forbidden
            else min(costs[node], capacity_weight)
        )

    scale = 1
    for weight in rational_weights.values():
        scale = _lcm(scale, weight.denominator)
    integer_weights = {
        node: int(weight * scale) for node, weight in rational_weights.items()
    }
    infinity = 1 + sum(integer_weights.values())

    weighted = nx.DiGraph()
    weighted.add_nodes_from((SOURCE, SINK))
    node_in: dict[NodeId, FlowNode] = {}
    node_out: dict[NodeId, FlowNode] = {}
    for node in stable_nodes(graph.nodes):
        incoming = FlowNode("in", node)
        outgoing = FlowNode("out", node)
        node_in[node] = incoming
        node_out[node] = outgoing
        weighted.add_edge(incoming, outgoing, capacity=integer_weights[node])
        role = graph.nodes[node]["role"]
        if role == "S":
            weighted.add_edge(SOURCE, incoming, capacity=infinity)
        elif role == "E":
            weighted.add_edge(outgoing, SINK, capacity=infinity)
    for source, target in sorted(
        graph.edges,
        key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1])),
    ):
        weighted.add_edge(node_out[source], node_in[target], capacity=infinity)

    cut_value, (source_side, sink_side) = nx.minimum_cut(
        weighted, SOURCE, SINK, capacity="capacity"
    )
    cut_nodes = stable_nodes(
        node
        for node in graph
        if node_in[node] in source_side and node_out[node] in sink_side
    )
    dual_value = Fraction(int(cut_value), scale) - lambda_value * (K - 1)

    strict_attack: set[NodeId] = set()
    ties: set[NodeId] = set()
    for node in cut_nodes:
        if node in forbidden:
            continue
        capacity_weight = lambda_value * int(graph.nodes[node]["capacity"])
        if costs[node] < capacity_weight:
            strict_attack.add(node)
        elif costs[node] == capacity_weight:
            ties.add(node)

    attack_variants = (frozenset(strict_attack),)
    if ties:
        attack_variants = (
            frozenset(strict_attack),
            frozenset(strict_attack | ties),
        )
    lines: list[PCDLine] = []
    for attacked in attack_variants:
        intercept = sum((costs[node] for node in attacked), Fraction(0, 1))
        residual = sum(
            int(graph.nodes[node]["capacity"])
            for node in cut_nodes
            if node not in attacked
        )
        line = PCDLine(intercept, residual - (K - 1), cut_nodes, attacked)
        if line.value(lambda_value) != dual_value:
            raise RuntimeError("PCD oracle line does not reproduce its dual value.")
        if line not in lines:
            lines.append(line)
    return _OracleResult(lambda_value, dual_value, cut_nodes, tuple(lines))


def _exact_cut_cover(
    graph: nx.DiGraph,
    cut_nodes: tuple[NodeId, ...],
    K: int,
    forbidden: frozenset[NodeId],
) -> tuple[frozenset[NodeId], Fraction] | None:
    """Solve one fixed-cut minimum-cost capacity cover with exact costs."""

    total_capacity = sum(int(graph.nodes[node]["capacity"]) for node in cut_nodes)
    required = max(0, total_capacity - (K - 1))
    if required == 0:
        return frozenset(), Fraction(0, 1)

    states: dict[int, tuple[Fraction, tuple[NodeId, ...]]] = {
        0: (Fraction(0, 1), ())
    }
    for node in cut_nodes:
        capacity = int(graph.nodes[node]["capacity"])
        if node in forbidden or capacity <= 0:
            continue
        cost = _fraction(graph.nodes[node]["attack_cost"])
        updated = dict(states)
        for removed, (current_cost, selected) in states.items():
            new_removed = min(required, removed + capacity)
            candidate = (current_cost + cost, (*selected, node))
            incumbent = updated.get(new_removed)
            if incumbent is None or candidate[0] < incumbent[0]:
                updated[new_removed] = candidate
            elif incumbent is not None and candidate[0] == incumbent[0]:
                if tuple(map(repr, candidate[1])) < tuple(map(repr, incumbent[1])):
                    updated[new_removed] = candidate
        states = updated
    if required not in states:
        return None
    cost, selected = states[required]
    return frozenset(selected), cost


def _master_solution(
    lines: tuple[PCDLine, ...], lambda_max: Fraction
) -> tuple[Fraction, Fraction]:
    """Maximize the discovered-line lower envelope on a bounded 1-D domain.

    Because the discovered line set is only a subset of all configurations,
    this lower envelope is an upper approximation of the true dual function.
    """

    if not lines:
        raise ValueError("PCD master requires at least one affine line.")
    candidates = {Fraction(0, 1), lambda_max}
    for left, right in combinations(lines, 2):
        slope_delta = left.slope - right.slope
        if slope_delta == 0:
            continue
        crossing = (right.intercept - left.intercept) / slope_delta
        if 0 <= crossing <= lambda_max:
            candidates.add(crossing)

    best_lambda = Fraction(0, 1)
    best_value: Fraction | None = None
    for candidate in sorted(candidates):
        value = min(line.value(candidate) for line in lines)
        if best_value is None or value > best_value:
            best_lambda = candidate
            best_value = value
    assert best_value is not None
    return best_lambda, best_value


def _unit_homogeneous_cost(graph: nx.DiGraph) -> Fraction | None:
    capacities = [int(graph.nodes[node]["capacity"]) for node in graph]
    if any(capacity != 1 for capacity in capacities):
        return None
    costs = [_fraction(graph.nodes[node]["attack_cost"]) for node in graph]
    if not costs or any(cost != costs[0] for cost in costs[1:]):
        return None
    return costs[0]


def _closed_form_unit_result(
    graph: nx.DiGraph, K: int, initial_capacity: int, common_cost: Fraction
) -> PCDResult:
    started = perf_counter()
    initial_flow = motif_capacity(graph)
    number_to_attack = initial_capacity - K + 1
    attack = frozenset(initial_flow.cut.cut_nodes[:number_to_attack])
    residual = motif_capacity(graph, attack)
    expected = common_cost * number_to_attack
    if residual.value != K - 1:
        raise RuntimeError("Unit closed-form RMCD result failed residual verification.")
    runtime = perf_counter() - started
    solver = SolverInfo(
        SolveStatus.OPTIMAL,
        True,
        message="Certified by the unit-capacity/unit-cost degeneration theorem.",
        objective=float(expected),
        lower_bound=float(expected),
        upper_bound=float(expected),
        mip_gap=0.0,
        runtime_seconds=runtime,
    )
    rmcd = RMCDResult(
        K,
        attack,
        float(expected),
        residual.value,
        residual.cut,
        solver,
        method="unit-cut-closed-form",
        _graph=graph,
    )
    return PCDResult(
        rmcd=rmcd,
        initial_capacity=initial_capacity,
        lower_bound=float(expected),
        upper_bound=float(expected),
        absolute_gap=0.0,
        relative_gap=0.0,
        dual_master_upper=None,
        dual_search_closed=False,
        certified_by_pcd=False,
        certification_source="unit_homogeneous_closed_form",
        exact_fallback_used=False,
        homogeneous_closed_form=True,
        objective_cost_quantum=common_cost,
        lattice_closed=False,
        termination_reason="unit_homogeneous_closed_form",
        trace=(),
        line_pool=(),
        cut_pool=(initial_flow.cut.cut_nodes,),
    )


def solve_rmcd_pcd(
    graph: nx.Graph,
    K: int,
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
    max_oracle_calls: int = 64,
    fallback: Literal["none", "exact"] = "none",
    time_limit: float | None = None,
) -> PCDResult:
    """Solve RMCD with exact-rational parametric cut decomposition.

    ``fallback='exact'`` invokes the compact MILP only when PCD does not close
    the primal gap.  The returned metadata keeps that source of certification
    explicit.
    """

    if max_oracle_calls < 2:
        raise ValueError("max_oracle_calls must be at least 2.")
    if fallback not in {"none", "exact"}:
        raise ValueError("fallback must be 'none' or 'exact'.")

    started = perf_counter()
    equipment = validate_graph(graph)
    threshold, initial_capacity = _validate_threshold(equipment, K)
    protected_set = ensure_node_subset(equipment, protected, label="protected")
    excluded_set = ensure_node_subset(equipment, excluded, label="excluded")
    forbidden = frozenset(protected_set | excluded_set)

    if not forbidden:
        common_cost = _unit_homogeneous_cost(equipment)
        if common_cost is not None:
            return _closed_form_unit_result(
                equipment, threshold, initial_capacity, common_cost
            )

    attackable = frozenset(
        node
        for node in equipment
        if node not in forbidden and int(equipment.nodes[node]["capacity"]) > 0
    )
    objective_cost_quantum = _objective_cost_quantum(equipment, attackable)
    all_removed = motif_capacity(equipment, attackable)
    if all_removed.value >= threshold:
        runtime = perf_counter() - started
        solver = SolverInfo(
            SolveStatus.UNBREAKABLE,
            True,
            message="Deleting every attackable positive-capacity node cannot breach K.",
            runtime_seconds=runtime,
        )
        rmcd = RMCDResult(
            threshold,
            frozenset(),
            None,
            None,
            None,
            solver,
            protected_set,
            excluded_set,
            "pcd",
            equipment,
        )
        return PCDResult(
            rmcd=rmcd,
            initial_capacity=initial_capacity,
            lower_bound=None,
            upper_bound=None,
            absolute_gap=None,
            relative_gap=None,
            dual_master_upper=None,
            dual_search_closed=False,
            certified_by_pcd=False,
            certification_source="unbreakable_monotonicity",
            exact_fallback_used=False,
            homogeneous_closed_form=False,
            objective_cost_quantum=objective_cost_quantum,
            lattice_closed=False,
            termination_reason="unbreakable",
            trace=(),
            line_pool=(),
            cut_pool=(),
        )

    best_attack = attackable
    best_upper = sum(
        (_fraction(equipment.nodes[node]["attack_cost"]) for node in best_attack),
        Fraction(0, 1),
    )
    best_lower = Fraction(0, 1)
    lambda_max = best_upper
    line_pool: list[PCDLine] = []
    cut_pool: list[tuple[NodeId, ...]] = []
    trace: list[PCDIteration] = []
    evaluated: set[Fraction] = set()
    dual_closed = False
    lattice_closed = False
    master_upper: Fraction | None = None
    next_lambda = Fraction(0, 1)
    repeated_multiplier_without_closure = False

    for oracle_call in range(1, max_oracle_calls + 1):
        if next_lambda in evaluated:
            # Re-visiting a multiplier is a valid dual-closure certificate only
            # when the restricted master upper bound has actually met the best
            # true oracle value.  A repeated point alone is not a lower bound.
            dual_closed = (
                master_upper is not None and master_upper <= best_lower
            )
            repeated_multiplier_without_closure = not dual_closed
            break
        master_before = master_upper
        oracle = _weighted_cut_oracle(
            equipment, threshold, next_lambda, forbidden
        )
        evaluated.add(next_lambda)
        best_lower = max(best_lower, oracle.dual_value)
        if oracle.cut_nodes not in cut_pool:
            cut_pool.append(oracle.cut_nodes)
        new_lines = 0
        for line in oracle.lines:
            if line not in line_pool:
                line_pool.append(line)
                new_lines += 1

        cover = _exact_cut_cover(
            equipment, oracle.cut_nodes, threshold, forbidden
        )
        candidate_attack: frozenset[NodeId] = frozenset()
        candidate_cost: Fraction | None = None
        if cover is not None:
            candidate_attack, candidate_cost = cover
            candidate_flow = motif_capacity(equipment, candidate_attack)
            if candidate_flow.value > threshold - 1:
                raise RuntimeError("PCD cut-cover candidate failed flow verification.")
            if candidate_cost < best_upper or (
                candidate_cost == best_upper
                and tuple(map(repr, stable_nodes(candidate_attack)))
                < tuple(map(repr, stable_nodes(best_attack)))
            ):
                best_attack = candidate_attack
                best_upper = candidate_cost
                lambda_max = min(lambda_max, best_upper)

        trace.append(
            PCDIteration(
                oracle_call,
                next_lambda,
                oracle.dual_value,
                master_before,
                oracle.cut_nodes,
                candidate_attack,
                candidate_cost,
                best_lower,
                best_upper,
                new_lines,
            )
        )

        if best_lower == best_upper:
            dual_closed = True
            break
        lattice_lower = _lattice_ceiling(best_lower, objective_cost_quantum)
        if lattice_lower == best_upper:
            lattice_closed = True
            break
        next_lambda, master_upper = _master_solution(
            tuple(line_pool), lambda_max
        )
        if master_upper <= best_lower:
            dual_closed = True
            break

    certified_by_pcd = best_lower == best_upper or lattice_closed
    exact_fallback_used = False
    termination_reason = (
        "pcd_primal_dual_closed"
        if best_lower == best_upper
        else "pcd_objective_lattice_closed"
        if lattice_closed
        else "dual_closed_with_incumbent_gap"
        if dual_closed
        else "repeated_multiplier_without_bound_closure"
        if repeated_multiplier_without_closure
        else "oracle_limit_reached"
    )

    if not certified_by_pcd and fallback == "exact":
        exact = solve_rmcd_exact(
            equipment,
            threshold,
            protected=protected_set,
            excluded=excluded_set,
            time_limit=time_limit,
            mip_rel_gap=0.0,
        )
        exact_fallback_used = True
        exact = replace(exact, method="pcd+exact-fallback")
        lower = exact.solver.lower_bound
        upper = exact.solver.upper_bound
        absolute_gap = None if lower is None or upper is None else upper - lower
        relative_gap = (
            None
            if absolute_gap is None or upper is None
            else absolute_gap / max(1.0, abs(upper))
        )
        return PCDResult(
            rmcd=exact,
            initial_capacity=initial_capacity,
            lower_bound=lower,
            upper_bound=upper,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            dual_master_upper=(
                float(master_upper) if master_upper is not None else None
            ),
            dual_search_closed=dual_closed,
            certified_by_pcd=False,
            certification_source=("exact_fallback" if exact.optimal else "none"),
            exact_fallback_used=True,
            homogeneous_closed_form=False,
            objective_cost_quantum=objective_cost_quantum,
            lattice_closed=lattice_closed,
            termination_reason="exact_fallback_" + exact.status.value.lower(),
            trace=tuple(trace),
            line_pool=tuple(line_pool),
            cut_pool=tuple(cut_pool),
        )

    residual = motif_capacity(equipment, best_attack)
    if residual.value > threshold - 1:
        raise RuntimeError("PCD incumbent failed final flow verification.")
    runtime = perf_counter() - started
    if best_lower > best_upper:
        raise RuntimeError(
            "PCD weak-duality violation: lower bound exceeds feasible upper bound."
        )
    gap_fraction = best_upper - best_lower
    relative_gap_fraction = gap_fraction / max(Fraction(1, 1), abs(best_upper))
    certified_lower = best_upper if certified_by_pcd else best_lower
    certified_relative_gap = (
        Fraction(0, 1) if certified_by_pcd else relative_gap_fraction
    )
    status = (
        SolveStatus.OPTIMAL
        if certified_by_pcd
        else SolveStatus.DUAL_CLOSED_GAP
        if dual_closed
        else SolveStatus.LIMIT_REACHED
    )
    solver = SolverInfo(
        status,
        certified_by_pcd,
        message=(
            "PCD lower and upper bounds closed exactly."
            if best_lower == best_upper
            else (
                "PCD lower bound and verified incumbent were closed by the "
                "exact rational attack-cost objective lattice."
            )
            if lattice_closed
            else (
                "PCD solved the bounded Lagrangian dual but returned a verified "
                "incumbent separated by a nonzero duality gap."
                if dual_closed
                else "PCD reached its oracle-call limit with a verified incumbent "
                "and an explicit certificate interval."
            )
        ),
        objective=float(best_upper),
        lower_bound=float(certified_lower),
        upper_bound=float(best_upper),
        mip_gap=float(certified_relative_gap),
        runtime_seconds=runtime,
    )
    rmcd = RMCDResult(
        threshold,
        best_attack,
        float(best_upper),
        residual.value,
        residual.cut,
        solver,
        protected_set,
        excluded_set,
        "pcd",
        equipment,
    )
    return PCDResult(
        rmcd=rmcd,
        initial_capacity=initial_capacity,
        lower_bound=float(best_lower),
        upper_bound=float(best_upper),
        absolute_gap=float(gap_fraction),
        relative_gap=float(relative_gap_fraction),
        dual_master_upper=(float(master_upper) if master_upper is not None else None),
        dual_search_closed=dual_closed,
        certified_by_pcd=certified_by_pcd,
        certification_source=(
            "pcd_primal_dual_closure"
            if best_lower == best_upper
            else "objective_lattice_closure"
            if lattice_closed
            else "none"
        ),
        exact_fallback_used=exact_fallback_used,
        homogeneous_closed_form=False,
        objective_cost_quantum=objective_cost_quantum,
        lattice_closed=lattice_closed,
        termination_reason=termination_reason,
        trace=tuple(trace),
        line_pool=tuple(line_pool),
        cut_pool=tuple(cut_pool),
    )
