"""Exact and exhaustive solvers for Role-Motif Capacity Dismantling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from math import isfinite
from numbers import Integral
from time import perf_counter
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .flow import (
    CutCertificate,
    FlowNode,
    _verify_cut_certificate,
    build_role_flow_network,
    motif_capacity,
)
from .model import (
    NodeId,
    SolveStatus,
    SolverInfo,
    attack_cost,
    ensure_node_subset,
    stable_node_key,
    stable_nodes,
    validate_graph,
)


@dataclass(frozen=True)
class RMCDResult:
    """A finite attack, a certified unbreakable state, or a non-optimal incumbent."""

    threshold: int
    attack_set: frozenset[NodeId]
    attack_cost: float | None
    residual_capacity: int | None
    cut_certificate: CutCertificate | None
    solver: SolverInfo
    protected: frozenset[NodeId] = frozenset()
    excluded: frozenset[NodeId] = frozenset()
    method: str = "milp"
    _graph: nx.DiGraph | None = field(default=None, repr=False, compare=False)

    @property
    def status(self) -> SolveStatus:
        return self.solver.status

    @property
    def optimal(self) -> bool:
        return self.solver.optimal

    @property
    def is_unbreakable(self) -> bool:
        return self.status == SolveStatus.UNBREAKABLE

    @property
    def is_feasible_attack(self) -> bool:
        return (
            self.attack_cost is not None
            and self.residual_capacity is not None
            and self.residual_capacity <= self.threshold - 1
        )


@dataclass(frozen=True)
class FrontierPoint:
    """One independently optimized point on the cost-capacity frontier."""

    max_remaining_capacity: int
    result: RMCDResult


@dataclass(frozen=True)
class FrontierResult:
    """Complete threshold sweep; attack sets are not asserted to be nested."""

    initial_capacity: int
    points: tuple[FrontierPoint, ...]
    complete: bool


@dataclass(frozen=True)
class _VariableLayout:
    equipment_nodes: tuple[NodeId, ...]
    flow_nodes: tuple[FlowNode, ...]
    x: dict[NodeId, int]
    a: dict[FlowNode, int]
    q: dict[NodeId, int]
    size: int


def _flow_node_key(node: FlowNode) -> tuple[str, str, str]:
    return (node.kind, repr(node.original), type(node.original).__qualname__)


def _layout(graph: nx.DiGraph) -> tuple[_VariableLayout, object]:
    network = build_role_flow_network(graph)
    equipment_nodes = stable_nodes(graph.nodes)
    remaining_flow_nodes = sorted(
        (node for node in network.graph if node not in {network.source, network.sink}),
        key=_flow_node_key,
    )
    flow_nodes = (network.source, network.sink, *remaining_flow_nodes)
    cursor = 0
    x = {node: cursor + index for index, node in enumerate(equipment_nodes)}
    cursor += len(equipment_nodes)
    a = {node: cursor + index for index, node in enumerate(flow_nodes)}
    cursor += len(flow_nodes)
    q = {node: cursor + index for index, node in enumerate(equipment_nodes)}
    cursor += len(equipment_nodes)
    return _VariableLayout(equipment_nodes, flow_nodes, x, a, q, cursor), network


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


_BOUND_CERTIFICATE_TOLERANCE = 1e-7


def _solve_rmcd_milp(
    graph: nx.DiGraph,
    threshold: int,
    *,
    protected: frozenset[NodeId],
    excluded: frozenset[NodeId],
    required_attack: frozenset[NodeId] = frozenset(),
    no_good_sets: Sequence[frozenset[NodeId]] = (),
    fixed_objective: float | None = None,
    objective_tolerance: float = 1e-8,
    time_limit: float | None = None,
    mip_rel_gap: float | None = 0.0,
    node_limit: int | None = None,
) -> RMCDResult:
    if required_attack.intersection(protected | excluded):
        raise ValueError(
            "required_attack cannot overlap protected or excluded nodes."
        )
    layout, network = _layout(graph)
    c = np.zeros(layout.size, dtype=float)
    lower = np.zeros(layout.size, dtype=float)
    upper = np.ones(layout.size, dtype=float)
    integrality = np.zeros(layout.size, dtype=np.uint8)

    for node in layout.equipment_nodes:
        c[layout.x[node]] = float(graph.nodes[node]["attack_cost"])
        integrality[layout.x[node]] = 1
        if node in protected or node in excluded:
            upper[layout.x[node]] = 0.0
        if node in required_attack:
            lower[layout.x[node]] = 1.0
    for node in layout.flow_nodes:
        integrality[layout.a[node]] = 1

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_row(coefficients: dict[int, float], lb: float, ub: float) -> None:
        rows.append(coefficients)
        row_lower.append(lb)
        row_upper.append(ub)

    add_row({layout.a[network.source]: 1.0}, 1.0, 1.0)
    add_row({layout.a[network.sink]: 1.0}, 0.0, 0.0)

    for source, target in network.infinite_arcs:
        add_row({layout.a[source]: 1.0, layout.a[target]: -1.0}, -np.inf, 0.0)

    for node in layout.equipment_nodes:
        incoming = network.node_in[node]
        outgoing = network.node_out[node]
        x_index = layout.x[node]
        a_in = layout.a[incoming]
        a_out = layout.a[outgoing]
        q_index = layout.q[node]

        # q >= a_in - a_out - x
        add_row({q_index: 1.0, a_in: -1.0, a_out: 1.0, x_index: 1.0}, 0.0, np.inf)
        # q <= a_in; q <= 1-a_out; q <= 1-x
        add_row({q_index: 1.0, a_in: -1.0}, -np.inf, 0.0)
        add_row({q_index: 1.0, a_out: 1.0}, -np.inf, 1.0)
        add_row({q_index: 1.0, x_index: 1.0}, -np.inf, 1.0)
        # Positive attack costs guarantee an optimum localized on a certificate cut.
        add_row({x_index: 1.0, a_in: -1.0}, -np.inf, 0.0)
        add_row({x_index: 1.0, a_out: 1.0}, -np.inf, 1.0)

    add_row(
        {
            layout.q[node]: float(graph.nodes[node]["capacity"])
            for node in layout.equipment_nodes
        },
        -np.inf,
        float(threshold - 1),
    )

    for attack in no_good_sets:
        coefficients = {
            layout.x[node]: (-1.0 if node in attack else 1.0)
            for node in layout.equipment_nodes
        }
        add_row(coefficients, 1.0 - float(len(attack)), np.inf)

    if fixed_objective is not None:
        objective_row = {
            layout.x[node]: float(graph.nodes[node]["attack_cost"])
            for node in layout.equipment_nodes
        }
        add_row(
            objective_row,
            float(fixed_objective) - objective_tolerance,
            float(fixed_objective) + objective_tolerance,
        )

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, coefficients in enumerate(rows):
        for column_index, value in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)), shape=(len(rows), layout.size)
    ).tocsr()

    options: dict[str, object] = {"presolve": True}
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive when provided.")
        options["time_limit"] = float(time_limit)
    if mip_rel_gap is not None:
        if mip_rel_gap < 0:
            raise ValueError("mip_rel_gap must be non-negative.")
        options["mip_rel_gap"] = float(mip_rel_gap)
    if node_limit is not None:
        if node_limit <= 0:
            raise ValueError("node_limit must be positive when provided.")
        options["node_limit"] = int(node_limit)

    started = perf_counter()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.array(row_lower), np.array(row_upper)),
        options=options,
    )
    runtime = perf_counter() - started

    scipy_status = int(result.status)
    message = str(result.message)
    dual_bound = _finite_float(getattr(result, "mip_dual_bound", None))
    mip_gap = _finite_float(getattr(result, "mip_gap", None))

    if scipy_status == 2:
        attackable = set(graph) - set(protected) - set(excluded)
        protected_only_capacity = motif_capacity(graph, attackable).value
        genuinely_unbreakable = protected_only_capacity >= threshold
        solver = SolverInfo(
            status=(
                SolveStatus.UNBREAKABLE
                if genuinely_unbreakable
                else SolveStatus.INFEASIBLE
            ),
            optimal=genuinely_unbreakable,
            scipy_status=scipy_status,
            message=(
                message
                if genuinely_unbreakable
                else message
                + " The base attack problem remains breakable; infeasibility is "
                "caused by additional objective/no-good restrictions."
            ),
            lower_bound=None,
            upper_bound=None,
            mip_gap=mip_gap,
            runtime_seconds=runtime,
        )
        return RMCDResult(
            threshold,
            frozenset(),
            None,
            None,
            None,
            solver,
            protected,
            excluded,
            "milp",
            graph,
        )

    if scipy_status == 3:
        status = SolveStatus.UNBOUNDED
    elif scipy_status == 1:
        status = (
            SolveStatus.TIME_LIMIT
            if "time limit" in message.lower()
            else SolveStatus.LIMIT_REACHED
        )
    elif scipy_status == 0:
        # A successful solver exit is not, by itself, the certificate exposed by
        # this API.  OPTIMAL is assigned below only after a feasible incumbent
        # has been independently checked and a finite global lower bound closes
        # against its cost.
        status = SolveStatus.FEASIBLE
    else:
        status = SolveStatus.ERROR

    vector = getattr(result, "x", None)
    candidate_attack: frozenset[NodeId] = frozenset()
    finite_attack_cost: float | None = None
    residual_capacity: int | None = None
    cut_certificate: CutCertificate | None = None
    valid_incumbent = False
    if vector is not None:
        candidate_attack = frozenset(
            node for node in layout.equipment_nodes if float(vector[layout.x[node]]) > 0.5
        )
        if (
            not candidate_attack.intersection(protected | excluded)
            and required_attack.issubset(candidate_attack)
        ):
            flow = motif_capacity(graph, candidate_attack)
            finite_attack_cost = attack_cost(graph, candidate_attack)
            residual_capacity = flow.value
            cut_certificate = flow.cut
            valid_incumbent = residual_capacity <= threshold - 1

    if not valid_incumbent:
        candidate_attack = frozenset()
        finite_attack_cost = None
        residual_capacity = None
        cut_certificate = None
        if status == SolveStatus.OPTIMAL:
            status = SolveStatus.ERROR
            message = "Solver returned an optimal vector that failed independent flow verification."

    if scipy_status == 0 and valid_incumbent:
        gap_is_closed = (
            mip_gap is not None
            and mip_gap <= _BOUND_CERTIFICATE_TOLERANCE
        )
        bound_is_closed = (
            dual_bound is not None
            and finite_attack_cost is not None
            and abs(dual_bound - finite_attack_cost)
            <= _BOUND_CERTIFICATE_TOLERANCE
            * max(1.0, abs(finite_attack_cost))
        )
        if gap_is_closed and bound_is_closed:
            status = SolveStatus.OPTIMAL
        else:
            status = SolveStatus.FEASIBLE
            message += (
                " A finite, closed global lower bound and a closed MIP gap are "
                "required for a global-optimality claim."
            )

    optimal = status == SolveStatus.OPTIMAL and valid_incumbent
    if status in {SolveStatus.TIME_LIMIT, SolveStatus.LIMIT_REACHED} and not valid_incumbent:
        message += " No independently verified feasible incumbent was available."
    upper_bound = finite_attack_cost if valid_incumbent else None
    solver = SolverInfo(
        status=status,
        optimal=optimal,
        scipy_status=scipy_status,
        message=message,
        objective=finite_attack_cost,
        lower_bound=dual_bound,
        upper_bound=upper_bound,
        mip_gap=mip_gap,
        runtime_seconds=runtime,
    )
    return RMCDResult(
        threshold,
        candidate_attack,
        finite_attack_cost,
        residual_capacity,
        cut_certificate,
        solver,
        protected,
        excluded,
        "milp",
        graph,
    )


def _validate_threshold(graph: nx.DiGraph, threshold: int) -> int:
    if isinstance(threshold, bool) or not isinstance(threshold, Integral):
        raise ValueError("K must be an integer.")
    value = int(threshold)
    initial_capacity = motif_capacity(graph).value
    if not 1 <= value <= initial_capacity:
        raise ValueError(
            f"K must satisfy 1 <= K <= Omega_0={initial_capacity}; got {value}."
        )
    return value


def _require_discrete_attack_costs(graph: nx.DiGraph) -> None:
    """Require integer-valued objective units for exact optimal-set claims."""

    costs = [float(graph.nodes[node]["attack_cost"]) for node in graph]
    if any(abs(cost - round(cost)) > 1e-10 for cost in costs):
        raise ValueError(
            "Exact optimal-set enumeration requires integer-valued attack_cost "
            "units. Scale rational costs to integers before critical-set analysis."
        )
    if sum(costs) > 2**52:
        raise ValueError(
            "Integer attack_cost units are too large for exact float representation."
        )


def solve_rmcd(
    graph: nx.Graph,
    K: int,
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
    forbidden_attack: Iterable[NodeId] = (),
    time_limit: float | None = None,
    mip_rel_gap: float | None = 0.0,
    node_limit: int | None = None,
) -> RMCDResult:
    """Solve the compact RMCD MILP with a verifiable residual cut.

    This backward-compatible name denotes the exact reference oracle.  New
    academic code should prefer :func:`solve_rmcd_exact` so that results from
    the generic MILP oracle cannot be confused with the dedicated PCD
    algorithm.
    """

    equipment = validate_graph(graph)
    threshold = _validate_threshold(equipment, K)
    protected_set = ensure_node_subset(equipment, protected, label="protected")
    excluded_set = ensure_node_subset(equipment, excluded, label="excluded")
    forbidden_set = ensure_node_subset(
        equipment, forbidden_attack, label="forbidden_attack"
    )
    excluded_set = frozenset(excluded_set | forbidden_set)
    return _solve_rmcd_milp(
        equipment,
        threshold,
        protected=protected_set,
        excluded=excluded_set,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
        node_limit=node_limit,
    )


def solve_rmcd_exact(
    graph: nx.Graph,
    K: int,
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
    forbidden_attack: Iterable[NodeId] = (),
    time_limit: float | None = None,
    mip_rel_gap: float | None = 0.0,
    node_limit: int | None = None,
) -> RMCDResult:
    """Run the exact compact-MILP oracle (``RMCD-Exact``).

    The wrapper is intentionally explicit rather than a plain alias: its
    result records ``rmcd-exact`` as the method, which keeps solver provenance
    visible in exported experiment tables and certificates.
    """

    result = solve_rmcd(
        graph,
        K,
        protected=protected,
        excluded=excluded,
        forbidden_attack=forbidden_attack,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
        node_limit=node_limit,
    )
    return replace(result, method="rmcd-exact")


def brute_force_rmcd(
    graph: nx.Graph,
    K: int,
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
    forbidden_attack: Iterable[NodeId] = (),
    max_nodes: int = 20,
) -> RMCDResult:
    """Exhaustively solve RMCD; intended only as a small-graph oracle."""

    equipment = validate_graph(graph)
    threshold = _validate_threshold(equipment, K)
    protected_set = ensure_node_subset(equipment, protected, label="protected")
    excluded_set = ensure_node_subset(equipment, excluded, label="excluded")
    excluded_set |= ensure_node_subset(
        equipment, forbidden_attack, label="forbidden_attack"
    )
    candidates = stable_nodes(set(equipment) - protected_set - excluded_set)
    if len(candidates) > max_nodes:
        raise ValueError(
            f"Brute-force oracle is limited to {max_nodes} attackable nodes; "
            f"received {len(candidates)}."
        )

    started = perf_counter()
    best_set: frozenset[NodeId] | None = None
    best_cost = np.inf
    tolerance = 1e-10
    for size in range(len(candidates) + 1):
        for subset_tuple in combinations(candidates, size):
            subset = frozenset(subset_tuple)
            cost = attack_cost(equipment, subset)
            if cost > best_cost + tolerance:
                continue
            if motif_capacity(equipment, subset).value <= threshold - 1:
                if cost < best_cost - tolerance:
                    best_cost = cost
                    best_set = subset
                elif best_set is not None and abs(cost - best_cost) <= tolerance:
                    if tuple(map(repr, stable_nodes(subset))) < tuple(
                        map(repr, stable_nodes(best_set))
                    ):
                        best_set = subset

    runtime = perf_counter() - started
    if best_set is None:
        solver = SolverInfo(
            SolveStatus.UNBREAKABLE,
            True,
            message="Complete attack-set enumeration found no successful attack.",
            runtime_seconds=runtime,
        )
        return RMCDResult(
            threshold,
            frozenset(),
            None,
            None,
            None,
            solver,
            protected_set,
            frozenset(excluded_set),
            "brute_force",
            equipment,
        )

    flow = motif_capacity(equipment, best_set)
    solver = SolverInfo(
        SolveStatus.OPTIMAL,
        True,
        message="Certified by complete attack-set enumeration.",
        objective=float(best_cost),
        lower_bound=float(best_cost),
        upper_bound=float(best_cost),
        mip_gap=0.0,
        runtime_seconds=runtime,
    )
    return RMCDResult(
        threshold,
        best_set,
        float(best_cost),
        flow.value,
        flow.cut,
        solver,
        protected_set,
        frozenset(excluded_set),
        "brute_force",
        equipment,
    )


def solve_frontier(
    graph: nx.Graph,
    *,
    method: str = "milp",
    protected: Iterable[NodeId] = (),
    time_limit: float | None = None,
    max_bruteforce_nodes: int = 20,
    progress: Callable[[str], None] | None = None,
) -> FrontierResult:
    """Compute every independent cost-capacity frontier point Phi(k)."""

    equipment = validate_graph(graph)
    initial_capacity = motif_capacity(equipment).value
    points: list[FrontierPoint] = []
    for maximum_remaining in range(initial_capacity):
        threshold = maximum_remaining + 1
        if method == "milp":
            result = solve_rmcd(
                equipment, threshold, protected=protected, time_limit=time_limit
            )
        elif method == "brute_force":
            result = brute_force_rmcd(
                equipment,
                threshold,
                protected=protected,
                max_nodes=max_bruteforce_nodes,
            )
        else:
            raise ValueError("method must be 'milp' or 'brute_force'.")
        points.append(FrontierPoint(maximum_remaining, result))
        if progress is not None:
            progress(
                f"frontier K={threshold}/{initial_capacity}: "
                f"status={result.status.value}, optimal={result.optimal}"
            )
    return FrontierResult(
        initial_capacity,
        tuple(points),
        all(point.result.optimal for point in points),
    )


def enumerate_optimal_attacks(
    graph: nx.Graph,
    K: int,
    *,
    protected: Iterable[NodeId] = (),
    excluded: Iterable[NodeId] = (),
    limit: int = 100,
    objective_tolerance: float = 1e-8,
    time_limit_per_solve: float | None = None,
) -> tuple[tuple[frozenset[NodeId], ...], bool]:
    """Enumerate up to ``limit`` optimal sets and report whether enumeration ended."""

    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not 0.0 <= objective_tolerance < 0.5:
        raise ValueError("objective_tolerance must satisfy 0 <= tolerance < 0.5.")
    equipment = validate_graph(graph)
    _require_discrete_attack_costs(equipment)
    threshold = _validate_threshold(equipment, K)
    protected_set = ensure_node_subset(equipment, protected, label="protected")
    excluded_set = ensure_node_subset(equipment, excluded, label="excluded")
    base = _solve_rmcd_milp(
        equipment,
        threshold,
        protected=protected_set,
        excluded=excluded_set,
        time_limit=time_limit_per_solve,
        mip_rel_gap=0.0,
    )
    if not base.optimal or base.is_unbreakable or base.attack_cost is None:
        return (), base.optimal and base.is_unbreakable

    found: list[frozenset[NodeId]] = []
    no_good: list[frozenset[NodeId]] = []
    while len(found) < limit:
        result = _solve_rmcd_milp(
            equipment,
            threshold,
            protected=protected_set,
            excluded=excluded_set,
            no_good_sets=no_good,
            fixed_objective=base.attack_cost,
            objective_tolerance=objective_tolerance,
            time_limit=time_limit_per_solve,
            mip_rel_gap=0.0,
        )
        if result.status in {SolveStatus.INFEASIBLE, SolveStatus.UNBREAKABLE}:
            return tuple(found), True
        if not result.optimal or result.attack_cost is None:
            return tuple(found), False
        if abs(result.attack_cost - base.attack_cost) > objective_tolerance:
            return tuple(found), True
        if result.attack_set in no_good:
            return tuple(found), False
        found.append(result.attack_set)
        no_good.append(result.attack_set)

    probe = _solve_rmcd_milp(
        equipment,
        threshold,
        protected=protected_set,
        excluded=excluded_set,
        no_good_sets=no_good,
        fixed_objective=base.attack_cost,
        objective_tolerance=objective_tolerance,
        time_limit=time_limit_per_solve,
        mip_rel_gap=0.0,
    )
    complete = probe.status in {SolveStatus.INFEASIBLE, SolveStatus.UNBREAKABLE}
    return tuple(found), complete


def verify_rmcd_certificate(
    result: RMCDResult, graph: nx.Graph | None = None, *, tolerance: float = 1e-7
) -> bool:
    """Validate feasibility, cost, cut, and status semantics of an RMCD result."""

    source_graph = graph if graph is not None else result._graph
    if source_graph is None:
        raise ValueError("A graph is required to verify this RMCD result.")
    equipment = validate_graph(source_graph)
    try:
        protected = ensure_node_subset(equipment, result.protected, label="protected")
        excluded = ensure_node_subset(equipment, result.excluded, label="excluded")
        attack_set = ensure_node_subset(equipment, result.attack_set, label="attack")
        _validate_threshold(equipment, result.threshold)
    except ValueError:
        return False
    if result.is_unbreakable:
        unavailable = set(equipment) - set(protected) - set(excluded)
        return (
            result.optimal
            and not result.attack_set
            and result.attack_cost is None
            and result.residual_capacity is None
            and result.cut_certificate is None
            and result.solver.objective is None
            and result.solver.lower_bound is None
            and result.solver.upper_bound is None
            and motif_capacity(equipment, unavailable).value >= result.threshold
        )
    if not result.is_feasible_attack or result.attack_cost is None:
        return False
    if result.status not in {
        SolveStatus.OPTIMAL,
        SolveStatus.FEASIBLE,
        SolveStatus.TIME_LIMIT,
        SolveStatus.LIMIT_REACHED,
        SolveStatus.DUAL_CLOSED_GAP,
    }:
        return False
    if result.optimal != (result.status == SolveStatus.OPTIMAL):
        return False
    if attack_set.intersection(protected | excluded):
        return False
    flow = motif_capacity(equipment, attack_set)
    if (
        flow.value > result.threshold - 1
        or result.residual_capacity != flow.value
    ):
        return False
    if result.cut_certificate is None or not _verify_cut_certificate(
        equipment,
        attack_set,
        result.cut_certificate,
        expected_value=flow.value,
    ):
        return False
    if abs(attack_cost(equipment, result.attack_set) - result.attack_cost) > tolerance:
        return False
    if (
        result.solver.objective is None
        or abs(result.solver.objective - result.attack_cost) > tolerance
        or result.solver.upper_bound is None
        or abs(result.solver.upper_bound - result.attack_cost) > tolerance
        or (
            result.solver.lower_bound is not None
            and result.solver.lower_bound > result.attack_cost + tolerance
        )
    ):
        return False
    if result.optimal:
        if result.solver.mip_gap is None or result.solver.mip_gap > tolerance:
            return False
        if result.solver.lower_bound is None or result.solver.upper_bound is None:
            return False
        if abs(result.solver.lower_bound - result.solver.upper_bound) > tolerance:
            return False
        if abs(result.solver.lower_bound - result.attack_cost) > tolerance:
            return False
        reference = solve_rmcd(
            equipment,
            result.threshold,
            protected=protected,
            excluded=excluded,
            mip_rel_gap=0.0,
        )
        if (
            not reference.optimal
            or reference.attack_cost is None
            or abs(reference.attack_cost - result.attack_cost) > tolerance
        ):
            return False
    return True
