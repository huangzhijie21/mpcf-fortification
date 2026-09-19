"""Exact adaptive RMCD-F fortification by constraint generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import fsum, isfinite
from time import perf_counter
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .flow import motif_capacity
from .model import (
    NodeId,
    SolveStatus,
    SolverInfo,
    attack_cost,
    protection_cost,
    stable_nodes,
    validate_graph,
)
from .rmcd import (
    RMCDResult,
    brute_force_rmcd,
    solve_rmcd,
    verify_rmcd_certificate,
)


@dataclass(frozen=True)
class FortificationIteration:
    """One exact master/adaptive-oracle exchange."""

    iteration: int
    protection_set: frozenset[NodeId]
    protection_cost: float
    master_bound: float
    master_runtime_seconds: float
    oracle_status: SolveStatus
    oracle_cost: float | None
    oracle_attack: frozenset[NodeId]
    oracle_runtime_seconds: float


@dataclass(frozen=True)
class FortificationResult:
    """Certified maximin defense or an explicitly incomplete CCG incumbent."""

    threshold: int
    budget: float
    protection_set: frozenset[NodeId]
    protection_cost: float
    adaptive_attack: RMCDResult | None
    defended_attack_cost: float | None
    censored_unbreakable: bool
    solver: SolverInfo
    trace: tuple[FortificationIteration, ...]
    attack_pool: tuple[frozenset[NodeId], ...]
    bar_c_a: float
    _graph: nx.DiGraph | None = field(default=None, repr=False, compare=False)

    @property
    def status(self) -> SolveStatus:
        return self.solver.status

    @property
    def optimal(self) -> bool:
        return self.solver.optimal


@dataclass(frozen=True)
class _MasterResult:
    protection_set: frozenset[NodeId]
    protection_cost: float
    theta: float | None
    solver: SolverInfo


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _fortification_big_m(graph: nx.DiGraph) -> float:
    costs = [float(graph.nodes[node]["attack_cost"]) for node in graph]
    return fsum((*costs, min(costs)))


def _solve_master(
    graph: nx.DiGraph,
    budget: float,
    attacks: Sequence[frozenset[NodeId]],
    bar_c_a: float,
    *,
    time_limit: float | None,
) -> _MasterResult:
    nodes = stable_nodes(graph.nodes)
    z_index = {node: index for index, node in enumerate(nodes)}
    theta_index = len(nodes)
    variable_count = len(nodes) + 1
    objective = np.zeros(variable_count, dtype=float)
    objective[theta_index] = -1.0
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[: len(nodes)] = 1
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    upper[theta_index] = float(bar_c_a)

    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []

    rows.append(
        {
            z_index[node]: float(graph.nodes[node]["protect_cost"])
            for node in nodes
        }
    )
    lbs.append(-np.inf)
    ubs.append(float(budget))

    for attack in attacks:
        cost = attack_cost(graph, attack)
        relaxation = float(bar_c_a) - cost
        row = {theta_index: 1.0}
        for node in attack:
            row[z_index[node]] = -relaxation
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(cost)

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for column_index, value in row.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)), shape=(len(rows), variable_count)
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
    vector = getattr(result, "x", None)
    protection = frozenset()
    theta: float | None = None
    if vector is not None:
        protection = frozenset(
            node for node in nodes if float(vector[z_index[node]]) > 0.5
        )
        theta = float(vector[theta_index])

    reported_gap = _finite_float(getattr(result, "mip_gap", None))
    if result.status == 0 and vector is not None and (
        reported_gap is not None and reported_gap <= 1e-10
    ):
        status = SolveStatus.OPTIMAL
        optimal = True
    elif result.status == 0 and vector is not None:
        status = SolveStatus.FEASIBLE
        optimal = False
    elif result.status == 1:
        status = (
            SolveStatus.TIME_LIMIT
            if "time limit" in str(result.message).lower()
            else SolveStatus.LIMIT_REACHED
        )
        optimal = False
    elif result.status == 2:
        status = SolveStatus.INFEASIBLE
        optimal = False
    elif result.status == 3:
        status = SolveStatus.UNBOUNDED
        optimal = False
    else:
        status = SolveStatus.ERROR
        optimal = False

    dual_min = _finite_float(getattr(result, "mip_dual_bound", None))
    master_upper = -dual_min if dual_min is not None else None
    solver = SolverInfo(
        status=status,
        optimal=optimal,
        scipy_status=int(result.status),
        message=(
            str(result.message)
            if status != SolveStatus.FEASIBLE
            else str(result.message)
            + " A non-zero master MIP gap prevents a global-optimality claim."
        ),
        objective=theta,
        lower_bound=theta,
        upper_bound=master_upper if not optimal else theta,
        mip_gap=reported_gap,
        runtime_seconds=runtime,
    )
    return _MasterResult(
        protection,
        protection_cost(graph, protection),
        theta,
        solver,
    )


def _partial_result(
    graph: nx.DiGraph,
    K: int,
    budget: float,
    protection: frozenset[NodeId],
    adaptive_attack: RMCDResult | None,
    trace: list[FortificationIteration],
    attack_pool: list[frozenset[NodeId]],
    bar_c_a: float,
    status: SolveStatus,
    message: str,
    runtime: float,
    source_solver: SolverInfo | None = None,
) -> FortificationResult:
    defended_cost = (
        adaptive_attack.attack_cost
        if adaptive_attack is not None and adaptive_attack.optimal
        else None
    )
    diagnostic_message = message
    if source_solver is not None and source_solver.message:
        diagnostic_message += f" Backend: {source_solver.message}"
    solver = SolverInfo(
        status=status,
        optimal=False,
        scipy_status=(source_solver.scipy_status if source_solver else None),
        message=diagnostic_message,
        objective=defended_cost,
        lower_bound=defended_cost,
        upper_bound=trace[-1].master_bound if trace else None,
        mip_gap=(source_solver.mip_gap if source_solver else None),
        runtime_seconds=runtime,
    )
    return FortificationResult(
        K,
        budget,
        protection,
        protection_cost(graph, protection),
        adaptive_attack,
        defended_cost,
        False,
        solver,
        tuple(trace),
        tuple(attack_pool),
        bar_c_a,
        graph,
    )


def solve_rmcd_f(
    graph: nx.Graph,
    K: int,
    Bp: float,
    *,
    time_limit_per_solve: float | None = None,
    max_iterations: int = 10_000,
    tolerance: float = 1e-7,
    progress: Callable[[str], None] | None = None,
) -> FortificationResult:
    """Solve adaptive binary fortification with an exact RMCD attack oracle."""

    equipment = validate_graph(graph)
    if not isfinite(Bp) or Bp < 0:
        raise ValueError("Bp must be finite and non-negative.")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    started = perf_counter()
    base_attack = solve_rmcd(
        equipment,
        K,
        time_limit=time_limit_per_solve,
        mip_rel_gap=0.0,
    )
    if not base_attack.optimal or base_attack.is_unbreakable:
        raise RuntimeError(
            "RMCD-F requires a finite globally optimal undefended RMCD attack."
        )

    bar_c_a = _fortification_big_m(equipment)
    attack_pool: list[frozenset[NodeId]] = [base_attack.attack_set]
    trace: list[FortificationIteration] = []
    previous_master_bound = np.inf
    current_protection = frozenset()
    current_oracle: RMCDResult | None = base_attack

    for iteration in range(1, max_iterations + 1):
        master = _solve_master(
            equipment,
            float(Bp),
            attack_pool,
            bar_c_a,
            time_limit=time_limit_per_solve,
        )
        if not master.solver.optimal or master.theta is None:
            return _partial_result(
                equipment,
                int(K),
                float(Bp),
                current_protection,
                current_oracle,
                trace,
                attack_pool,
                bar_c_a,
                master.solver.status,
                "Fortification master was not solved to global optimality.",
                perf_counter() - started,
                source_solver=master.solver,
            )
        if master.theta > previous_master_bound + tolerance:
            return _partial_result(
                equipment,
                int(K),
                float(Bp),
                current_protection,
                current_oracle,
                trace,
                attack_pool,
                bar_c_a,
                SolveStatus.ERROR,
                "CCG master upper bound increased beyond numerical tolerance.",
                perf_counter() - started,
            )
        previous_master_bound = master.theta
        current_protection = master.protection_set
        oracle = solve_rmcd(
            equipment,
            K,
            protected=current_protection,
            time_limit=time_limit_per_solve,
            mip_rel_gap=0.0,
        )
        current_oracle = oracle
        trace.append(
            FortificationIteration(
                iteration,
                current_protection,
                master.protection_cost,
                master.theta,
                master.solver.runtime_seconds,
                oracle.status,
                oracle.attack_cost,
                oracle.attack_set,
                oracle.solver.runtime_seconds,
            )
        )
        if progress is not None:
            oracle_value = (
                "UNBREAKABLE"
                if oracle.is_unbreakable
                else str(oracle.attack_cost)
            )
            progress(
                f"CCG iteration {iteration}: master_bound={master.theta:.10g}, "
                f"oracle={oracle_value}, oracle_status={oracle.status.value}, "
                f"attacks={len(attack_pool)}"
            )

        if oracle.is_unbreakable and oracle.optimal:
            attack_all_unprotected = set(equipment) - set(current_protection)
            protected_only_capacity = motif_capacity(
                equipment, attack_all_unprotected
            ).value
            if protected_only_capacity < int(K):
                return _partial_result(
                    equipment,
                    int(K),
                    float(Bp),
                    current_protection,
                    oracle,
                    trace,
                    attack_pool,
                    bar_c_a,
                    SolveStatus.ERROR,
                    "Oracle claimed UNBREAKABLE but protected-only capacity is below K.",
                    perf_counter() - started,
                )
            solver = SolverInfo(
                SolveStatus.UNBREAKABLE,
                True,
                message=(
                    "Certified unbreakable: deleting every unprotected node leaves "
                    f"role-motif capacity {protected_only_capacity} >= K."
                ),
                objective=None,
                lower_bound=None,
                upper_bound=None,
                mip_gap=0.0,
                runtime_seconds=perf_counter() - started,
            )
            return FortificationResult(
                int(K),
                float(Bp),
                current_protection,
                master.protection_cost,
                oracle,
                None,
                True,
                solver,
                tuple(trace),
                tuple(attack_pool),
                bar_c_a,
                equipment,
            )

        if not oracle.optimal or oracle.attack_cost is None:
            return _partial_result(
                equipment,
                int(K),
                float(Bp),
                current_protection,
                oracle,
                trace,
                attack_pool,
                bar_c_a,
                oracle.status,
                "Adaptive RMCD oracle was not solved to global optimality.",
                perf_counter() - started,
                source_solver=oracle.solver,
            )

        if oracle.attack_cost > master.theta:
            return _partial_result(
                equipment,
                int(K),
                float(Bp),
                current_protection,
                oracle,
                trace,
                attack_pool,
                bar_c_a,
                SolveStatus.ERROR,
                "Adaptive oracle value exceeds the CCG master upper bound.",
                perf_counter() - started,
            )

        gap = master.theta - oracle.attack_cost
        if gap <= tolerance:
            solver = SolverInfo(
                SolveStatus.OPTIMAL,
                True,
                message=(
                    "CCG converged within the reported numerical tolerance with "
                    "an adaptive re-optimized attack."
                ),
                objective=oracle.attack_cost,
                lower_bound=oracle.attack_cost,
                upper_bound=master.theta,
                mip_gap=gap / max(1.0, abs(oracle.attack_cost)),
                runtime_seconds=perf_counter() - started,
            )
            return FortificationResult(
                int(K),
                float(Bp),
                current_protection,
                master.protection_cost,
                oracle,
                oracle.attack_cost,
                False,
                solver,
                tuple(trace),
                tuple(attack_pool),
                bar_c_a,
                equipment,
            )

        if oracle.attack_set.intersection(current_protection):
            return _partial_result(
                equipment,
                int(K),
                float(Bp),
                current_protection,
                oracle,
                trace,
                attack_pool,
                bar_c_a,
                SolveStatus.ERROR,
                "Adaptive oracle attacked a protected node.",
                perf_counter() - started,
            )
        if oracle.attack_set in attack_pool:
            return _partial_result(
                equipment,
                int(K),
                float(Bp),
                current_protection,
                oracle,
                trace,
                attack_pool,
                bar_c_a,
                SolveStatus.ERROR,
                "CCG produced a duplicate violated attack constraint.",
                perf_counter() - started,
            )
        attack_pool.append(oracle.attack_set)

    return _partial_result(
        equipment,
        int(K),
        float(Bp),
        current_protection,
        current_oracle,
        trace,
        attack_pool,
        bar_c_a,
        SolveStatus.FEASIBLE,
        "Maximum CCG iteration count reached before certification.",
        perf_counter() - started,
    )


def brute_force_fortification(
    graph: nx.Graph,
    K: int,
    Bp: float,
    *,
    max_nodes: int = 16,
) -> FortificationResult:
    """Enumerate all budget-feasible defenses and all adaptive attacks."""

    equipment = validate_graph(graph)
    nodes = stable_nodes(equipment.nodes)
    if len(nodes) > max_nodes:
        raise ValueError(
            f"Brute-force fortification is limited to {max_nodes} nodes; got {len(nodes)}."
        )
    if not isfinite(Bp) or Bp < 0:
        raise ValueError("Bp must be finite and non-negative.")
    started = perf_counter()
    best_protection = frozenset()
    best_oracle: RMCDResult | None = None
    best_value = -np.inf
    best_unbreakable = False
    tolerance = 1e-10

    for size in range(len(nodes) + 1):
        for subset_tuple in combinations(nodes, size):
            protection = frozenset(subset_tuple)
            p_cost = protection_cost(equipment, protection)
            if p_cost > float(Bp) + tolerance:
                continue
            oracle = brute_force_rmcd(
                equipment,
                K,
                protected=protection,
                max_nodes=max_nodes,
            )
            candidate_unbreakable = oracle.is_unbreakable
            candidate_value = np.inf if candidate_unbreakable else float(oracle.attack_cost)
            incumbent_cost = protection_cost(equipment, best_protection)
            candidate_key = tuple(map(repr, stable_nodes(protection)))
            incumbent_key = tuple(map(repr, stable_nodes(best_protection)))
            better = candidate_value > best_value + tolerance
            tied = (
                candidate_unbreakable == best_unbreakable
                and (
                    (candidate_unbreakable and best_unbreakable)
                    or abs(candidate_value - best_value) <= tolerance
                )
            )
            if better or (
                tied
                and (p_cost < incumbent_cost - tolerance or (abs(p_cost - incumbent_cost) <= tolerance and candidate_key < incumbent_key))
            ):
                best_protection = protection
                best_oracle = oracle
                best_value = candidate_value
                best_unbreakable = candidate_unbreakable

    if best_oracle is None:
        raise RuntimeError("No budget-feasible protection set exists; empty set should be feasible.")
    status = SolveStatus.UNBREAKABLE if best_unbreakable else SolveStatus.OPTIMAL
    defended_cost = None if best_unbreakable else best_oracle.attack_cost
    solver = SolverInfo(
        status,
        True,
        message="Certified by complete defense and adaptive-attack enumeration.",
        objective=defended_cost,
        lower_bound=defended_cost,
        upper_bound=defended_cost,
        mip_gap=0.0,
        runtime_seconds=perf_counter() - started,
    )
    bar_c_a = _fortification_big_m(equipment)
    return FortificationResult(
        int(K),
        float(Bp),
        best_protection,
        protection_cost(equipment, best_protection),
        best_oracle,
        defended_cost,
        best_unbreakable,
        solver,
        (),
        (),
        bar_c_a,
        equipment,
    )


def verify_fortification_certificate(
    result: FortificationResult,
    graph: nx.Graph | None = None,
    *,
    tolerance: float = 1e-7,
) -> bool:
    """Validate the complete CCG upper/lower-bound certificate."""

    source_graph = graph if graph is not None else result._graph
    if source_graph is None:
        raise ValueError("A graph is required to verify this fortification result.")
    equipment = validate_graph(source_graph)
    recomputed_protection_cost = protection_cost(equipment, result.protection_set)
    if recomputed_protection_cost > result.budget + tolerance:
        return False
    if abs(recomputed_protection_cost - result.protection_cost) > tolerance:
        return False
    if not result.optimal:
        return False
    if result.adaptive_attack is None:
        return False
    if (
        result.adaptive_attack.protected != result.protection_set
        or not result.adaptive_attack.optimal
        or not verify_rmcd_certificate(result.adaptive_attack, equipment, tolerance=tolerance)
    ):
        return False
    oracle = solve_rmcd(
        equipment,
        result.threshold,
        protected=result.protection_set,
        mip_rel_gap=0.0,
    )
    if result.censored_unbreakable:
        if (
            result.status != SolveStatus.UNBREAKABLE
            or result.defended_attack_cost is not None
            or not result.adaptive_attack.is_unbreakable
            or not result.trace
            or result.trace[-1].protection_set != result.protection_set
            or result.trace[-1].oracle_status != SolveStatus.UNBREAKABLE
            or result.trace[-1].oracle_cost is not None
            or result.trace[-1].oracle_attack != result.adaptive_attack.attack_set
            or result.solver.objective is not None
            or result.solver.lower_bound is not None
            or result.solver.upper_bound is not None
            or not oracle.is_unbreakable
            or not oracle.optimal
        ):
            return False
        expected_bar = _fortification_big_m(equipment)
        if abs(expected_bar - result.bar_c_a) > tolerance:
            return False
        for attack in result.attack_pool:
            if any(node not in equipment for node in attack):
                return False
            if motif_capacity(equipment, attack).value > result.threshold - 1:
                return False
        all_unprotected = set(equipment) - set(result.protection_set)
        return motif_capacity(equipment, all_unprotected).value >= result.threshold
    if (
        not oracle.optimal
        or oracle.attack_cost is None
        or result.defended_attack_cost is None
        or not result.trace
        or not result.attack_pool
    ):
        return False
    if result.trace[-1].protection_set != result.protection_set:
        return False
    if (
        result.trace[-1].oracle_attack != result.adaptive_attack.attack_set
        or result.trace[-1].oracle_status != result.adaptive_attack.status
        or result.adaptive_attack.attack_cost is None
        or abs(result.adaptive_attack.attack_cost - result.defended_attack_cost) > tolerance
    ):
        return False
    expected_bar = _fortification_big_m(equipment)
    if abs(expected_bar - result.bar_c_a) > tolerance:
        return False
    for attack in result.attack_pool:
        if any(node not in equipment for node in attack):
            return False
        if motif_capacity(equipment, attack).value > result.threshold - 1:
            return False

    master = _solve_master(
        equipment,
        result.budget,
        result.attack_pool,
        result.bar_c_a,
        time_limit=None,
    )
    if not master.solver.optimal or master.theta is None:
        return False
    if abs(master.theta - result.trace[-1].master_bound) > tolerance:
        return False
    if oracle.attack_cost > master.theta + tolerance:
        return False
    if abs(master.theta - oracle.attack_cost) > tolerance:
        return False
    if abs(oracle.attack_cost - result.defended_attack_cost) > tolerance:
        return False
    if (
        result.solver.lower_bound is None
        or result.solver.upper_bound is None
        or abs(result.solver.lower_bound - oracle.attack_cost) > tolerance
        or abs(result.solver.upper_bound - master.theta) > tolerance
    ):
        return False
    return True
