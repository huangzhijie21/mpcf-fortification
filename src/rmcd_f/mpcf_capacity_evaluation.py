"""Out-of-objective service-capacity evaluation for frozen MPCF selections.

MPCF continues to select a fortification set against complete task-path
disconnection.  This module does not feed capacity information back into that
selection.  It only asks how much adaptive attack cost is required, after the
selection is frozen, to reduce the surviving S-C-L-E role-motif capacity to a
predeclared integer threshold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from math import isclose, isfinite
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .flow import motif_capacity, verify_flow_certificate
from .model import (
    NodeId,
    SolveStatus,
    attack_cost,
    ensure_node_subset,
    stable_nodes,
    validate_graph,
)
from .operational_motif import build_operational_motif_system
from .pcd import solve_rmcd_pcd
from .rmcd import RMCDResult, solve_rmcd_exact, verify_rmcd_certificate
from .task_path_fortification import (
    evaluate_task_path_fortification,
    fortified_attack_costs,
)


CAPACITY_EVALUATION_VERSION = "mpcf-out-of-objective-capacity-v1"
CAPACITY_THRESHOLD_SEMANTICS = (
    "minimum attack cost such that residual integer role-motif capacity is "
    "at most floor(requested_fraction * initial_capacity)"
)
SUPPORTED_CAPACITY_SOLVERS = frozenset(
    {"rmcd-exact", "pcd", "pcd-exact-fallback"}
)


class CapacityEvaluationNotApplicableError(ValueError):
    """Raised when the graph is outside the fixed S-C-L-E capacity model."""


@dataclass(frozen=True)
class CapacityThresholdPoint:
    requested_remaining_fraction: float
    threshold_k: int
    max_remaining_capacity: int
    realized_remaining_fraction: float
    threshold_alias_id: str
    canonical_threshold: bool
    solver_method: str
    status: str
    feasible_attack: bool
    optimal: bool
    unbreakable: bool
    attack_set: frozenset[NodeId]
    attack_cost: float | None
    residual_capacity: int | None
    residual_fraction: float | None
    lower_bound: float | None
    upper_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    certification_source: str
    runtime_seconds: float | None
    residual_flow_certificate_valid: bool
    intrinsic_result_valid: bool
    independent_reference_verified: bool
    global_optimum_certified: bool
    r0_pathcut_margin: float
    r0_absolute_error: float | None
    r0_crosscheck_valid: bool | None


@dataclass(frozen=True)
class CapacityThresholdEvaluation:
    version: str
    threshold_semantics: str
    solver_method: str
    initial_capacity: int
    initial_flow_certificate_valid: bool
    protection_set: frozenset[NodeId]
    fortification_multiplier: float
    uplift_map_sha256: str
    r0_pathcut_margin: float
    points: tuple[CapacityThresholdPoint, ...]


def _fraction(value: float) -> Fraction:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Remaining-capacity fractions must be numeric.") from exc
    if not isfinite(number) or number < 0.0 or number >= 1.0:
        raise ValueError("Remaining-capacity fractions must lie in [0, 1).")
    return Fraction(str(number))


def _uplift_hash(
    graph: nx.DiGraph,
    protected: frozenset[NodeId],
    fortified_cost: Mapping[NodeId, float],
) -> str:
    rows = [
        {
            "node_type": type(node).__qualname__,
            "node_repr": repr(node),
            "base_attack_cost": float(graph.nodes[node]["attack_cost"]),
            "fortified_attack_cost": float(fortified_cost[node]),
        }
        for node in stable_nodes(protected)
    ]
    payload = json.dumps(
        rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fortified_graph(
    graph: nx.DiGraph,
    protection_set: Iterable[NodeId],
    *,
    fortification_multiplier: float,
    uplifts: Mapping[NodeId, float] | None,
) -> tuple[nx.DiGraph, frozenset[NodeId], str]:
    try:
        equipment = validate_graph(graph)
    except (TypeError, ValueError) as exc:
        raise CapacityEvaluationNotApplicableError(
            "Capacity thresholds require a valid fixed S-C-L-E equipment graph."
        ) from exc
    roles = {equipment.nodes[node].get("role") for node in equipment}
    if roles != {"S", "C", "L", "E"}:
        raise CapacityEvaluationNotApplicableError(
            "Capacity thresholds require all four fixed roles S, C, L and E."
        )
    system = build_operational_motif_system(equipment)
    protected = ensure_node_subset(
        equipment, protection_set, label="protection_set"
    )
    unknown = protected - frozenset(system.removable_nodes)
    if unknown:
        raise ValueError(
            "Protection set contains non-removable nodes: "
            f"{stable_nodes(unknown)!r}."
        )
    fortified_cost = fortified_attack_costs(
        system,
        protected,
        fortification_multiplier=fortification_multiplier,
        uplifts=uplifts,
    )
    fortified = equipment.copy()
    for node, value in fortified_cost.items():
        fortified.nodes[node]["attack_cost"] = float(value)
    return fortified, protected, _uplift_hash(equipment, protected, fortified_cost)


def _closed_bounds(result: RMCDResult, *, tolerance: float = 1e-7) -> bool:
    if not result.optimal or result.attack_cost is None:
        return False
    lower = result.solver.lower_bound
    upper = result.solver.upper_bound
    return bool(
        lower is not None
        and upper is not None
        and abs(lower - upper) <= tolerance
        and abs(lower - result.attack_cost) <= tolerance
    )


def _intrinsic_result_valid(
    graph: nx.DiGraph,
    result: RMCDResult,
    *,
    tolerance: float = 1e-7,
) -> tuple[bool, bool]:
    if result.is_unbreakable:
        attackable = frozenset(
            node
            for node in graph
            if int(graph.nodes[node]["capacity"]) > 0
        )
        valid = bool(
            result.optimal
            and result.attack_cost is None
            and motif_capacity(graph, attackable).value >= result.threshold
        )
        return valid, valid
    if not result.is_feasible_attack or result.attack_cost is None:
        return False, False
    residual = motif_capacity(graph, result.attack_set)
    residual_valid = verify_flow_certificate(graph, residual)
    cost_valid = isclose(
        attack_cost(graph, result.attack_set),
        result.attack_cost,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    result_valid = bool(
        residual_valid
        and cost_valid
        and result.residual_capacity == residual.value
        and residual.value <= result.threshold - 1
        and result.solver.objective is not None
        and isclose(
            result.solver.objective,
            result.attack_cost,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    )
    return result_valid, residual_valid


def evaluate_role_motif_capacity_thresholds(
    graph: nx.DiGraph,
    protection_set: Iterable[NodeId],
    remaining_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    solver: str = "rmcd-exact",
    time_limit: float | None = None,
    max_oracle_calls: int = 64,
    independent_verification: bool = False,
) -> CapacityThresholdEvaluation:
    """Evaluate a frozen protection set at predeclared capacity thresholds.

    The function never changes or re-optimizes ``protection_set``.  Protection
    is represented by finite attack-cost uplift, not by RMCD's invulnerable
    ``protected`` argument.
    """

    if solver not in SUPPORTED_CAPACITY_SOLVERS:
        raise ValueError(
            f"solver must be one of {sorted(SUPPORTED_CAPACITY_SOLVERS)}."
        )
    requested = tuple(float(value) for value in remaining_fractions)
    if not requested:
        raise ValueError("At least one remaining-capacity fraction is required.")
    if len(requested) != len(set(requested)):
        raise ValueError("Remaining-capacity fractions must be unique.")
    rational = {value: _fraction(value) for value in requested}
    fortified, protected, uplift_hash = _fortified_graph(
        graph,
        protection_set,
        fortification_multiplier=fortification_multiplier,
        uplifts=uplifts,
    )
    initial_flow = motif_capacity(fortified)
    if initial_flow.value < 1:
        raise CapacityEvaluationNotApplicableError(
            "The intact graph has zero S-C-L-E role-motif capacity."
        )
    initial_capacity = initial_flow.value
    initial_valid = verify_flow_certificate(fortified, initial_flow)
    pathcut = evaluate_task_path_fortification(
        graph,
        protected,
        fortification_multiplier=fortification_multiplier,
        uplifts=uplifts,
    )
    if not pathcut.optimal or pathcut.objective is None:
        raise RuntimeError("Exact fortified PathCut cross-check failed.")
    r0_margin = float(pathcut.objective)

    threshold_by_fraction = {
        value: int(rational[value] * initial_capacity) + 1
        for value in requested
    }
    aliases: dict[int, list[float]] = {}
    for value, threshold in threshold_by_fraction.items():
        aliases.setdefault(threshold, []).append(value)

    solved: dict[int, tuple[RMCDResult, str, bool]] = {}
    for threshold in sorted(aliases):
        if solver == "rmcd-exact":
            result = solve_rmcd_exact(
                fortified,
                threshold,
                time_limit=time_limit,
                mip_rel_gap=0.0,
            )
            source = "rmcd_exact_closed_bounds"
            solver_certified = _closed_bounds(result)
        else:
            pcd = solve_rmcd_pcd(
                fortified,
                threshold,
                max_oracle_calls=max_oracle_calls,
                fallback=("exact" if solver == "pcd-exact-fallback" else "none"),
                time_limit=time_limit,
            )
            result = pcd.rmcd
            source = pcd.certification_source
            solver_certified = bool(result.optimal and _closed_bounds(result))
        solved[threshold] = (result, source, solver_certified)

    points: list[CapacityThresholdPoint] = []
    for value in requested:
        threshold = threshold_by_fraction[value]
        result, source, solver_certified = solved[threshold]
        intrinsic_valid, residual_valid = _intrinsic_result_valid(
            fortified, result
        )
        independently_verified = False
        if independent_verification and result.optimal:
            independently_verified = verify_rmcd_certificate(result, fortified)
        global_certified = bool(
            solver_certified
            and intrinsic_valid
            and (not independent_verification or independently_verified)
        )
        max_remaining = threshold - 1
        residual_fraction = (
            None
            if result.residual_capacity is None
            else result.residual_capacity / initial_capacity
        )
        r0_error = (
            None
            if threshold != 1 or result.attack_cost is None
            else abs(result.attack_cost - r0_margin)
        )
        r0_valid = None if threshold != 1 else bool(
            r0_error is not None
            and r0_error <= 1e-7 * max(1.0, abs(r0_margin))
        )
        canonical_fraction = min(aliases[threshold])
        lower = result.solver.lower_bound
        upper = result.solver.upper_bound
        absolute_gap = (
            None if lower is None or upper is None else max(0.0, upper - lower)
        )
        relative_gap = (
            None
            if absolute_gap is None or upper is None or abs(upper) <= 1e-15
            else absolute_gap / abs(upper)
        )
        points.append(
            CapacityThresholdPoint(
                requested_remaining_fraction=value,
                threshold_k=threshold,
                max_remaining_capacity=max_remaining,
                realized_remaining_fraction=max_remaining / initial_capacity,
                threshold_alias_id=f"K{threshold}",
                canonical_threshold=isclose(
                    value, canonical_fraction, rel_tol=0.0, abs_tol=0.0
                ),
                solver_method=solver,
                status=result.status.value,
                feasible_attack=result.is_feasible_attack,
                optimal=result.optimal,
                unbreakable=result.status == SolveStatus.UNBREAKABLE,
                attack_set=result.attack_set,
                attack_cost=result.attack_cost,
                residual_capacity=result.residual_capacity,
                residual_fraction=residual_fraction,
                lower_bound=lower,
                upper_bound=upper,
                absolute_gap=absolute_gap,
                relative_gap=relative_gap,
                certification_source=source,
                runtime_seconds=result.solver.runtime_seconds,
                residual_flow_certificate_valid=residual_valid,
                intrinsic_result_valid=intrinsic_valid,
                independent_reference_verified=independently_verified,
                global_optimum_certified=global_certified,
                r0_pathcut_margin=r0_margin,
                r0_absolute_error=r0_error,
                r0_crosscheck_valid=r0_valid,
            )
        )
    return CapacityThresholdEvaluation(
        version=CAPACITY_EVALUATION_VERSION,
        threshold_semantics=CAPACITY_THRESHOLD_SEMANTICS,
        solver_method=solver,
        initial_capacity=initial_capacity,
        initial_flow_certificate_valid=initial_valid,
        protection_set=protected,
        fortification_multiplier=float(fortification_multiplier),
        uplift_map_sha256=uplift_hash,
        r0_pathcut_margin=r0_margin,
        points=tuple(points),
    )
