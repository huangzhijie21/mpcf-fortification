"""Certified finite-budget fortification of complete task-path families.

The attacker removes a minimum-cost equipment vertex cut that intersects every
directed S-C-L-E task path.  Fortifying node ``v`` does not make it invulnerable;
it increases its removal cost from ``a_v`` to ``a_v + delta_v``.  The defender
chooses a budget-feasible set before the attacker recomputes its cut.

Two exact formulations are exposed:

``MPCF-Exact``
    A compact mixed-integer maximum-flow formulation on the node-split task
    network.  Max-flow/min-cut duality certifies the defended dismantling
    margin.

``MPCF-CG``
    A cut-generation formulation whose master contains observed task-path
    cuts and whose separation oracle is the exact weighted PathCut solver.

``MPCF-Greedy`` is a deterministic marginal-gain approximation and is never
reported as globally optimal.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import combinations
from math import floor, fsum, gcd, isclose, isfinite, lcm
from time import perf_counter
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .model import NodeId, SolveStatus, SolverInfo, protection_cost, stable_node_key, stable_nodes
from .operational_motif import (
    OperationalMotifSystem,
    audit_path_closed_motif_system,
    build_operational_motif_system,
    solve_path_closed_motif_cut,
    task_path_endpoints,
)


MPCF_VERSION = "task-path-cut-fortification-v1"

#: One solver thread per solve.  Timing comparisons across methods and across
#: machine sizes are only meaningful when the numerical backend does not
#: oversubscribe the machine, and the multi-worker launchers rely on it.
DEFAULT_SOLVER_THREADS = 1

#: Fixed solver seed, recorded in ``metadata/environment.json``.
DEFAULT_SOLVER_SEED = 0


def _milp_options(
    *,
    time_limit: float | None,
    solver_threads: int = DEFAULT_SOLVER_THREADS,
    solver_seed: int = DEFAULT_SOLVER_SEED,
) -> dict[str, object]:
    """Options shared by every MILP call in this module."""

    options: dict[str, object] = {
        "presolve": True,
        "mip_rel_gap": 0.0,
        "threads": int(solver_threads),
        "random_seed": int(solver_seed),
    }
    if time_limit is not None:
        if time_limit <= 0.0:
            raise ValueError("time_limit must be positive when supplied.")
        options["time_limit"] = float(time_limit)
    return options


@contextmanager
def _quiet_highs_options():
    """Silence SciPy's notice that ``threads``/``random_seed`` go to HiGHS raw.

    Both are genuine HiGHS options; SciPy simply does not validate them itself,
    so the warning is expected and carries no information.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected",
            category=RuntimeWarning,
        )
        yield


@dataclass(frozen=True)
class MPCFIteration:
    """One defender/oracle exchange or one greedy selection step."""

    iteration: int
    protection_set: frozenset[NodeId]
    protection_cost: float
    selected_node: NodeId | None
    master_upper_bound: float | None
    oracle_margin: float
    oracle_cut: frozenset[NodeId]
    absolute_gap: float | None
    marginal_gain: float | None
    score: float | None
    master_runtime_seconds: float
    oracle_runtime_seconds: float


@dataclass(frozen=True)
class SolverLog:
    """Solver-level quantities retained for the revision's audit tables.

    The reviewers require more than a median runtime: every Exact run must keep
    its incumbent, global bound, final MIP gap and branch-and-bound node count
    together with an independent replay of the frozen protection set; every
    cut-generation run must keep its ``L``/``U`` interval, iteration count and
    generated-cut count.  Fields a method does not produce stay ``None``.
    """

    incumbent: float | None = None
    best_bound: float | None = None
    final_gap: float | None = None
    bb_nodes: int | None = None
    replay_kappa: float | None = None
    certificate_mode: str = "open"
    cg_L: float | None = None
    cg_U: float | None = None
    cg_iterations: int | None = None
    cg_cuts: int | None = None
    time_limit_s: float | None = None
    replay_matches: bool | None = None


@dataclass(frozen=True)
class MPCFResult:
    """A task-path fortification solution and its adaptive attack certificate."""

    method: str
    budget: float
    fortification_multiplier: float
    protection_set: frozenset[NodeId]
    protection_cost: float
    undefended_margin: float
    defended_margin: float
    adaptive_cut: frozenset[NodeId]
    solver: SolverInfo
    trace: tuple[MPCFIteration, ...]
    cut_pool: tuple[frozenset[NodeId], ...]
    uplift_by_node: tuple[tuple[NodeId, float], ...]
    path_closure_classification: str
    version: str = MPCF_VERSION
    solver_log: SolverLog = field(default_factory=SolverLog)
    _system: OperationalMotifSystem | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def status(self) -> SolveStatus:
        return self.solver.status

    @property
    def optimal(self) -> bool:
        return self.solver.optimal


@dataclass(frozen=True)
class MPCFCostQuantum:
    """Audited lattice quantum for every attainable task-path cut cost."""

    valid: bool
    quantum: float | None
    numerator: int | None
    denominator: int | None
    generator_count: int
    max_reconstruction_error: float
    reason: str

    @property
    def fraction_label(self) -> str:
        if not self.valid or self.numerator is None or self.denominator is None:
            return ""
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class MPCFObjectiveCertificate:
    """Bound ledger for one feasible MPCF incumbent.

    ``raw_bounds_closed`` records ordinary numerical bound closure.  The
    separate ``lattice_bounds_closed`` flag is true only when all attack costs
    and fortification uplifts share an audited rational quantum and no larger
    attainable objective value lies below the solver's raw upper bound.
    """

    incumbent_value: float
    raw_lower_bound: float | None
    raw_upper_bound: float | None
    raw_absolute_gap: float | None
    cost_quantum: MPCFCostQuantum
    incumbent_lattice_index: int | None
    lattice_upper_index: int | None
    lattice_upper_bound: float | None
    effective_absolute_gap: float | None
    raw_bounds_closed: bool
    lattice_bounds_closed: bool
    certified_optimal: bool
    certificate_source: str
    reason: str


@dataclass(frozen=True)
class MPCFFamilyMembershipRecord:
    """Exact or explicitly unresolved membership status for one node."""

    node: NodeId
    mandatory: bool | None
    can_appear: bool | None
    exclusion_status: SolveStatus
    inclusion_status: SolveStatus


@dataclass(frozen=True)
class MPCFOptimalFamilyResult:
    """Normalized family of maximum-margin, minimum-cost defenses.

    The primary objective is the defended PathCut margin.  Among all defenses
    attaining that margin, the family retains only those with minimum total
    protection cost.  This removes arbitrary budget-filling supersets before
    node frequencies or substitutability are interpreted.
    """

    optimum_margin: float
    minimum_protection_cost: float
    reference_set: frozenset[NodeId]
    membership_records: tuple[MPCFFamilyMembershipRecord, ...]
    mandatory_nodes: frozenset[NodeId]
    possible_nodes: frozenset[NodeId]
    interchangeable_nodes: frozenset[NodeId]
    membership_exact: bool
    enumerated_sets: tuple[frozenset[NodeId], ...]
    enumeration_complete: bool
    objective_cost_quantum: float | None
    protection_cost_quantum: float | None
    objective_membership_tolerance: float
    protection_cost_membership_tolerance: float
    lattice_membership_valid: bool
    runtime_seconds: float


@dataclass(frozen=True)
class _Arc:
    tail: object
    head: object
    base_capacity: float
    uplift_node: NodeId | None = None


@dataclass(frozen=True)
class _CompactSolve:
    protection_set: frozenset[NodeId]
    flow_value: float | None
    maximin_upper_bound: float | None
    status: SolveStatus
    optimal: bool
    scipy_status: int
    mip_gap: float | None
    message: str
    runtime_seconds: float
    bb_nodes: int | None = None


@dataclass(frozen=True)
class _MasterSolve:
    protection_set: frozenset[NodeId]
    theta: float | None
    upper_bound: float | None
    status: SolveStatus
    optimal: bool
    scipy_status: int
    mip_gap: float | None
    message: str
    runtime_seconds: float
    bb_nodes: int | None = None


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _node_count(result: Any) -> int | None:
    """Branch-and-bound node count reported by the MILP backend."""

    value = getattr(result, "mip_node_count", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _solver_status(result: Any) -> tuple[SolveStatus, bool]:
    gap = _finite_float(getattr(result, "mip_gap", None))
    if int(result.status) == 0 and getattr(result, "x", None) is not None:
        if gap is None or gap <= 1e-9:
            return SolveStatus.OPTIMAL, True
        return SolveStatus.FEASIBLE, False
    if int(result.status) == 1:
        if "time limit" in str(result.message).lower():
            return SolveStatus.TIME_LIMIT, False
        return SolveStatus.LIMIT_REACHED, False
    if int(result.status) == 2:
        return SolveStatus.INFEASIBLE, False
    if int(result.status) == 3:
        return SolveStatus.UNBOUNDED, False
    return SolveStatus.ERROR, False


def _system(graph: nx.Graph | OperationalMotifSystem) -> OperationalMotifSystem:
    return (
        graph
        if isinstance(graph, OperationalMotifSystem)
        else build_operational_motif_system(graph)
    )


def _validate_budget(value: float) -> float:
    budget = float(value)
    if not isfinite(budget) or budget < 0.0:
        raise ValueError("Fortification budget must be finite and non-negative.")
    return budget


def _excluded_protection_nodes(
    system: OperationalMotifSystem,
    protectable_nodes: Iterable[NodeId] | None,
) -> frozenset[NodeId]:
    """Translate an optional protection domain into compact-MILP exclusions."""

    if protectable_nodes is None:
        return frozenset()
    removable = frozenset(system.removable_nodes)
    allowed = frozenset(protectable_nodes)
    unknown = allowed - removable
    if unknown:
        raise ValueError(
            "protectable_nodes contains unknown/non-removable nodes: "
            f"{stable_nodes(unknown)!r}."
        )
    return removable - allowed


def _uplifts(
    system: OperationalMotifSystem,
    *,
    multiplier: float,
    uplifts: Mapping[NodeId, float] | None,
) -> dict[NodeId, float]:
    if not isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("fortification_multiplier must be finite and non-negative.")
    removable = frozenset(system.removable_nodes)
    if uplifts is not None:
        unknown = set(uplifts) - removable
        if unknown:
            raise ValueError(f"Uplifts contain unknown/non-removable nodes: {stable_nodes(unknown)!r}.")
    result = {
        node: float(
            uplifts[node]
            if uplifts is not None and node in uplifts
            else multiplier * float(system.graph.nodes[node]["attack_cost"])
        )
        for node in system.removable_nodes
    }
    bad = [node for node, value in result.items() if not isfinite(value) or value < 0.0]
    if bad:
        raise ValueError(f"Fortification uplifts must be finite and non-negative: {bad!r}.")
    return result


def mpcf_objective_cost_quantum(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    max_denominator: int = 1_000_000,
    max_common_denominator: int = 1_000_000_000,
    tolerance: float = 1e-10,
) -> MPCFCostQuantum:
    """Return a conservative rational lattice quantum for PathCut values.

    Every post-fortification node cost is a base attack cost plus either zero
    or one declared uplift.  Consequently, every feasible PathCut objective is
    an integer combination of those generators.  Rational reconstruction is
    accepted only when every generator is reproduced within ``tolerance`` and
    their common denominator stays bounded; otherwise no lattice claim is
    made.
    """

    if max_denominator < 1 or max_common_denominator < 1:
        raise ValueError("Lattice denominator limits must be positive.")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Lattice tolerance must be finite and non-negative.")
    system = _system(graph)
    delta = _uplifts(
        system,
        multiplier=fortification_multiplier,
        uplifts=uplifts,
    )
    fractions: list[Fraction] = []
    max_error = 0.0

    def declared_fraction(value: float) -> Fraction | None:
        nonlocal max_error
        reconstructed = Fraction(str(value))
        if reconstructed.denominator > max_denominator:
            return None
        error = abs(float(reconstructed) - value)
        max_error = max(max_error, error)
        if error > tolerance * max(1.0, abs(value)):
            return None
        return reconstructed

    base_fractions: dict[NodeId, Fraction] = {}
    for node in system.removable_nodes:
        value = float(system.graph.nodes[node]["attack_cost"])
        reconstructed = declared_fraction(value)
        if reconstructed is None:
            return MPCFCostQuantum(
                False,
                None,
                None,
                None,
                len(system.removable_nodes) * 2,
                max_error,
                f"base attack cost {value:.17g} exceeds the rational denominator limit",
            )
        base_fractions[node] = reconstructed
        fractions.append(reconstructed)
    if uplifts is None:
        multiplier_fraction = declared_fraction(float(fortification_multiplier))
        if multiplier_fraction is None:
            return MPCFCostQuantum(
                False,
                None,
                None,
                None,
                len(system.removable_nodes) * 2,
                max_error,
                "fortification multiplier exceeds the rational denominator limit",
            )
        uplift_fractions = {
            node: base_fractions[node] * multiplier_fraction
            for node in system.removable_nodes
        }
    else:
        uplift_fractions: dict[NodeId, Fraction] = {}
        for node in system.removable_nodes:
            value = delta[node]
            reconstructed = declared_fraction(value)
            if reconstructed is None:
                return MPCFCostQuantum(
                    False,
                    None,
                    None,
                    None,
                    len(system.removable_nodes) * 2,
                    max_error,
                    f"uplift {value:.17g} exceeds the rational denominator limit",
                )
            uplift_fractions[node] = reconstructed
    fractions.extend(value for value in uplift_fractions.values() if value > 0)
    generator_count = len(fractions)
    if not fractions:
        return MPCFCostQuantum(
            False, None, None, None, 0, 0.0, "no positive cost generators"
        )
    common_denominator = 1
    for value in fractions:
        common_denominator = lcm(common_denominator, value.denominator)
        if common_denominator > max_common_denominator:
            return MPCFCostQuantum(
                False,
                None,
                None,
                None,
                generator_count,
                max_error,
                "rational generators require an excessive common denominator",
            )
    integer_generators = [
        value.numerator * (common_denominator // value.denominator)
        for value in fractions
    ]
    common_numerator = 0
    for value in integer_generators:
        common_numerator = gcd(common_numerator, abs(value))
    if common_numerator <= 0:
        return MPCFCostQuantum(
            False,
            None,
            None,
            None,
            generator_count,
            max_error,
            "cost generators do not define a positive lattice quantum",
        )
    quantum = Fraction(common_numerator, common_denominator)
    return MPCFCostQuantum(
        True,
        float(quantum),
        quantum.numerator,
        quantum.denominator,
        generator_count,
        max_error,
        "all base attack costs and finite uplifts lie on the audited rational lattice",
    )


def mpcf_protection_cost_quantum(
    graph: nx.Graph | OperationalMotifSystem,
    *,
    max_denominator: int = 1_000_000,
    max_common_denominator: int = 1_000_000_000,
    tolerance: float = 1e-10,
) -> MPCFCostQuantum:
    """Return the audited rational lattice quantum for protection costs."""

    if max_denominator < 1 or max_common_denominator < 1:
        raise ValueError("Lattice denominator limits must be positive.")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Lattice tolerance must be finite and non-negative.")
    system = _system(graph)
    fractions: list[Fraction] = []
    max_error = 0.0
    for node in system.removable_nodes:
        value = float(system.graph.nodes[node]["protect_cost"])
        reconstructed = Fraction(str(value))
        error = abs(float(reconstructed) - value)
        max_error = max(max_error, error)
        if (
            reconstructed.denominator > max_denominator
            or error > tolerance * max(1.0, abs(value))
        ):
            return MPCFCostQuantum(
                False,
                None,
                None,
                None,
                len(system.removable_nodes),
                max_error,
                f"protection cost {value:.17g} exceeds the rational reconstruction limit",
            )
        fractions.append(reconstructed)
    if not fractions:
        return MPCFCostQuantum(
            False, None, None, None, 0, 0.0, "no protection-cost generators"
        )
    common_denominator = 1
    for value in fractions:
        common_denominator = lcm(common_denominator, value.denominator)
        if common_denominator > max_common_denominator:
            return MPCFCostQuantum(
                False,
                None,
                None,
                None,
                len(fractions),
                max_error,
                "protection costs require an excessive common denominator",
            )
    integer_generators = [
        value.numerator * (common_denominator // value.denominator)
        for value in fractions
    ]
    common_numerator = 0
    for value in integer_generators:
        common_numerator = gcd(common_numerator, abs(value))
    if common_numerator <= 0:
        return MPCFCostQuantum(
            False,
            None,
            None,
            None,
            len(fractions),
            max_error,
            "protection costs do not define a positive lattice quantum",
        )
    quantum = Fraction(common_numerator, common_denominator)
    return MPCFCostQuantum(
        True,
        float(quantum),
        quantum.numerator,
        quantum.denominator,
        len(fractions),
        max_error,
        "all protection costs lie on the audited rational lattice",
    )


def certify_mpcf_objective_bounds(
    incumbent_value: float,
    raw_lower_bound: float | None,
    raw_upper_bound: float | None,
    cost_quantum: MPCFCostQuantum,
    *,
    tolerance: float = 1e-7,
) -> MPCFObjectiveCertificate:
    """Certify a maximin MPCF objective from raw and integer-lattice bounds."""

    incumbent = float(incumbent_value)
    lower = _finite_float(raw_lower_bound)
    upper = _finite_float(raw_upper_bound)
    if not isfinite(incumbent) or incumbent < 0.0:
        raise ValueError("MPCF incumbent must be finite and non-negative.")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Certificate tolerance must be finite and non-negative.")
    scale = max(1.0, abs(incumbent), abs(lower or 0.0), abs(upper or 0.0))
    absolute_tolerance = tolerance * scale
    raw_gap = None if upper is None else max(0.0, upper - incumbent)
    lower_consistent = lower is None or lower <= incumbent + absolute_tolerance
    raw_closed = (
        upper is not None
        and lower_consistent
        and upper >= incumbent - absolute_tolerance
        and upper - incumbent <= absolute_tolerance
    )
    incumbent_index: int | None = None
    lattice_upper_index: int | None = None
    lattice_upper: float | None = None
    lattice_closed = False
    reason = "raw solver bounds remain open"
    if not lower_consistent:
        reason = "raw lower bound is above the verified feasible incumbent"
    elif upper is not None and upper < incumbent - absolute_tolerance:
        reason = "raw upper bound is below the verified feasible incumbent"
    elif cost_quantum.valid and cost_quantum.quantum is not None and upper is not None:
        quantum = cost_quantum.quantum
        lattice_tolerance = min(absolute_tolerance, 0.25 * quantum)
        candidate_index = int(round(incumbent / quantum))
        candidate_value = candidate_index * quantum
        if abs(candidate_value - incumbent) <= lattice_tolerance:
            incumbent_index = candidate_index
            lattice_upper_index = int(floor((upper + lattice_tolerance) / quantum))
            lattice_upper = lattice_upper_index * quantum
            lattice_closed = (
                lower_consistent
                and upper >= incumbent - absolute_tolerance
                and lattice_upper_index <= incumbent_index
            )
            reason = (
                "no higher lattice-attainable objective lies below the raw upper bound"
                if lattice_closed
                else "at least one higher lattice-attainable objective remains below the raw upper bound"
            )
        else:
            reason = "verified incumbent is not on the audited objective lattice"
    elif not cost_quantum.valid:
        reason = "objective lattice unavailable: " + cost_quantum.reason
    certified = raw_closed or lattice_closed
    source = (
        "RAW_BOUND_CLOSURE"
        if raw_closed
        else "OBJECTIVE_LATTICE_CLOSURE"
        if lattice_closed
        else "OPEN_GAP"
    )
    effective_gap = (
        0.0
        if certified
        else None
        if lattice_upper is None
        else max(0.0, lattice_upper - incumbent)
    )
    return MPCFObjectiveCertificate(
        incumbent_value=incumbent,
        raw_lower_bound=lower,
        raw_upper_bound=upper,
        raw_absolute_gap=raw_gap,
        cost_quantum=cost_quantum,
        incumbent_lattice_index=incumbent_index,
        lattice_upper_index=lattice_upper_index,
        lattice_upper_bound=lattice_upper,
        effective_absolute_gap=effective_gap,
        raw_bounds_closed=raw_closed,
        lattice_bounds_closed=lattice_closed,
        certified_optimal=certified,
        certificate_source=source,
        reason=reason,
    )


def mpcf_objective_certificate(
    result: MPCFResult,
    graph: nx.Graph | OperationalMotifSystem | None = None,
    *,
    tolerance: float = 1e-7,
) -> MPCFObjectiveCertificate:
    """Build the auditable objective certificate for one MPCF result."""

    system = _system(graph) if graph is not None else result._system
    if system is None:
        raise ValueError("A graph is required to derive the MPCF objective lattice.")
    recorded_uplifts = dict(result.uplift_by_node)
    default_uplifts = {
        node: result.fortification_multiplier
        * float(system.graph.nodes[node]["attack_cost"])
        for node in system.removable_nodes
    }
    uses_default_uplifts = all(
        isclose(
            recorded_uplifts[node],
            default_uplifts[node],
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, abs(default_uplifts[node])),
        )
        for node in system.removable_nodes
    )
    quantum = mpcf_objective_cost_quantum(
        system,
        fortification_multiplier=result.fortification_multiplier,
        uplifts=None if uses_default_uplifts else recorded_uplifts,
    )
    return certify_mpcf_objective_bounds(
        result.defended_margin,
        result.solver.lower_bound,
        result.solver.upper_bound,
        quantum,
        tolerance=tolerance,
    )


def fortified_attack_costs(
    graph: nx.Graph | OperationalMotifSystem,
    protection_set: Iterable[NodeId],
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
) -> dict[NodeId, float]:
    """Return finite post-fortification attack costs for one protection set."""

    system = _system(graph)
    protected = frozenset(protection_set)
    unknown = protected - frozenset(system.removable_nodes)
    if unknown:
        raise ValueError(f"Unknown/non-removable protection nodes: {stable_nodes(unknown)!r}.")
    delta = _uplifts(
        system, multiplier=fortification_multiplier, uplifts=uplifts
    )
    return {
        node: float(system.graph.nodes[node]["attack_cost"])
        + (delta[node] if node in protected else 0.0)
        for node in system.removable_nodes
    }


def evaluate_task_path_fortification(
    graph: nx.Graph | OperationalMotifSystem,
    protection_set: Iterable[NodeId],
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
):
    """Recompute the exact adaptive PathCut after finite fortification."""

    system = _system(graph)
    costs = fortified_attack_costs(
        system,
        protection_set,
        fortification_multiplier=fortification_multiplier,
        uplifts=uplifts,
    )
    return solve_path_closed_motif_cut(system, costs=costs)


def _require_path_closed(system: OperationalMotifSystem) -> str:
    audit = audit_path_closed_motif_system(system)
    if not audit.path_closed:
        raise ValueError(
            "MPCF requires the complete directed task-path family: " + audit.reason
        )
    return audit.classification


def _base_costs(system: OperationalMotifSystem) -> dict[NodeId, float]:
    return {
        node: float(system.graph.nodes[node]["attack_cost"])
        for node in system.removable_nodes
    }


def _build_node_split_arcs(
    system: OperationalMotifSystem,
    delta: Mapping[NodeId, float],
) -> tuple[object, object, tuple[_Arc, ...], float]:
    base = _base_costs(system)
    maximum_node_cut = fsum(base[node] + delta[node] for node in system.removable_nodes)
    positive = [base[node] + delta[node] for node in system.removable_nodes]
    # ``C_INF`` is the finite capacity placed on every non-removable arc.  It is
    # named for what it is -- a finite stand-in for infinity -- rather than ``M``,
    # which collided with the fortification multiplier in the manuscript.
    c_inf = fsum((maximum_node_cut, min(positive))) if positive else 1.0
    if not isfinite(c_inf) or c_inf <= maximum_node_cut:
        raise ValueError("Node costs cannot form a finite PathCut C_INF; rescale costs.")
    source = ("__mpcf_source__",)
    sink = ("__mpcf_sink__",)
    task_sources, task_targets = task_path_endpoints(system)
    task_source_set = frozenset(task_sources)
    task_target_set = frozenset(task_targets)
    removable = frozenset(system.removable_nodes)
    arcs: list[_Arc] = []
    for node in stable_nodes(system.graph.nodes):
        node_in = ("__mpcf_in__", node)
        node_out = ("__mpcf_out__", node)
        arcs.append(
            _Arc(
                node_in,
                node_out,
                base[node] if node in removable else c_inf,
                node if node in removable else None,
            )
        )
        if node in task_source_set:
            arcs.append(_Arc(source, node_in, c_inf))
        if node in task_target_set:
            arcs.append(_Arc(node_out, sink, c_inf))
    for left, right in sorted(
        system.graph.edges,
        key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1])),
    ):
        arcs.append(
            _Arc(("__mpcf_out__", left), ("__mpcf_in__", right), c_inf)
        )
    return source, sink, tuple(arcs), c_inf


def _compact_milp(
    system: OperationalMotifSystem,
    budget: float,
    delta: Mapping[NodeId, float],
    *,
    time_limit: float | None,
    margin_floor: float | None = None,
    minimize_protection_cost: bool = False,
    required_protection: Iterable[NodeId] = (),
    excluded_protection: Iterable[NodeId] = (),
    no_good_sets: Sequence[frozenset[NodeId]] = (),
    protection_cost_ceiling: float | None = None,
) -> _CompactSolve:
    nodes = system.removable_nodes
    node_index = {node: index for index, node in enumerate(nodes)}
    removable = frozenset(nodes)
    required = frozenset(required_protection)
    excluded = frozenset(excluded_protection)
    if required & excluded:
        raise ValueError("required_protection and excluded_protection must be disjoint.")
    unknown = (required | excluded) - removable
    if unknown:
        raise ValueError(f"Unknown forced protection nodes: {stable_nodes(unknown)!r}.")
    normalized_no_good: list[frozenset[NodeId]] = []
    for candidate in no_good_sets:
        frozen = frozenset(candidate)
        unknown = frozen - removable
        if unknown:
            raise ValueError(f"No-good set contains unknown nodes: {stable_nodes(unknown)!r}.")
        normalized_no_good.append(frozen)
    if protection_cost_ceiling is not None:
        protection_cost_ceiling = float(protection_cost_ceiling)
        if not isfinite(protection_cost_ceiling) or protection_cost_ceiling < 0.0:
            raise ValueError("protection_cost_ceiling must be finite and non-negative.")
    source, sink, arcs, c_inf = _build_node_split_arcs(system, delta)
    arc_offset = len(nodes)
    variable_count = len(nodes) + len(arcs)
    objective = np.zeros(variable_count, dtype=float)
    if minimize_protection_cost:
        for node in nodes:
            objective[node_index[node]] = float(system.graph.nodes[node]["protect_cost"])
    else:
        for index, arc in enumerate(arcs):
            if arc.tail == source:
                objective[arc_offset + index] = -1.0

    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[: len(nodes)] = 1
    lower = np.zeros(variable_count, dtype=float)
    upper = np.empty(variable_count, dtype=float)
    upper[: len(nodes)] = 1.0
    for node in required:
        lower[node_index[node]] = 1.0
    for node in excluded:
        upper[node_index[node]] = 0.0
    for index, arc in enumerate(arcs):
        upper[arc_offset + index] = (
            arc.base_capacity + delta[arc.uplift_node]
            if arc.uplift_node is not None
            else c_inf
        )

    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    rows.append(
        {
            node_index[node]: float(system.graph.nodes[node]["protect_cost"])
            for node in nodes
        }
    )
    lbs.append(-np.inf)
    ubs.append(budget)

    if protection_cost_ceiling is not None:
        rows.append(
            {
                node_index[node]: float(system.graph.nodes[node]["protect_cost"])
                for node in nodes
            }
        )
        lbs.append(-np.inf)
        ubs.append(protection_cost_ceiling)

    for forbidden in normalized_no_good:
        # Exclude exactly one binary vector:
        # sum_{v in S} x_v - sum_{v not in S} x_v <= |S| - 1.
        rows.append(
            {
                node_index[node]: (1.0 if node in forbidden else -1.0)
                for node in nodes
            }
        )
        lbs.append(-np.inf)
        ubs.append(float(len(forbidden) - 1))

    for arc_index, arc in enumerate(arcs):
        if arc.uplift_node is None:
            continue
        rows.append(
            {
                arc_offset + arc_index: 1.0,
                node_index[arc.uplift_node]: -delta[arc.uplift_node],
            }
        )
        lbs.append(-np.inf)
        ubs.append(arc.base_capacity)

    auxiliary_nodes = {source, sink}
    for arc in arcs:
        auxiliary_nodes.add(arc.tail)
        auxiliary_nodes.add(arc.head)
    for vertex in sorted(auxiliary_nodes - {source, sink}, key=repr):
        row: dict[int, float] = {}
        for arc_index, arc in enumerate(arcs):
            variable = arc_offset + arc_index
            if arc.head == vertex:
                row[variable] = row.get(variable, 0.0) + 1.0
            if arc.tail == vertex:
                row[variable] = row.get(variable, 0.0) - 1.0
        rows.append(row)
        lbs.append(0.0)
        ubs.append(0.0)

    if margin_floor is not None:
        row = {
            arc_offset + index: 1.0
            for index, arc in enumerate(arcs)
            if arc.tail == source
        }
        rows.append(row)
        lbs.append(float(margin_floor))
        ubs.append(np.inf)

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
    options = _milp_options(time_limit=time_limit)
    started = perf_counter()
    with _quiet_highs_options():
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
    protected = frozenset(
        node
        for node in nodes
        if vector is not None and float(vector[node_index[node]]) >= 0.5
    )
    flow_value: float | None = None
    if vector is not None:
        flow_value = float(
            sum(
                float(vector[arc_offset + index])
                for index, arc in enumerate(arcs)
                if arc.tail == source
            )
        )
    dual = _finite_float(getattr(result, "mip_dual_bound", None))
    maximin_upper = None
    if not minimize_protection_cost:
        maximin_upper = -dual if dual is not None else None
    return _CompactSolve(
        protected,
        flow_value,
        maximin_upper,
        status,
        optimal,
        int(result.status),
        _finite_float(getattr(result, "mip_gap", None)),
        str(result.message),
        runtime,
        _node_count(result),
    )


def _result(
    *,
    method: str,
    system: OperationalMotifSystem,
    budget: float,
    multiplier: float,
    delta: Mapping[NodeId, float],
    protection: frozenset[NodeId],
    baseline_margin: float,
    solver: SolverInfo,
    trace: Sequence[MPCFIteration],
    cut_pool: Sequence[frozenset[NodeId]],
    classification: str,
    solver_log: SolverLog | None = None,
) -> MPCFResult:
    oracle = evaluate_task_path_fortification(
        system,
        protection,
        fortification_multiplier=multiplier,
        uplifts=delta,
    )
    if not oracle.optimal or oracle.objective is None:
        raise RuntimeError("Adaptive PathCut oracle did not return an optimal finite cut.")
    replay = float(oracle.objective)
    if solver_log is None:
        solver_log = SolverLog(replay_kappa=replay)
    else:
        tolerance = 1e-7 * max(1.0, abs(replay), abs(solver.objective or 0.0))
        solver_log = replace(
            solver_log,
            replay_kappa=replay,
            replay_matches=(
                solver.objective is None
                or abs(replay - float(solver.objective)) <= tolerance
            ),
        )
    return MPCFResult(
        method=method,
        budget=budget,
        fortification_multiplier=multiplier,
        protection_set=protection,
        protection_cost=protection_cost(system.graph, protection),
        undefended_margin=baseline_margin,
        defended_margin=replay,
        adaptive_cut=oracle.selected_set,
        solver=solver,
        trace=tuple(trace),
        cut_pool=tuple(cut_pool),
        uplift_by_node=tuple((node, delta[node]) for node in system.removable_nodes),
        path_closure_classification=classification,
        solver_log=solver_log,
        _system=system,
    )


def solve_mpcf_exact(
    graph: nx.Graph | OperationalMotifSystem,
    budget: float,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    time_limit: float | None = None,
    lexicographic_tie_break: bool = True,
    protectable_nodes: Iterable[NodeId] | None = None,
) -> MPCFResult:
    """Solve finite task-path fortification by compact max-flow MILP."""

    system = _system(graph)
    classification = _require_path_closed(system)
    budget = _validate_budget(budget)
    delta = _uplifts(
        system, multiplier=fortification_multiplier, uplifts=uplifts
    )
    excluded = _excluded_protection_nodes(system, protectable_nodes)
    baseline = solve_path_closed_motif_cut(system)
    if not baseline.optimal or baseline.objective is None:
        raise RuntimeError("MPCF requires a finite optimal undefended PathCut.")
    started = perf_counter()
    primary = _compact_milp(
        system,
        budget,
        delta,
        time_limit=time_limit,
        excluded_protection=excluded,
    )
    protection = primary.protection_set
    message = primary.message
    extra_runtime = 0.0
    if primary.optimal and primary.flow_value is not None and lexicographic_tie_break:
        tolerance = 1e-8 * max(1.0, abs(primary.flow_value))
        # ``time_limit`` bounds this *method*, not each internal solve.  The
        # lexicographic tie-break therefore gets whatever is left of the budget
        # rather than a second full limit, so a reported Exact runtime can never
        # exceed the unified limit the experiment declares.
        secondary_limit = time_limit
        if secondary_limit is not None:
            secondary_limit = max(0.0, float(time_limit) - (perf_counter() - started))
        secondary = None
        if secondary_limit is None or secondary_limit > 0.0:
            secondary = _compact_milp(
                system,
                budget,
                delta,
                time_limit=secondary_limit,
                margin_floor=primary.flow_value - tolerance,
                minimize_protection_cost=True,
                excluded_protection=excluded,
            )
        if secondary is None:
            message += " Lexicographic tie-break skipped: unified time limit exhausted."
        else:
            extra_runtime = secondary.runtime_seconds
            if secondary.optimal:
                protection = secondary.protection_set
                message += " Lexicographic minimum-protection-cost tie-break solved."
            else:
                message += " Lexicographic tie-break incomplete; primary incumbent retained."

    oracle = evaluate_task_path_fortification(
        system,
        protection,
        fortification_multiplier=fortification_multiplier,
        uplifts=delta,
    )
    if not oracle.optimal or oracle.objective is None:
        raise RuntimeError("Adaptive PathCut verification failed after compact MILP.")
    tolerance = 1e-7 * max(
        1.0, abs(float(oracle.objective)), abs(primary.flow_value or 0.0)
    )
    bound_consistent = (
        primary.flow_value is not None
        and abs(float(oracle.objective) - primary.flow_value) <= tolerance
    )
    optimal = primary.optimal and bound_consistent
    status = SolveStatus.OPTIMAL if optimal else primary.status
    upper = (
        float(oracle.objective)
        if optimal
        else primary.maximin_upper_bound
    )
    solver = SolverInfo(
        status=status,
        optimal=optimal,
        scipy_status=primary.scipy_status,
        message=(
            message
            + (
                " Compact max-flow and adaptive PathCut values agree."
                if bound_consistent
                else " Compact max-flow and adaptive PathCut values do not close."
            )
        ),
        objective=float(oracle.objective),
        lower_bound=float(oracle.objective),
        upper_bound=upper,
        mip_gap=primary.mip_gap,
        runtime_seconds=perf_counter() - started,
    )
    return _result(
        method="MPCF-Exact",
        system=system,
        budget=budget,
        multiplier=fortification_multiplier,
        delta=delta,
        protection=protection,
        baseline_margin=float(baseline.objective),
        solver=solver,
        trace=(),
        cut_pool=(baseline.selected_set, oracle.selected_set),
        classification=classification,
        solver_log=SolverLog(
            incumbent=primary.flow_value,
            best_bound=primary.maximin_upper_bound,
            final_gap=primary.mip_gap,
            bb_nodes=primary.bb_nodes,
            certificate_mode=(
                "solver_closed" if optimal else "open"
            ),
            time_limit_s=time_limit,
        ),
    )


def _solve_cut_master(
    system: OperationalMotifSystem,
    budget: float,
    delta: Mapping[NodeId, float],
    cuts: Sequence[frozenset[NodeId]],
    *,
    time_limit: float | None,
) -> _MasterSolve:
    nodes = system.removable_nodes
    index = {node: position for position, node in enumerate(nodes)}
    theta_index = len(nodes)
    count = len(nodes) + 1
    objective = np.zeros(count, dtype=float)
    objective[theta_index] = -1.0
    integrality = np.zeros(count, dtype=np.uint8)
    integrality[: len(nodes)] = 1
    lower = np.zeros(count, dtype=float)
    maximum = fsum(
        float(system.graph.nodes[node]["attack_cost"]) + delta[node]
        for node in nodes
    )
    upper = np.ones(count, dtype=float)
    upper[theta_index] = maximum
    rows: list[dict[int, float]] = [
        {
            index[node]: float(system.graph.nodes[node]["protect_cost"])
            for node in nodes
        }
    ]
    lbs = [-np.inf]
    ubs = [budget]
    for cut in cuts:
        row = {theta_index: 1.0}
        for node in cut:
            row[index[node]] = -delta[node]
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(
            float(sum(float(system.graph.nodes[node]["attack_cost"]) for node in cut))
        )
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_number, row in enumerate(rows):
        for column, value in row.items():
            row_indices.append(row_number)
            column_indices.append(column)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)), shape=(len(rows), count)
    ).tocsr()
    options = _milp_options(time_limit=time_limit)
    started = perf_counter()
    with _quiet_highs_options():
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
    protected = frozenset(
        node
        for node in nodes
        if vector is not None and float(vector[index[node]]) >= 0.5
    )
    theta = float(vector[theta_index]) if vector is not None else None
    dual = _finite_float(getattr(result, "mip_dual_bound", None))
    return _MasterSolve(
        protected,
        theta,
        -dual if dual is not None else None,
        status,
        optimal,
        int(result.status),
        _finite_float(getattr(result, "mip_gap", None)),
        str(result.message),
        runtime,
        _node_count(result),
    )


def solve_mpcf_cut_generation(
    graph: nx.Graph | OperationalMotifSystem,
    budget: float,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    time_limit_per_solve: float | None = None,
    overall_time_limit: float | None = None,
    max_iterations: int = 10_000,
    tolerance: float = 1e-7,
    progress: Callable[[str], None] | None = None,
) -> MPCFResult:
    """Solve MPCF exactly by master/separation cut generation.

    ``time_limit_per_solve`` bounds one master solve; ``overall_time_limit``
    bounds the whole method, master solves *and* separation oracles together.
    Both are supplied by the experiment harness, so the reported CG runtime
    respects the same unified limit as Exact and Greedy: cut generation is an
    iterative algorithm and a per-solve limit alone would let its total grow
    with the number of cuts it happens to need.
    """

    system = _system(graph)
    classification = _require_path_closed(system)
    budget = _validate_budget(budget)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    if overall_time_limit is not None and overall_time_limit <= 0.0:
        raise ValueError("overall_time_limit must be positive when supplied.")
    delta = _uplifts(
        system, multiplier=fortification_multiplier, uplifts=uplifts
    )
    baseline = solve_path_closed_motif_cut(system)
    if not baseline.optimal or baseline.objective is None:
        raise RuntimeError("MPCF-CG requires a finite optimal undefended PathCut.")
    cuts: list[frozenset[NodeId]] = [baseline.selected_set]
    trace: list[MPCFIteration] = []
    incumbent = frozenset()
    incumbent_margin = float(baseline.objective)
    incumbent_cut = baseline.selected_set
    started = perf_counter()
    last_master: _MasterSolve | None = None
    previous_upper = float("inf")
    exhausted_limit = False
    for iteration in range(1, max_iterations + 1):
        solve_limit = time_limit_per_solve
        if overall_time_limit is not None:
            remaining = float(overall_time_limit) - (perf_counter() - started)
            if remaining <= 0.0:
                exhausted_limit = True
                break
            solve_limit = remaining if solve_limit is None else min(solve_limit, remaining)
        master = _solve_cut_master(
            system,
            budget,
            delta,
            cuts,
            time_limit=solve_limit,
        )
        last_master = master
        if not master.optimal or master.theta is None:
            break
        if isfinite(previous_upper):
            monotonicity_scale = max(
                1.0, abs(master.theta), abs(previous_upper)
            )
            monotonicity_tolerance = tolerance * monotonicity_scale
            if master.theta > previous_upper + monotonicity_tolerance:
                raise RuntimeError(
                    "MPCF-CG restricted-master bound increased beyond the "
                    "scale-aware numerical tolerance: "
                    f"previous={previous_upper:.17g}, "
                    f"current={master.theta:.17g}, "
                    f"tolerance={monotonicity_tolerance:.17g}."
                )
        # Added cuts can only tighten the mathematical master problem.  Keep
        # the strongest historical bound when HiGHS reports harmless jitter.
        previous_upper = min(previous_upper, master.theta)
        # The separation oracle is an indivisible exact max-flow call, so it
        # cannot be interrupted part-way.  Checking the deadline here keeps the
        # overrun to at most one oracle call instead of one per iteration.
        if overall_time_limit is not None:
            if float(overall_time_limit) - (perf_counter() - started) <= 0.0:
                exhausted_limit = True
                break
        oracle_started = perf_counter()
        oracle = evaluate_task_path_fortification(
            system,
            master.protection_set,
            fortification_multiplier=fortification_multiplier,
            uplifts=delta,
        )
        oracle_runtime = perf_counter() - oracle_started
        if not oracle.optimal or oracle.objective is None:
            raise RuntimeError("MPCF-CG PathCut separation oracle failed.")
        incumbent = master.protection_set
        incumbent_margin = float(oracle.objective)
        incumbent_cut = oracle.selected_set
        gap = max(0.0, master.theta - incumbent_margin)
        trace.append(
            MPCFIteration(
                iteration,
                incumbent,
                protection_cost(system.graph, incumbent),
                None,
                master.theta,
                incumbent_margin,
                incumbent_cut,
                gap,
                None,
                None,
                master.runtime_seconds,
                oracle_runtime,
            )
        )
        if progress is not None:
            progress(
                f"MPCF-CG iteration {iteration}: upper={master.theta:.10g}, "
                f"oracle={incumbent_margin:.10g}, gap={gap:.3g}, cuts={len(cuts)}"
            )
        scale = max(1.0, abs(master.theta), abs(incumbent_margin))
        if gap <= tolerance * scale:
            solver = SolverInfo(
                status=SolveStatus.OPTIMAL,
                optimal=True,
                scipy_status=master.scipy_status,
                message="Restricted-master bound closed by exact PathCut separation.",
                objective=incumbent_margin,
                lower_bound=incumbent_margin,
                upper_bound=incumbent_margin,
                mip_gap=0.0,
                runtime_seconds=perf_counter() - started,
            )
            return _result(
                method="MPCF-CG",
                system=system,
                budget=budget,
                multiplier=fortification_multiplier,
                delta=delta,
                protection=incumbent,
                baseline_margin=float(baseline.objective),
                solver=solver,
                trace=trace,
                cut_pool=cuts,
                classification=classification,
                solver_log=SolverLog(
                    incumbent=incumbent_margin,
                    best_bound=master.theta,
                    final_gap=gap,
                    bb_nodes=master.bb_nodes,
                    certificate_mode="cg_closed",
                    cg_L=incumbent_margin,
                    cg_U=master.theta,
                    cg_iterations=len(trace),
                    cg_cuts=len(cuts),
                    time_limit_s=(
                        overall_time_limit
                        if overall_time_limit is not None
                        else time_limit_per_solve
                    ),
                ),
            )
        if incumbent_cut in cuts:
            raise RuntimeError(
                "MPCF-CG separation returned a duplicate cut before closing the gap."
            )
        cuts.append(incumbent_cut)

    status = last_master.status if last_master is not None else SolveStatus.ERROR
    if exhausted_limit:
        status = SolveStatus.TIME_LIMIT
    upper_candidates = [
        value
        for value in (
            last_master.upper_bound if last_master is not None else None,
            previous_upper if isfinite(previous_upper) else None,
        )
        if value is not None
    ]
    upper = min(upper_candidates) if upper_candidates else None
    solver = SolverInfo(
        status=status if status != SolveStatus.OPTIMAL else SolveStatus.LIMIT_REACHED,
        optimal=False,
        scipy_status=last_master.scipy_status if last_master else None,
        message=(
            "MPCF-CG stopped before the restricted-master and oracle bounds closed. "
            + (
                "Unified method time limit reached."
                if exhausted_limit
                else (last_master.message if last_master else "No master solution.")
            )
        ),
        objective=incumbent_margin,
        lower_bound=incumbent_margin,
        upper_bound=upper,
        mip_gap=last_master.mip_gap if last_master else None,
        runtime_seconds=perf_counter() - started,
    )
    return _result(
        method="MPCF-CG",
        system=system,
        budget=budget,
        multiplier=fortification_multiplier,
        delta=delta,
        protection=incumbent,
        baseline_margin=float(baseline.objective),
        solver=solver,
        trace=trace,
        cut_pool=cuts,
        classification=classification,
        solver_log=SolverLog(
            incumbent=incumbent_margin,
            best_bound=upper,
            final_gap=(upper - incumbent_margin) if upper is not None else None,
            bb_nodes=last_master.bb_nodes if last_master is not None else None,
            certificate_mode="open",
            cg_L=incumbent_margin,
            cg_U=upper,
            cg_iterations=len(trace),
            cg_cuts=len(cuts),
            time_limit_s=(
                overall_time_limit
                if overall_time_limit is not None
                else time_limit_per_solve
            ),
        ),
    )


def solve_mpcf_greedy(
    graph: nx.Graph | OperationalMotifSystem,
    budget: float,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    progress: Callable[[str], None] | None = None,
    protectable_nodes: Iterable[NodeId] | None = None,
    time_limit: float | None = None,
) -> MPCFResult:
    """Select nodes by exact adaptive-margin gain per protection cost.

    ``time_limit`` applies the same wall-clock rule the two exact solvers use,
    so that a scaling comparison can hold the instance set *and* the time rule
    fixed across all three MPCF methods.  On expiry the method returns the
    protection set built so far, reported as a time-limited incumbent rather
    than as an optimal solution.
    """

    system = _system(graph)
    classification = _require_path_closed(system)
    budget = _validate_budget(budget)
    if time_limit is not None and time_limit <= 0.0:
        raise ValueError("time_limit must be positive when supplied.")
    delta = _uplifts(
        system, multiplier=fortification_multiplier, uplifts=uplifts
    )
    excluded = _excluded_protection_nodes(system, protectable_nodes)
    allowed = frozenset(system.removable_nodes) - excluded
    baseline = solve_path_closed_motif_cut(system)
    if not baseline.optimal or baseline.objective is None:
        raise RuntimeError("MPCF-Greedy requires a finite optimal PathCut.")
    started = perf_counter()
    timed_out = False
    selected: set[NodeId] = set()
    current_margin = float(baseline.objective)
    current_cut = baseline.selected_set
    trace: list[MPCFIteration] = []
    while True:
        if time_limit is not None and (perf_counter() - started) >= time_limit:
            timed_out = True
            break
        spent = protection_cost(system.graph, selected)
        affordable = [
            node
            for node in allowed
            if node not in selected
            and spent + float(system.graph.nodes[node]["protect_cost"]) <= budget + 1e-9
        ]
        if not affordable:
            break
        evaluated: list[tuple[float, float, int, tuple[str, str], NodeId, Any]] = []
        for node in affordable:
            if time_limit is not None and (perf_counter() - started) >= time_limit:
                timed_out = True
                break
            oracle = evaluate_task_path_fortification(
                system,
                (*selected, node),
                fortification_multiplier=fortification_multiplier,
                uplifts=delta,
            )
            if not oracle.optimal or oracle.objective is None:
                raise RuntimeError("MPCF-Greedy candidate PathCut failed.")
            gain = float(oracle.objective) - current_margin
            cost = float(system.graph.nodes[node]["protect_cost"])
            evaluated.append(
                (
                    gain / cost,
                    gain,
                    int(node in current_cut),
                    stable_node_key(node),
                    node,
                    oracle,
                )
            )
        if timed_out or not evaluated:
            break
        score, gain, _in_cut, _key, node, oracle = min(
            evaluated,
            key=lambda item: (-item[0], -item[1], -item[2], item[3]),
        )
        selected.add(node)
        current_margin = float(oracle.objective)
        current_cut = oracle.selected_set
        trace.append(
            MPCFIteration(
                len(trace) + 1,
                frozenset(selected),
                protection_cost(system.graph, selected),
                node,
                None,
                current_margin,
                current_cut,
                None,
                gain,
                score,
                0.0,
                oracle.runtime_seconds,
            )
        )
        if progress is not None:
            progress(
                f"MPCF-Greedy step {len(trace)}: node={node!r}, "
                f"margin={current_margin:.10g}, gain={gain:.3g}"
            )
    runtime = perf_counter() - started
    solver = SolverInfo(
        status=SolveStatus.TIME_LIMIT if timed_out else SolveStatus.FEASIBLE,
        optimal=False,
        message=(
            "Deterministic exact-marginal greedy heuristic; unified time limit "
            "reached, incumbent protection set retained; no global certificate."
            if timed_out
            else "Deterministic exact-marginal greedy heuristic; no global certificate."
        ),
        objective=current_margin,
        lower_bound=current_margin,
        upper_bound=None,
        mip_gap=None,
        runtime_seconds=runtime,
    )
    return _result(
        method="MPCF-Greedy",
        system=system,
        budget=budget,
        multiplier=fortification_multiplier,
        delta=delta,
        protection=frozenset(selected),
        baseline_margin=float(baseline.objective),
        solver=solver,
        trace=trace,
        cut_pool=tuple(dict.fromkeys((baseline.selected_set, *(row.oracle_cut for row in trace)))),
        classification=classification,
        solver_log=SolverLog(
            incumbent=current_margin,
            best_bound=None,
            final_gap=None,
            bb_nodes=None,
            certificate_mode="open",
            cg_iterations=len(trace),
            time_limit_s=time_limit,
        ),
    )


def analyze_mpcf_optimal_family(
    graph: nx.Graph | OperationalMotifSystem,
    budget: float,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    enumerate_limit: int = 100,
    time_limit_per_solve: float | None = None,
    tolerance: float = 1e-8,
    progress: Callable[[str], None] | None = None,
) -> MPCFOptimalFamilyResult:
    """Certify and sample the normalized family of optimal MPCF defenses.

    Maximum defended margin is the primary objective.  Minimum protection cost
    is a lexicographic normalization, not a new performance objective.  Node
    membership is certified with forced-in/forced-out solves; bounded no-good
    enumeration is used only for frequencies and exchange diagnostics.
    """

    if enumerate_limit < 0:
        raise ValueError("enumerate_limit must be non-negative.")
    if not 0.0 < tolerance < 0.5:
        raise ValueError("tolerance must satisfy 0 < tolerance < 0.5.")
    system = _system(graph)
    _require_path_closed(system)
    budget = _validate_budget(budget)
    delta = _uplifts(
        system, multiplier=fortification_multiplier, uplifts=uplifts
    )
    started = perf_counter()
    reference = solve_mpcf_exact(
        system,
        budget,
        fortification_multiplier=fortification_multiplier,
        uplifts=delta,
        time_limit=time_limit_per_solve,
        lexicographic_tie_break=False,
    )
    if not reference.optimal:
        raise RuntimeError("Optimal-family analysis requires a certified MPCF optimum.")
    optimum = float(reference.defended_margin)
    objective_quantum = mpcf_objective_cost_quantum(
        system,
        fortification_multiplier=fortification_multiplier,
        uplifts=uplifts,
    )
    requested_objective_tolerance = tolerance * max(1.0, abs(optimum))
    objective_tolerance = (
        min(requested_objective_tolerance, 0.25 * objective_quantum.quantum)
        if objective_quantum.valid and objective_quantum.quantum is not None
        else requested_objective_tolerance
    )
    margin_floor = optimum - objective_tolerance

    secondary = _compact_milp(
        system,
        budget,
        delta,
        time_limit=time_limit_per_solve,
        margin_floor=margin_floor,
        minimize_protection_cost=True,
    )
    if not secondary.optimal:
        raise RuntimeError("Minimum-cost normalization of the optimal family failed.")
    normalized_reference = secondary.protection_set
    minimum_cost = protection_cost(system.graph, normalized_reference)
    protection_quantum = mpcf_protection_cost_quantum(system)
    requested_cost_tolerance = tolerance * max(1.0, abs(minimum_cost))
    cost_tolerance = (
        min(requested_cost_tolerance, 0.25 * protection_quantum.quantum)
        if protection_quantum.valid and protection_quantum.quantum is not None
        else requested_cost_tolerance
    )
    cost_ceiling = minimum_cost + cost_tolerance
    objective_on_lattice = (
        objective_quantum.valid
        and objective_quantum.quantum is not None
        and abs(
            optimum
            - round(optimum / objective_quantum.quantum)
            * objective_quantum.quantum
        )
        <= objective_tolerance
    )
    protection_on_lattice = (
        protection_quantum.valid
        and protection_quantum.quantum is not None
        and abs(
            minimum_cost
            - round(minimum_cost / protection_quantum.quantum)
            * protection_quantum.quantum
        )
        <= cost_tolerance
    )
    lattice_membership_valid = bool(
        objective_on_lattice
        and protection_on_lattice
        and objective_tolerance < 0.5 * float(objective_quantum.quantum)
        and cost_tolerance < 0.5 * float(protection_quantum.quantum)
    )
    normalized_oracle = evaluate_task_path_fortification(
        system,
        normalized_reference,
        fortification_multiplier=fortification_multiplier,
        uplifts=delta,
    )
    if (
        not normalized_oracle.optimal
        or normalized_oracle.objective is None
        or float(normalized_oracle.objective) < margin_floor
    ):
        raise RuntimeError("Normalized optimal-family witness failed PathCut verification.")

    records: list[MPCFFamilyMembershipRecord] = []
    mandatory_nodes: set[NodeId] = set()
    possible_nodes: set[NodeId] = set()
    membership_exact = lattice_membership_valid
    nodes = tuple(system.removable_nodes)
    for index, node in enumerate(nodes, start=1):
        without = _compact_milp(
            system,
            budget,
            delta,
            time_limit=time_limit_per_solve,
            margin_floor=margin_floor,
            minimize_protection_cost=True,
            excluded_protection=(node,),
            protection_cost_ceiling=cost_ceiling,
        )
        if without.status == SolveStatus.INFEASIBLE:
            mandatory: bool | None = True
            mandatory_nodes.add(node)
        elif without.optimal:
            mandatory = False
        else:
            mandatory = None
            membership_exact = False

        with_node = _compact_milp(
            system,
            budget,
            delta,
            time_limit=time_limit_per_solve,
            margin_floor=margin_floor,
            minimize_protection_cost=True,
            required_protection=(node,),
            protection_cost_ceiling=cost_ceiling,
        )
        if with_node.optimal:
            can_appear: bool | None = True
            possible_nodes.add(node)
        elif with_node.status == SolveStatus.INFEASIBLE:
            can_appear = False
        else:
            can_appear = None
            membership_exact = False
        records.append(
            MPCFFamilyMembershipRecord(
                node=node,
                mandatory=mandatory,
                can_appear=can_appear,
                exclusion_status=without.status,
                inclusion_status=with_node.status,
            )
        )
        if progress is not None:
            progress(
                f"MPCF family membership {index}/{len(nodes)}: node={node!r}, "
                f"mandatory={mandatory}, can_appear={can_appear}"
            )

    enumerated: list[frozenset[NodeId]] = []
    complete = False
    if enumerate_limit > 0:
        enumerated.append(normalized_reference)
        no_good = [normalized_reference]
        while len(enumerated) < enumerate_limit:
            candidate = _compact_milp(
                system,
                budget,
                delta,
                time_limit=time_limit_per_solve,
                margin_floor=margin_floor,
                minimize_protection_cost=True,
                no_good_sets=no_good,
                protection_cost_ceiling=cost_ceiling,
            )
            if candidate.status == SolveStatus.INFEASIBLE:
                complete = True
                break
            if not candidate.optimal or candidate.protection_set in no_good:
                break
            enumerated.append(candidate.protection_set)
            no_good.append(candidate.protection_set)
        if len(enumerated) == enumerate_limit and not complete:
            probe = _compact_milp(
                system,
                budget,
                delta,
                time_limit=time_limit_per_solve,
                margin_floor=margin_floor,
                minimize_protection_cost=True,
                no_good_sets=no_good,
                protection_cost_ceiling=cost_ceiling,
            )
            complete = probe.status == SolveStatus.INFEASIBLE
    if progress is not None:
        progress(
            f"MPCF family enumeration: sets={len(enumerated)}, complete={complete}"
        )

    return MPCFOptimalFamilyResult(
        optimum_margin=optimum,
        minimum_protection_cost=minimum_cost,
        reference_set=normalized_reference,
        membership_records=tuple(records),
        mandatory_nodes=frozenset(mandatory_nodes),
        possible_nodes=frozenset(possible_nodes),
        interchangeable_nodes=frozenset(possible_nodes - mandatory_nodes),
        membership_exact=membership_exact,
        enumerated_sets=tuple(enumerated),
        enumeration_complete=complete,
        objective_cost_quantum=objective_quantum.quantum,
        protection_cost_quantum=protection_quantum.quantum,
        objective_membership_tolerance=objective_tolerance,
        protection_cost_membership_tolerance=cost_tolerance,
        lattice_membership_valid=lattice_membership_valid,
        runtime_seconds=perf_counter() - started,
    )


def brute_force_mpcf(
    graph: nx.Graph | OperationalMotifSystem,
    budget: float,
    *,
    fortification_multiplier: float = 1.0,
    uplifts: Mapping[NodeId, float] | None = None,
    max_nodes: int = 20,
) -> MPCFResult:
    """Enumerate tiny fortification sets for independent test certification."""

    system = _system(graph)
    classification = _require_path_closed(system)
    budget = _validate_budget(budget)
    if len(system.removable_nodes) > max_nodes:
        raise ValueError(f"Brute-force MPCF is limited to {max_nodes} removable nodes.")
    delta = _uplifts(
        system, multiplier=fortification_multiplier, uplifts=uplifts
    )
    baseline = solve_path_closed_motif_cut(system)
    if not baseline.optimal or baseline.objective is None:
        raise RuntimeError("Brute-force MPCF requires a finite PathCut.")
    started = perf_counter()
    best_set = frozenset()
    best_margin = float(baseline.objective)
    ordered_costs = sorted(
        float(system.graph.nodes[node]["protect_cost"])
        for node in system.removable_nodes
    )
    for size in range(len(system.removable_nodes) + 1):
        if fsum(ordered_costs[:size]) > budget + 1e-9:
            break
        for subset in combinations(system.removable_nodes, size):
            candidate = frozenset(subset)
            candidate_cost = protection_cost(system.graph, candidate)
            if candidate_cost > budget + 1e-9:
                continue
            oracle = evaluate_task_path_fortification(
                system,
                candidate,
                fortification_multiplier=fortification_multiplier,
                uplifts=delta,
            )
            if not oracle.optimal or oracle.objective is None:
                raise RuntimeError("Brute-force MPCF PathCut failed.")
            margin = float(oracle.objective)
            if (
                margin > best_margin + 1e-9
                or isclose(margin, best_margin, rel_tol=0.0, abs_tol=1e-9)
                and (
                    candidate_cost < protection_cost(system.graph, best_set) - 1e-9
                    or isclose(
                        candidate_cost,
                        protection_cost(system.graph, best_set),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and tuple(stable_node_key(node) for node in stable_nodes(candidate))
                    < tuple(stable_node_key(node) for node in stable_nodes(best_set))
                )
            ):
                best_set = candidate
                best_margin = margin
    solver = SolverInfo(
        status=SolveStatus.OPTIMAL,
        optimal=True,
        message="Certified by exhaustive enumeration of every budget-feasible defense.",
        objective=best_margin,
        lower_bound=best_margin,
        upper_bound=best_margin,
        mip_gap=0.0,
        runtime_seconds=perf_counter() - started,
    )
    return _result(
        method="MPCF-BruteForce",
        system=system,
        budget=budget,
        multiplier=fortification_multiplier,
        delta=delta,
        protection=best_set,
        baseline_margin=float(baseline.objective),
        solver=solver,
        trace=(),
        cut_pool=(baseline.selected_set,),
        classification=classification,
    )


def verify_mpcf_certificate(
    result: MPCFResult,
    graph: nx.Graph | OperationalMotifSystem | None = None,
    *,
    tolerance: float = 1e-7,
) -> bool:
    """Verify budget, adaptive cut, and any claimed closed optimality bounds."""

    try:
        system = _system(graph) if graph is not None else result._system
        if system is None:
            return False
        _require_path_closed(system)
        if protection_cost(system.graph, result.protection_set) > result.budget + tolerance:
            return False
        delta = dict(result.uplift_by_node)
        oracle = evaluate_task_path_fortification(
            system,
            result.protection_set,
            fortification_multiplier=result.fortification_multiplier,
            uplifts=delta,
        )
        if not oracle.optimal or oracle.objective is None:
            return False
        scale = max(1.0, abs(result.defended_margin), abs(float(oracle.objective)))
        if abs(result.defended_margin - float(oracle.objective)) > tolerance * scale:
            return False
        if result.adaptive_cut != oracle.selected_set:
            return False
        if result.optimal:
            if result.solver.lower_bound is None or result.solver.upper_bound is None:
                return False
            if result.solver.upper_bound < result.solver.lower_bound - tolerance * scale:
                return False
            if result.solver.upper_bound - result.solver.lower_bound > tolerance * scale:
                return False
            if abs(result.solver.objective - result.defended_margin) > tolerance * scale:
                return False
        return True
    except (TypeError, ValueError, RuntimeError, KeyError):
        return False


def verify_mpcf_objective_certificate(
    result: MPCFResult,
    graph: nx.Graph | OperationalMotifSystem | None = None,
    *,
    tolerance: float = 1e-7,
) -> bool:
    """Verify the adaptive cut and a raw- or lattice-closed objective bound."""

    if not verify_mpcf_certificate(result, graph, tolerance=tolerance):
        return False
    try:
        return mpcf_objective_certificate(
            result, graph, tolerance=tolerance
        ).certified_optimal
    except (TypeError, ValueError, RuntimeError, KeyError):
        return False
