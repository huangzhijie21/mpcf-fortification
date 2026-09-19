"""Collective critical-set interpretation for certified RMCD optima."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Callable, Iterable

import networkx as nx

from .model import NodeId, SolveStatus, stable_nodes, validate_graph
from .rmcd import (
    RMCDResult,
    _require_discrete_attack_costs,
    _solve_rmcd_milp,
    enumerate_optimal_attacks,
    solve_rmcd,
    verify_rmcd_certificate,
)


@dataclass(frozen=True)
class ExclusionRecord:
    """Cost of the best successful attack when one node cannot be attacked."""

    node: NodeId
    restricted_cost: float | None
    exclusion_penalty: float | None
    restricted_status: SolveStatus
    exact: bool


@dataclass(frozen=True)
class InclusionRecord:
    """Whether one node can occur in a globally optimal attack set."""

    node: NodeId
    forced_cost: float | None
    forced_status: SolveStatus
    in_optimal_union: bool | None
    exact: bool


@dataclass(frozen=True)
class CriticalSetResult:
    """Exact core plus an exact or explicitly partial view of the optimal shell."""

    threshold: int
    base_result: RMCDResult
    exclusion_records: tuple[ExclusionRecord, ...]
    core: frozenset[NodeId]
    core_exact: bool
    enumerated_optimal_sets: tuple[frozenset[NodeId], ...]
    discovered_union: frozenset[NodeId]
    discovered_shell: frozenset[NodeId]
    enumeration_complete: bool
    inclusion_records: tuple[InclusionRecord, ...]
    union_membership_exact: bool

    @property
    def union_exact(self) -> bool:
        return self.union_membership_exact or self.enumeration_complete

    @property
    def shell_exact(self) -> bool:
        return self.union_exact and self.core_exact

    def exclusion_by_node(self) -> dict[NodeId, ExclusionRecord]:
        return {record.node: record for record in self.exclusion_records}

    def inclusion_by_node(self) -> dict[NodeId, InclusionRecord]:
        return {record.node: record for record in self.inclusion_records}


def analyze_exclusion(
    graph: nx.Graph,
    K: int,
    base_result: RMCDResult | None = None,
    *,
    enumerate_limit: int = 100,
    certify_union: bool = True,
    time_limit_per_solve: float | None = None,
    tolerance: float = 1e-8,
    progress: Callable[[str], None] | None = None,
) -> CriticalSetResult:
    """Compute the exact core and, by default, exact optimal-set union.

    Complete enumeration can be exponentially large.  Union membership is
    instead certified node by node by fixing the globally optimal objective
    value and requiring the node to be attacked.  Bounded enumeration remains
    available only as a diagnostic sample of the solution family.
    """

    equipment = validate_graph(graph)
    _require_discrete_attack_costs(equipment)
    if enumerate_limit < 0:
        raise ValueError("enumerate_limit must be non-negative.")
    if not 0.0 <= tolerance < 0.5:
        raise ValueError("tolerance must satisfy 0 <= tolerance < 0.5.")
    if base_result is None:
        base = solve_rmcd(
            equipment, K, time_limit=time_limit_per_solve, mip_rel_gap=0.0
        )
        if progress is not None:
            progress(
                f"criticality base RMCD: status={base.status.value}, "
                f"optimal={base.optimal}"
            )
        if not base.optimal or base.is_unbreakable or base.attack_cost is None:
            return CriticalSetResult(
                threshold=int(K),
                base_result=base,
                exclusion_records=(),
                core=frozenset(),
                core_exact=False,
                enumerated_optimal_sets=(),
                discovered_union=frozenset(),
                discovered_shell=frozenset(),
                enumeration_complete=False,
                inclusion_records=(),
                union_membership_exact=False,
            )
    else:
        if base_result.threshold != int(K):
            raise ValueError("base_result threshold does not match K.")
        if not verify_rmcd_certificate(base_result, equipment):
            raise ValueError("base_result is not a valid certificate for the current graph.")
        reference = solve_rmcd(
            equipment,
            K,
            protected=base_result.protected,
            excluded=base_result.excluded,
            time_limit=time_limit_per_solve,
            mip_rel_gap=0.0,
        )
        if (
            not reference.optimal
            or reference.attack_cost is None
            or base_result.attack_cost is None
            or abs(reference.attack_cost - base_result.attack_cost) > tolerance
        ):
            raise ValueError(
                "base_result is not globally optimal for the current graph and constraints."
            )
        base = base_result
    if not base.optimal or base.is_unbreakable or base.attack_cost is None:
        raise ValueError(
            "Critical-set analysis requires a finite, globally optimal RMCD base result."
        )
    records: list[ExclusionRecord] = []
    core_nodes: set[NodeId] = set()
    all_core_checks_exact = True
    exclusion_total = len(base.attack_set)
    exclusion_index = 0
    for node in stable_nodes(equipment.nodes):
        if node not in base.attack_set:
            # The certified base optimum itself excludes this node.
            records.append(
                ExclusionRecord(node, base.attack_cost, 0.0, SolveStatus.OPTIMAL, True)
            )
            continue

        restricted = solve_rmcd(
            equipment,
            K,
            protected=base.protected,
            excluded=set(base.excluded) | {node},
            time_limit=time_limit_per_solve,
            mip_rel_gap=0.0,
        )
        exclusion_index += 1
        if progress is not None:
            progress(
                f"criticality exclusion {exclusion_index}/{exclusion_total} "
                f"node={node!r}: status={restricted.status.value}, "
                f"optimal={restricted.optimal}"
            )
        if restricted.is_unbreakable and restricted.optimal:
            penalty = inf
            exact = True
            restricted_cost = None
            core_nodes.add(node)
        elif restricted.optimal and restricted.attack_cost is not None:
            restricted_cost = restricted.attack_cost
            penalty = max(0.0, restricted.attack_cost - base.attack_cost)
            exact = True
            if penalty > tolerance:
                core_nodes.add(node)
        else:
            restricted_cost = restricted.attack_cost
            penalty = None
            exact = False
            all_core_checks_exact = False
        records.append(
            ExclusionRecord(
                node,
                restricted_cost,
                penalty,
                restricted.status,
                exact,
            )
        )

    inclusion_records: list[InclusionRecord] = []
    certified_union: set[NodeId] = set(base.attack_set)
    all_union_checks_exact = bool(certify_union)
    if certify_union:
        candidates = stable_nodes(equipment.nodes)
        inclusion_total = sum(
            node not in base.attack_set
            and node not in (base.protected | base.excluded)
            for node in candidates
        )
        inclusion_index = 0
        for node in candidates:
            if node in base.attack_set:
                inclusion_records.append(
                    InclusionRecord(
                        node,
                        base.attack_cost,
                        SolveStatus.OPTIMAL,
                        True,
                        True,
                    )
                )
                continue
            if node in base.protected or node in base.excluded:
                inclusion_records.append(
                    InclusionRecord(
                        node,
                        None,
                        SolveStatus.INFEASIBLE,
                        False,
                        True,
                    )
                )
                continue

            forced = _solve_rmcd_milp(
                equipment,
                int(K),
                protected=base.protected,
                excluded=base.excluded,
                required_attack=frozenset({node}),
                fixed_objective=base.attack_cost,
                objective_tolerance=tolerance,
                time_limit=time_limit_per_solve,
                mip_rel_gap=0.0,
            )
            inclusion_index += 1
            if progress is not None:
                progress(
                    f"criticality inclusion {inclusion_index}/{inclusion_total} "
                    f"node={node!r}: status={forced.status.value}, "
                    f"optimal={forced.optimal}"
                )

            if forced.optimal and forced.attack_cost is not None:
                member: bool | None = True
                exact = True
                certified_union.add(node)
            elif forced.status == SolveStatus.INFEASIBLE:
                member = False
                exact = True
            else:
                member = None
                exact = False
                all_union_checks_exact = False
            inclusion_records.append(
                InclusionRecord(
                    node,
                    forced.attack_cost,
                    forced.status,
                    member,
                    exact,
                )
            )

    if enumerate_limit > 0:
        enumerated, complete = enumerate_optimal_attacks(
            equipment,
            K,
            protected=base.protected,
            excluded=base.excluded,
            limit=enumerate_limit,
            objective_tolerance=tolerance,
            time_limit_per_solve=time_limit_per_solve,
        )
        sets = list(enumerated)
        if base.attack_set not in sets:
            sets.insert(0, base.attack_set)
            if len(sets) > enumerate_limit:
                sets = sets[:enumerate_limit]
                complete = False
    else:
        sets = []
        complete = False
    if progress is not None:
        progress(
            f"criticality optimal-set enumeration: sets={len(sets)}, "
            f"complete={complete}"
        )
    enumerated_union = frozenset().union(*sets) if sets else frozenset()
    discovered_union = frozenset(certified_union | set(enumerated_union))
    core = frozenset(core_nodes)
    return CriticalSetResult(
        threshold=int(K),
        base_result=base,
        exclusion_records=tuple(records),
        core=core,
        core_exact=base.optimal and all_core_checks_exact,
        enumerated_optimal_sets=tuple(sets),
        discovered_union=frozenset(discovered_union),
        discovered_shell=frozenset(set(discovered_union) - set(core)),
        enumeration_complete=complete,
        inclusion_records=tuple(inclusion_records),
        union_membership_exact=all_union_checks_exact,
    )
