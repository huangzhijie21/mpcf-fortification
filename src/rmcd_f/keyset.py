"""Unified identification of role-aware collective critical equipment sets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal

import networkx as nx

from .criticality import CriticalSetResult, analyze_exclusion
from .flow import motif_capacity
from .model import NodeId, ROLES, stable_nodes, validate_graph
from .pcd import PCDResult, solve_rmcd_pcd
from .rmcd import RMCDResult, solve_rmcd_exact


@dataclass(frozen=True)
class CriticalEquipmentNode:
    """One physical equipment node in an identified collective critical set."""

    node: NodeId
    role: str
    capacity: int
    attack_cost: float


@dataclass(frozen=True)
class CriticalEquipmentSetResult:
    """One certified or explicitly bounded role-aware critical-node result."""

    threshold: int
    initial_capacity: int
    solve_result: RMCDResult
    nodes: tuple[CriticalEquipmentNode, ...]
    role_counts: tuple[tuple[str, int], ...]
    role_capacity: tuple[tuple[str, int], ...]
    pcd_result: PCDResult | None = None
    family_analysis: CriticalSetResult | None = None

    @property
    def critical_set(self) -> frozenset[NodeId]:
        return self.solve_result.attack_set

    @property
    def critical_cost(self) -> float | None:
        return self.solve_result.attack_cost

    @property
    def residual_capacity(self) -> int | None:
        return self.solve_result.residual_capacity

    @property
    def optimal(self) -> bool:
        return self.solve_result.optimal

    @property
    def capacity_breached(self) -> bool:
        residual = self.residual_capacity
        return residual is not None and residual <= self.threshold - 1


def _node_records(
    graph: nx.DiGraph, nodes: frozenset[NodeId]
) -> tuple[CriticalEquipmentNode, ...]:
    return tuple(
        CriticalEquipmentNode(
            node=node,
            role=str(graph.nodes[node]["role"]),
            capacity=int(graph.nodes[node]["capacity"]),
            attack_cost=float(graph.nodes[node]["attack_cost"]),
        )
        for node in stable_nodes(nodes)
    )


def _role_totals(
    records: tuple[CriticalEquipmentNode, ...],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    counts = Counter(record.role for record in records)
    capacities = Counter()
    for record in records:
        capacities[record.role] += record.capacity
    return (
        tuple((role, int(counts[role])) for role in ROLES),
        tuple((role, int(capacities[role])) for role in ROLES),
    )


def identify_critical_equipment_set(
    graph: nx.Graph,
    K: int,
    *,
    method: Literal["exact", "pcd"] = "pcd",
    max_oracle_calls: int = 64,
    pcd_fallback: Literal["none", "exact"] = "none",
    time_limit_per_solve: float | None = None,
    analyze_optimal_family: bool = False,
    enumerate_limit: int = 100,
    progress: Callable[[str], None] | None = None,
) -> CriticalEquipmentSetResult:
    """Identify the minimum-cost collective set breaching ``Omega_R >= K``.

    The public result deliberately uses the language of *critical equipment
    sets*.  Internally, the same set remains an RMCD attack set so that all
    existing mathematical certificates and verification routines are reused.
    """

    if method not in {"exact", "pcd"}:
        raise ValueError("method must be 'exact' or 'pcd'.")
    equipment = validate_graph(graph)
    initial_capacity = motif_capacity(equipment).value

    pcd_result: PCDResult | None = None
    if method == "pcd":
        if progress is not None:
            progress(f"solving collective critical set by RMCD-PCD for K={K}")
        pcd_result = solve_rmcd_pcd(
            equipment,
            K,
            max_oracle_calls=max_oracle_calls,
            fallback=pcd_fallback,
            time_limit=time_limit_per_solve,
        )
        solve_result = pcd_result.rmcd
    else:
        if progress is not None:
            progress(f"solving collective critical set by RMCD-Exact for K={K}")
        solve_result = solve_rmcd_exact(
            equipment,
            K,
            time_limit=time_limit_per_solve,
            mip_rel_gap=0.0,
        )

    family: CriticalSetResult | None = None
    if analyze_optimal_family:
        if not solve_result.optimal or solve_result.is_unbreakable:
            raise ValueError(
                "Optimal-set family analysis requires a finite globally "
                "certified critical set."
            )
        if progress is not None:
            progress("analyzing exact critical core and substitutable shell")
        family = analyze_exclusion(
            equipment,
            K,
            base_result=solve_result,
            enumerate_limit=enumerate_limit,
            time_limit_per_solve=time_limit_per_solve,
            progress=progress,
        )

    records = _node_records(equipment, solve_result.attack_set)
    role_counts, role_capacity = _role_totals(records)
    return CriticalEquipmentSetResult(
        threshold=int(K),
        initial_capacity=initial_capacity,
        solve_result=solve_result,
        nodes=records,
        role_counts=role_counts,
        role_capacity=role_capacity,
        pcd_result=pcd_result,
        family_analysis=family,
    )
