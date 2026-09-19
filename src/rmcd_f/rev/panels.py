"""Frozen instance panels for the revision experiments.

Five panels share one generator so that every comparison inside a panel runs on
identical graphs:

``main``
    45 frozen graphs = 3 scale tiers x 3 topologies x 5 seeds.  ``graph_id``
    (topology + N + seed) is the independent experimental unit and ``budget`` is
    a repeated measure inside it.

``role``
    45 bases x 6 role compositions x 3 budgets = 810 records.  All six
    compositions of one ``base_id`` are one repeated-measures cluster.

``scaling``
    N in {200, 300, 500, 1000} with the 4:3:3:4 composition and ``3N`` directed
    interface edges, evaluated under one unified time limit.

``heterogeneity``
    Joint manipulation of the protection cost ``b_v`` and the relative
    fortification increment ``m_v = delta_v / a_v``, under three correlation
    profiles plus a homogeneous control.

``capacity`` / ``public``
    Reserved slots so the archive layout is fixed even when those studies are
    reported elsewhere.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from ..cli import load_graph, save_graph
from ..cost_profiles import apply_attack_cost_profile, apply_protection_cost_profile
from ..official_review import graph_fingerprint
from ..operational_motif import build_path_closed_system
from ..provenance import full_instance_fingerprint
from ..synthetic import TOPOLOGIES, generate_equipment_network

PANEL_VERSION = "1.0"

DEFAULT_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
DEFAULT_CAPACITY_LEVELS: tuple[int, ...] = (1, 2, 3)
DEFAULT_COST_LEVELS: tuple[int, ...] = (1, 2, 4)
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.02, 0.05, 0.10)
DEFAULT_PROTECTION_COST_LEVELS: tuple[int, ...] = (1, 2, 4)
DEFAULT_EFFECT_LEVELS: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ScaleTier:
    name: str
    role_sizes: tuple[int, int, int, int]
    edge_budget: int

    @property
    def node_count(self) -> int:
        return sum(self.role_sizes)

    @property
    def composition(self) -> str:
        return "-".join(str(value) for value in self.role_sizes)


#: The frozen 4:3:3:4 production tiers used by the paper's main panel.
MAIN_TIERS: tuple[ScaleTier, ...] = (
    ScaleTier("n42", (12, 9, 9, 12), 126),
    ScaleTier("n70", (20, 15, 15, 20), 210),
    ScaleTier("n112", (32, 24, 24, 32), 336),
)

#: Six role compositions.  Every profile sums to 14 units so that one base
#: instance supports all six at the same N, matching the role-ratio protocol.
ROLE_COMPOSITIONS: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("4-3-3-4", (4, 3, 3, 4)),
    ("3-4-4-3", (3, 4, 4, 3)),
    ("2-4-4-4", (2, 4, 4, 4)),
    ("4-2-4-4", (4, 2, 4, 4)),
    ("4-4-2-4", (4, 4, 2, 4)),
    ("4-4-4-2", (4, 4, 4, 2)),
)

#: Scaling targets.  Role sizes come from largest-remainder apportionment of
#: 4:3:3:4 so that N is hit exactly, and the edge budget stays ``3N``.
SCALING_SIZES: tuple[int, ...] = (200, 300, 500, 1000)

#: Correlation profiles for the joint heterogeneity experiment.
CORRELATION_PROFILES: tuple[str, ...] = ("independent", "positive_corr", "negative_corr")

CORRELATION_ALIASES: Mapping[str, str] = {
    "independent": "independent",
    "positive": "positive_corr",
    "positive_corr": "positive_corr",
    "negative": "negative_corr",
    "negative_corr": "negative_corr",
}


def apportion(ratio: Sequence[int], total: int) -> tuple[int, ...]:
    """Largest-remainder apportionment of ``ratio`` to exactly ``total`` units."""

    if total < len(ratio):
        raise ValueError("total must be at least the number of roles")
    quotient = sum(ratio)
    exact = [total * value / quotient for value in ratio]
    floors = [int(math.floor(value)) for value in exact]
    remainder = total - sum(floors)
    order = sorted(
        range(len(ratio)), key=lambda index: (-(exact[index] - floors[index]), index)
    )
    for index in order[:remainder]:
        floors[index] += 1
    if any(value < 1 for value in floors):
        raise ValueError(f"apportionment produced an empty role: {floors}")
    return tuple(floors)


@dataclass
class InstanceSpec:
    """One frozen graph view that solvers consume."""

    experiment: str
    instance_id: str
    graph_id: str
    base_id: str
    topology: str
    node_count: int
    edge_count: int
    seed: int
    composition: str
    scale_tier: str
    instance_file: str
    graph_fingerprint: str
    full_fingerprint: str
    total_protection_cost: float
    total_attack_cost: float
    task_path_count: int
    uid_scale: str
    budget_ratios: tuple[float, ...] = DEFAULT_BUDGET_RATIOS
    attack_cost_profile: str = "balanced-heterogeneous"
    attack_cost_levels: tuple[int, ...] = DEFAULT_COST_LEVELS
    protection_cost_profile: str = "unit"
    protection_cost_levels: tuple[int, ...] = DEFAULT_PROTECTION_COST_LEVELS
    effect_levels: tuple[float, ...] = ()
    correlation_profile: str = ""
    spearman_b_effect: float | None = None
    spearman_b_uplift: float | None = None
    uplift_multiplier: float = 1.0
    homogeneous_control: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_manifest_row(self) -> dict[str, Any]:
        row = {
            "experiment": self.experiment,
            "instance_id": self.instance_id,
            "graph_id": self.graph_id,
            "base_id": self.base_id,
            "topology": self.topology,
            "N": self.node_count,
            "num_edges": self.edge_count,
            "seed": self.seed,
            "composition": self.composition,
            "scale_tier": self.scale_tier,
            "instance_file": self.instance_file,
            "graph_fingerprint": self.graph_fingerprint,
            "full_instance_fingerprint": self.full_fingerprint,
            "total_protection_cost": self.total_protection_cost,
            "total_attack_cost": self.total_attack_cost,
            "task_path_count": self.task_path_count,
            "budget_ratios": json.dumps(list(self.budget_ratios)),
            "attack_cost_profile": self.attack_cost_profile,
            "attack_cost_levels": json.dumps(list(self.attack_cost_levels)),
            "protection_cost_profile": self.protection_cost_profile,
            "protection_cost_levels": json.dumps(list(self.protection_cost_levels)),
            "effect_levels": json.dumps(list(self.effect_levels)),
            "correlation_profile": self.correlation_profile,
            "spearman_b_effect": self.spearman_b_effect,
            "spearman_b_uplift": self.spearman_b_uplift,
            "uplift_multiplier": self.uplift_multiplier,
            "homogeneous_control": self.homogeneous_control,
        }
        row.update(self.extra)
        return row


# ---------------------------------------------------------------------------
# ranking helpers used by the correlation profiles
# ---------------------------------------------------------------------------


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties.

    Returns ``nan`` when the correlation is undefined (fewer than two points or
    a constant vector); callers serialise that as ``None``.
    """

    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator > 0 else float("nan")


def _optional(value: float) -> float | None:
    return float(value) if isinstance(value, float) and math.isfinite(value) else None


def _level_sequence(levels: Sequence[float], count: int) -> list[float]:
    return [float(levels[index % len(levels)]) for index in range(count)]


def coupled_levels(
    costs: Sequence[float],
    levels: Sequence[float],
    profile: str,
    rng: random.Random,
) -> list[float]:
    """Assign effectiveness levels to nodes to realise one correlation profile.

    The multiset of effectiveness levels is fixed by ``levels``; only the
    assignment to nodes changes, so the three profiles are exactly comparable
    and the realised Spearman correlation can be reported as measured.
    """

    count = len(costs)
    pool = _level_sequence(levels, count)
    if profile == "positive_corr":
        order = sorted(range(count), key=lambda index: (costs[index], index))
        pool.sort()
        assignment = [0.0] * count
        for position, index in enumerate(order):
            assignment[index] = pool[position]
        return assignment
    if profile == "negative_corr":
        order = sorted(range(count), key=lambda index: (costs[index], index))
        pool.sort(reverse=True)
        assignment = [0.0] * count
        for position, index in enumerate(order):
            assignment[index] = pool[position]
        return assignment
    if profile == "independent":
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown correlation profile {profile!r}.")


def apply_joint_heterogeneity(
    graph: nx.Graph,
    *,
    profile: str,
    cost_levels: Sequence[int] = DEFAULT_PROTECTION_COST_LEVELS,
    effect_levels: Sequence[float] = DEFAULT_EFFECT_LEVELS,
    seed: int = 0,
    homogeneous: bool = False,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Return a graph with joint ``b_v`` / ``m_v`` heterogeneity and its audit.

    ``b_v`` is the protection cost and ``m_v = delta_v / a_v`` is the relative
    fortification increment.  Both are assigned role-balanced multisets; only
    their *coupling* changes across profiles.  The returned audit records the
    realised Spearman correlations instead of asserting them by name.
    """

    profile = CORRELATION_ALIASES.get(profile, profile)
    if not homogeneous and profile not in CORRELATION_PROFILES:
        raise ValueError(
            f"profile must be one of {CORRELATION_PROFILES}; got {profile!r}."
        )

    if homogeneous:
        equipment = apply_protection_cost_profile(
            graph, "unit", cost_seed=seed, cost_levels=(1,)
        )
    else:
        equipment = apply_protection_cost_profile(
            graph, "balanced-heterogeneous", cost_seed=seed, cost_levels=cost_levels
        )

    rng = random.Random(f"{seed}|joint-heterogeneity|{PANEL_VERSION}|{profile}")
    nodes = sorted(
        equipment.nodes,
        key=lambda node: (str(equipment.nodes[node].get("role", "")), repr(node)),
    )
    by_role: dict[str, list[Any]] = {}
    for node in nodes:
        by_role.setdefault(str(equipment.nodes[node].get("role", "")), []).append(node)

    costs: list[float] = []
    effectiveness: list[float] = []
    uplifts: dict[Any, float] = {}
    for role in sorted(by_role):
        role_nodes = by_role[role]
        role_costs = [
            float(equipment.nodes[node]["protect_cost"]) for node in role_nodes
        ]
        if homogeneous:
            role_effect = [1.0] * len(role_nodes)
        else:
            role_effect = coupled_levels(
                role_costs, effect_levels, profile, random.Random(f"{seed}|{profile}|{role}")
            )
        for node, effect in zip(role_nodes, role_effect):
            attack = float(equipment.nodes[node]["attack_cost"])
            uplift = float(effect) * attack
            uplifts[node] = uplift
            # Persist the per-node uplift on the frozen graph so that a later
            # process can reproduce the instance exactly without re-deriving
            # the coupling from a seed.
            equipment.nodes[node]["uplift"] = uplift
            equipment.nodes[node]["uplift_multiplier"] = float(effect)
            costs.append(float(equipment.nodes[node]["protect_cost"]))
            effectiveness.append(float(effect))

    absolute_uplifts = [uplifts[node] for node in nodes]
    audit: dict[str, Any] = {
        "b_profile": "homogeneous" if homogeneous else "heterogeneous",
        "effect_levels": [float(value) for value in (() if homogeneous else effect_levels)],
        "correlation_profile": "homogeneous" if homogeneous else profile,
        "b_levels": [1] if homogeneous else [int(value) for value in cost_levels],
        "spearman_b_effect": _optional(spearman(costs, effectiveness)),
        "spearman_b_uplift": _optional(spearman(costs, absolute_uplifts)),
        "sum_b": float(sum(costs)),
        "sum_uplift": float(sum(absolute_uplifts)),
    }
    equipment.graph["joint_heterogeneity"] = dict(audit)
    return equipment, audit


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _freeze(
    graph: nx.DiGraph,
    path: Path,
) -> tuple[nx.DiGraph, str, str]:
    """Persist one frozen instance and return it with its fingerprints.

    Instance generation is deterministic, so concurrent budget shards compute
    byte-identical graphs.  If the file already holds that same graph it is
    reused instead of rewritten, and :func:`rmcd_f.cli.save_graph` writes
    atomically, so no reader can ever observe a partially written instance.
    """

    if path.exists():
        try:
            existing = load_graph(path)
            if graph_fingerprint(existing) == graph_fingerprint(graph):
                return (
                    existing,
                    graph_fingerprint(existing),
                    full_instance_fingerprint(existing),
                )
        except Exception:
            # A truncated or stale file is simply regenerated below.
            pass
    save_graph(graph, path)
    reloaded = load_graph(path)
    return reloaded, graph_fingerprint(reloaded), full_instance_fingerprint(reloaded)


def _finalize(
    *,
    experiment: str,
    graph: nx.DiGraph,
    instance_file: Path,
    topology: str,
    seed: int,
    composition: str,
    scale_tier: str,
    budget_ratios: Sequence[float],
    attack_cost_profile: str,
    attack_cost_levels: Sequence[int],
    protection_cost_profile: str,
    protection_cost_levels: Sequence[int],
    effect_levels: Sequence[float] = (),
    correlation_profile: str = "",
    spearman_b_effect: float | None = None,
    spearman_b_uplift: float | None = None,
    uplift_multiplier: float = 1.0,
    homogeneous_control: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> InstanceSpec:
    from .ids import base_id as make_base_id
    from .ids import instance_id as make_instance_id

    system = build_path_closed_system(graph)
    node_count = graph.number_of_nodes()
    base = make_base_id(topology, node_count, seed)
    identifier = make_instance_id(
        experiment,
        base,
        None if experiment == "main" else composition,
        correlation_profile if experiment == "heterogeneity" else None,
    )
    payload = {
        "experiment": experiment,
        "instance_id": identifier,
        "graph_id": base,
        "base_id": base,
        "topology": topology,
        "node_count": node_count,
        "edge_count": graph.number_of_edges(),
        "seed": seed,
        "composition": composition,
        "scale_tier": scale_tier,
        "budget_ratios": tuple(float(value) for value in budget_ratios),
        "attack_cost_profile": attack_cost_profile,
        "attack_cost_levels": tuple(int(value) for value in attack_cost_levels),
        "protection_cost_profile": protection_cost_profile,
        "protection_cost_levels": tuple(int(value) for value in protection_cost_levels),
        "effect_levels": tuple(float(value) for value in effect_levels),
        "correlation_profile": correlation_profile,
        "spearman_b_effect": spearman_b_effect,
        "spearman_b_uplift": spearman_b_uplift,
        "uplift_multiplier": float(uplift_multiplier),
        "homogeneous_control": int(homogeneous_control),
    }
    if experiment != "main":
        # A composition is part of the instance identity, so the same
        # (topology, N, seed) base yields several distinct instance_ids that
        # nevertheless share one base_id cluster.
        payload["graph_id"] = identifier
    reloaded, fingerprint, full = _freeze(graph, instance_file)
    spec = InstanceSpec(
        **payload,
        instance_file=str(instance_file),
        graph_fingerprint=fingerprint,
        full_fingerprint=full,
        total_protection_cost=float(
            sum(
                float(reloaded.nodes[node]["protect_cost"])
                for node in system.removable_nodes
            )
        ),
        total_attack_cost=float(
            sum(
                float(reloaded.nodes[node]["attack_cost"])
                for node in system.removable_nodes
            )
        ),
        task_path_count=len(system.motifs),
        uid_scale=scale_tier,
        extra=dict(extra or {}),
    )
    return spec


def build_main_panel(
    output: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    tiers: Sequence[ScaleTier] = MAIN_TIERS,
    topologies: Sequence[str] = TOPOLOGIES,
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    capacity_levels: Sequence[int] = DEFAULT_CAPACITY_LEVELS,
    cost_levels: Sequence[int] = DEFAULT_COST_LEVELS,
    fortification_multiplier: float = 1.0,
    labels: Mapping[str, Mapping[str, int]] | None = None,
) -> list[InstanceSpec]:
    """The 45-graph main panel; each graph is one independent unit."""

    root = Path(output) / "instances" / "main"
    root.mkdir(parents=True, exist_ok=True)
    specs: list[InstanceSpec] = []
    for tier in tiers:
        for topology in topologies:
            for seed in seeds:
                graph = generate_equipment_network(
                    topology,
                    dict(zip(("S", "C", "L", "E"), tier.role_sizes)),
                    seed=int(seed),
                    edge_budget=int(tier.edge_budget),
                    capacity_levels=tuple(int(value) for value in capacity_levels),
                )
                graph = apply_attack_cost_profile(
                    graph,
                    "balanced-heterogeneous",
                    cost_levels=tuple(int(value) for value in cost_levels),
                )
                spec = _finalize(
                    experiment="main",
                    graph=graph,
                    instance_file=root / f"{topology}_seed{seed}_{tier.name}.json",
                    topology=topology,
                    seed=int(seed),
                    composition=tier.composition,
                    scale_tier=tier.name,
                    budget_ratios=budget_ratios,
                    attack_cost_profile="balanced-heterogeneous",
                    attack_cost_levels=cost_levels,
                    protection_cost_profile="unit",
                    protection_cost_levels=(1,),
                    uplift_multiplier=fortification_multiplier,
                    extra={"role_sizes": json.dumps(list(tier.role_sizes))},
                )
                specs.append(spec)
    return specs


def scaling_tier(node_count: int) -> ScaleTier:
    role_sizes = apportion((4, 3, 3, 4), int(node_count))
    return ScaleTier(f"n{int(node_count)}", role_sizes, 3 * int(node_count))


def build_scaling_panel(
    output: Path,
    *,
    sizes: Sequence[int] = SCALING_SIZES,
    seeds: Sequence[int] = (11, 22, 33),
    topologies: Sequence[str] = TOPOLOGIES,
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    capacity_levels: Sequence[int] = DEFAULT_CAPACITY_LEVELS,
    cost_levels: Sequence[int] = DEFAULT_COST_LEVELS,
    fortification_multiplier: float = 1.0,
) -> list[InstanceSpec]:
    """Scaling panel with a fixed 4:3:3:4 composition and ``3N`` edges."""

    root = Path(output) / "instances" / "scaling"
    root.mkdir(parents=True, exist_ok=True)
    specs: list[InstanceSpec] = []
    for size in sizes:
        tier = scaling_tier(int(size))
        for topology in topologies:
            for seed in seeds:
                graph = generate_equipment_network(
                    topology,
                    dict(zip(("S", "C", "L", "E"), tier.role_sizes)),
                    seed=int(seed),
                    edge_budget=int(tier.edge_budget),
                    capacity_levels=tuple(int(value) for value in capacity_levels),
                )
                graph = apply_attack_cost_profile(
                    graph,
                    "balanced-heterogeneous",
                    cost_levels=tuple(int(value) for value in cost_levels),
                )
                specs.append(
                    _finalize(
                        experiment="scaling",
                        graph=graph,
                        instance_file=root
                        / f"{topology}_seed{seed}_{tier.name}.json",
                        topology=topology,
                        seed=int(seed),
                        composition=tier.composition,
                        scale_tier=tier.name,
                        budget_ratios=budget_ratios,
                        attack_cost_profile="balanced-heterogeneous",
                        attack_cost_levels=cost_levels,
                        protection_cost_profile="unit",
                        protection_cost_levels=(1,),
                        uplift_multiplier=fortification_multiplier,
                        extra={"role_sizes": json.dumps(list(tier.role_sizes))},
                    )
                )
    return specs


def build_role_panel(
    output: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    tiers: Sequence[ScaleTier] = MAIN_TIERS,
    topologies: Sequence[str] = TOPOLOGIES,
    compositions: Sequence[tuple[str, tuple[int, int, int, int]]] = ROLE_COMPOSITIONS,
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    capacity_levels: Sequence[int] = DEFAULT_CAPACITY_LEVELS,
    cost_levels: Sequence[int] = DEFAULT_COST_LEVELS,
    fortification_multiplier: float = 1.0,
) -> list[InstanceSpec]:
    """45 bases x 6 compositions; all six share one ``base_id`` cluster."""

    root = Path(output) / "instances" / "role"
    root.mkdir(parents=True, exist_ok=True)
    specs: list[InstanceSpec] = []
    for tier in tiers:
        unit = tier.node_count // 14
        for topology in topologies:
            for seed in seeds:
                for label, ratio in compositions:
                    role_sizes = tuple(int(value) * unit for value in ratio)
                    if sum(role_sizes) != tier.node_count:
                        raise ValueError(
                            f"composition {label} does not preserve N={tier.node_count}"
                        )
                    graph = generate_equipment_network(
                        topology,
                        dict(zip(("S", "C", "L", "E"), role_sizes)),
                        seed=int(seed),
                        edge_budget=int(tier.edge_budget),
                        capacity_levels=tuple(int(value) for value in capacity_levels),
                    )
                    graph = apply_attack_cost_profile(
                        graph,
                        "balanced-heterogeneous",
                        cost_levels=tuple(int(value) for value in cost_levels),
                    )
                    specs.append(
                        _finalize(
                            experiment="role",
                            graph=graph,
                            instance_file=root
                            / f"{topology}_seed{seed}_{tier.name}_{label}.json",
                            topology=topology,
                            seed=int(seed),
                            composition=label,
                            scale_tier=tier.name,
                            budget_ratios=budget_ratios,
                            attack_cost_profile="balanced-heterogeneous",
                            attack_cost_levels=cost_levels,
                            protection_cost_profile="unit",
                            protection_cost_levels=(1,),
                            uplift_multiplier=fortification_multiplier,
                            extra={
                                "role_sizes": json.dumps(list(role_sizes)),
                                "role_ratio": label,
                            },
                        )
                    )
    return specs


def build_heterogeneity_panel(
    output: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    tiers: Sequence[ScaleTier] = MAIN_TIERS,
    topologies: Sequence[str] = TOPOLOGIES,
    profiles: Sequence[str] = CORRELATION_PROFILES,
    include_homogeneous: bool = True,
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    capacity_levels: Sequence[int] = DEFAULT_CAPACITY_LEVELS,
    attack_cost_levels: Sequence[int] = DEFAULT_COST_LEVELS,
    protection_cost_levels: Sequence[int] = DEFAULT_PROTECTION_COST_LEVELS,
    effect_levels: Sequence[float] = DEFAULT_EFFECT_LEVELS,
    fortification_multiplier: float = 1.0,
) -> list[InstanceSpec]:
    """Joint ``b_v`` x ``m_v`` panel with measured Spearman correlations."""

    root = Path(output) / "instances" / "heterogeneity"
    root.mkdir(parents=True, exist_ok=True)
    specs: list[InstanceSpec] = []
    variants: list[tuple[str, bool]] = []
    if include_homogeneous:
        variants.append(("homogeneous", True))
    variants.extend(
        (CORRELATION_ALIASES.get(name, name), False) for name in profiles
    )
    for tier in tiers:
        for topology in topologies:
            for seed in seeds:
                base_graph = generate_equipment_network(
                    topology,
                    dict(zip(("S", "C", "L", "E"), tier.role_sizes)),
                    seed=int(seed),
                    edge_budget=int(tier.edge_budget),
                    capacity_levels=tuple(int(value) for value in capacity_levels),
                )
                base_graph = apply_attack_cost_profile(
                    base_graph,
                    "balanced-heterogeneous",
                    cost_levels=tuple(int(value) for value in attack_cost_levels),
                )
                for label, homogeneous in variants:
                    graph, audit = apply_joint_heterogeneity(
                        base_graph,
                        profile=label,
                        cost_levels=protection_cost_levels,
                        effect_levels=effect_levels,
                        seed=int(seed),
                        homogeneous=homogeneous,
                    )
                    specs.append(
                        _finalize(
                            experiment="heterogeneity",
                            graph=graph,
                            instance_file=root
                            / f"{topology}_seed{seed}_{tier.name}_{label}.json",
                            topology=topology,
                            seed=int(seed),
                            composition=tier.composition,
                            scale_tier=tier.name,
                            budget_ratios=budget_ratios,
                            attack_cost_profile="balanced-heterogeneous",
                            attack_cost_levels=attack_cost_levels,
                            protection_cost_profile=(
                                "unit" if homogeneous else "balanced-heterogeneous"
                            ),
                            protection_cost_levels=(
                                (1,) if homogeneous else protection_cost_levels
                            ),
                            effect_levels=() if homogeneous else effect_levels,
                            correlation_profile=label,
                            spearman_b_effect=audit["spearman_b_effect"],
                            spearman_b_uplift=audit["spearman_b_uplift"],
                            uplift_multiplier=fortification_multiplier,
                            homogeneous_control=homogeneous,
                            extra={
                                "role_sizes": json.dumps(list(tier.role_sizes)),
                                "b_profile": audit["b_profile"],
                                "sum_b": audit["sum_b"],
                            },
                        )
                    )
    return specs


def uplift_for_spec(graph: nx.DiGraph) -> dict[Any, float] | None:
    """Per-node uplift map read back from a frozen heterogeneous graph.

    Returns ``None`` for the homogeneous case, where the solver's scalar
    fortification multiplier already reproduces ``delta_v = m * a_v``.
    """

    marked = [node for node, data in graph.nodes(data=True) if "uplift" in data]
    if not marked:
        return None
    return {node: float(graph.nodes[node]["uplift"]) for node in marked}


PANEL_BUILDERS = {
    "main": build_main_panel,
    "role": build_role_panel,
    "scaling": build_scaling_panel,
    "heterogeneity": build_heterogeneity_panel,
}
