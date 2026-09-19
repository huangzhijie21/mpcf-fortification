"""Pre-registered attack-cost overlays for paired RMCD experiments."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Integral
import random
from typing import Iterable

import networkx as nx

from .model import ROLES, stable_nodes, validate_graph


COST_PROFILES: tuple[str, ...] = ("unit", "balanced-heterogeneous")
PROTECTION_COST_PROFILES: tuple[str, ...] = COST_PROFILES
DEFAULT_HETEROGENEOUS_LEVELS: tuple[int, ...] = (1, 2, 4)
COST_PROFILE_VERSION = "1.0"


def _positive_integer_levels(values: Iterable[int]) -> tuple[int, ...]:
    levels = tuple(values)
    if not levels:
        raise ValueError("cost_levels must contain at least one positive integer.")
    for value in levels:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(
                "cost_levels must contain only positive integers; "
                f"got {value!r}."
            )
    return tuple(int(value) for value in levels)


def _cost_seed(graph: nx.DiGraph, requested: int | None) -> int:
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, Integral):
            raise ValueError("cost_seed must be a non-negative integer.")
        seed = int(requested)
    else:
        synthetic = graph.graph.get("synthetic", {})
        seed = int(synthetic.get("seed", 0))
    if seed < 0:
        raise ValueError("cost_seed must be a non-negative integer.")
    return seed


def apply_attack_cost_profile(
    graph: nx.Graph,
    profile: str,
    *,
    cost_seed: int | None = None,
    cost_levels: Iterable[int] = DEFAULT_HETEROGENEOUS_LEVELS,
) -> nx.DiGraph:
    """Return a paired copy with a pre-registered attack-cost assignment.

    ``balanced-heterogeneous`` repeats the same integer level multiset inside
    every role and shuffles it with an RNG namespace that depends only on the
    cost seed and role.  The assignment therefore does not inspect topology,
    degree, capacity, solver output, or functional outcomes.
    """

    if profile not in COST_PROFILES:
        raise ValueError(f"profile must be one of {COST_PROFILES}; got {profile!r}.")
    equipment = deepcopy(validate_graph(graph))
    seed = _cost_seed(equipment, cost_seed)
    levels = _positive_integer_levels(cost_levels)

    assigned: dict[object, int] = {}
    for role in ROLES:
        nodes = list(
            stable_nodes(
                node
                for node, data in equipment.nodes(data=True)
                if data["role"] == role
            )
        )
        if profile == "unit":
            values = [1] * len(nodes)
        else:
            values = [levels[index % len(levels)] for index in range(len(nodes))]
            random.Random(
                f"{seed}|attack-cost-profile|{COST_PROFILE_VERSION}|{role}"
            ).shuffle(values)
        for node, value in zip(nodes, values):
            equipment.nodes[node]["attack_cost"] = float(value)
            assigned[node] = value

    role_multisets = {}
    for role in ROLES:
        counts = Counter(
            assigned[node]
            for node, data in equipment.nodes(data=True)
            if data["role"] == role
        )
        role_multisets[role] = [
            [level, counts[level]] for level in sorted(counts)
        ]
    all_counts = Counter(assigned.values())
    metadata = {
        "name": profile,
        "version": COST_PROFILE_VERSION,
        "cost_seed": seed,
        "integer_levels": [1] if profile == "unit" else list(levels),
        "cost_multiset": [
            [level, all_counts[level]] for level in sorted(all_counts)
        ],
        "cost_multiset_by_role": role_multisets,
        "assignment_namespace": "seed|attack-cost-profile|version|role",
        "topology_oblivious": True,
        "capacity_oblivious": True,
        "outcome_screening": False,
    }
    equipment.graph["attack_cost_profile"] = metadata
    synthetic = deepcopy(equipment.graph.get("synthetic", {}))
    if synthetic:
        synthetic["attack_cost"] = None
        synthetic["cost_mode"] = profile
        synthetic["attack_cost_profile"] = metadata
        equipment.graph["synthetic"] = synthetic
    return validate_graph(equipment)


def apply_protection_cost_profile(
    graph: nx.Graph,
    profile: str,
    *,
    cost_seed: int | None = None,
    cost_levels: Iterable[int] = DEFAULT_HETEROGENEOUS_LEVELS,
) -> nx.DiGraph:
    """Return a paired copy with a topology-oblivious protection-cost profile.

    The heterogeneous multiset is balanced separately within every role and
    assigned before any protection method is evaluated.  Attack costs,
    capacities, edges and role labels remain unchanged.
    """

    if profile not in PROTECTION_COST_PROFILES:
        raise ValueError(
            f"profile must be one of {PROTECTION_COST_PROFILES}; got {profile!r}."
        )
    equipment = deepcopy(validate_graph(graph))
    seed = _cost_seed(equipment, cost_seed)
    levels = _positive_integer_levels(cost_levels)

    assigned: dict[object, int] = {}
    for role in ROLES:
        nodes = list(
            stable_nodes(
                node
                for node, data in equipment.nodes(data=True)
                if data["role"] == role
            )
        )
        if profile == "unit":
            values = [1] * len(nodes)
        else:
            values = [levels[index % len(levels)] for index in range(len(nodes))]
            random.Random(
                f"{seed}|protect-cost-profile|{COST_PROFILE_VERSION}|{role}"
            ).shuffle(values)
        for node, value in zip(nodes, values):
            equipment.nodes[node]["protect_cost"] = float(value)
            assigned[node] = value

    role_multisets = {}
    for role in ROLES:
        counts = Counter(
            assigned[node]
            for node, data in equipment.nodes(data=True)
            if data["role"] == role
        )
        role_multisets[role] = [
            [level, counts[level]] for level in sorted(counts)
        ]
    all_counts = Counter(assigned.values())
    metadata = {
        "name": profile,
        "version": COST_PROFILE_VERSION,
        "cost_seed": seed,
        "integer_levels": [1] if profile == "unit" else list(levels),
        "cost_multiset": [
            [level, all_counts[level]] for level in sorted(all_counts)
        ],
        "cost_multiset_by_role": role_multisets,
        "assignment_namespace": "seed|protect-cost-profile|version|role",
        "topology_oblivious": True,
        "capacity_oblivious": True,
        "attack_cost_oblivious": True,
        "outcome_screening": False,
    }
    equipment.graph["protect_cost_profile"] = metadata
    synthetic = deepcopy(equipment.graph.get("synthetic", {}))
    if synthetic:
        synthetic["protect_cost"] = None
        synthetic["protect_cost_profile"] = metadata
        equipment.graph["synthetic"] = synthetic
    return validate_graph(equipment)
