"""Controlled synthetic four-role equipment-network generation.

The generator is deliberately independent of RMCD objectives and functional
outcomes.  A seed fixes node attributes first; the requested topology then
only changes how a fixed number of legal role-interface edges are arranged.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite
from numbers import Integral, Real
import random
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .model import ROLES, validate_graph


TOPOLOGIES: tuple[str, ...] = ("centralized", "modular", "distributed")
INTERFACES: tuple[tuple[str, str], ...] = (("S", "C"), ("C", "L"), ("L", "E"))
INTERFACE_KEYS: tuple[str, ...] = tuple(f"{left}-{right}" for left, right in INTERFACES)
DEFAULT_INTERFACE_DENSITY = 0.20


def _gini(values: Sequence[int]) -> float:
    ordered = sorted(int(value) for value in values)
    total = sum(ordered)
    if not ordered or total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (len(ordered) * total) - (
        len(ordered) + 1
    ) / len(ordered)


def _require_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {result}.")
    return result


def _require_positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number; got {value!r}.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}.")
    return result


def _normalize_role_sizes(role_sizes: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(role_sizes, Mapping):
        raise ValueError("role_sizes must be a mapping with keys S, C, L, and E.")
    unknown = sorted(set(role_sizes) - set(ROLES))
    missing = [role for role in ROLES if role not in role_sizes]
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError("role_sizes must contain exactly S, C, L, and E (" + ", ".join(details) + ").")
    return {
        role: _require_integer(f"role_sizes[{role!r}]", role_sizes[role], minimum=1)
        for role in ROLES
    }


def _normalize_one_capacity_sequence(
    values: Sequence[int], *, label: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a non-empty sequence of positive integers.")
    levels = tuple(
        _require_integer(f"{label}[{index}]", value, minimum=1)
        for index, value in enumerate(values)
    )
    if not levels:
        raise ValueError(f"{label} must not be empty.")
    return levels


def _normalize_capacity_levels(
    capacity_levels: Sequence[int] | Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    if isinstance(capacity_levels, Mapping):
        raw = _normalize_role_mapping("capacity_levels", capacity_levels)
        return {
            role: _normalize_one_capacity_sequence(
                raw[role], label=f"capacity_levels[{role!r}]"
            )
            for role in ROLES
        }
    common = _normalize_one_capacity_sequence(
        capacity_levels, label="capacity_levels"
    )
    return {role: common for role in ROLES}


def _normalize_role_mapping(
    name: str, values: Mapping[str, object]
) -> dict[str, object]:
    unknown = sorted(set(values) - set(ROLES))
    missing = [role for role in ROLES if role not in values]
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"{name} must contain exactly {list(ROLES)} (" + ", ".join(details) + ")."
        )
    return {role: values[role] for role in ROLES}


def _normalize_interface_mapping(
    name: str, values: Mapping[str, object]
) -> dict[str, object]:
    unknown = sorted(set(values) - set(INTERFACE_KEYS))
    missing = [key for key in INTERFACE_KEYS if key not in values]
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"{name} must contain exactly {list(INTERFACE_KEYS)} (" + ", ".join(details) + ")."
        )
    return {key: values[key] for key in INTERFACE_KEYS}


def _allocate_total_edge_budget(total: int, maxima: Sequence[int]) -> tuple[int, ...]:
    minimum_total = len(maxima)
    maximum_total = sum(maxima)
    if not minimum_total <= total <= maximum_total:
        raise ValueError(
            f"edge_budget must be between {minimum_total} and {maximum_total}; got {total}."
        )

    counts = [1] * len(maxima)
    remaining = total - minimum_total
    while remaining:
        residual = [maximum - count for maximum, count in zip(maxima, counts)]
        residual_total = sum(residual)
        raw = [remaining * value / residual_total for value in residual]
        additions = [min(value, int(share)) for value, share in zip(residual, raw)]
        if sum(additions) == 0:
            index = max(
                (index for index, value in enumerate(residual) if value),
                key=lambda index: (raw[index], residual[index], -index),
            )
            additions[index] = 1
        for index, addition in enumerate(additions):
            counts[index] += addition
            remaining -= addition
    return tuple(counts)


def _interface_edge_counts(
    role_sizes: Mapping[str, int],
    edge_budget: int | Mapping[str, int] | None,
    interface_density: float | Mapping[str, float] | None,
) -> tuple[dict[str, int], dict[str, object]]:
    if edge_budget is not None and interface_density is not None:
        raise ValueError("Specify edge_budget or interface_density, not both.")

    maxima = {
        f"{left}-{right}": role_sizes[left] * role_sizes[right]
        for left, right in INTERFACES
    }
    if edge_budget is not None:
        if isinstance(edge_budget, Mapping):
            raw = _normalize_interface_mapping("edge_budget", edge_budget)
            counts = {
                key: _require_integer(f"edge_budget[{key!r}]", raw[key], minimum=1)
                for key in INTERFACE_KEYS
            }
            for key, count in counts.items():
                if count > maxima[key]:
                    raise ValueError(
                        f"edge_budget[{key!r}] exceeds the {maxima[key]} possible edges."
                    )
            control = {"mode": "per_interface_edge_budget", "requested": dict(counts)}
        else:
            total = _require_integer("edge_budget", edge_budget, minimum=len(INTERFACES))
            allocated = _allocate_total_edge_budget(total, tuple(maxima[key] for key in INTERFACE_KEYS))
            counts = dict(zip(INTERFACE_KEYS, allocated))
            control = {"mode": "total_edge_budget", "requested": total}
        return counts, control

    density = DEFAULT_INTERFACE_DENSITY if interface_density is None else interface_density
    if isinstance(density, Mapping):
        raw_density = _normalize_interface_mapping("interface_density", density)
        densities = {
            key: _require_positive_real(f"interface_density[{key!r}]", raw_density[key])
            for key in INTERFACE_KEYS
        }
        control = {"mode": "per_interface_density", "requested": dict(densities)}
    else:
        value = _require_positive_real("interface_density", density)
        densities = {key: value for key in INTERFACE_KEYS}
        control = {"mode": "uniform_interface_density", "requested": value}
    for key, value in densities.items():
        if value > 1.0:
            raise ValueError(f"interface_density[{key!r}] must not exceed 1; got {value}.")
    counts = {
        key: max(1, min(maxima[key], int(maxima[key] * densities[key] + 0.5)))
        for key in INTERFACE_KEYS
    }
    return counts, control


def _seeded_random(seed: int, *parts: str) -> random.Random:
    return random.Random("|".join((str(seed), *parts)))


def _append_unique(
    ordered: list[tuple[str, str]], seen: set[tuple[str, str]], edge: tuple[str, str]
) -> None:
    if edge not in seen:
        ordered.append(edge)
        seen.add(edge)


def _centralized_edges(
    left_nodes: Sequence[str], right_nodes: Sequence[str], count: int, seed: int, key: str
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    rng = _seeded_random(seed, "centralized", key)
    left = list(left_nodes)
    right = list(right_nodes)
    rng.shuffle(left)
    rng.shuffle(right)
    left_hub, right_hub = left[0], right[0]

    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index in range(max(len(left), len(right))):
        if index < len(left):
            _append_unique(ordered, seen, (left[index], right_hub))
        if index < len(right):
            _append_unique(ordered, seen, (left_hub, right[index]))

    left_rank = {node: index for index, node in enumerate(left)}
    right_rank = {node: index for index, node in enumerate(right)}
    remainder = [
        (source, target)
        for source in left
        for target in right
        if (source, target) not in seen
    ]
    tie_break = {edge: rng.random() for edge in remainder}
    remainder.sort(
        key=lambda edge: (
            left_rank[edge[0]] + right_rank[edge[1]],
            max(left_rank[edge[0]], right_rank[edge[1]]),
            tie_break[edge],
        )
    )
    ordered.extend(remainder)
    return ordered[:count], {"left_hub": left_hub, "right_hub": right_hub}


def _distributed_edges(
    left_nodes: Sequence[str], right_nodes: Sequence[str], count: int, seed: int, key: str
) -> list[tuple[str, str]]:
    rng = _seeded_random(seed, "distributed", key)
    left = list(left_nodes)
    right = list(right_nodes)
    rng.shuffle(left)
    rng.shuffle(right)
    ordered = [
        (source, right[(position + offset) % len(right)])
        for offset in range(len(right))
        for position, source in enumerate(left)
    ]
    return ordered[:count]


def _module_assignments(
    nodes_by_role: Mapping[str, Sequence[str]], seed: int, module_count: int
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for role in ROLES:
        ordered = list(nodes_by_role[role])
        _seeded_random(seed, "modular", "assignment", role).shuffle(ordered)
        assignments.update({node: index % module_count for index, node in enumerate(ordered)})
    return assignments


def _modular_edges(
    left_nodes: Sequence[str],
    right_nodes: Sequence[str],
    count: int,
    seed: int,
    key: str,
    assignments: Mapping[str, int],
    module_count: int,
) -> list[tuple[str, str]]:
    rng = _seeded_random(seed, "modular", key)
    candidates = [(source, target) for source in left_nodes for target in right_nodes]
    tie_break = {edge: rng.random() for edge in candidates}

    def module_distance(edge: tuple[str, str]) -> int:
        difference = abs(assignments[edge[0]] - assignments[edge[1]])
        return min(difference, module_count - difference)

    candidates.sort(key=lambda edge: (module_distance(edge), tie_break[edge]))
    return candidates[:count]


def _nodes_and_capacities(
    role_sizes: Mapping[str, int],
    capacity_levels: Mapping[str, Sequence[int]],
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    nodes_by_role = {
        role: [f"{role}{index:04d}" for index in range(role_sizes[role])]
        for role in ROLES
    }
    assigned: dict[str, int] = {}
    for role in ROLES:
        nodes = nodes_by_role[role]
        levels = capacity_levels[role]
        capacities = [levels[index % len(levels)] for index in range(len(nodes))]
        _seeded_random(seed, "capacities", role).shuffle(capacities)
        assigned.update(dict(zip(nodes, capacities)))
    return nodes_by_role, assigned


def _guaranteed_chain_nodes(
    nodes_by_role: Mapping[str, Sequence[str]],
    topology: str,
    seed: int,
    assignments: Mapping[str, int],
) -> dict[str, str]:
    """Choose one role-complete chain before any outcome is evaluated."""

    selected: dict[str, str] = {}
    for role in ROLES:
        candidates = list(nodes_by_role[role])
        if topology == "modular":
            candidates = [node for node in candidates if assignments[node] == 0]
        rng = _seeded_random(seed, "guaranteed-chain", role)
        selected[role] = candidates[rng.randrange(len(candidates))]
    return selected


def _force_interface_edge(
    edges: Sequence[tuple[str, str]],
    forced: tuple[str, str],
    count: int,
) -> list[tuple[str, str]]:
    """Insert a preregistered chain edge while preserving the edge budget."""

    if forced in edges:
        return list(edges)
    retained = [edge for edge in edges if edge != forced]
    return [forced, *retained[: count - 1]]


def generate_equipment_network(
    topology: str,
    role_sizes: Mapping[str, int],
    *,
    seed: int = 0,
    edge_budget: int | Mapping[str, int] | None = None,
    interface_density: float | Mapping[str, float] | None = None,
    capacity_levels: Sequence[int] | Mapping[str, Sequence[int]] = (1, 2, 3),
    attack_cost: float = 1.0,
    protect_cost: float = 1.0,
) -> nx.DiGraph:
    """Generate one validated synthetic S->C->L->E equipment network.

    ``edge_budget`` may be a total integer or an exact mapping for ``S-C``,
    ``C-L``, and ``L-E``.  Alternatively, ``interface_density`` may be one
    density or a mapping for those interfaces. ``capacity_levels`` may be one
    common sequence or an exact S/C/L/E mapping. Capacity assignment is made
    within roles and is topology-neutral, so matched calls with the same seed
    have identical node-level capacities.
    """

    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}; got {topology!r}.")
    normalized_seed = _require_integer("seed", seed, minimum=0)
    sizes = _normalize_role_sizes(role_sizes)
    levels = _normalize_capacity_levels(capacity_levels)
    unit_attack_cost = _require_positive_real("attack_cost", attack_cost)
    unit_protect_cost = _require_positive_real("protect_cost", protect_cost)
    edge_counts, edge_control = _interface_edge_counts(
        sizes, edge_budget, interface_density
    )
    nodes_by_role, capacities = _nodes_and_capacities(sizes, levels, normalized_seed)

    graph = nx.DiGraph()
    for role in ROLES:
        for node in nodes_by_role[role]:
            graph.add_node(
                node,
                role=role,
                capacity=capacities[node],
                attack_cost=unit_attack_cost,
                protect_cost=unit_protect_cost,
            )

    module_count = min(3, *(sizes[role] for role in ROLES))
    assignments = (
        _module_assignments(nodes_by_role, normalized_seed, module_count)
        if topology == "modular"
        else {}
    )
    guaranteed_chain = _guaranteed_chain_nodes(
        nodes_by_role, topology, normalized_seed, assignments
    )
    central_hubs: dict[str, object] = {}
    for (left_role, right_role), key in zip(INTERFACES, INTERFACE_KEYS):
        left_nodes = nodes_by_role[left_role]
        right_nodes = nodes_by_role[right_role]
        count = edge_counts[key]
        if topology == "centralized":
            edges, hubs = _centralized_edges(
                left_nodes, right_nodes, count, normalized_seed, key
            )
            central_hubs[key] = hubs
        elif topology == "modular":
            edges = _modular_edges(
                left_nodes,
                right_nodes,
                count,
                normalized_seed,
                key,
                assignments,
                module_count,
            )
        else:
            edges = _distributed_edges(
                left_nodes, right_nodes, count, normalized_seed, key
            )
        edges = _force_interface_edge(
            edges,
            (guaranteed_chain[left_role], guaranteed_chain[right_role]),
            count,
        )
        graph.add_edges_from(edges)

    if assignments:
        nx.set_node_attributes(graph, assignments, "synthetic_module")

    capacity_counts = Counter(capacities.values())
    capacity_counts_by_role = {
        role: Counter(capacities[node] for node in nodes_by_role[role])
        for role in ROLES
    }
    common_levels = len({levels[role] for role in ROLES}) == 1
    degrees = [int(degree) for _, degree in graph.to_undirected().degree()]
    weak_components = list(nx.weakly_connected_components(graph))
    role_incident_coverage = {
        role: sum(graph.degree(node) > 0 for node in nodes_by_role[role])
        / len(nodes_by_role[role])
        for role in ROLES
    }
    within_module_edges = (
        sum(
            assignments[source] == assignments[target]
            for source, target in graph.edges
        )
        if assignments
        else 0
    )
    graph.graph["synthetic"] = {
        "generator": "rmcd_f.synthetic.generate_equipment_network",
        "generator_version": 1,
        "topology": topology,
        "seed": normalized_seed,
        "role_sizes": dict(sizes),
        "edge_control": edge_control,
        "interface_edge_counts": dict(edge_counts),
        "total_edge_count": sum(edge_counts.values()),
        "realized_interface_density": {
            key: edge_counts[key]
            / (sizes[left_role] * sizes[right_role])
            for (left_role, right_role), key in zip(INTERFACES, INTERFACE_KEYS)
        },
        "capacity_levels": list(levels["S"]) if common_levels else None,
        "capacity_levels_by_role": {
            role: list(levels[role]) for role in ROLES
        },
        "capacity_multiset": [
            [level, capacity_counts[level]] for level in sorted(capacity_counts)
        ],
        "capacity_multiset_by_role": {
            role: [
                [level, capacity_counts_by_role[role][level]]
                for level in sorted(capacity_counts_by_role[role])
            ]
            for role in ROLES
        },
        "attack_cost": unit_attack_cost,
        "protect_cost": unit_protect_cost,
        "cost_mode": "unit" if unit_attack_cost == unit_protect_cost == 1.0 else "uniform",
        "module_count": module_count if topology == "modular" else None,
        "central_hubs": central_hubs if topology == "centralized" else None,
        "guaranteed_role_chain": [guaranteed_chain[role] for role in ROLES],
        "positive_motif_capacity_by_construction": True,
        "generation_attempts": 1,
        "generation_failures": 0,
        "isolated_node_count": sum(degree == 0 for degree in degrees),
        "degree_gini": _gini(degrees),
        "largest_weak_component_fraction": max(map(len, weak_components))
        / graph.number_of_nodes(),
        "role_incident_coverage": role_incident_coverage,
        "within_module_edge_ratio": (
            within_module_edges / graph.number_of_edges() if assignments else None
        ),
        "screening_or_outcome_selection": False,
    }
    return validate_graph(graph)


def generate_equipment_networks(
    role_sizes: Mapping[str, int],
    *,
    seeds: Iterable[int],
    topologies: Iterable[str] = TOPOLOGIES,
    edge_budget: int | Mapping[str, int] | None = None,
    interface_density: float | Mapping[str, float] | None = None,
    capacity_levels: Sequence[int] | Mapping[str, Sequence[int]] = (1, 2, 3),
    attack_cost: float = 1.0,
    protect_cost: float = 1.0,
) -> dict[tuple[str, int], nx.DiGraph]:
    """Generate a matched topology-by-seed batch without outcome screening."""

    seed_values = tuple(seeds)
    topology_values = tuple(topologies)
    if not seed_values:
        raise ValueError("seeds must contain at least one seed.")
    if not topology_values:
        raise ValueError("topologies must contain at least one topology.")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must not contain duplicates.")
    if len(set(topology_values)) != len(topology_values):
        raise ValueError("topologies must not contain duplicates.")

    return {
        (topology, int(seed)): generate_equipment_network(
            topology,
            role_sizes,
            seed=seed,
            edge_budget=edge_budget,
            interface_density=interface_density,
            capacity_levels=capacity_levels,
            attack_cost=attack_cost,
            protect_cost=protect_cost,
        )
        for seed in seed_values
        for topology in topology_values
    }


def role_interface_degree_preserving_rewire(
    graph: nx.DiGraph,
    *,
    seed: int,
    swaps_per_edge: float = 5.0,
    max_attempt_factor: int = 50,
) -> nx.DiGraph:
    """Randomize endpoints within each legal interface while preserving degrees.

    A swap ``u->v, x->y`` becomes ``u->y, x->v``.  Consequently every
    node's directed degree, every S-C/C-L/L-E edge count, and all node
    attributes remain exactly fixed.  No RMCD outcome is inspected while
    accepting swaps.
    """

    equipment = validate_graph(graph)
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be nonnegative.")
    attempts_factor = _require_integer(
        "max_attempt_factor", max_attempt_factor, minimum=1
    )
    rewired = equipment.copy()
    rng = _seeded_random(_require_integer("seed", seed, minimum=0), "role-null")
    swap_audit: dict[str, dict[str, int]] = {}

    for left_role, right_role in INTERFACES:
        key = f"{left_role}-{right_role}"
        edges = [
            (source, target)
            for source, target in rewired.edges
            if rewired.nodes[source]["role"] == left_role
            and rewired.nodes[target]["role"] == right_role
        ]
        target_swaps = int(round(len(edges) * float(swaps_per_edge)))
        max_attempts = max(1, target_swaps * attempts_factor)
        swaps = 0
        attempts = 0
        while swaps < target_swaps and attempts < max_attempts and len(edges) >= 2:
            attempts += 1
            first_index, second_index = rng.sample(range(len(edges)), 2)
            source_a, target_a = edges[first_index]
            source_b, target_b = edges[second_index]
            if source_a == source_b or target_a == target_b:
                continue
            proposed_a = (source_a, target_b)
            proposed_b = (source_b, target_a)
            if proposed_a in rewired.edges or proposed_b in rewired.edges:
                continue
            rewired.remove_edge(source_a, target_a)
            rewired.remove_edge(source_b, target_b)
            rewired.add_edge(*proposed_a)
            rewired.add_edge(*proposed_b)
            edges[first_index] = proposed_a
            edges[second_index] = proposed_b
            swaps += 1
        swap_audit[key] = {
            "edge_count": len(edges),
            "requested_swaps": target_swaps,
            "successful_swaps": swaps,
            "attempts": attempts,
        }

    rewired.graph = dict(equipment.graph)
    rewired.graph["role_degree_null"] = {
        "generator": "rmcd_f.synthetic.role_interface_degree_preserving_rewire",
        "generator_version": 1,
        "seed": int(seed),
        "swaps_per_edge": float(swaps_per_edge),
        "max_attempt_factor": attempts_factor,
        "interfaces": swap_audit,
        "outcome_screening": False,
    }
    return validate_graph(rewired)
