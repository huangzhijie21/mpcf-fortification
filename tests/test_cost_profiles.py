from __future__ import annotations

from collections import Counter

from rmcd_f import (
    apply_attack_cost_profile,
    full_instance_fingerprint,
    generate_equipment_network,
    official_projection_fingerprint,
    role_structure_fingerprint,
)


def _base(topology: str):
    return generate_equipment_network(
        topology,
        {"S": 3, "C": 3, "L": 3, "E": 3},
        seed=11,
        interface_density=0.4,
        capacity_levels=(1, 2, 3),
    )


def test_heterogeneous_profile_preserves_structure_and_projection():
    base = _base("centralized")
    unit = apply_attack_cost_profile(base, "unit", cost_seed=11)
    heterogeneous = apply_attack_cost_profile(
        base, "balanced-heterogeneous", cost_seed=11
    )

    assert role_structure_fingerprint(unit) == role_structure_fingerprint(heterogeneous)
    assert official_projection_fingerprint(unit) == official_projection_fingerprint(
        heterogeneous
    )
    assert full_instance_fingerprint(unit) != full_instance_fingerprint(heterogeneous)
    assert any(
        unit.nodes[node]["attack_cost"]
        != heterogeneous.nodes[node]["attack_cost"]
        for node in unit
    )


def test_cost_assignment_is_topology_oblivious_and_role_balanced():
    assignments = []
    for topology in ("centralized", "modular", "distributed"):
        graph = apply_attack_cost_profile(
            _base(topology), "balanced-heterogeneous", cost_seed=22
        )
        assignments.append(
            {node: graph.nodes[node]["attack_cost"] for node in graph}
        )
        for role in ("S", "C", "L", "E"):
            counts = Counter(
                int(graph.nodes[node]["attack_cost"])
                for node, data in graph.nodes(data=True)
                if data["role"] == role
            )
            assert counts == Counter({1: 1, 2: 1, 4: 1})

    assert assignments[0] == assignments[1] == assignments[2]


def test_cost_profile_is_deterministic():
    base = _base("distributed")
    first = apply_attack_cost_profile(
        base, "balanced-heterogeneous", cost_seed=33
    )
    second = apply_attack_cost_profile(
        base, "balanced-heterogeneous", cost_seed=33
    )
    assert full_instance_fingerprint(first) == full_instance_fingerprint(second)
