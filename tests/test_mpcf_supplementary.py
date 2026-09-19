from __future__ import annotations

import csv

import networkx as nx
import pytest

from rmcd_f.cost_profiles import apply_protection_cost_profile
from rmcd_f.synthetic import generate_equipment_network
from rmcd_f.mpcf_capacity_evaluation import evaluate_role_motif_capacity_thresholds
from rmcd_f.task_path_fortification import analyze_mpcf_optimal_family, fortified_attack_costs
from rmcd_f.mpcf_supplementary import (
    brute_force_capacity_attack,
    exchange_groups,
    protection_scores,
    select_score_policy,
)
from scripts.analyze_mpcf_stratified_explainability import _role_rows
from scripts.run_mpcf_small_exhaustive_validation import main as exhaustive_main


def _parallel_chains(count: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    for index in range(count):
        path = []
        for role in ("S", "C", "L", "E"):
            node = f"{role}{index}"
            graph.add_node(
                node,
                role=role,
                capacity=1,
                attack_cost=1.0,
                protect_cost=1.0,
                removable=True,
            )
            path.append(node)
        graph.add_edges_from(zip(path, path[1:]))
    return graph


def test_mpcf_normalized_family_enumerates_all_exchange_alternatives() -> None:
    graph = generate_equipment_network(
        "centralized",
        {"S": 2, "C": 2, "L": 2, "E": 2},
        seed=1,
        interface_density=0.65,
        capacity_levels=(1,),
    )
    family = analyze_mpcf_optimal_family(
        graph, 4.0, enumerate_limit=20, time_limit_per_solve=30.0
    )

    assert family.optimum_margin == pytest.approx(3.0)
    assert family.minimum_protection_cost == pytest.approx(4.0)
    assert family.membership_exact
    assert family.enumeration_complete
    assert len(family.enumerated_sets) == 7
    assert family.mandatory_nodes == frozenset()
    assert family.possible_nodes == frozenset(graph.nodes)
    assert all(len(chosen) == 4 for chosen in family.enumerated_sets)
    assert len(exchange_groups(family.enumerated_sets)) == 4


def test_balanced_protection_cost_overlay_preserves_graph_and_attack_costs() -> None:
    graph = _parallel_chains(3)
    overlaid = apply_protection_cost_profile(
        graph, "balanced-heterogeneous", cost_seed=11
    )

    assert set(overlaid.edges) == set(graph.edges)
    assert {
        node: overlaid.nodes[node]["attack_cost"] for node in overlaid
    } == {node: graph.nodes[node]["attack_cost"] for node in graph}
    assert set(overlaid.nodes[node]["protect_cost"] for node in overlaid) == {
        1.0,
        2.0,
        4.0,
    }
    assert overlaid.graph["protect_cost_profile"]["outcome_screening"] is False


def test_score_cost_and_knapsack_policies_are_budget_feasible() -> None:
    graph = _parallel_chains(2)
    for index, node in enumerate(sorted(graph.nodes)):
        graph.nodes[node]["protect_cost"] = (1.0, 2.0, 4.0)[index % 3]
    scores = protection_scores(graph, "betweenness")
    for policy in ("raw", "per_cost", "prefix_knapsack"):
        selected = select_score_policy(
            graph, scores, 4.0, method=f"test-{policy}", policy=policy
        )
        assert selected.protection_cost <= 4.0 + 1e-9
        assert selected.protection_set <= frozenset(graph.nodes)


def test_capacity_attack_enumeration_matches_exact_response() -> None:
    graph = _parallel_chains(2)
    protected = {"S0"}
    evaluation = evaluate_role_motif_capacity_thresholds(
        graph, protected, remaining_fractions=(0.0, 0.5), solver="rmcd-exact"
    )
    fortified = graph.copy()
    for node, value in fortified_attack_costs(graph, protected).items():
        fortified.nodes[node]["attack_cost"] = value

    for point in evaluation.points:
        brute = brute_force_capacity_attack(
            fortified, point.threshold_k, max_nodes=8
        )
        assert brute.complete
        assert brute.attack_cost == pytest.approx(point.attack_cost)


def test_small_exhaustive_script_writes_both_certificate_tables(tmp_path) -> None:
    output = tmp_path / "tiny"
    assert exhaustive_main(
        [
            "--sizes", "8",
            "--capacity-sizes", "8",
            "--topologies", "distributed",
            "--seeds", "11",
            "--budgets", "1",
            "--capacity-fractions", "0,0.5",
            "--workers", "1",
            "--output", str(output),
        ]
    ) == 0
    with (output / "small_defense_exhaustive_validation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        defense = list(csv.DictReader(handle))
    with (output / "small_capacity_exhaustive_validation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        capacity = list(csv.DictReader(handle))
    assert defense and all(row["gate_pass"] == "1" for row in defense)
    assert capacity and all(row["gate_pass"] == "1" for row in capacity)
    assert (output / "_COMPLETE").is_file()


def test_role_distribution_emits_explicit_zero_rows() -> None:
    protected = [
        {
            "graph_fingerprint": "frozen",
            "method": "MPCF-Exact",
            "topology": "centralized",
            "scale_tier": "n42",
            "budget_fraction": "0.05",
            "role": "C",
        }
    ]
    rows = _role_rows(
        protected,
        [{"graph_fingerprint": "frozen", "scale_tier": "n42"}],
    )

    by_role = {row["role"]: row for row in rows}
    assert set(by_role) == {"S", "C", "L", "E"}
    assert by_role["C"]["selected_fraction"] == pytest.approx(1.0)
    assert by_role["S"]["selected_count"] == 0
    assert by_role["L"]["selected_fraction"] == pytest.approx(0.0)
