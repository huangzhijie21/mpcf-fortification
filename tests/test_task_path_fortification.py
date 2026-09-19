from __future__ import annotations

import networkx as nx
import pytest

from rmcd_f.model import SolveStatus, SolverInfo

from rmcd_f.protection_baselines import (
    LOCAL_PROTECTION_METHODS,
    select_budget_feasible_ranking,
    select_local_protection,
)
from rmcd_f.task_path_fortification import (
    brute_force_mpcf,
    certify_mpcf_objective_bounds,
    evaluate_task_path_fortification,
    solve_mpcf_cut_generation,
    solve_mpcf_exact,
    solve_mpcf_greedy,
    mpcf_objective_cost_quantum,
    verify_mpcf_certificate,
    verify_mpcf_objective_certificate,
)
from rmcd_f.synthetic import generate_equipment_network


def _node(graph: nx.DiGraph, name: str, role: str) -> None:
    graph.add_node(
        name,
        role=role,
        capacity=1,
        attack_cost=1.0,
        protect_cost=1.0,
        removable=True,
    )


def shared_command_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for name, role in (
        ("S1", "S"),
        ("S2", "S"),
        ("C1", "C"),
        ("L1", "L"),
        ("L2", "L"),
        ("E1", "E"),
        ("E2", "E"),
    ):
        _node(graph, name, role)
    graph.add_edges_from(
        (
            ("S1", "C1"),
            ("S2", "C1"),
            ("C1", "L1"),
            ("C1", "L2"),
            ("L1", "E1"),
            ("L2", "E2"),
        )
    )
    return graph


def test_mpcf_exact_cg_and_enumeration_close_the_same_optimum() -> None:
    graph = shared_command_graph()
    exact = solve_mpcf_exact(graph, 1.0)
    cg = solve_mpcf_cut_generation(graph, 1.0)
    brute = brute_force_mpcf(graph, 1.0)

    assert exact.optimal and cg.optimal and brute.optimal
    assert exact.undefended_margin == pytest.approx(1.0)
    assert exact.defended_margin == pytest.approx(2.0)
    assert exact.defended_margin == pytest.approx(cg.defended_margin)
    assert exact.defended_margin == pytest.approx(brute.defended_margin)
    assert exact.protection_set == frozenset({"C1"})
    assert cg.protection_set == frozenset({"C1"})
    assert verify_mpcf_certificate(exact, graph)
    assert verify_mpcf_certificate(cg, graph)


def test_integer_lattice_closes_subunit_time_limit_gap() -> None:
    quantum = mpcf_objective_cost_quantum(
        shared_command_graph(), fortification_multiplier=1.0
    )
    assert quantum.valid
    assert quantum.quantum == 1.0
    certificate = certify_mpcf_objective_bounds(
        176.0, 176.0, 176.932043, quantum
    )
    assert certificate.certified_optimal
    assert not certificate.raw_bounds_closed
    assert certificate.lattice_bounds_closed
    assert certificate.certificate_source == "OBJECTIVE_LATTICE_CLOSURE"
    assert certificate.lattice_upper_bound == 176.0


def test_integer_lattice_does_not_hide_a_three_unit_open_gap() -> None:
    quantum = mpcf_objective_cost_quantum(
        shared_command_graph(), fortification_multiplier=1.0
    )
    certificate = certify_mpcf_objective_bounds(
        182.0, 182.0, 185.0, quantum
    )
    assert not certificate.certified_optimal
    assert certificate.certificate_source == "OPEN_GAP"
    assert certificate.effective_absolute_gap == 3.0


def test_rational_uplifts_produce_a_fractional_quantum() -> None:
    quantum = mpcf_objective_cost_quantum(
        shared_command_graph(), fortification_multiplier=0.5
    )
    assert quantum.valid
    assert quantum.quantum == 0.5
    certificate = certify_mpcf_objective_bounds(
        4.5, 4.5, 4.99, quantum
    )
    assert certificate.certified_optimal


def test_lattice_certificate_verifier_accepts_a_time_limited_closed_result() -> None:
    result = solve_mpcf_exact(shared_command_graph(), 1.0)
    open_solver = SolverInfo(
        status=SolveStatus.TIME_LIMIT,
        optimal=False,
        objective=result.defended_margin,
        lower_bound=result.defended_margin,
        upper_bound=result.defended_margin + 0.9,
        runtime_seconds=result.solver.runtime_seconds,
    )
    replay = type(result)(
        **{**result.__dict__, "solver": open_solver}
    )
    assert verify_mpcf_certificate(replay, shared_command_graph())
    assert verify_mpcf_objective_certificate(replay, shared_command_graph())


def test_mpcf_is_finite_cost_uplift_not_invulnerability() -> None:
    graph = shared_command_graph()
    result = solve_mpcf_exact(graph, 1.0, fortification_multiplier=0.5)
    cg = solve_mpcf_cut_generation(
        graph, 1.0, fortification_multiplier=0.5
    )
    oracle = evaluate_task_path_fortification(
        graph, result.protection_set, fortification_multiplier=0.5
    )

    assert result.protection_set == frozenset({"C1"})
    assert result.defended_margin == pytest.approx(1.5)
    assert cg.optimal
    assert cg.defended_margin == pytest.approx(result.defended_margin)
    assert verify_mpcf_certificate(cg, graph)
    assert oracle.optimal and oracle.objective == pytest.approx(1.5)
    assert result.adaptive_cut


def test_zero_budget_returns_the_undefended_margin() -> None:
    graph = shared_command_graph()
    exact = solve_mpcf_exact(graph, 0.0)
    cg = solve_mpcf_cut_generation(graph, 0.0)

    assert exact.protection_set == frozenset()
    assert cg.protection_set == frozenset()
    assert exact.defended_margin == pytest.approx(exact.undefended_margin)
    assert cg.defended_margin == pytest.approx(cg.undefended_margin)


def test_greedy_is_feasible_and_does_not_claim_optimality() -> None:
    result = solve_mpcf_greedy(shared_command_graph(), 1.0)
    assert not result.optimal
    assert result.protection_cost <= result.budget
    assert result.defended_margin == pytest.approx(2.0)
    assert verify_mpcf_certificate(result, shared_command_graph())


def test_ranking_scan_skips_unaffordable_nodes_without_reordering() -> None:
    graph = shared_command_graph()
    graph.nodes["S1"]["protect_cost"] = 3.0
    graph.nodes["S2"]["protect_cost"] = 1.0
    selected = select_budget_feasible_ranking(
        graph,
        ("S1", "S2", "C1"),
        2.0,
        method="test",
    )
    assert selected.protection_sequence == ("S2", "C1")
    assert selected.skipped_unaffordable == 1
    assert selected.protection_cost == pytest.approx(2.0)


@pytest.mark.parametrize("method", LOCAL_PROTECTION_METHODS)
def test_every_local_baseline_returns_a_budget_feasible_set(method: str) -> None:
    result = select_local_protection(
        shared_command_graph(), method, 2.0, seed=11
    )
    assert result.method == method
    assert result.protection_cost <= 2.0 + 1e-9
    assert result.protection_set <= frozenset(shared_command_graph().nodes)


@pytest.mark.parametrize(
    ("topology", "seed"),
    (("centralized", 11), ("modular", 22), ("distributed", 33)),
)
def test_random_small_instances_match_exhaustive_defense(
    topology: str, seed: int
) -> None:
    graph = generate_equipment_network(
        topology,
        {"S": 2, "C": 2, "L": 2, "E": 2},
        seed=seed,
        interface_density=0.65,
        capacity_levels=(1,),
    )
    exact = solve_mpcf_exact(graph, 2.0)
    cg = solve_mpcf_cut_generation(graph, 2.0)
    brute = brute_force_mpcf(graph, 2.0, max_nodes=8)

    assert exact.optimal and cg.optimal and brute.optimal
    assert exact.defended_margin == pytest.approx(brute.defended_margin)
    assert cg.defended_margin == pytest.approx(brute.defended_margin)
    assert verify_mpcf_certificate(exact, graph)
    assert verify_mpcf_certificate(cg, graph)
