"""Unit tests for the three theoretical properties of ``MPCF-Greedy``.

The revision does not convert greedy into an exact algorithm.  It pins the
properties the manuscript actually claims, plus the minimal counterexample that
shows one-step exact-marginal greedy can be strictly suboptimal:

1. every greedy step keeps the protection set budget-feasible;
2. the defended margin ``kappa(P)`` is monotone non-decreasing along the trace;
3. the procedure selects at most ``|V_R|`` nodes before terminating;
4. on a three-removable-node instance, greedy attains 3 while the exact solver
   attains 4.
"""

from __future__ import annotations

import networkx as nx
import pytest

from rmcd_f.model import protection_cost, stable_nodes
from rmcd_f.operational_motif import build_path_closed_system
from rmcd_f.synthetic import TOPOLOGIES, generate_equipment_network
from rmcd_f.task_path_fortification import (
    solve_mpcf_exact,
    solve_mpcf_greedy,
)


def counterexample_graph() -> nx.DiGraph:
    """Three removable nodes: ``c0``, ``c1`` (command) and ``l0`` (relay).

    The task boundaries ``s`` and ``t`` are fixed non-removable endpoints, and
    the interface is complete, so both ``s -> c0 -> l0 -> t`` and
    ``s -> c1 -> l0 -> t`` are complete task paths.
    """

    graph = nx.DiGraph()
    graph.add_node("s", role="S", capacity=1, attack_cost=1.0, protect_cost=1.0, removable=False)
    graph.add_node("t", role="E", capacity=1, attack_cost=1.0, protect_cost=1.0, removable=False)
    graph.add_node("c0", role="C", capacity=1, attack_cost=1.0, protect_cost=1.0)
    graph.add_node("c1", role="C", capacity=1, attack_cost=1.0, protect_cost=1.0)
    graph.add_node("l0", role="L", capacity=1, attack_cost=3.0, protect_cost=1.0)
    graph.add_edge("s", "c0")
    graph.add_edge("s", "c1")
    graph.add_edge("c0", "l0")
    graph.add_edge("c1", "l0")
    graph.add_edge("l0", "t")
    return graph


#: Uplifts for the counterexample: fortifying ``c0`` adds 1, ``c1`` adds 2 and
#: ``l0`` adds 1 to the attack cost of the respective node.
COUNTEREXAMPLE_UPLIFTS = {"c0": 1.0, "c1": 2.0, "l0": 1.0}


def practice_instances():
    """Small frozen equipment networks used for the property checks."""

    instances = []
    for topology in TOPOLOGIES:
        graph = generate_equipment_network(
            topology, {"S": 4, "C": 3, "L": 3, "E": 4}, seed=11, edge_budget=30
        )
        instances.append((topology, graph))
    return instances


# ---------------------------------------------------------------------------
# 1. budget feasibility at every step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology,graph", practice_instances())
def test_every_greedy_step_respects_the_budget(topology: str, graph: nx.DiGraph) -> None:
    system = build_path_closed_system(graph)
    total = sum(
        float(graph.nodes[node]["protect_cost"]) for node in system.removable_nodes
    )
    for fraction in (0.05, 0.10, 0.20):
        budget = fraction * total
        result = solve_mpcf_greedy(graph, budget)
        for step in result.trace:
            spent = protection_cost(graph, step.protection_set)
            assert spent <= budget + 1e-9, (
                f"{topology}: step {step.iteration} spent {spent} > budget {budget}"
            )
        assert result.protection_cost <= budget + 1e-9
        assert result.protection_set <= frozenset(system.removable_nodes)


# ---------------------------------------------------------------------------
# 2. monotonicity of kappa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology,graph", practice_instances())
def test_greedy_margin_is_monotone_non_decreasing(topology: str, graph: nx.DiGraph) -> None:
    system = build_path_closed_system(graph)
    total = sum(
        float(graph.nodes[node]["protect_cost"]) for node in system.removable_nodes
    )
    for fraction in (0.05, 0.10, 0.20):
        result = solve_mpcf_greedy(graph, fraction * total)
        margins = [result.undefended_margin] + [
            step.oracle_margin for step in result.trace
        ]
        for previous, current in zip(margins, margins[1:]):
            assert current >= previous - 1e-9, (
                f"{topology}: kappa decreased from {previous} to {current}"
            )
        if result.trace:
            assert result.defended_margin == pytest.approx(
                result.trace[-1].oracle_margin
            )


# ---------------------------------------------------------------------------
# 3. termination bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology,graph", practice_instances())
def test_greedy_selects_at_most_all_removable_nodes(topology: str, graph: nx.DiGraph) -> None:
    system = build_path_closed_system(graph)
    removable = len(system.removable_nodes)
    total = sum(
        float(graph.nodes[node]["protect_cost"]) for node in system.removable_nodes
    )
    # A budget large enough to cover every removable node cannot make the
    # procedure run longer than the number of removable nodes.
    result = solve_mpcf_greedy(graph, total)
    assert len(result.trace) <= removable
    assert len(result.protection_set) <= removable
    assert len({step.selected_node for step in result.trace}) == len(result.trace)


def test_greedy_stops_when_no_node_is_affordable() -> None:
    graph = counterexample_graph()
    result = solve_mpcf_greedy(graph, 0.0)
    assert result.protection_set == frozenset()
    assert result.trace == ()
    assert result.defended_margin == result.undefended_margin


# ---------------------------------------------------------------------------
# 4. the three-node counterexample
# ---------------------------------------------------------------------------


def test_three_node_counterexample_greedy_objective_is_three() -> None:
    """Greedy attains 3 on the three-removable-node instance."""

    graph = counterexample_graph()
    result = solve_mpcf_greedy(graph, 2.0, uplifts=COUNTEREXAMPLE_UPLIFTS)
    assert result.defended_margin == pytest.approx(3.0)
    assert result.undefended_margin == pytest.approx(2.0)
    assert stable_nodes(result.protection_set) == ("c0", "l0")
    assert result.protection_cost == pytest.approx(2.0)
    assert result.protection_cost <= 2.0 + 1e-9


def test_three_node_counterexample_exact_objective_is_four() -> None:
    """The exact solver attains 4 on the same instance and budget."""

    graph = counterexample_graph()
    result = solve_mpcf_exact(graph, 2.0, uplifts=COUNTEREXAMPLE_UPLIFTS)
    assert result.optimal
    assert result.defended_margin == pytest.approx(4.0)
    assert stable_nodes(result.protection_set) == ("c1", "l0")
    assert result.protection_cost == pytest.approx(2.0)


def test_greedy_is_strictly_suboptimal_on_the_counterexample() -> None:
    """The gap between the heuristic and the optimum is exactly one unit."""

    graph = counterexample_graph()
    greedy = solve_mpcf_greedy(graph, 2.0, uplifts=COUNTEREXAMPLE_UPLIFTS)
    exact = solve_mpcf_exact(graph, 2.0, uplifts=COUNTEREXAMPLE_UPLIFTS)
    assert greedy.defended_margin == pytest.approx(3.0)
    assert exact.defended_margin == pytest.approx(4.0)
    assert greedy.defended_margin < exact.defended_margin
    assert greedy.protection_cost <= 2.0 + 1e-9
    # Both sets are budget-feasible; only the objective differs.
    assert exact.protection_cost <= 2.0 + 1e-9
    # Greedy never claims global optimality.
    assert not greedy.optimal
    assert greedy.solver_log.certificate_mode == "open"


def test_counterexample_removable_node_count_is_three() -> None:
    """The counterexample really is a minimal three-removable-node instance."""

    system = build_path_closed_system(counterexample_graph())
    assert len(system.removable_nodes) == 3
    assert stable_nodes(system.removable_nodes) == ("c0", "c1", "l0")
    assert len(system.motifs) == 2
