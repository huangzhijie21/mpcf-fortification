from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest


ROOT = Path(__file__).resolve().parents[1]


def equipment_graph(nodes, edges) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node, role, capacity, attack_cost, protect_cost in nodes:
        graph.add_node(
            node,
            role=role,
            capacity=capacity,
            attack_cost=attack_cost,
            protect_cost=protect_cost,
        )
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def two_disjoint_paths() -> nx.DiGraph:
    nodes = []
    edges = []
    for index in (1, 2):
        path = [f"S{index}", f"C{index}", f"L{index}", f"E{index}"]
        nodes.extend(
            (node, role, 1, 1.0, 1.0)
            for node, role in zip(path, ("S", "C", "L", "E"))
        )
        edges.extend(zip(path, path[1:]))
    return equipment_graph(nodes, edges)


@pytest.fixture
def shared_command_graph() -> nx.DiGraph:
    return equipment_graph(
        [
            ("S1", "S", 1, 4.0, 2.0),
            ("S2", "S", 1, 4.0, 2.0),
            ("C0", "C", 2, 1.0, 1.0),
            ("L1", "L", 1, 4.0, 2.0),
            ("L2", "L", 1, 4.0, 2.0),
            ("E1", "E", 1, 4.0, 2.0),
            ("E2", "E", 1, 4.0, 2.0),
        ],
        [
            ("S1", "C0"),
            ("S2", "C0"),
            ("C0", "L1"),
            ("C0", "L2"),
            ("L1", "E1"),
            ("L2", "E2"),
        ],
    )
