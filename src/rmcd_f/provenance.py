"""Stable instance identities used by paired and official RMCD studies."""

from __future__ import annotations

import hashlib
import json

import networkx as nx

from .model import NodeId, stable_node_key, stable_nodes, validate_graph


def _token(node: NodeId) -> tuple[str, str]:
    return (
        f"{type(node).__module__}.{type(node).__qualname__}",
        repr(node),
    )


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def role_structure_fingerprint(graph: nx.Graph) -> str:
    """Hash directed structure, node IDs, roles, and capacities, excluding costs."""

    equipment = validate_graph(graph)
    nodes = [
        {
            "id": _token(node),
            "role": str(equipment.nodes[node]["role"]),
            "capacity": int(equipment.nodes[node]["capacity"]),
        }
        for node in stable_nodes(equipment.nodes)
    ]
    edges = sorted(
        ((_token(left), _token(right)) for left, right in equipment.edges),
        key=repr,
    )
    return _digest({"directed": True, "nodes": nodes, "edges": edges})


def official_projection_fingerprint(graph: nx.Graph) -> str:
    """Hash exactly the simple undirected projection seen by official baselines."""

    if str(graph.graph.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        from .operational_motif import build_path_closed_system

        equipment = build_path_closed_system(graph).graph
    else:
        equipment = validate_graph(graph)
    nodes = [_token(node) for node in stable_nodes(equipment.nodes)]
    edges = sorted(
        (
            tuple(sorted((_token(left), _token(right)), key=repr))
            for left, right in equipment.to_undirected().edges
        ),
        key=repr,
    )
    return _digest({"directed": False, "nodes": nodes, "edges": edges})


def full_instance_fingerprint(graph: nx.Graph) -> str:
    """Hash structure and all attack/protection attributes affecting RMCD."""

    equipment = validate_graph(graph)
    nodes = [
        {
            "id": _token(node),
            "role": str(equipment.nodes[node]["role"]),
            "capacity": int(equipment.nodes[node]["capacity"]),
            "attack_cost": format(float(equipment.nodes[node]["attack_cost"]), ".17g"),
            "protect_cost": format(float(equipment.nodes[node]["protect_cost"]), ".17g"),
        }
        for node in stable_nodes(equipment.nodes)
    ]
    edges = sorted(
        ((_token(left), _token(right)) for left, right in equipment.edges),
        key=repr,
    )
    return _digest({"directed": True, "nodes": nodes, "edges": edges})
