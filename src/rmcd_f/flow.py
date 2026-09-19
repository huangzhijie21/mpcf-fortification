"""Role-expanded flow network and verifiable motif-capacity certificates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Hashable, Iterable, Mapping

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from .model import (
    GraphValidationError,
    NodeId,
    ensure_node_subset,
    stable_node_key,
    stable_nodes,
    validate_graph,
)


@dataclass(frozen=True)
class FlowNode:
    """Collision-free internal identifier in a node-split graph."""

    kind: str
    original: NodeId | None = None


SOURCE = FlowNode("source")
SINK = FlowNode("sink")


@dataclass(frozen=True)
class RoleFlowNetwork:
    """Node-split representation of the fixed S-C-L-E role chain."""

    equipment_graph: nx.DiGraph
    graph: nx.DiGraph
    source: FlowNode
    sink: FlowNode
    node_in: Mapping[NodeId, FlowNode]
    node_out: Mapping[NodeId, FlowNode]
    capacity_arcs: Mapping[NodeId, tuple[FlowNode, FlowNode]]
    infinite_arcs: tuple[tuple[FlowNode, FlowNode], ...]
    u_inf_flow: int
    attacked: frozenset[NodeId]


@dataclass(frozen=True)
class FlowPath:
    """An integral amount routed through one complete role motif."""

    nodes: tuple[NodeId, ...]
    amount: int


@dataclass(frozen=True)
class CutCertificate:
    """A minimum-cut certificate in equipment-node coordinates."""

    source_side: frozenset[FlowNode]
    sink_side: frozenset[FlowNode]
    cut_nodes: tuple[NodeId, ...]
    cut_capacity: int
    crossing_arcs: tuple[tuple[FlowNode, FlowNode, int], ...]


@dataclass(frozen=True)
class FlowCertificate:
    """Integral max-flow value, path decomposition, and matching min cut."""

    value: int
    attacked: frozenset[NodeId]
    paths: tuple[FlowPath, ...]
    cut: CutCertificate
    u_inf_flow: int


@dataclass(frozen=True)
class MotifPackingCertificate:
    """Explicit motif-packing ILP result used as an independent small-graph oracle."""

    value: int
    motifs: tuple[tuple[NodeId, NodeId, NodeId, NodeId], ...]
    multiplicities: tuple[int, ...]
    optimal: bool
    message: str


def _internal_node_key(node: FlowNode) -> tuple[str, str, str]:
    return (node.kind, repr(node.original), type(node.original).__qualname__)


def build_role_flow_network(
    graph: nx.Graph, attacked: Iterable[NodeId] = ()
) -> RoleFlowNetwork:
    """Build the frozen role-expanded node-split graph."""

    equipment = validate_graph(graph)
    attacked_set = ensure_node_subset(equipment, attacked, label="attacked")
    u_inf_flow = 1 + sum(int(equipment.nodes[node]["capacity"]) for node in equipment)

    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from((SOURCE, SINK))
    node_in: dict[NodeId, FlowNode] = {}
    node_out: dict[NodeId, FlowNode] = {}
    capacity_arcs: dict[NodeId, tuple[FlowNode, FlowNode]] = {}
    infinite_arcs: list[tuple[FlowNode, FlowNode]] = []

    for node in stable_nodes(equipment.nodes):
        incoming = FlowNode("in", node)
        outgoing = FlowNode("out", node)
        node_in[node] = incoming
        node_out[node] = outgoing
        capacity_arcs[node] = (incoming, outgoing)
        capacity = 0 if node in attacked_set else int(equipment.nodes[node]["capacity"])
        flow_graph.add_edge(incoming, outgoing, capacity=capacity, kind="node", node=node)

        role = equipment.nodes[node]["role"]
        if role == "S":
            flow_graph.add_edge(SOURCE, incoming, capacity=u_inf_flow, kind="infinite")
            infinite_arcs.append((SOURCE, incoming))
        elif role == "E":
            flow_graph.add_edge(outgoing, SINK, capacity=u_inf_flow, kind="infinite")
            infinite_arcs.append((outgoing, SINK))

    for source, target in sorted(
        equipment.edges,
        key=lambda edge: (stable_node_key(edge[0]), stable_node_key(edge[1])),
    ):
        arc = (node_out[source], node_in[target])
        flow_graph.add_edge(*arc, capacity=u_inf_flow, kind="infinite")
        infinite_arcs.append(arc)

    return RoleFlowNetwork(
        equipment_graph=equipment,
        graph=flow_graph,
        source=SOURCE,
        sink=SINK,
        node_in=node_in,
        node_out=node_out,
        capacity_arcs=capacity_arcs,
        infinite_arcs=tuple(infinite_arcs),
        u_inf_flow=u_inf_flow,
        attacked=attacked_set,
    )


def _decompose_flow(
    network: RoleFlowNetwork,
    flow: Mapping[FlowNode, Mapping[FlowNode, float]],
    *,
    expected_value: int,
) -> tuple[FlowPath, ...]:
    """Extract a certificate packing from a solver flow dictionary.

    Some NetworkX flow implementations can retain positive, noncontributing
    preflow branches in the returned dictionary.  A local greedy walk may enter
    such a branch even though a complete positive-flow source-sink path still
    exists.  We therefore search the whole positive-flow support for each path
    and stop after extracting the independently certified max-flow value.
    """

    if expected_value < 0:
        raise ValueError("expected_value must be non-negative.")
    remaining: dict[FlowNode, dict[FlowNode, int]] = {}
    for source, targets in flow.items():
        positive: dict[FlowNode, int] = {}
        for target, value in targets.items():
            numeric = float(value)
            rounded = int(round(numeric))
            if abs(numeric - rounded) > 1e-7:
                raise RuntimeError(
                    "Maximum-flow backend returned a non-integral arc flow."
                )
            if rounded > 0:
                positive[target] = rounded
        if positive:
            remaining[source] = positive

    paths: list[FlowPath] = []
    decomposed_value = 0

    while decomposed_value < expected_value:
        parent: dict[FlowNode, FlowNode | None] = {network.source: None}
        frontier = deque((network.source,))
        while frontier and network.sink not in parent:
            current = frontier.popleft()
            candidates = sorted(
                (
                    target
                    for target, value in remaining.get(current, {}).items()
                    if value > 0 and target not in parent
                ),
                key=_internal_node_key,
            )
            for target in candidates:
                parent[target] = current
                frontier.append(target)

        if network.sink not in parent:
            raise RuntimeError(
                "Integral max flow could not be decomposed into source-sink "
                f"paths: extracted {decomposed_value} of {expected_value}."
            )

        reversed_path = [network.sink]
        current = network.sink
        while current != network.source:
            predecessor = parent[current]
            assert predecessor is not None
            reversed_path.append(predecessor)
            current = predecessor
        internal_path = list(reversed(reversed_path))

        amount = min(
            expected_value - decomposed_value,
            min(
                remaining[source][target]
                for source, target in zip(internal_path, internal_path[1:])
            ),
        )
        for source, target in zip(internal_path, internal_path[1:]):
            remaining[source][target] -= amount

        equipment_nodes = tuple(
            node.original
            for node in internal_path
            if node.kind == "in" and node.original is not None
        )
        if len(equipment_nodes) != 4:
            raise RuntimeError(
                "Flow decomposition produced a path outside the frozen four-role motif."
            )
        paths.append(FlowPath(nodes=equipment_nodes, amount=amount))
        decomposed_value += amount

    return tuple(paths)


def motif_capacity(
    graph: nx.Graph, attacked: Iterable[NodeId] = ()
) -> FlowCertificate:
    """Compute integer role-motif capacity and a minimum-cut certificate."""

    network = build_role_flow_network(graph, attacked)
    value, flow = nx.maximum_flow(
        network.graph,
        network.source,
        network.sink,
        capacity="capacity",
        flow_func=nx.algorithms.flow.edmonds_karp,
    )
    cut_value, partition = nx.minimum_cut(
        network.graph,
        network.source,
        network.sink,
        capacity="capacity",
        flow_func=nx.algorithms.flow.edmonds_karp,
    )
    source_side, sink_side = partition

    crossing: list[tuple[FlowNode, FlowNode, int]] = []
    for source in sorted(source_side, key=_internal_node_key):
        for target in sorted(network.graph.successors(source), key=_internal_node_key):
            if target in sink_side:
                capacity = int(network.graph[source][target]["capacity"])
                crossing.append((source, target, capacity))

    cut_nodes = stable_nodes(
        node
        for node, (incoming, outgoing) in network.capacity_arcs.items()
        if incoming in source_side and outgoing in sink_side
    )
    integral_value = int(round(value))
    integral_cut = int(round(cut_value))
    if integral_value != integral_cut:
        raise RuntimeError(
            f"Max-flow/min-cut mismatch: flow={integral_value}, cut={integral_cut}."
        )

    cut = CutCertificate(
        source_side=frozenset(source_side),
        sink_side=frozenset(sink_side),
        cut_nodes=cut_nodes,
        cut_capacity=integral_cut,
        crossing_arcs=tuple(crossing),
    )
    return FlowCertificate(
        value=integral_value,
        attacked=network.attacked,
        paths=_decompose_flow(network, flow, expected_value=integral_value),
        cut=cut,
        u_inf_flow=network.u_inf_flow,
    )


def enumerate_role_motifs(
    graph: nx.Graph,
) -> tuple[tuple[NodeId, NodeId, NodeId, NodeId], ...]:
    """Enumerate all S-C-L-E paths for independent small-graph validation."""

    equipment = validate_graph(graph)
    motifs: list[tuple[NodeId, NodeId, NodeId, NodeId]] = []
    sensors = stable_nodes(
        node for node, data in equipment.nodes(data=True) if data["role"] == "S"
    )
    for sensor in sensors:
        for command in stable_nodes(equipment.successors(sensor)):
            for relay in stable_nodes(equipment.successors(command)):
                for executor in stable_nodes(equipment.successors(relay)):
                    motifs.append((sensor, command, relay, executor))
    return tuple(motifs)


def explicit_motif_packing(
    graph: nx.Graph, attacked: Iterable[NodeId] = ()
) -> MotifPackingCertificate:
    """Solve the explicit integer motif-packing model for validation."""

    equipment = validate_graph(graph)
    attacked_set = ensure_node_subset(equipment, attacked, label="attacked")
    motifs = enumerate_role_motifs(equipment)
    if not motifs:
        return MotifPackingCertificate(0, (), (), True, "No role motif exists.")

    nodes = stable_nodes(equipment.nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(motifs)), dtype=float)
    for motif_index, motif in enumerate(motifs):
        for node in motif:
            matrix[node_index[node], motif_index] = 1.0
    capacities = np.array(
        [
            0.0 if node in attacked_set else float(equipment.nodes[node]["capacity"])
            for node in nodes
        ]
    )
    result = milp(
        c=-np.ones(len(motifs), dtype=float),
        integrality=np.ones(len(motifs), dtype=int),
        bounds=Bounds(np.zeros(len(motifs)), np.full(len(motifs), np.inf)),
        constraints=LinearConstraint(csr_matrix(matrix), -np.inf, capacities),
        options={"presolve": True},
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"Explicit motif-packing ILP failed: {result.message}")
    multiplicities = tuple(max(0, int(round(value))) for value in result.x)
    value = int(sum(multiplicities))
    return MotifPackingCertificate(value, motifs, multiplicities, True, str(result.message))


def verify_flow_certificate(graph: nx.Graph, certificate: FlowCertificate) -> bool:
    """Recompute and fully validate the flow decomposition and cut witness."""

    equipment = validate_graph(graph)
    try:
        attacked = ensure_node_subset(equipment, certificate.attacked, label="attacked")
    except ValueError:
        return False
    recomputed = motif_capacity(equipment, attacked)
    if certificate.value != recomputed.value:
        return False
    if not _verify_cut_certificate(
        equipment,
        attacked,
        certificate.cut,
        expected_value=certificate.value,
        expected_u_inf=certificate.u_inf_flow,
    ):
        return False

    usage = {node: 0 for node in equipment}
    path_total = 0
    for path in certificate.paths:
        if (
            isinstance(path.amount, bool)
            or not isinstance(path.amount, Integral)
            or path.amount <= 0
            or len(path.nodes) != 4
        ):
            return False
        if any(node not in equipment or node in attacked for node in path.nodes):
            return False
        roles = tuple(equipment.nodes[node]["role"] for node in path.nodes)
        if roles != ("S", "C", "L", "E"):
            return False
        if any(
            not equipment.has_edge(source, target)
            for source, target in zip(path.nodes, path.nodes[1:])
        ):
            return False
        path_total += int(path.amount)
        for node in path.nodes:
            usage[node] += int(path.amount)

    if path_total != certificate.value:
        return False
    return all(
        amount <= int(equipment.nodes[node]["capacity"])
        for node, amount in usage.items()
    )


def _verify_cut_certificate(
    graph: nx.Graph,
    attacked: Iterable[NodeId],
    cut: CutCertificate,
    *,
    expected_value: int,
    expected_u_inf: int | None = None,
) -> bool:
    """Validate one minimum-cut witness without requiring a canonical cut.

    Networks with several minimum cuts may legitimately return different
    partitions across solver/library versions.  The certificate is therefore
    checked intrinsically against its own partition and the independently
    computed max-flow value.
    """

    try:
        network = build_role_flow_network(graph, attacked)
    except (GraphValidationError, ValueError):
        return False
    if expected_u_inf is not None and expected_u_inf != network.u_inf_flow:
        return False
    source_side = frozenset(cut.source_side)
    sink_side = frozenset(cut.sink_side)
    all_nodes = frozenset(network.graph.nodes)
    if (
        source_side & sink_side
        or source_side | sink_side != all_nodes
        or network.source not in source_side
        or network.sink not in sink_side
    ):
        return False

    crossing: list[tuple[FlowNode, FlowNode, int]] = []
    for source in sorted(source_side, key=_internal_node_key):
        for target in sorted(network.graph.successors(source), key=_internal_node_key):
            if target in sink_side:
                crossing.append(
                    (
                        source,
                        target,
                        int(network.graph[source][target]["capacity"]),
                    )
                )
    crossing_tuple = tuple(crossing)
    cut_nodes = stable_nodes(
        node
        for node, (incoming, outgoing) in network.capacity_arcs.items()
        if incoming in source_side and outgoing in sink_side
    )
    cut_capacity = sum(capacity for _, _, capacity in crossing_tuple)
    return (
        cut.crossing_arcs == crossing_tuple
        and cut.cut_nodes == cut_nodes
        and cut.cut_capacity == cut_capacity
        and cut_capacity == expected_value
    )
