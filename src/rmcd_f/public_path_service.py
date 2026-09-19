"""Source-backed public path-service adapters for MPCF generalization.

The adapters do not infer military roles.  They freeze one OD service, retain
only source-declared path sequences, and require the declared family to equal
every path enabled by the resulting directed graph.  Candidate selection uses
source-side demand or graph-size fields only and never consults MPCF outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from .directed_task_thread import DIRECTED_TASK_THREAD_MAPPING_VERSION
from .public_data import SndlibDemand, SndlibNetwork, parse_sndlib_native


MAX_DECLARED_PATHS = 100


@dataclass(frozen=True)
class PublicPathCandidate:
    dataset_id: str
    task_id: str
    source_node: str
    target_node: str
    paths: tuple[tuple[str, tuple[str, ...], str], ...]
    source_score: float
    node_count: int
    edge_count: int
    node_connectivity: int
    selection_rank: int = 0

    @property
    def path_count(self) -> int:
        return len(self.paths)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_graph(paths: Iterable[Sequence[str]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for path in paths:
        nx.add_path(graph, path)
    return graph


def exact_path_closure(
    paths: Iterable[Sequence[str]], source: str, target: str
) -> tuple[bool, int, int, int]:
    """Return exact-closure status and source-only nontriviality statistics."""

    declared = {tuple(path) for path in paths}
    if not declared:
        return False, 0, 0, 0
    graph = _path_graph(declared)
    enumerated: set[tuple[str, ...]] = set()
    for path in nx.all_simple_paths(graph, source, target):
        enumerated.add(tuple(path))
        if len(enumerated) > len(declared):
            return False, graph.number_of_nodes(), graph.number_of_edges(), 0
    if enumerated != declared:
        return False, graph.number_of_nodes(), graph.number_of_edges(), 0
    connectivity = int(nx.node_connectivity(graph, source, target))
    return True, graph.number_of_nodes(), graph.number_of_edges(), connectivity


def _ordered_sndlib_nodes(
    demand: SndlibDemand,
    link_ids: Sequence[str],
    links: Mapping[str, Any],
) -> tuple[str, ...] | None:
    current = demand.source
    ordered = [current]
    for link_id in link_ids:
        link = links.get(link_id)
        if link is None:
            return None
        if link.source == current:
            current = link.target
        elif link.target == current:
            current = link.source
        else:
            return None
        ordered.append(current)
    return tuple(ordered) if current == demand.target else None


def sndlib_candidates(path: str | Path) -> tuple[PublicPathCandidate, ...]:
    """Find closure-exact SNDlib OD demands using all declared admissible paths."""

    network: SndlibNetwork = parse_sndlib_native(path)
    links = {link.link_id: link for link in network.links}
    by_demand: dict[str, list[Any]] = defaultdict(list)
    for admissible in network.admissible_paths:
        by_demand[admissible.demand_id].append(admissible)
    candidates: list[PublicPathCandidate] = []
    for demand in network.demands:
        records: list[tuple[str, tuple[str, ...], str]] = []
        invalid = False
        for admissible in sorted(
            by_demand.get(demand.demand_id, ()), key=lambda item: item.path_id
        ):
            ordered = _ordered_sndlib_nodes(demand, admissible.link_ids, links)
            if ordered is None:
                invalid = True
                break
            locator = (
                f"ADMISSIBLE_PATHS demand_id={demand.demand_id} "
                f"path_id={admissible.path_id}; links={','.join(admissible.link_ids)}"
            )
            records.append((admissible.path_id, ordered, locator))
        unique = {record[1] for record in records}
        if (
            invalid
            or not 4 <= len(unique) <= MAX_DECLARED_PATHS
            or any(len(sequence) < 3 for sequence in unique)
        ):
            continue
        exact, nodes, edges, connectivity = exact_path_closure(
            unique, demand.source, demand.target
        )
        if not exact or connectivity < 2:
            continue
        candidates.append(
            PublicPathCandidate(
                dataset_id=f"sndlib_{network.name}",
                task_id=demand.demand_id,
                source_node=demand.source,
                target_node=demand.target,
                paths=tuple(records),
                source_score=float(demand.demand_value),
                node_count=nodes,
                edge_count=edges,
                node_connectivity=connectivity,
            )
        )
    candidates.sort(
        key=lambda item: (-item.source_score, item.task_id, item.source_node, item.target_node)
    )
    return tuple(
        PublicPathCandidate(**{**item.__dict__, "selection_rank": index})
        for index, item in enumerate(candidates, start=1)
    )


def _representative_gtfs_paths(
    archive: str | Path,
) -> tuple[
    list[tuple[str, tuple[str, ...], tuple[str, str, str, str]]],
    dict[str, str],
    str,
]:
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        required = {"route_patterns.txt", "stop_times.txt", "stops.txt", "feed_info.txt"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"GTFS archive is missing required files: {missing}.")
        representatives: dict[str, tuple[str, str, str, str]] = {}
        with bundle.open("route_patterns.txt") as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            )
            for row in reader:
                trip = str(row.get("representative_trip_id", "")).strip()
                if not trip:
                    continue
                representatives[trip] = (
                    str(row.get("route_pattern_id", "")),
                    str(row.get("route_id", "")),
                    str(row.get("direction_id", "")),
                    str(row.get("route_pattern_name", "")),
                )
        sequences: dict[str, list[tuple[int, str]]] = defaultdict(list)
        with bundle.open("stop_times.txt") as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            )
            for row in reader:
                trip = str(row.get("trip_id", ""))
                if trip in representatives:
                    sequences[trip].append(
                        (int(row["stop_sequence"]), str(row["stop_id"]))
                    )
        stops: dict[str, str] = {}
        with bundle.open("stops.txt") as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            )
            for row in reader:
                stop = str(row.get("stop_id", ""))
                if stop:
                    stops[stop] = str(row.get("stop_name", stop))
        with bundle.open("feed_info.txt") as raw:
            rows = list(
                csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            )
        feed_version = str(rows[0].get("feed_version", "published feed")) if rows else "published feed"
    patterns: list[tuple[str, tuple[str, ...], tuple[str, str, str, str]]] = []
    for trip, records in sequences.items():
        sequence = tuple(stop for _, stop in sorted(records))
        if len(sequence) >= 3:
            patterns.append((trip, sequence, representatives[trip]))
    patterns.sort(key=lambda item: (item[2][0], item[0]))
    return patterns, stops, feed_version


def gtfs_candidates(path: str | Path) -> tuple[PublicPathCandidate, ...]:
    """Find closure-exact one-seat OD path families in one MBTA-style GTFS feed."""

    patterns, _stops, _version = _representative_gtfs_paths(path)
    endpoint_counts: Counter[tuple[str, str]] = Counter()
    for _trip, sequence, _metadata in patterns:
        seen: set[tuple[str, str]] = set()
        for left in range(len(sequence) - 2):
            for right in range(left + 2, len(sequence)):
                seen.add((sequence[left], sequence[right]))
        endpoint_counts.update(seen)
    eligible_pairs = {
        pair for pair, count in endpoint_counts.items() if 4 <= count <= 200
    }
    grouped: dict[
        tuple[str, str], dict[tuple[str, ...], tuple[str, str, str, str, str]]
    ] = defaultdict(dict)
    for trip, sequence, metadata in patterns:
        positions: dict[str, list[int]] = defaultdict(list)
        for index, stop in enumerate(sequence):
            positions[stop].append(index)
        for source, target in eligible_pairs:
            if source not in positions or target not in positions:
                continue
            chosen: tuple[str, ...] | None = None
            for left in positions[source]:
                rights = [right for right in positions[target] if right > left + 1]
                if rights:
                    chosen = sequence[left : min(rights) + 1]
                    break
            if chosen is None:
                continue
            route_pattern, route_id, direction, name = metadata
            existing = grouped[(source, target)].get(chosen)
            record = (trip, route_pattern, route_id, direction, name)
            if existing is None or record < existing:
                grouped[(source, target)][chosen] = record
    candidates: list[PublicPathCandidate] = []
    for (source, target), records in grouped.items():
        if not 4 <= len(records) <= MAX_DECLARED_PATHS:
            continue
        exact, nodes, edges, connectivity = exact_path_closure(
            records, source, target
        )
        if not exact or connectivity < 2:
            continue
        paths = []
        for sequence, metadata in sorted(records.items(), key=lambda item: item[1]):
            trip, route_pattern, route_id, direction, name = metadata
            path_id = f"{route_pattern}::{trip}"
            locator = (
                f"route_patterns.txt route_pattern_id={route_pattern} "
                f"representative_trip_id={trip}; stop_times.txt ordered subsequence "
                f"{source}->{target}; route_id={route_id}; direction_id={direction}; "
                f"name={name}"
            )
            paths.append((path_id, sequence, locator))
        candidates.append(
            PublicPathCandidate(
                dataset_id="mbta_gtfs",
                task_id=f"one_seat_{source}_to_{target}",
                source_node=source,
                target_node=target,
                paths=tuple(paths),
                source_score=float(nodes),
                node_count=nodes,
                edge_count=edges,
                node_connectivity=connectivity,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.node_count,
            -item.path_count,
            -item.node_connectivity,
            -item.edge_count,
            item.source_node,
            item.target_node,
        )
    )
    return tuple(
        PublicPathCandidate(**{**item.__dict__, "selection_rank": index})
        for index, item in enumerate(candidates, start=1)
    )


def _edge_locators(candidate: PublicPathCandidate) -> dict[tuple[str, str], list[str]]:
    locators: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _path_id, sequence, locator in candidate.paths:
        for edge in zip(sequence, sequence[1:]):
            if locator not in locators[edge]:
                locators[edge].append(locator)
    return locators


def build_mapping(
    candidate: PublicPathCandidate,
    *,
    source_file: str | Path,
    source_family: str,
    independence_basis: str,
    task_completion: str,
    task_mode: str,
    operating_condition: str,
    node_labels: Mapping[str, str] | None = None,
    evidence_member_prefix: str = "",
) -> dict[str, Any]:
    """Build one v2 mapping from a source-only selected public OD candidate."""

    source = Path(source_file)
    source_name = source.name
    labels = dict(node_labels or {})
    evidence = lambda locator: [{"source_path": source_name, "locator": locator}]
    all_nodes = sorted(
        {node for _path_id, sequence, _locator in candidate.paths for node in sequence}
    )
    edge_locators = _edge_locators(candidate)
    prefix = f"{evidence_member_prefix} " if evidence_member_prefix else ""
    nodes = []
    for node in all_nodes:
        boundary = node in {candidate.source_node, candidate.target_node}
        stage = (
            "OD service origin"
            if node == candidate.source_node
            else "OD service destination"
            if node == candidate.target_node
            else "intermediate path-service component"
        )
        nodes.append(
            {
                "id": node,
                "component_instance_id": f"{candidate.dataset_id}::{node}",
                "node_kind": "task_boundary" if boundary else "implemented_component",
                "task_stage": stage,
                "capacity": 1,
                "attack_cost": 1.0,
                "protect_cost": 1.0,
                "removable": not boundary,
                "public_label": labels.get(node, node),
                "task_stage_basis": (
                    f"{prefix}published OD endpoint"
                    if boundary
                    else f"{prefix}source-declared path sequence"
                ),
                "attack_cost_basis": "unit-cost structural sensitivity",
                "protect_cost_basis": "unit-cost structural sensitivity",
                "evidence": evidence(f"{prefix}node/component id={node}"),
            }
        )
    edges = [
        {
            "source": source_node,
            "target": target_node,
            "evidence": evidence(" | ".join(locators)),
        }
        for (source_node, target_node), locators in sorted(edge_locators.items())
    ]
    declared_paths = [
        {
            "path_id": path_id,
            "task_mode_id": task_mode,
            "operating_condition": operating_condition,
            "ordered_node_ids": list(sequence),
            "evidence": evidence(locator),
        }
        for path_id, sequence, locator in candidate.paths
    ]
    return {
        "schema_version": DIRECTED_TASK_THREAD_MAPPING_VERSION,
        "dataset_id": candidate.dataset_id,
        "source_family": source_family,
        "mapping_version": "1.0.0",
        "statistically_independent_testbed": True,
        "independence_basis": independence_basis,
        "evidence_domain": "public_path_service",
        "screening_or_outcome_selection": False,
        "candidate_selection_rank": candidate.selection_rank,
        "scope_statement": (
            "ONE BOUNDED PUBLIC OD PATH-SERVICE MODE; SOURCE-DECLARED PATHS "
            "EXACTLY EQUAL EVERY PATH ENABLED BY THE FROZEN DIRECTED EDGE SET"
        ),
        "mapping_policy": {
            "node_unit": "implemented_component_with_boundary_terminals",
            "edge_semantics": "source-declared directed consecutive service-path contribution",
            "no_inferred_interfaces": True,
            "cost_policy": "unit_cost_sensitivity",
        },
        "source_files": [{"path": source_name, "sha256": sha256_file(source)}],
        "task": {
            "task_id": candidate.task_id,
            "completion_statement": task_completion,
            "source_nodes": [candidate.source_node],
            "target_nodes": [candidate.target_node],
            "evidence": evidence(
                f"source-declared OD service {candidate.source_node}->{candidate.target_node}"
            ),
        },
        "nodes": nodes,
        "edges": edges,
        "declared_legal_paths": declared_paths,
    }


def write_mapping(mapping: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
