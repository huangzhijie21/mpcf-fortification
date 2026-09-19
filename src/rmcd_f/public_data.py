"""Auditable adapters for public datasets used around the RMCD study.

Only a source-backed four-role mapping may produce an RMCD equipment graph.
Topology-only datasets are parsed for boundary audits and are deliberately not
coerced into the frozen S-C-L-E semantics.
"""

from __future__ import annotations

import ast
import bz2
import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from .model import ROLES, GraphValidationError, validate_graph


class PublicDataError(ValueError):
    """Raised when public source data or a declared mapping fails its audit."""


PUBLIC_PROVENANCE_SCHEME = "rmcd-public-provenance-v1"


@dataclass(frozen=True)
class SndlibNode:
    node_id: str
    longitude: float | None
    latitude: float | None


@dataclass(frozen=True)
class SndlibLink:
    link_id: str
    source: str
    target: str
    preinstalled_capacity: float
    routing_cost: float


@dataclass(frozen=True)
class SndlibDemand:
    demand_id: str
    source: str
    target: str
    routing_unit: float
    demand_value: float
    max_path_length: str


@dataclass(frozen=True)
class SndlibAdmissiblePath:
    demand_id: str
    path_id: str
    link_ids: tuple[str, ...]


@dataclass(frozen=True)
class SndlibNetwork:
    name: str
    nodes: tuple[SndlibNode, ...]
    links: tuple[SndlibLink, ...]
    demands: tuple[SndlibDemand, ...]
    admissible_paths: tuple[SndlibAdmissiblePath, ...]


@dataclass(frozen=True)
class CaidaAsInventory:
    nodes: int
    edges: int
    provider_customer_edges: int
    peer_edges: int


@dataclass(frozen=True)
class PeeringDbInventory:
    pages: int
    records: int
    unique_record_ids: int
    unique_asns: int


@dataclass(frozen=True)
class SndlibDynamicInventory:
    archive_name: str
    matrix_count: int
    audited_matrix_count: int
    node_count: int
    demand_count_min: int
    demand_count_max: int
    node_set_matches_static: bool


@dataclass(frozen=True)
class HaiMappingResult:
    graph: nx.DiGraph
    mapping: Mapping[str, Any]
    source_audit: tuple[Mapping[str, Any], ...]
    mapping_sha256: str


@dataclass(frozen=True)
class PublicSourceRegistryAudit:
    registry_id: str
    registry_version: str
    registry_sha256: str
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PublicCandidateAudit:
    """One preregistered decision about a public dataset's RMCD eligibility."""

    dataset_id: str
    source_family: str
    access_status: str
    physical_equipment: str
    role_separation: str
    legal_role_order: str
    directed_interfaces_documented: str
    comparable_costs: str
    statistical_unit: str
    rmcd_semantic_gate: str
    permitted_panel: str
    reason: str


_SECTION_RE = re.compile(r"^([A-Z_]+)\s*\(\s*$")
_NODE_RE = re.compile(
    r"^(\S+?)(?:\s+\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\))?\s*$"
)
_PAIR_RECORD_RE = re.compile(r"^(\S+)\s+\(\s*(\S+)\s+(\S+)\s*\)\s+(.*)$")


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_json(payload: Any) -> str:
    """Hash a JSON-compatible object using a stable, whitespace-free encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hai_source_bundle_sha256(source_audit: Iterable[Mapping[str, Any]]) -> str:
    """Hash the complete, ordered HAI source-evidence bundle."""

    records = [
        {
            "path": str(record["path"]),
            "format": str(record["format"]),
            "sha256": str(record["sha256"]).lower(),
            "node_count": int(record["node_count"]),
            "checksum_ok": bool(record["checksum_ok"]),
        }
        for record in source_audit
    ]
    records.sort(key=lambda record: (record["path"], record["format"]))
    return sha256_canonical_json(records)


def public_provenance_fingerprint(
    *,
    dataset_id: str,
    solver_instance_fingerprint: str,
    mapping_sha256: str,
    source_bundle_sha256: str,
) -> str:
    """Bind public sources, semantic mapping, and solved instance together."""

    return sha256_canonical_json(
        {
            "fingerprint_scheme": PUBLIC_PROVENANCE_SCHEME,
            "dataset_id": dataset_id,
            "solver_instance_fingerprint": solver_instance_fingerprint.lower(),
            "mapping_sha256": mapping_sha256.lower(),
            "source_bundle_sha256": source_bundle_sha256.lower(),
        }
    )


def audit_public_source_registry(
    registry_path: str | Path,
    dataset_root: str | Path,
) -> PublicSourceRegistryAudit:
    """Verify every acquired source against the frozen public-source registry."""

    registry_file = Path(registry_path)
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDataError(
            f"Could not parse public source registry {registry_file}: {exc}"
        ) from exc
    registry = _require_mapping(payload, label="public source registry")
    registry_id = registry.get("registry_id")
    registry_version = registry.get("registry_version")
    if not isinstance(registry_id, str) or not registry_id:
        raise PublicDataError("Public source registry requires a nonempty registry_id.")
    if not isinstance(registry_version, str) or not registry_version:
        raise PublicDataError("Public source registry requires a nonempty registry_version.")

    root = Path(dataset_root).resolve()
    records: list[Mapping[str, Any]] = []
    observed_paths: set[str] = set()
    for index, raw in enumerate(_require_list(registry.get("files"), label="files")):
        record = _require_mapping(raw, label=f"files[{index}]")
        required = ("dataset", "path", "size_bytes", "sha256", "source_url", "access")
        missing = [field for field in required if field not in record]
        if missing:
            raise PublicDataError(f"files[{index}] is missing fields: {missing}.")
        relative = record["path"]
        if not isinstance(relative, str) or not relative:
            raise PublicDataError(f"files[{index}].path must be nonempty.")
        normalized = Path(relative).as_posix()
        if normalized in observed_paths:
            raise PublicDataError(f"Duplicate public source path {normalized!r}.")
        observed_paths.add(normalized)
        source = (root / Path(relative)).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise PublicDataError(
                f"Public source path escapes the dataset root: {relative!r}."
            ) from exc
        if not source.is_file():
            raise PublicDataError(f"Required public source file is missing: {source}.")
        expected_size = record["size_bytes"]
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            raise PublicDataError(f"files[{index}].size_bytes must be an integer.")
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            raise PublicDataError(
                f"Public source size mismatch for {relative}: expected "
                f"{expected_size}, got {actual_size}."
            )
        expected_hash = record["sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise PublicDataError(f"files[{index}].sha256 must be a SHA-256 digest.")
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash.lower():
            raise PublicDataError(
                f"Public source checksum mismatch for {relative}: expected "
                f"{expected_hash.lower()}, got {actual_hash}."
            )
        records.append(
            {
                "dataset": str(record["dataset"]),
                "path": normalized,
                "size_bytes": actual_size,
                "sha256": actual_hash,
                "source_url": str(record["source_url"]),
                "access": str(record["access"]),
                "checksum_ok": True,
            }
        )
    if not records:
        raise PublicDataError("Public source registry contains no files.")
    return PublicSourceRegistryAudit(
        registry_id=registry_id,
        registry_version=registry_version,
        registry_sha256=sha256_file(registry_file),
        records=tuple(records),
    )


def audit_sha256_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Verify every file listed by a conventional SHA256SUMS manifest."""

    manifest = Path(manifest_path)
    root = Path(dataset_root).resolve()
    records: list[Mapping[str, Any]] = []
    observed: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicDataError(f"Could not read checksum manifest {manifest}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9A-Fa-f]{64})\s+\*?(.+)", line)
        if match is None:
            raise PublicDataError(
                f"Malformed SHA-256 manifest record at {manifest}:{line_number}."
            )
        expected, relative = match.group(1).lower(), match.group(2).strip()
        normalized = Path(relative).as_posix()
        if normalized in observed:
            raise PublicDataError(f"Duplicate checksum-manifest path {normalized!r}.")
        observed.add(normalized)
        source = (root / Path(relative)).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise PublicDataError(
                f"Checksum-manifest path escapes the dataset root: {relative!r}."
            ) from exc
        if not source.is_file():
            raise PublicDataError(f"Checksum-manifest file is missing: {source}.")
        actual = sha256_file(source)
        if actual != expected:
            raise PublicDataError(
                f"Checksum-manifest mismatch for {relative}: expected {expected}, got {actual}."
            )
        records.append(
            {
                "path": normalized,
                "size_bytes": source.stat().st_size,
                "sha256": actual,
                "checksum_ok": True,
            }
        )
    if not records:
        raise PublicDataError(f"Checksum manifest {manifest} contains no records.")
    return tuple(records)


def audit_sndlib_static_extraction(
    archive_path: str | Path,
    extracted_dir: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Bind every parsed SNDlib static file to its frozen archive member."""

    archive_file = Path(archive_path)
    extracted_root = Path(extracted_dir)
    records: list[Mapping[str, Any]] = []
    with tarfile.open(archive_file, mode="r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".txt")
            ),
            key=lambda member: member.name,
        )
        member_names = [Path(member.name).name for member in members]
        if len(member_names) != len(set(member_names)):
            raise PublicDataError(
                f"SNDlib archive {archive_file} has duplicate member basenames."
            )
        extracted_names = sorted(path.name for path in extracted_root.glob("*.txt"))
        if sorted(member_names) != extracted_names:
            raise PublicDataError(
                "SNDlib extracted-file set does not match the frozen static archive."
            )
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise PublicDataError(f"Could not read SNDlib member {member.name}.")
            member_hash = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                member_hash.update(block)
            extracted = extracted_root / Path(member.name).name
            extracted_hash = sha256_file(extracted)
            if extracted_hash != member_hash.hexdigest():
                raise PublicDataError(
                    f"Extracted SNDlib file differs from {member.name} in {archive_file}."
                )
            records.append(
                {
                    "archive_member": member.name,
                    "extracted_file": extracted.name,
                    "size_bytes": extracted.stat().st_size,
                    "sha256": extracted_hash,
                    "checksum_ok": True,
                }
            )
    if not records:
        raise PublicDataError(f"SNDlib archive {archive_file} contains no native files.")
    return tuple(records)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicDataError(f"{label} must be an object.")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PublicDataError(f"{label} must be a list.")
    return value


def load_hai_literal_graph(path: str | Path) -> Mapping[str, Any]:
    """Safely load HAIEnd's Python-literal node-link files.

    The upstream ``dcs_*.json`` files are Python literals rather than JSON.
    ``ast.literal_eval`` accepts their booleans and single quotes without ever
    executing source text.
    """

    source = Path(path)
    try:
        payload = ast.literal_eval(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        raise PublicDataError(f"Could not parse HAI literal graph {source}: {exc}") from exc
    root = _require_mapping(payload, label=f"HAI graph {source}")
    nodes = _require_list(root.get("nodes"), label=f"{source}.nodes")
    links = _require_list(root.get("links"), label=f"{source}.links")
    node_ids: set[str] = set()
    for index, record in enumerate(nodes):
        node = _require_mapping(record, label=f"{source}.nodes[{index}]")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise PublicDataError(f"{source}.nodes[{index}].id must be a nonempty string.")
        if node_id in node_ids:
            raise PublicDataError(f"Duplicate HAI node ID {node_id!r} in {source}.")
        node_ids.add(node_id)
    for index, record in enumerate(links):
        edge = _require_mapping(record, label=f"{source}.links[{index}]")
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            raise PublicDataError(f"{source}.links[{index}] references an unknown endpoint.")
    return root


def load_hai_json_graph(path: str | Path) -> Mapping[str, Any]:
    """Load and validate a strict-JSON HAI node-link graph."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDataError(f"Could not parse HAI JSON graph {source}: {exc}") from exc
    root = _require_mapping(payload, label=f"HAI graph {source}")
    nodes = _require_list(root.get("nodes"), label=f"{source}.nodes")
    links = _require_list(root.get("links"), label=f"{source}.links")
    node_ids: set[str] = set()
    for index, record in enumerate(nodes):
        node = _require_mapping(record, label=f"{source}.nodes[{index}]")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise PublicDataError(f"{source}.nodes[{index}].id must be a nonempty string.")
        if node_id in node_ids:
            raise PublicDataError(f"Duplicate HAI node ID {node_id!r} in {source}.")
        node_ids.add(node_id)
    for index, record in enumerate(links):
        edge = _require_mapping(record, label=f"{source}.links[{index}]")
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            raise PublicDataError(f"{source}.links[{index}] references an unknown endpoint.")
    return root


def _hai_source_node_ids(path: Path, source_format: str) -> set[str]:
    if source_format == "json-node-link":
        payload = load_hai_json_graph(path)
    elif source_format == "python-literal-node-link":
        payload = load_hai_literal_graph(path)
    elif source_format == "pdf":
        return set()
    else:
        raise PublicDataError(f"Unsupported HAI source format {source_format!r}.")
    return {str(record["id"]) for record in payload["nodes"]}


def hai_mapping_instance_ids(mapping_path: str | Path) -> tuple[str, ...]:
    """List declared HAI instance views without touching source data."""

    mapping_file = Path(mapping_path)
    try:
        payload = json.loads(mapping_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDataError(f"Could not parse HAI mapping {mapping_file}: {exc}") from exc
    mapping = _require_mapping(payload, label="HAI mapping")
    if "instances" not in mapping:
        dataset_id = mapping.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise PublicDataError("HAI mapping requires a nonempty dataset_id.")
        return (dataset_id,)
    instance_ids: list[str] = []
    for index, raw in enumerate(_require_list(mapping["instances"], label="instances")):
        record = _require_mapping(raw, label=f"instances[{index}]")
        instance_id = record.get("dataset_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise PublicDataError(f"instances[{index}].dataset_id must be nonempty.")
        if instance_id in instance_ids:
            raise PublicDataError(f"Duplicate HAI instance ID {instance_id!r}.")
        instance_ids.append(instance_id)
    if not instance_ids:
        raise PublicDataError("HAI mapping catalog contains no instances.")
    return tuple(instance_ids)


def _resolve_hai_catalog_mapping(
    mapping: Mapping[str, Any],
    instance_id: str | None,
) -> Mapping[str, Any]:
    if "instances" not in mapping:
        if instance_id is not None and instance_id != mapping.get("dataset_id"):
            raise PublicDataError(
                f"Requested HAI instance {instance_id!r} does not match "
                f"{mapping.get('dataset_id')!r}."
            )
        return mapping

    instances = _require_list(mapping["instances"], label="instances")
    if instance_id is None:
        if len(instances) != 1:
            raise PublicDataError(
                "HAI mapping catalog contains multiple instances; instance_id is required."
            )
        selected = _require_mapping(instances[0], label="instances[0]")
    else:
        matches = [
            _require_mapping(raw, label=f"instances[{index}]")
            for index, raw in enumerate(instances)
            if isinstance(raw, Mapping) and raw.get("dataset_id") == instance_id
        ]
        if len(matches) != 1:
            raise PublicDataError(f"Unknown or duplicate HAI instance ID {instance_id!r}.")
        selected = matches[0]

    common = _require_mapping(mapping.get("common_equipment"), label="common_equipment")
    controller = _require_mapping(common.get("controller"), label="common_equipment.controller")
    relay = _require_mapping(common.get("relay"), label="common_equipment.relay")
    sensors = _require_list(selected.get("sensors"), label="selected instance sensors")
    actuators = _require_list(selected.get("actuators"), label="selected instance actuators")
    edge_evidence = _require_mapping(
        selected.get("edge_evidence"), label="selected instance edge_evidence"
    )
    for name in ("sensor_to_controller", "controller_to_relay", "relay_to_actuator"):
        _require_list(edge_evidence.get(name), label=f"edge_evidence.{name}")

    nodes = [*sensors, dict(controller), dict(relay), *actuators]
    edges: list[Mapping[str, Any]] = []
    for sensor_raw in sensors:
        sensor = _require_mapping(sensor_raw, label="sensor")
        edges.append(
            {
                "source": sensor["id"],
                "target": controller["id"],
                "relation": "feedback measurement",
                "evidence": edge_evidence["sensor_to_controller"],
            }
        )
    edges.append(
        {
            "source": controller["id"],
            "target": relay["id"],
            "relation": "downstream command delivery",
            "evidence": edge_evidence["controller_to_relay"],
        }
    )
    for actuator_raw in actuators:
        actuator = _require_mapping(actuator_raw, label="actuator")
        edges.append(
            {
                "source": relay["id"],
                "target": actuator["id"],
                "relation": "actuation command",
                "evidence": edge_evidence["relay_to_actuator"],
            }
        )
    global_exclusions = _require_list(
        mapping.get("global_exclusions", []), label="global_exclusions"
    )
    instance_exclusions = _require_list(
        selected.get("exclusions", []), label="selected instance exclusions"
    )
    return {
        "dataset_id": selected["dataset_id"],
        "mapping_version": mapping["mapping_version"],
        "scope": selected["scope"],
        "source_files": mapping["source_files"],
        "nodes": nodes,
        "edges": edges,
        "exclusions": [*global_exclusions, *instance_exclusions],
        "shared_equipment_catalog_id": mapping.get("catalog_id", ""),
        "statistically_independent_instance": False,
    }


def build_hai_equipment_graph(
    mapping_path: str | Path,
    source_root: str | Path,
    *,
    instance_id: str | None = None,
) -> HaiMappingResult:
    """Build an RMCD graph only from an explicit, source-checked HAI mapping."""

    mapping_file = Path(mapping_path)
    try:
        mapping_payload = json.loads(mapping_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDataError(f"Could not parse HAI mapping {mapping_file}: {exc}") from exc
    catalog_mapping = _require_mapping(mapping_payload, label="HAI mapping")
    mapping = _resolve_hai_catalog_mapping(catalog_mapping, instance_id)
    required_top = {
        "dataset_id",
        "mapping_version",
        "scope",
        "source_files",
        "nodes",
        "edges",
        "exclusions",
    }
    missing_top = sorted(required_top - set(mapping))
    if missing_top:
        raise PublicDataError(f"HAI mapping is missing fields: {missing_top}.")

    root = Path(source_root)
    source_ids: dict[str, set[str]] = {}
    source_audit: list[Mapping[str, Any]] = []
    for index, record in enumerate(
        _require_list(mapping["source_files"], label="source_files")
    ):
        source = _require_mapping(record, label=f"source_files[{index}]")
        relative = source.get("path")
        expected_hash = source.get("sha256")
        source_format = source.get("format")
        if not all(isinstance(value, str) and value for value in (relative, expected_hash, source_format)):
            raise PublicDataError(
                f"source_files[{index}] requires nonempty path, sha256, and format."
            )
        path = root / relative
        if not path.is_file():
            raise PublicDataError(f"Required HAI source file is missing: {path}.")
        actual_hash = sha256_file(path)
        if actual_hash.lower() != expected_hash.lower():
            raise PublicDataError(
                f"HAI source checksum mismatch for {relative}: expected "
                f"{expected_hash.lower()}, got {actual_hash}."
            )
        ids = _hai_source_node_ids(path, source_format)
        source_ids[relative] = ids
        source_audit.append(
            {
                "path": relative,
                "format": source_format,
                "sha256": actual_hash,
                "node_count": len(ids),
                "checksum_ok": True,
            }
        )

    graph = nx.DiGraph()
    node_records = _require_list(mapping["nodes"], label="nodes")
    for index, raw in enumerate(node_records):
        record = _require_mapping(raw, label=f"nodes[{index}]")
        required = {
            "id",
            "role",
            "capacity",
            "attack_cost",
            "protect_cost",
            "physical_entity",
            "capacity_basis",
            "cost_basis",
            "evidence",
        }
        missing = sorted(required - set(record))
        if missing:
            raise PublicDataError(f"nodes[{index}] is missing fields: {missing}.")
        if record["role"] not in ROLES:
            raise PublicDataError(f"nodes[{index}].role must be one of {ROLES}.")
        if record["physical_entity"] is not True:
            raise PublicDataError(
                f"nodes[{index}] is not declared as a physical equipment entity."
            )
        evidence_items = _require_list(record["evidence"], label=f"nodes[{index}].evidence")
        if not evidence_items:
            raise PublicDataError(f"nodes[{index}] requires at least one evidence record.")
        for evidence_index, evidence_raw in enumerate(evidence_items):
            evidence = _require_mapping(
                evidence_raw,
                label=f"nodes[{index}].evidence[{evidence_index}]",
            )
            source_path = evidence.get("source_path")
            locator = evidence.get("locator")
            if source_path not in source_ids or not isinstance(locator, str) or not locator:
                raise PublicDataError(
                    f"nodes[{index}].evidence[{evidence_index}] has an invalid source or locator."
                )
            source_node_id = evidence.get("source_node_id")
            if source_node_id is not None and source_node_id not in source_ids[source_path]:
                raise PublicDataError(
                    f"Source node {source_node_id!r} for {record['id']!r} is absent from "
                    f"{source_path}."
                )
        graph.add_node(
            record["id"],
            role=record["role"],
            capacity=record["capacity"],
            attack_cost=record["attack_cost"],
            protect_cost=record["protect_cost"],
            public_label=record.get("public_label", record["id"]),
            source_kind=record.get("source_kind", "HAI documented equipment"),
        )

    edge_records = _require_list(mapping["edges"], label="edges")
    for index, raw in enumerate(edge_records):
        record = _require_mapping(raw, label=f"edges[{index}]")
        required = {"source", "target", "relation", "evidence"}
        missing = sorted(required - set(record))
        if missing:
            raise PublicDataError(f"edges[{index}] is missing fields: {missing}.")
        if record["source"] not in graph or record["target"] not in graph:
            raise PublicDataError(f"edges[{index}] references an unknown mapped node.")
        evidence_items = _require_list(record["evidence"], label=f"edges[{index}].evidence")
        if not evidence_items:
            raise PublicDataError(f"edges[{index}] requires at least one evidence record.")
        for evidence_index, evidence_raw in enumerate(evidence_items):
            evidence = _require_mapping(
                evidence_raw,
                label=f"edges[{index}].evidence[{evidence_index}]",
            )
            if evidence.get("source_path") not in source_ids:
                raise PublicDataError(
                    f"edges[{index}].evidence[{evidence_index}] references an unknown source."
                )
            if not isinstance(evidence.get("locator"), str) or not evidence["locator"]:
                raise PublicDataError(
                    f"edges[{index}].evidence[{evidence_index}] requires a locator."
                )
        graph.add_edge(record["source"], record["target"])

    mapping_hash = sha256_file(mapping_file)
    graph.graph.update(
        {
            "dataset_id": mapping["dataset_id"],
            "mapping_version": mapping["mapping_version"],
            "mapping_sha256": mapping_hash,
            "public_source": "HAI security dataset",
            "scope": mapping["scope"],
            "role_semantics": "frozen S-C-L-E physical equipment chain",
            "shared_equipment_catalog_id": mapping.get("shared_equipment_catalog_id", ""),
            "statistically_independent_instance": bool(
                mapping.get("statistically_independent_instance", True)
            ),
        }
    )
    try:
        equipment = validate_graph(graph)
    except GraphValidationError as exc:
        raise PublicDataError(f"Declared HAI mapping violates RMCD semantics: {exc}") from exc
    return HaiMappingResult(
        graph=equipment,
        mapping=mapping,
        source_audit=tuple(source_audit),
        mapping_sha256=mapping_hash,
    )


def build_hai_catalog_equipment_graph(
    mapping_path: str | Path,
    source_root: str | Path,
    *,
    dataset_id: str = "hai_p1_multiloop_v1",
) -> HaiMappingResult:
    """Merge all declared views of one HAI testbed into one audited graph.

    The catalog views share the same controller and remote-I/O equipment and
    are therefore repeated views, not independent observations.  This helper
    merges identical physical nodes and edges before any RMCD evaluation.
    """

    instance_ids = hai_mapping_instance_ids(mapping_path)
    results = [
        build_hai_equipment_graph(mapping_path, source_root, instance_id=value)
        for value in instance_ids
    ]
    if not results:
        raise PublicDataError("HAI mapping catalog contains no mergeable views.")

    graph = nx.DiGraph()
    node_records: dict[str, Mapping[str, Any]] = {}
    node_view_capacity: dict[str, int] = {}
    edge_records: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    exclusion_records: list[Mapping[str, Any]] = []
    source_records: dict[tuple[str, str], Mapping[str, Any]] = {}

    for result in results:
        for node, attributes in result.graph.nodes(data=True):
            observed = dict(attributes)
            if node in graph and dict(graph.nodes[node]) != observed:
                raise PublicDataError(
                    f"Shared HAI equipment {node!r} has conflicting attributes."
                )
            graph.add_node(node, **observed)
        graph.add_edges_from(result.graph.edges)

        for record in result.mapping["nodes"]:
            node_id = str(record["id"])
            previous = node_records.get(node_id)
            if previous is not None and previous != record:
                raise PublicDataError(
                    f"Shared HAI equipment record {node_id!r} is inconsistent."
                )
            node_records[node_id] = record
            node_view_capacity[node_id] = node_view_capacity.get(node_id, 0) + int(
                record["capacity"]
            )
        for record in result.mapping["edges"]:
            key = (
                str(record["source"]),
                str(record["target"]),
                str(record["relation"]),
            )
            edge_records.setdefault(key, record)
        exclusion_records.extend(result.mapping["exclusions"])
        for record in result.source_audit:
            key = (str(record["path"]), str(record["format"]))
            previous = source_records.get(key)
            if previous is not None and previous != record:
                raise PublicDataError(
                    f"HAI source audit for {key[0]!r} is inconsistent across views."
                )
            source_records[key] = record

    mapping_hashes = {result.mapping_sha256 for result in results}
    if len(mapping_hashes) != 1:
        raise PublicDataError("HAI catalog views do not share one frozen mapping hash.")
    mapping_hash = mapping_hashes.pop()
    source_audit = tuple(source_records[key] for key in sorted(source_records))
    aggregate_nodes: list[Mapping[str, Any]] = []
    for node_id in sorted(node_records):
        record = dict(node_records[node_id])
        record["capacity"] = node_view_capacity[node_id]
        record["capacity_basis"] = (
            "Sum of unit support incidences across documented HAI feedback-loop "
            "views; this is not a hardware-throughput claim."
        )
        graph.nodes[node_id]["capacity"] = node_view_capacity[node_id]
        aggregate_nodes.append(record)
    aggregate_mapping: Mapping[str, Any] = {
        "dataset_id": dataset_id,
        "mapping_version": results[0].mapping["mapping_version"],
        "scope": {
            "process": "HAI P1 boiler",
            "view": "source-audited union of declared feedback loops",
            "source_view_ids": list(instance_ids),
            "panel": "descriptive_real_case_study",
        },
        "source_files": results[0].mapping["source_files"],
        "nodes": aggregate_nodes,
        "edges": [edge_records[key] for key in sorted(edge_records)],
        "exclusions": exclusion_records,
        "source_view_ids": list(instance_ids),
        "statistically_independent_instance": False,
    }
    graph.graph.update(
        {
            "dataset_id": dataset_id,
            "mapping_version": aggregate_mapping["mapping_version"],
            "mapping_sha256": mapping_hash,
            "public_source": "HAI security dataset",
            "scope": aggregate_mapping["scope"],
            "role_semantics": "frozen S-C-L-E physical equipment chain",
            "source_view_ids": list(instance_ids),
            "source_view_count": len(instance_ids),
            "statistically_independent_instance": False,
            "evidence_panel": "descriptive_real_case_study",
            "capacity_mode": "documented_loop_support_incidence",
        }
    )
    try:
        equipment = validate_graph(graph)
    except GraphValidationError as exc:
        raise PublicDataError(f"Merged HAI catalog violates RMCD semantics: {exc}") from exc
    return HaiMappingResult(
        graph=equipment,
        mapping=aggregate_mapping,
        source_audit=source_audit,
        mapping_sha256=mapping_hash,
    )


def audit_public_candidate_registry(
    registry_path: str | Path,
) -> tuple[PublicCandidateAudit, ...]:
    """Validate and load the preregistered semantic eligibility decisions."""

    path = Path(registry_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDataError(f"Could not parse public candidate registry {path}: {exc}") from exc
    root = _require_mapping(payload, label="public candidate registry")
    rows = _require_list(root.get("datasets"), label="public candidate registry datasets")
    required = set(PublicCandidateAudit.__dataclass_fields__)
    audits: list[PublicCandidateAudit] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        record = _require_mapping(raw, label=f"datasets[{index}]")
        missing = sorted(required - set(record))
        if missing:
            raise PublicDataError(f"datasets[{index}] is missing fields: {missing}.")
        values = {field: record[field] for field in required}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise PublicDataError(f"datasets[{index}] fields must be nonempty strings.")
        dataset_key = str(values["dataset_id"])
        if dataset_key in seen:
            raise PublicDataError(f"Duplicate public candidate {dataset_key!r}.")
        seen.add(dataset_key)
        gate = str(values["rmcd_semantic_gate"])
        if gate not in {"PASS", "PENDING", "FAIL"}:
            raise PublicDataError(
                f"datasets[{index}].rmcd_semantic_gate must be PASS, PENDING, or FAIL."
            )
        audits.append(PublicCandidateAudit(**values))
    if not audits:
        raise PublicDataError("Public candidate registry is empty.")
    return tuple(audits)


def _read_sndlib_sections_text(text: str, *, source_label: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    depth = 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if current is None:
            match = _SECTION_RE.match(line)
            if match:
                current = match.group(1)
                sections.setdefault(current, [])
                depth = 1
            continue
        opening = line.count("(")
        closing = line.count(")")
        if line == ")" and depth == 1:
            current = None
            depth = 0
            continue
        sections[current].append(line)
        depth += opening - closing
        if depth < 1:
            raise PublicDataError(
                f"Unbalanced SNDlib parentheses at {source_label}:{line_number}."
            )
    if current is not None:
        raise PublicDataError(
            f"Unclosed SNDlib section {current!r} in {source_label}."
        )
    return sections


def _read_sndlib_sections(path: Path) -> dict[str, list[str]]:
    return _read_sndlib_sections_text(
        path.read_text(encoding="utf-8-sig"),
        source_label=str(path),
    )


def _parse_sndlib_sections(
    source: Path,
    sections: Mapping[str, list[str]],
) -> SndlibNetwork:
    for required in ("NODES", "LINKS", "DEMANDS"):
        if required not in sections:
            raise PublicDataError(f"SNDlib file {source} has no {required} section.")

    nodes: list[SndlibNode] = []
    for record in sections["NODES"]:
        match = _NODE_RE.match(record)
        if not match:
            raise PublicDataError(f"Malformed SNDlib node record in {source}: {record!r}.")
        longitude = float(match.group(2)) if match.group(2) is not None else None
        latitude = float(match.group(3)) if match.group(3) is not None else None
        nodes.append(SndlibNode(match.group(1), longitude, latitude))

    links: list[SndlibLink] = []
    for record in sections["LINKS"]:
        match = _PAIR_RECORD_RE.match(record)
        if not match:
            raise PublicDataError(f"Malformed SNDlib link record in {source}: {record!r}.")
        prefix = match.group(4).split("(", 1)[0].split()
        if len(prefix) < 4:
            raise PublicDataError(f"Incomplete SNDlib link record in {source}: {record!r}.")
        try:
            capacity = float(prefix[0])
            routing_cost = float(prefix[2])
        except ValueError as exc:
            raise PublicDataError(f"Non-numeric SNDlib link value in {source}: {record!r}.") from exc
        links.append(
            SndlibLink(match.group(1), match.group(2), match.group(3), capacity, routing_cost)
        )

    demands: list[SndlibDemand] = []
    for record in sections["DEMANDS"]:
        match = _PAIR_RECORD_RE.match(record)
        if not match:
            raise PublicDataError(f"Malformed SNDlib demand record in {source}: {record!r}.")
        values = match.group(4).split()
        if len(values) < 3:
            raise PublicDataError(f"Incomplete SNDlib demand record in {source}: {record!r}.")
        try:
            routing_unit = float(values[0])
            demand_value = float(values[1])
        except ValueError as exc:
            raise PublicDataError(f"Non-numeric SNDlib demand value in {source}: {record!r}.") from exc
        demands.append(
            SndlibDemand(
                match.group(1),
                match.group(2),
                match.group(3),
                routing_unit,
                demand_value,
                values[2],
            )
        )

    node_ids = {node.node_id for node in nodes}
    if len(node_ids) != len(nodes):
        raise PublicDataError(f"Duplicate node IDs in SNDlib file {source}.")
    for link in links:
        if link.source not in node_ids or link.target not in node_ids:
            raise PublicDataError(f"Link {link.link_id!r} has an unknown endpoint in {source}.")
    for demand in demands:
        if demand.source not in node_ids or demand.target not in node_ids:
            raise PublicDataError(f"Demand {demand.demand_id!r} has an unknown endpoint in {source}.")
    admissible_paths: list[SndlibAdmissiblePath] = []
    current_demand: str | None = None
    demand_ids = {demand.demand_id for demand in demands}
    link_ids = {link.link_id for link in links}
    for record in sections.get("ADMISSIBLE_PATHS", []):
        if record == ")":
            if current_demand is None:
                raise PublicDataError(
                    f"Unexpected admissible-path group close in {source}."
                )
            current_demand = None
            continue
        group_match = re.match(r"^(\S+)\s+\(\s*$", record)
        if group_match:
            if current_demand is not None:
                raise PublicDataError(f"Nested admissible-path demand groups in {source}.")
            current_demand = group_match.group(1)
            if current_demand not in demand_ids:
                raise PublicDataError(
                    f"Admissible paths reference unknown demand {current_demand!r} in {source}."
                )
            continue
        path_match = re.match(r"^(\S+)\s+\(\s*(.*?)\s*\)\s*$", record)
        if not path_match or current_demand is None:
            raise PublicDataError(
                f"Malformed admissible-path record in {source}: {record!r}."
            )
        path_links = tuple(path_match.group(2).split()) if path_match.group(2) else ()
        if not path_links:
            raise PublicDataError(f"Empty admissible path in {source}: {record!r}.")
        unknown = sorted(set(path_links) - link_ids)
        if unknown:
            raise PublicDataError(
                f"Admissible path {path_match.group(1)!r} in {source} uses unknown "
                f"links: {unknown}."
            )
        admissible_paths.append(
            SndlibAdmissiblePath(current_demand, path_match.group(1), path_links)
        )
    if current_demand is not None:
        raise PublicDataError(f"Unclosed admissible-path demand group in {source}.")
    return SndlibNetwork(
        source.stem,
        tuple(nodes),
        tuple(links),
        tuple(demands),
        tuple(admissible_paths),
    )


def parse_sndlib_native(path: str | Path) -> SndlibNetwork:
    """Parse all supported sections of a SNDlib native-format file."""

    source = Path(path)
    return _parse_sndlib_sections(source, _read_sndlib_sections(source))


def parse_sndlib_native_text(text: str, *, name: str = "memory.txt") -> SndlibNetwork:
    """Parse an in-memory SNDlib member without extracting a dynamic archive."""

    source = Path(name)
    sections = _read_sndlib_sections_text(text, source_label=name)
    return _parse_sndlib_sections(source, sections)


def audit_sndlib_dynamic_archive(
    archive_path: str | Path,
    static_network: SndlibNetwork,
    *,
    full: bool = False,
) -> SndlibDynamicInventory:
    """Stream-audit a SNDlib demand-matrix archive.

    The default checks the first, middle, and last matrices plus the 16 member
    names with the smallest SHA-256 digests. ``full=True`` parses every matrix
    and is intended for the server audit.
    """

    source = Path(archive_path)
    with tarfile.open(source, mode="r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".txt")
                and "readme" not in member.name.lower()
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise PublicDataError(f"No demand matrices found in {source}.")
        if full:
            selected = members
        else:
            hash_sample = sorted(
                members,
                key=lambda member: hashlib.sha256(member.name.encode("utf-8")).digest(),
            )[:16]
            selected_by_name = {
                member.name: member
                for member in (
                    members[0],
                    members[len(members) // 2],
                    members[-1],
                    *hash_sample,
                )
            }
            selected = [selected_by_name[name] for name in sorted(selected_by_name)]
        expected_nodes = {node.node_id for node in static_network.nodes}
        demand_counts: list[int] = []
        node_set_matches = True
        for member in selected:
            handle = archive.extractfile(member)
            if handle is None:
                raise PublicDataError(f"Could not read SNDlib member {member.name}.")
            try:
                text = handle.read().decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise PublicDataError(
                    f"SNDlib member {member.name} is not UTF-8 text."
                ) from exc
            matrix = parse_sndlib_native_text(text, name=member.name)
            observed_nodes = {node.node_id for node in matrix.nodes}
            node_set_matches = node_set_matches and observed_nodes == expected_nodes
            if matrix.links:
                raise PublicDataError(
                    f"Demand-matrix member {member.name} unexpectedly contains links."
                )
            demand_counts.append(len(matrix.demands))
    return SndlibDynamicInventory(
        archive_name=source.name,
        matrix_count=len(members),
        audited_matrix_count=len(selected),
        node_count=len(expected_nodes),
        demand_count_min=min(demand_counts),
        demand_count_max=max(demand_counts),
        node_set_matches_static=node_set_matches,
    )


def inventory_caida_as_rel(path: str | Path) -> CaidaAsInventory:
    """Audit a CAIDA AS-relationship snapshot without assigning RMCD roles."""

    source = Path(path)
    opener = bz2.open if source.suffix.lower() == ".bz2" else open
    nodes: set[int] = set()
    edges = 0
    provider_customer = 0
    peer = 0
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("|")
            if len(fields) < 3:
                raise PublicDataError(f"Malformed CAIDA record at {source}:{line_number}.")
            try:
                left, right, relation = int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise PublicDataError(
                    f"Non-integer CAIDA record at {source}:{line_number}."
                ) from exc
            if relation not in (-1, 0):
                raise PublicDataError(
                    f"Unsupported CAIDA relationship {relation} at {source}:{line_number}."
                )
            nodes.update((left, right))
            edges += 1
            provider_customer += int(relation == -1)
            peer += int(relation == 0)
    if not nodes or not edges:
        raise PublicDataError(f"CAIDA snapshot {source} contains no relationships.")
    return CaidaAsInventory(len(nodes), edges, provider_customer, peer)


def inventory_peeringdb_pages(paths: Iterable[str | Path]) -> PeeringDbInventory:
    """Validate paginated PeeringDB API records and report unique IDs/ASNs."""

    pages = 0
    records = 0
    record_ids: set[int] = set()
    asns: set[int] = set()
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: path.name):
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicDataError(f"Could not parse PeeringDB page {raw_path}: {exc}") from exc
        root = _require_mapping(payload, label=f"PeeringDB page {raw_path}")
        data = _require_list(root.get("data"), label=f"{raw_path}.data")
        pages += 1
        records += len(data)
        for index, raw in enumerate(data):
            record = _require_mapping(raw, label=f"{raw_path}.data[{index}]")
            record_id = record.get("id")
            asn = record.get("asn")
            if not isinstance(record_id, int):
                raise PublicDataError(f"{raw_path}.data[{index}].id must be an integer.")
            record_ids.add(record_id)
            if isinstance(asn, int):
                asns.add(asn)
    if pages == 0:
        raise PublicDataError("No PeeringDB pages were supplied.")
    return PeeringDbInventory(pages, records, len(record_ids), len(asns))
