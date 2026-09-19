"""Adapters for the official NetworkDismantling/review implementations.

The official methods generate structural node-removal sequences on a simple
undirected projection of the physical equipment graph.  RMCD evaluates those
sequences separately against its own role-motif capacity threshold; this
module never substitutes the review repository's LCC objective for
``Omega_R(G-D) < K``.

Third-party source code is not embedded in the Python wheel.  A complete
server source bundle may provide it under ``external/review-main``; otherwise
it is discovered from ``NETWORK_DISMANTLING_REVIEW_PATH`` or an official
``review-main.zip`` archive and imported lazily.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from .model import NodeId, stable_nodes, validate_graph
from .nature_review_panel import (
    NATURE_REVIEW_TABLE1_ALL_OFFICIAL_VARIANTS,
    NATURE_REVIEW_TABLE1_CANONICAL_OFFICIAL_METHODS,
)
from .provenance import official_projection_fingerprint


REVIEW_REPOSITORY_URL = "https://github.com/NetworkDismantling/review"
OFFICIAL_ADAPTER_VERSION = "1.9"
GND_FAMILY_ADAPTER_VERSION = "1.8-gnd-family-v1"
FINDER_LEGACY_ADAPTER_VERSION = "legacy-finder-direct-1.5"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINDER_REQUIRED_SOURCE_FILES = (
    "PrepareBatchGraph.cpp",
    "PrepareBatchGraph.h",
    "config.cpp",
    "config.h",
    "decrease_strategy.cpp",
    "disjoint_set.cpp",
    "disjoint_set.h",
    "graph.cpp",
    "graph.h",
    "graph_struct.cpp",
    "graph_struct.h",
    "graph_utils.cpp",
    "graph_utils.h",
    "i_env.h",
    "msg_pass.cpp",
    "msg_pass.h",
    "mvc_env.cpp",
    "mvc_env.h",
    "nstep_replay_mem.cpp",
    "nstep_replay_mem.h",
    "nstep_replay_mem_prioritized.cpp",
    "nstep_replay_mem_prioritized.h",
    "utils.cpp",
    "utils.h",
)
LOGGER = logging.getLogger("rmcd_f.official_review")
LOGGER.addHandler(logging.NullHandler())
_SOURCE_IDENTITY_CACHE: dict[Path, tuple[str, str]] = {}
_NUMPY_RANDOM_LOCK = threading.Lock()
_SOURCE_IDENTITY_EXCLUDED_RELATIVE = frozenset(
    {
        "network_dismantling/decycler/COPYING",
        "network_dismantling/decycler/Makefile",
        "network_dismantling/decycler/gnp.py",
        "network_dismantling/EGND/config.h",
        "network_dismantling/EGND/config_r.h",
        "network_dismantling/FINDER_ND/FINDER.c",
        "network_dismantling/FINDER_ND/PrepareBatchGraph.cpp",
        "network_dismantling/FINDER_ND/graph.cpp",
        "network_dismantling/FINDER_ND/graph_struct.cpp",
        "network_dismantling/FINDER_ND/mvc_env.cpp",
        "network_dismantling/FINDER_ND/nstep_replay_mem.cpp",
        (
            "network_dismantling/FINDER_ND/"
            "nstep_replay_mem_prioritized.cpp"
        ),
        "network_dismantling/FINDER_ND/utils.cpp",
        "network_dismantling/GDM/reinsert.py",
    }
)
_SOURCE_IDENTITY_GENERATED_NAMES = frozenset(
    {
        "CI",
        "CMakeCache.txt",
        "EnsembleGND",
        "EnsembleGNDR",
        "GND",
        "coreHD",
        "decycler",
        "dismantler",
        "exploimmun",
        "gmon.out",
        "reinsertion",
        "reverse-greedy",
    }
)
_SOURCE_IDENTITY_GENERATED_PARTS = frozenset(
    {
        ".git",
        ".idea",
        "CMakeFiles",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "out",
    }
)
_SOURCE_IDENTITY_GENERATED_SUFFIXES = frozenset(
    {".a", ".dll", ".dylib", ".exe", ".o", ".pyc", ".so"}
)


@dataclass(frozen=True)
class OfficialMethodSpec:
    """One official review method, launch group and dependency environment."""

    name: str
    module: str
    function: str
    group: str
    kind: str = "wrapped"
    dependency: str | None = None
    dependency_argument: str | None = None
    required_modules: tuple[str, ...] = ()
    preparation: str | None = None
    experimental_upstream: bool = False
    environment: str = "dismantling"
    derived_output: bool = False


def _spec(
    name: str,
    module: str,
    function: str,
    group: str,
    **kwargs: Any,
) -> OfficialMethodSpec:
    return OfficialMethodSpec(name, module, function, group, **kwargs)


OFFICIAL_METHOD_SPECS: dict[str, OfficialMethodSpec] = {
    spec.name: spec
    for spec in (
        _spec(
            "ReviewBruteForceFrequencyRank",
            "network_dismantling.brute_force.dismantler",
            "optimal_threshold_dismantler",
            "derived",
            kind="bruteforce_target",
            derived_output=True,
        ),
        _spec(
            "OfficialCI_L1",
            "network_dismantling.CI.python_interface",
            "CollectiveInfluenceL1",
            "core",
        ),
        _spec(
            "OfficialCI_L2",
            "network_dismantling.CI.python_interface",
            "CollectiveInfluenceL2",
            "core",
        ),
        _spec(
            "OfficialCI_L3",
            "network_dismantling.CI.python_interface",
            "CollectiveInfluenceL3",
            "core",
        ),
        _spec(
            "OfficialCoreHD",
            "network_dismantling.CoreHD.python_interface",
            "CoreHD",
            "core",
        ),
        _spec(
            "OfficialGND",
            "network_dismantling.GND.python_interface",
            "GND",
            "core",
        ),
        _spec(
            "OfficialGNDR",
            "network_dismantling.GND.python_interface",
            "GNDR",
            "core",
            dependency="OfficialGND",
            dependency_argument="GND",
        ),
        _spec(
            "OfficialEI_S1",
            "network_dismantling.EI.python_interface",
            "EI_s1",
            "core",
        ),
        _spec(
            "OfficialEI_S2",
            "network_dismantling.EI.python_interface",
            "EI_s2",
            "core",
        ),
        _spec(
            "OfficialMinSum",
            "network_dismantling.decycler.python_interface",
            "MS",
            "core",
        ),
        _spec(
            "OfficialMinSumR",
            "network_dismantling.decycler.python_interface",
            "MSR",
            "core",
        ),
        _spec(
            "OfficialNetworkEntanglementSmall",
            "network_dismantling.multiscale_entanglement.python_interface",
            "network_entanglement_small",
            "core",
        ),
        _spec(
            "OfficialNetworkEntanglementSmallR",
            "network_dismantling.multiscale_entanglement.python_interface",
            "network_entanglement_small_reinsertion",
            "core",
            dependency="OfficialNetworkEntanglementSmall",
            dependency_argument="network_entanglement_small",
        ),
        _spec(
            "OfficialNetworkEntanglementMid",
            "network_dismantling.multiscale_entanglement.python_interface",
            "network_entanglement_mid",
            "core",
        ),
        _spec(
            "OfficialNetworkEntanglementMidR",
            "network_dismantling.multiscale_entanglement.python_interface",
            "network_entanglement_mid_reinsertion",
            "core",
            dependency="OfficialNetworkEntanglementMid",
            dependency_argument="network_entanglement_mid",
        ),
        _spec(
            "OfficialNetworkEntanglementLarge",
            "network_dismantling.multiscale_entanglement.python_interface",
            "network_entanglement_large",
            "core",
        ),
        _spec(
            "OfficialNetworkEntanglementLargeR",
            "network_dismantling.multiscale_entanglement.python_interface",
            "network_entanglement_large_reinsertion",
            "core",
            dependency="OfficialNetworkEntanglementLarge",
            dependency_argument="network_entanglement_large",
        ),
        _spec(
            "OfficialVertexEntanglement",
            "network_dismantling.vertex_entanglement.python_interface",
            "vertex_entanglement",
            "core",
            required_modules=("yaml",),
        ),
        _spec(
            "OfficialVertexEntanglementR",
            "network_dismantling.vertex_entanglement.python_interface",
            "vertex_entanglement_reinsertion",
            "core",
            dependency="OfficialVertexEntanglement",
            dependency_argument="vertex_entanglement",
            required_modules=("yaml",),
        ),
        _spec(
            "OfficialGDM",
            "network_dismantling.GDM.python_interface",
            "GDM",
            "ml",
            required_modules=("dill", "torch", "torch_geometric", "torch_sparse"),
            environment="gdm",
        ),
        _spec(
            "OfficialGDMR",
            "network_dismantling.GDM.python_interface",
            "GDMR",
            "ml",
            required_modules=("dill", "torch", "torch_geometric", "torch_sparse"),
            environment="gdm",
        ),
        _spec(
            "OfficialCoreGDM",
            "network_dismantling.CoreGDM.python_interface",
            "CoreGDM",
            "ml",
            required_modules=("dill", "torch", "torch_geometric", "torch_sparse"),
            experimental_upstream=True,
            environment="gdm",
        ),
        _spec(
            "OfficialEGND",
            "network_dismantling.EGND.python_interface",
            "EGND",
            "ml",
        ),
        _spec(
            "OfficialFINDER_R",
            "network_dismantling.FINDER_ND.python_interface",
            "FINDER_ND",
            "finder",
            preparation="finder",
            environment="finder",
        ),
        _spec(
            "OfficialDegreeStatic",
            "network_dismantling.heuristics.sorters",
            "get_degree",
            "heuristic",
            kind="heuristic_static",
        ),
        _spec(
            "OfficialDegreeDynamic",
            "network_dismantling.heuristics.sorters",
            "get_degree",
            "heuristic",
            kind="heuristic_dynamic",
        ),
        _spec(
            "OfficialBetweennessStatic",
            "network_dismantling.heuristics.sorters",
            "get_betweenness_centrality",
            "heuristic",
            kind="heuristic_static",
        ),
        _spec(
            "OfficialBetweennessDynamic",
            "network_dismantling.heuristics.sorters",
            "get_betweenness_centrality",
            "heuristic",
            kind="heuristic_dynamic",
        ),
        _spec(
            "OfficialEigenvectorStatic",
            "network_dismantling.heuristics.sorters",
            "get_eigenvector_centrality",
            "heuristic",
            kind="heuristic_static",
        ),
        _spec(
            "OfficialEigenvectorDynamic",
            "network_dismantling.heuristics.sorters",
            "get_eigenvector_centrality",
            "heuristic",
            kind="heuristic_dynamic",
        ),
        _spec(
            "OfficialPageRankStatic",
            "network_dismantling.heuristics.sorters",
            "get_pagerank",
            "heuristic",
            kind="heuristic_static",
        ),
        _spec(
            "OfficialPageRankDynamic",
            "network_dismantling.heuristics.sorters",
            "get_pagerank",
            "heuristic",
            kind="heuristic_dynamic",
        ),
        _spec(
            "OfficialRandomStatic",
            "network_dismantling.heuristics.sorters",
            "get_random",
            "heuristic",
            kind="heuristic_static",
        ),
    )
}

OFFICIAL_CORE_METHODS = tuple(
    name for name, spec in OFFICIAL_METHOD_SPECS.items() if spec.group == "core"
)
OFFICIAL_BRUTE_FORCE_METHODS = tuple(
    name
    for name, spec in OFFICIAL_METHOD_SPECS.items()
    if spec.kind == "bruteforce_target"
)
OFFICIAL_ML_METHODS = tuple(
    name for name, spec in OFFICIAL_METHOD_SPECS.items() if spec.group == "ml"
)
OFFICIAL_FINDER_METHODS = tuple(
    name for name, spec in OFFICIAL_METHOD_SPECS.items() if spec.group == "finder"
)
OFFICIAL_HEURISTIC_METHODS = tuple(
    name for name, spec in OFFICIAL_METHOD_SPECS.items() if spec.group == "heuristic"
)
OFFICIAL_ALL_METHODS = tuple(OFFICIAL_METHOD_SPECS)
OFFICIAL_EXPERIMENTAL_METHODS = tuple(
    name
    for name, spec in OFFICIAL_METHOD_SPECS.items()
    if spec.experimental_upstream
)
OFFICIAL_STABLE_METHODS = tuple(
    name
    for name, spec in OFFICIAL_METHOD_SPECS.items()
    if not spec.experimental_upstream and not spec.derived_output
)
OFFICIAL_METHOD_GROUPS: dict[str, tuple[str, ...]] = {
    "derived": OFFICIAL_BRUTE_FORCE_METHODS,
    "core": OFFICIAL_CORE_METHODS,
    "ml": OFFICIAL_ML_METHODS,
    "finder": OFFICIAL_FINDER_METHODS,
    "heuristic": OFFICIAL_HEURISTIC_METHODS,
    "experimental": OFFICIAL_EXPERIMENTAL_METHODS,
    "stable": OFFICIAL_STABLE_METHODS,
    "all_scalable": OFFICIAL_STABLE_METHODS,
    "all": OFFICIAL_ALL_METHODS,
    # The local BPDReference implementation is intentionally absent here:
    # this registry launches only upstream review adapters.  The operational-
    # motif study adds BPDReference locally when expanding the full panel.
    "nature_table1": NATURE_REVIEW_TABLE1_CANONICAL_OFFICIAL_METHODS,
    "nature_table1_variants": NATURE_REVIEW_TABLE1_ALL_OFFICIAL_VARIANTS,
}


@dataclass(frozen=True)
class OfficialSequenceResult:
    """One official structural removal sequence mapped to equipment node IDs."""

    method: str
    method_group: str
    sequence: tuple[NodeId, ...]
    graph_fingerprint: str
    node_count: int
    edge_count: int
    stop_condition: int
    verified_final_lcc_size: int
    runtime_seconds: float
    review_root: str
    projection: str = "simple_undirected_physical_equipment_projection"
    sequence_semantics: str = "official_structural_removal_order"
    source_identity_kind: str = ""
    source_identity: str = ""
    compatibility_patches: tuple[str, ...] = ()
    adapter_version: str = OFFICIAL_ADAPTER_VERSION
    projection_fingerprint: str = ""
    sequence_sha256: str = ""
    dependency_method: str = ""
    dependency_sequence_sha256: str = ""


@dataclass(frozen=True)
class OfficialMethodApplicability:
    """Whether an upstream method's documented structural domain is met."""

    applicable: bool
    code: str = "applicable"
    detail: str = ""


class OfficialMethodNotApplicableError(RuntimeError):
    """Raised before launch when an upstream structural prerequisite fails."""

    def __init__(self, method: str, code: str, detail: str):
        self.method = method
        self.code = code
        self.detail = detail
        super().__init__(f"{method} is not applicable ({code}): {detail}")


_CONNECTED_PROJECTION_METHODS = frozenset(
    {
        "OfficialEI_S2",
        "OfficialEGND",
        "OfficialEigenvectorStatic",
        "OfficialEigenvectorDynamic",
    }
)

_EIGENVECTOR_POWER_METHODS = frozenset(
    {
        "OfficialEigenvectorStatic",
        "OfficialEigenvectorDynamic",
    }
)


class _SerialExecutor:
    def map(self, function: Callable[..., Any], iterable: Iterable[Any], chunksize: int = 1):
        return map(function, iterable)


def expand_official_methods(value: str | Iterable[str]) -> tuple[str, ...]:
    """Expand comma-separated method names and group aliases deterministically."""

    raw = value.split(",") if isinstance(value, str) else list(value)
    expanded: list[str] = []
    for item in raw:
        name = str(item).strip()
        if not name:
            continue
        if name in OFFICIAL_METHOD_GROUPS:
            candidates = OFFICIAL_METHOD_GROUPS[name]
        elif name in OFFICIAL_METHOD_SPECS:
            candidates = (name,)
        else:
            known = ", ".join((*OFFICIAL_METHOD_GROUPS, *OFFICIAL_METHOD_SPECS))
            raise ValueError(f"Unknown official method or group {name!r}. Known: {known}")
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    if not expanded:
        raise ValueError("At least one official method is required.")
    return tuple(expanded)


def _official_projection_graph(graph: nx.Graph) -> nx.DiGraph:
    """Validate either supported input model before structural projection.

    Official review methods see only a simple undirected projection.  Legacy
    equipment graphs still pass the strict S-C-L-E validator; source-backed
    MPCF task-thread graphs pass their own path-closure validator.  Keeping
    this dispatch in one place prevents fingerprinting and execution from
    accepting different graph domains.
    """

    if str(graph.graph.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        from .operational_motif import build_path_closed_system

        return build_path_closed_system(graph).graph
    return validate_graph(graph)


def official_method_applicability(
    graph: nx.Graph,
    method: str,
    *,
    brute_force_max_nodes: int = 18,
) -> OfficialMethodApplicability:
    """Audit narrow upstream prerequisites without changing the input graph.

    The review implementations below fail pathologically outside these
    structural domains.  Recording that boundary is preferable to silently
    substituting another ranking or counting an adapter crash as a loss.
    """

    if method not in OFFICIAL_METHOD_SPECS:
        raise ValueError(f"Unknown official review method: {method!r}.")
    equipment = _official_projection_graph(graph)
    projection = nx.Graph(equipment.to_undirected())
    projection.remove_edges_from(nx.selfloop_edges(projection))
    component_sizes = sorted(
        (len(component) for component in nx.connected_components(projection)),
        reverse=True,
    )

    if brute_force_max_nodes < 4:
        raise ValueError("brute_force_max_nodes must be at least 4.")
    if (
        method in OFFICIAL_BRUTE_FORCE_METHODS
        and projection.number_of_nodes() > brute_force_max_nodes
    ):
        return OfficialMethodApplicability(
            False,
            "bruteforce_node_limit",
            (
                "the upstream exhaustive implementation enumerates node "
                f"combinations and is preregistered only through "
                f"n={brute_force_max_nodes}; received "
                f"n={projection.number_of_nodes()}"
            ),
        )

    if method in _CONNECTED_PROJECTION_METHODS and len(component_sizes) > 1:
        isolates = sum(size == 1 for size in component_sizes)
        return OfficialMethodApplicability(
            False,
            "initial_projection_disconnected",
            (
                "the official implementation requires one connected equipment "
                f"projection; component_sizes={component_sizes}, "
                f"isolates={isolates}"
            ),
        )

    if method in _EIGENVECTOR_POWER_METHODS and nx.is_bipartite(projection):
        return OfficialMethodApplicability(
            False,
            "bipartite_power_iteration_periodicity",
            (
                "the official graph-tool sorter uses unbounded power iteration; "
                "an undirected bipartite projection has equal-magnitude positive "
                "and negative dominant eigenvalues, so that implementation is not "
                "a reproducibly terminating baseline on this graph"
            ),
        )

    if method == "OfficialCoreGDM":
        two_core = nx.k_core(projection, k=2)
        if two_core.number_of_nodes() == 0:
            return OfficialMethodApplicability(
                False,
                "empty_two_core",
                "CoreGDM has no 2-core on which to construct its core ranking.",
            )

    return OfficialMethodApplicability(True)


def graph_fingerprint(graph: nx.Graph) -> str:
    """Return a stable content fingerprint for sequence/instance matching."""

    if str(graph.graph.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        from .operational_motif import build_path_closed_system

        system = build_path_closed_system(graph)

        def generic_token(node: NodeId) -> tuple[str, str]:
            return type(node).__qualname__, repr(node)

        generic_nodes = [
            {
                "id": generic_token(node),
                "task_stage": str(system.graph.nodes[node].get("task_stage", "")),
                "capacity": int(system.graph.nodes[node]["capacity"]),
                "attack_cost": str(system.graph.nodes[node]["attack_cost"]),
                "protect_cost": str(system.graph.nodes[node]["protect_cost"]),
                "removable": bool(system.graph.nodes[node].get("removable", True)),
            }
            for node in stable_nodes(system.graph.nodes)
        ]
        generic_edges = sorted(
            (
                (generic_token(left), generic_token(right))
                for left, right in system.graph.edges
            ),
            key=repr,
        )
        generic_payload = json.dumps(
            {
                "directed": True,
                "nodes": generic_nodes,
                "edges": generic_edges,
                "source_nodes": [generic_token(node) for node in system.source_nodes],
                "target_nodes": [generic_token(node) for node in system.target_nodes],
                "input_semantics": graph.graph.get("input_semantics"),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(generic_payload).hexdigest()

    equipment = _official_projection_graph(graph)

    def token(node: NodeId) -> tuple[str, str]:
        return type(node).__qualname__, repr(node)

    nodes = [
        {
            "id": token(node),
            "role": str(equipment.nodes[node]["role"]),
            "capacity": int(equipment.nodes[node]["capacity"]),
            "attack_cost": str(equipment.nodes[node]["attack_cost"]),
        }
        for node in stable_nodes(equipment.nodes)
    ]
    edges = sorted(
        ((token(left), token(right)) for left, right in equipment.edges),
        key=repr,
    )
    payload = json.dumps(
        {"directed": True, "nodes": nodes, "edges": edges},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sequence_sha256(sequence: Iterable[NodeId]) -> str:
    """Return a stable hash for a mapped equipment-node sequence."""

    payload = [
        {
            "type": f"{type(node).__module__}.{type(node).__qualname__}",
            "repr": repr(node),
        }
        for node in sequence
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _candidate_review_roots(review_path: str | Path | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if review_path is not None:
        candidates.append(Path(review_path))
    env_path = os.environ.get("NETWORK_DISMANTLING_REVIEW_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        (
            Path.cwd() / "review-main",
            Path.cwd() / "external" / "review-main",
            PROJECT_ROOT / "review-main",
            PROJECT_ROOT / "external" / "review-main",
        )
    )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _candidate_archives(zip_path: str | Path | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if zip_path is not None:
        candidates.append(Path(zip_path))
    env_path = os.environ.get("NETWORK_DISMANTLING_REVIEW_ZIP")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        (
            Path.cwd() / "review-main.zip",
            PROJECT_ROOT / "review-main.zip",
            PROJECT_ROOT.parent / "review-main.zip",
        )
    )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _validate_review_root(root: Path) -> None:
    package = root / "network_dismantling"
    init_path = package / "__init__.py"
    readme = root / "README.md"
    if not package.is_dir() or not init_path.is_file():
        raise RuntimeError(
            f"Not a NetworkDismantling/review source tree: {root}"
        )
    markers = ""
    for path in (readme, init_path):
        if path.is_file():
            markers += path.read_text(encoding="utf-8", errors="ignore")[:12000]
    if (
        "Robustness and resilience of complex networks" not in markers
        and "Network Dismantling review" not in markers
    ):
        raise RuntimeError(
            "The supplied source tree does not contain the official review "
            f"repository identity markers: {root}"
        )


def _is_generated_source_artifact(path: Path, root: Path) -> bool:
    """Return whether *path* is generated state, not upstream source.

    Official wrappers compile binaries and, in a few cases, write profiling or
    per-run configuration files inside the review checkout.  Those files must
    not make the recorded source identity depend on which method ran first.
    """

    relative = path.relative_to(root).as_posix()
    if relative in _SOURCE_IDENTITY_EXCLUDED_RELATIVE:
        return True
    if path.name in _SOURCE_IDENTITY_GENERATED_NAMES:
        return True
    if path.suffix in _SOURCE_IDENTITY_GENERATED_SUFFIXES:
        return True
    return any(
        part in _SOURCE_IDENTITY_GENERATED_PARTS
        or part.endswith(".egg-info")
        for part in path.relative_to(root).parts
    )


def _source_identity(root: Path) -> tuple[str, str]:
    resolved = root.resolve()
    cached = _SOURCE_IDENTITY_CACHE.get(resolved)
    if cached is not None:
        return cached
    try:
        completed = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        revision = completed.stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(resolved),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        if dirty:
            revision = ""
    except (OSError, subprocess.SubprocessError):
        revision = ""
    if revision:
        identity = ("git_commit", revision)
    else:
        digest = hashlib.sha256()
        for path in sorted(
            (
                item
                for item in (resolved / "network_dismantling").rglob("*")
                if item.is_file()
                and not _is_generated_source_artifact(item, resolved)
            ),
            key=lambda item: item.as_posix(),
        ):
            relative = path.relative_to(resolved).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        identity = ("source_tree_sha256", digest.hexdigest())
    _SOURCE_IDENTITY_CACHE[resolved] = identity
    return identity


def review_source_identity(
    *,
    review_path: str | Path | None = None,
    zip_path: str | Path | None = None,
    extraction_root: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, str, str]:
    """Resolve and identify the actual upstream source before a resumable run."""

    root = ensure_review_source(
        review_path=review_path,
        zip_path=zip_path,
        extraction_root=extraction_root,
        progress=progress,
    )
    kind, identity = _source_identity(root)
    return root, kind, identity


def ensure_review_source(
    *,
    review_path: str | Path | None = None,
    zip_path: str | Path | None = None,
    extraction_root: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Locate or safely extract the official review source tree."""

    for candidate in _candidate_review_roots(review_path):
        if (candidate / "network_dismantling").is_dir():
            resolved = candidate.resolve()
            _validate_review_root(resolved)
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved

    archive = next(
        (candidate for candidate in _candidate_archives(zip_path) if candidate.is_file()),
        None,
    )
    if archive is None:
        raise FileNotFoundError(
            "Official NetworkDismantling source was not found. Set "
            "NETWORK_DISMANTLING_REVIEW_PATH, pass --review-path, or place "
            "review-main.zip in the project root."
        )

    target = (
        Path(extraction_root)
        if extraction_root is not None
        else Path.cwd() / "external" / "review-main"
    )
    target.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()
    if progress is not None:
        progress(f"extracting official review source to {target_resolved}")

    with zipfile.ZipFile(archive) as zipped:
        members = [
            member
            for member in zipped.infolist()
            if member.filename.startswith("review-main/network_dismantling/")
            or member.filename
            in {
                "review-main/README.md",
                "review-main/CITATIONS.md",
                "review-main/setup.py",
                "review-main/__init__.py",
            }
        ]
        for index, member in enumerate(members, start=1):
            relative = Path(member.filename).relative_to("review-main")
            destination = (target_resolved / relative).resolve()
            try:
                destination.relative_to(target_resolved)
            except ValueError as exc:
                raise RuntimeError(
                    f"Unsafe path in review archive: {member.filename}"
                ) from exc
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == member.file_size:
                checksum = 0
                with destination.open("rb") as existing:
                    for block in iter(lambda: existing.read(1024 * 1024), b""):
                        checksum = zlib.crc32(block, checksum)
                if checksum & 0xFFFFFFFF == member.CRC:
                    continue
            with zipped.open(member) as source, destination.open("wb") as output:
                output.write(source.read())
            if progress is not None and index % 300 == 0:
                progress(f"extracted {index}/{len(members)} official source entries")

    if not (target_resolved / "network_dismantling").is_dir():
        raise RuntimeError("Extracted review archive has no network_dismantling package.")
    _validate_review_root(target_resolved)
    if str(target_resolved) not in sys.path:
        sys.path.insert(0, str(target_resolved))
    return target_resolved


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _require_modules(method: str, modules: Sequence[str]) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"{method} requires additional modules: {', '.join(missing)}. "
            "Run this method in its official review environment and export the "
            "canonical sequence CSV."
        )


def _patch_review_logpipe() -> None:
    try:
        import network_dismantling.common.logging.pipe as pipe
    except Exception:
        return
    logpipe = pipe.LogPipe
    if getattr(logpipe, "_rmcd_patched", False):
        return

    def patched_init(self, logger, level=logging.INFO):
        threading.Thread.__init__(self)
        self.daemon = True
        self.level = level
        self.logger = logger
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    def patched_close(self):
        try:
            os.close(self.fdWrite)
        except OSError:
            pass
        try:
            if self.is_alive():
                self.join(timeout=1.0)
        except RuntimeError:
            pass

    logpipe.__init__ = patched_init
    logpipe.close = patched_close
    logpipe._rmcd_patched = True


def _patch_review_makefiles(
    source_root: Path, progress: Callable[[str], None] | None
) -> None:
    makefile = source_root / "network_dismantling" / "decycler" / "Makefile"
    if not makefile.is_file():
        return
    text = makefile.read_text(encoding="utf-8")
    patched = text
    static_boost = "-Wl,-Bstatic -lboost_program_options -Wl,-Bdynamic"
    boost_static = (
        Path(os.environ.get("CONDA_PREFIX", ""))
        / "lib"
        / "libboost_program_options.a"
    )
    if static_boost in patched and not boost_static.is_file():
        patched = patched.replace(static_boost, "-lboost_program_options")
    old_reverse = "${CXX} ${FLAGS} reverse-greedy.cpp ${LIBS} -o $@ $(CLinkFlags)"
    new_reverse = (
        "${CXX} ${FLAGS} ${CInc} reverse-greedy.cpp ${LIBS} -o $@ $(CLinkFlags)"
    )
    patched = patched.replace(old_reverse, new_reverse)
    dynamic_flags = "CLinkFlags = -L$(BOOST_LIB_LOCATION) -lboost_program_options"
    rpath_flags = (
        "CLinkFlags = -L$(BOOST_LIB_LOCATION) "
        "-Wl,-rpath,$(BOOST_LIB_LOCATION) -lboost_program_options"
    )
    patched = patched.replace(dynamic_flags, rpath_flags)
    if patched != text:
        if progress is not None:
            progress("applying build-only compatibility patch to decycler Makefile")
        makefile.write_text(patched, encoding="utf-8")


def _patch_review_reinsert_parser(
    source_root: Path, progress: Callable[[str], None] | None
) -> None:
    path = source_root / "network_dismantling" / "GDM" / "reinsert.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "_rmcd_parse_removals" in text:
        return
    patched = text.replace("from ast import literal_eval\n", "from ast import literal_eval\nimport re\n")
    helper = r'''

def _rmcd_parse_removals(removals):
    if not isinstance(removals, str):
        return removals
    try:
        return literal_eval(removals)
    except (SyntaxError, ValueError):
        cleaned = removals
        scalar_pattern = (
            r"(?:np\.)?"
            r"(?:float64|float32|float16|int64|int32|int16|int8|"
            r"uint64|uint32|uint16|uint8|bool_)\(([^()]+)\)"
        )
        for _ in range(8):
            next_cleaned = re.sub(scalar_pattern, r"\1", cleaned)
            if next_cleaned == cleaned:
                break
            cleaned = next_cleaned
        cleaned = re.sub(
            r"(?:np\.)?array\((\[[^()]*\])(?:,\s*dtype=[^)]+)?\)",
            r"\1",
            cleaned,
        )
        return literal_eval(cleaned)
'''
    marker = "cached_networks: Dict[Path, str] = {}\n"
    if marker in patched:
        patched = patched.replace(marker, marker + helper, 1)
    else:
        patched = patched.replace("\n\ndef get_predictions(", helper + "\n\ndef get_predictions(", 1)
    patched = patched.replace(
        "removals = literal_eval(removals)",
        "removals = _rmcd_parse_removals(removals)",
    )
    if patched != text:
        if progress is not None:
            progress("applying NumPy scalar compatibility patch to GDM reinsertion parser")
        path.write_text(patched, encoding="utf-8")


def _ensure_external_dismantler(
    source_root: Path, progress: Callable[[str], None] | None
) -> None:
    directory = (
        source_root
        / "network_dismantling"
        / "common"
        / "external_dismantlers"
    )
    if not (directory / "Makefile").is_file():
        raise RuntimeError(f"Official external dismantler Makefile not found: {directory}")

    def complete_extension(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                header = handle.read(4)
            return path.stat().st_size >= 4096 and header == b"\x7fELF"
        except OSError:
            return False

    lock_path = directory / ".rmcd-external-dismantler-build.lock"
    with lock_path.open("a+b") as lock_handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            extensions = tuple(directory.glob("dismantler*.so"))
            if extensions and all(complete_extension(path) for path in extensions):
                return
            if progress is not None:
                progress("building official common external dismantler extension")
            subprocess.run(
                ["make", "clean"],
                cwd=directory,
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["make"],
                cwd=directory,
                text=True,
                capture_output=True,
                check=True,
            )
            extensions = tuple(directory.glob("dismantler*.so"))
            if not extensions or not all(
                complete_extension(path) for path in extensions
            ):
                raise RuntimeError(
                    "Official external dismantler build produced an incomplete "
                    "shared object."
                )
        except subprocess.CalledProcessError as exc:
            output = "\n".join(
                part.strip()
                for part in (exc.stdout or "", exc.stderr or "")
                if part.strip()
            )
            raise RuntimeError(
                "Failed to build the official external dismantler. Install GCC, "
                "make and Boost.Python.\n" + output
            ) from exc
        finally:
            if os.name == "posix":
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _ensure_finder(
    source_root: Path, progress: Callable[[str], None] | None
) -> None:
    directory = source_root / "network_dismantling" / "FINDER_ND"
    if any(directory.glob("FINDER*.so")):
        return
    source_directory = directory / "src" / "lib"
    missing = tuple(
        name
        for name in FINDER_REQUIRED_SOURCE_FILES
        if not (source_directory / name).is_file()
    )
    if missing:
        raise RuntimeError(
            "Official FINDER source is incomplete under FINDER_ND/src/lib; "
            "missing: {0}. Use the complete upstream FINDER source in the "
            "dedicated FINDER environment, then export its sequence CSV.".format(
                ", ".join(missing)
            )
        )
    if progress is not None:
        progress("building official FINDER Cython extension")
    try:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=directory,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part.strip()
            for part in (exc.stdout or "", exc.stderr or "")
            if part.strip()
        )
        raise RuntimeError("Failed to build official FINDER.\n" + output) from exc


def _repair_multiscale_reinsertion_executable(
    source_root: Path,
    progress: Callable[[str], None] | None,
) -> bool:
    """Rebuild a stale non-executable shared reinsertion binary on POSIX."""

    if os.name != "posix":
        return False
    directory = (
        source_root
        / "network_dismantling"
        / "multiscale_entanglement"
        / "reinsertion"
    )
    executable = directory / "reinsertion"
    if not executable.is_file() or os.access(executable, os.X_OK):
        return False
    if progress is not None:
        progress(
            "rebuilding non-executable multiscale-entanglement reinsertion binary"
        )
    try:
        subprocess.run(
            ["make", "clean"],
            cwd=directory,
            text=True,
            capture_output=True,
            check=True,
        )
        built = subprocess.run(
            ["make"],
            cwd=directory,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part.strip()
            for part in (exc.stdout or "", exc.stderr or "")
            if part.strip()
        )
        raise RuntimeError(
            "Failed to rebuild multiscale-entanglement reinsertion.\n" + output
        ) from exc
    if not executable.is_file() or not os.access(executable, os.X_OK):
        output = "\n".join(
            part.strip()
            for part in (built.stdout or "", built.stderr or "")
            if part.strip()
        )
        raise RuntimeError(
            "Rebuilt multiscale-entanglement reinsertion is not executable.\n"
            + output
        )
    return True


def _prepare_source(
    source_root: Path,
    spec: OfficialMethodSpec,
    progress: Callable[[str], None] | None,
) -> tuple[str, ...]:
    patches = ["runtime_logpipe_daemon_close"]
    _patch_review_logpipe()
    if spec.name in {"OfficialMinSum", "OfficialMinSumR"}:
        _patch_review_makefiles(source_root, progress)
        patches.append("decycler_makefile_boost_include_rpath_compat")
    if spec.name in {"OfficialGDM", "OfficialGDMR", "OfficialCoreGDM"}:
        _patch_review_reinsert_parser(source_root, progress)
        patches.append("gdm_numpy_scalar_reinsert_parser_compat")
    if spec.name in {
        "OfficialNetworkEntanglementSmallR",
        "OfficialNetworkEntanglementMidR",
        "OfficialNetworkEntanglementLargeR",
    } and _repair_multiscale_reinsertion_executable(source_root, progress):
        patches.append("multiscale_reinsertion_nonexecutable_rebuild")
    # Wrapped methods use the common LCC-threshold extension either through
    # @dismantler_wrapper, from GDM's network_dismantler.py, or during the
    # reinsertion phase shared by GDMR and CoreGDM.
    needs_external = spec.kind == "wrapped"
    if needs_external:
        _ensure_external_dismantler(source_root, progress)
    if spec.preparation == "finder":
        _ensure_finder(source_root, progress)
    return tuple(patches)


def _to_graph_tool(graph: nx.Graph):
    try:
        from graph_tool import Graph
    except Exception as exc:
        try:
            from graph_tool.all import Graph
        except Exception as nested:
            raise RuntimeError(
                "Official NetworkDismantling methods require graph-tool."
            ) from nested

    equipment = _official_projection_graph(graph)
    nodes = stable_nodes(equipment.nodes)
    node_to_static = {node: index for index, node in enumerate(nodes)}
    static_to_node = {index: node for node, index in node_to_static.items()}

    converted = Graph(directed=False)
    converted.add_vertex(len(nodes))
    static_id = converted.new_vertex_property("int")
    for node, index in node_to_static.items():
        static_id[converted.vertex(index)] = index
    converted.vertex_properties["static_id"] = static_id
    filename = converted.new_graph_property("string")
    filename[converted] = graph_fingerprint(equipment)
    converted.graph_properties["filename"] = filename

    edges: set[tuple[int, int]] = set()
    for left, right in equipment.edges:
        if left == right:
            continue
        first, second = sorted((node_to_static[left], node_to_static[right]))
        edges.add((first, second))
    for first, second in sorted(edges):
        converted.add_edge(converted.vertex(first), converted.vertex(second))
    return converted, static_to_node


def _import_function(spec: OfficialMethodSpec):
    module = importlib.import_module(spec.module)
    return getattr(module, spec.function)


def _coerce_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = [part for part in value.replace(",", " ").split() if part]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return []


def _removal_payload(run: Any) -> list[Any]:
    if isinstance(run, Mapping):
        return _coerce_rows(run.get("removals"))
    if hasattr(run, "columns") and "removals" in getattr(run, "columns"):
        values = getattr(run, "__getitem__")("removals")
        for value in _coerce_rows(values):
            rows = _coerce_rows(value)
            if rows:
                return rows
    return _coerce_rows(run)


def _static_ids_from_run(run: Any) -> tuple[int, ...]:
    payload = _removal_payload(run)
    if not payload:
        return ()
    scalar_payload = all(
        isinstance(item, (int, float, np.integer, np.floating, str))
        for item in payload
    )
    raw_ids: list[Any]
    if scalar_payload:
        raw_ids = payload
    else:
        raw_ids = []
        for position, row in enumerate(payload, start=1):
            if isinstance(row, str):
                try:
                    row = ast.literal_eval(row)
                except (SyntaxError, ValueError):
                    row = [part for part in row.replace(",", " ").split() if part]
            if hasattr(row, "tolist"):
                row = row.tolist()
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                raise RuntimeError(
                    "Official review output contains a malformed removal row "
                    f"at position {position}: {row!r}."
                )
            raw_ids.append(row[1])

    ordered: list[int] = []
    seen: set[int] = set()
    for position, raw in enumerate(raw_ids, start=1):
        if isinstance(raw, (bool, np.bool_)):
            raise RuntimeError(
                f"Official review output has a Boolean node ID at position {position}."
            )
        if isinstance(raw, (int, np.integer)):
            static_id = int(raw)
        elif isinstance(raw, (float, np.floating)):
            value = float(raw)
            if not np.isfinite(value) or not value.is_integer():
                raise RuntimeError(
                    "Official review output has a non-integral node ID "
                    f"{raw!r} at position {position}."
                )
            static_id = int(value)
        elif isinstance(raw, str):
            try:
                static_id = int(raw.strip(), 10)
            except ValueError as exc:
                raise RuntimeError(
                    "Official review output has an invalid node ID "
                    f"{raw!r} at position {position}."
                ) from exc
        else:
            raise RuntimeError(
                "Official review output has an unsupported node ID "
                f"{raw!r} at position {position}."
            )
        if static_id < 0:
            raise RuntimeError(
                f"Official review output has negative node ID {static_id} "
                f"at position {position}."
            )
        if static_id in seen:
            raise RuntimeError(
                f"Official review output repeats node ID {static_id} "
                f"at position {position}."
            )
        ordered.append(static_id)
        seen.add(static_id)
    return tuple(ordered)


def _verified_final_lcc_size(
    graph: nx.Graph,
    sequence: Sequence[NodeId],
    stop_condition: int,
) -> int:
    if len(sequence) != len(set(sequence)):
        raise RuntimeError(
            "Official review method returned duplicate node removals."
        )
    active = nx.Graph(graph)
    active.remove_nodes_from(sequence)
    final_lcc = (
        max(
            (len(component) for component in nx.connected_components(active)),
            default=0,
        )
        if active.number_of_nodes()
        else 0
    )
    if final_lcc > stop_condition:
        raise RuntimeError(
            "Official review sequence did not reach its structural stop "
            f"condition: final LCC={final_lcc}, target<={stop_condition}."
        )
    return final_lcc


def _aligned_heuristic_values(
    vertex_indices: Sequence[int],
    raw_values: Any,
) -> tuple[float, ...]:
    """Align an upstream sorter array with active graph-tool vertices.

    Graph-tool property maps may expose either one value per active vertex or
    one value per vertex in the underlying graph.  The distinction matters
    after removals because filtered vertex indices are no longer contiguous.
    """

    indices = tuple(int(index) for index in vertex_indices)
    values = np.asarray(raw_values, dtype=float)
    if values.ndim != 1:
        raise RuntimeError(
            "Official heuristic sorter returned a non-vector score array "
            f"with shape {values.shape!r}."
        )
    if values.size == len(indices):
        aligned = tuple(float(value) for value in values)
    elif indices and values.size > max(indices):
        aligned = tuple(float(values[index]) for index in indices)
    elif not indices and values.size == 0:
        aligned = ()
    else:
        raise RuntimeError(
            "Official heuristic sorter returned an incompatible score vector: "
            f"{values.size} scores for active vertex indices {indices!r}."
        )
    if not all(np.isfinite(value) for value in aligned):
        raise RuntimeError("Official heuristic sorter returned non-finite scores.")
    return aligned


def _active_graph_view(network: Any) -> tuple[Any, Any]:
    from graph_tool import GraphView

    vertex_filter = network.new_vertex_property("bool")
    for vertex in network.vertices():
        vertex_filter[vertex] = True
    return GraphView(network, vfilt=vertex_filter), vertex_filter


def _active_component_sizes(network: Any) -> tuple[int, int]:
    from graph_tool.topology import label_components

    _belonging, raw_counts = label_components(network)
    counts = sorted((int(value) for value in raw_counts), reverse=True)
    largest = counts[0] if counts else 0
    second_largest = counts[1] if len(counts) > 1 else 0
    return largest, second_largest


def _heuristic_scores(network: Any, sorter: Callable[..., Any]) -> tuple[
    tuple[int, int, float], ...
]:
    vertices = tuple(network.vertices())
    vertex_indices = tuple(int(vertex) for vertex in vertices)
    values = _aligned_heuristic_values(vertex_indices, sorter(network))
    static_id = network.vertex_properties["static_id"]
    return tuple(
        (vertex_index, int(static_id[vertex]), value)
        for vertex_index, vertex, value in zip(vertex_indices, vertices, values)
    )


@contextmanager
def _seeded_numpy_random(seed: int, *, enabled: bool):
    if not enabled:
        yield
        return
    with _NUMPY_RANDOM_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


def _run_heuristic(
    network: Any,
    spec: OfficialMethodSpec,
    stop_condition: int,
    network_name: str,
    seed: int,
) -> Mapping[str, Any]:
    """Run official heuristic scores on the surviving-node graph.

    The upstream dynamic generator clears removed vertices but leaves them in
    the graph as zero-degree isolates.  Tied centrality values can then select
    the same static ID repeatedly, and eigenvector centrality can stall on the
    accumulating isolates.  This adapter keeps the upstream sorter unchanged
    while applying it to an active GraphView and reproducing the same LCC
    stopping rule in Python.
    """

    del network_name
    sorter = _import_function(spec)
    working = network.copy()
    network_size = int(working.num_vertices())
    active_indices = {int(vertex) for vertex in working.vertices()}
    active_view, active_filter = _active_graph_view(working)
    removals: list[tuple[int, int, float, float, float]] = []
    score_seconds = 0.0
    ordering_seconds = 0.0
    lcc_update_seconds = 0.0
    started = perf_counter()

    with _seeded_numpy_random(seed, enabled=spec.function == "get_random"):
        static_order: tuple[tuple[int, int, float], ...] | None = None
        static_cursor = 0
        if spec.kind == "heuristic_static":
            prediction_started = perf_counter()
            initial_scores = _heuristic_scores(active_view, sorter)
            score_seconds += perf_counter() - prediction_started
            ordering_started = perf_counter()
            static_order = tuple(
                sorted(
                    initial_scores,
                    key=lambda item: item[2],
                    reverse=True,
                )
            )
            ordering_seconds += perf_counter() - ordering_started

        while active_indices:
            if static_order is None:
                prediction_started = perf_counter()
                candidates = _heuristic_scores(active_view, sorter)
                score_seconds += perf_counter() - prediction_started
                if not candidates:
                    break
                ordering_started = perf_counter()
                selected = max(candidates, key=lambda item: item[2])
                ordering_seconds += perf_counter() - ordering_started
            else:
                if static_cursor >= len(static_order):
                    break
                selected = static_order[static_cursor]
                static_cursor += 1

            vertex_index, selected_static_id, score = selected
            if vertex_index not in active_indices:
                continue
            active_indices.remove(vertex_index)
            active_filter[working.vertex(vertex_index)] = False
            lcc_started = perf_counter()
            largest, second_largest = _active_component_sizes(active_view)
            lcc_update_seconds += perf_counter() - lcc_started
            step = len(removals) + 1
            denominator = max(1, network_size)
            removals.append(
                (
                    step,
                    selected_static_id,
                    score,
                    largest / denominator,
                    second_largest / denominator,
                )
            )
            # Preserve the upstream threshold driver: the stopping event is
            # checked after a removal, not before the first score evaluation.
            if largest <= stop_condition:
                break

    total_time = perf_counter() - started
    return {
        "removals": removals,
        "adapter_wall_seconds": total_time,
        "score_seconds": score_seconds,
        "ordering_seconds": ordering_seconds,
        "active_lcc_update_seconds": lcc_update_seconds,
    }


def _run_bruteforce_target_ranking(
    network: Any,
    spec: OfficialMethodSpec,
    stop_condition: int,
    max_nodes: int,
) -> Mapping[str, Any]:
    """Convert the official exhaustive LCC target property to a ranking.

    The upstream brute-force routine does not expose a removal sequence.  It
    returns each node's occurrence frequency across the best LCC-dismantling
    combinations.  We preserve that official output and apply a deterministic
    descending-score/static-id order so it can be evaluated at the RMCD event.
    """

    node_count = int(network.num_vertices())
    if node_count > int(max_nodes):
        raise RuntimeError(
            "ReviewBruteForceFrequencyRank is intentionally limited to "
            f"{max_nodes} nodes because its upstream implementation enumerates "
            f"node combinations; received {node_count}. Run it only in the "
            "small-scale exact-reference experiment."
        )
    function = _import_function(spec)
    targets = function(
        network.copy(),
        stop_condition=stop_condition,
        k_range=range(1, node_count + 1),
    )
    static_id = network.vertex_properties["static_id"]
    scored: list[tuple[float, int]] = []
    for vertex in network.vertices():
        raw = targets[vertex]
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = 0.0
        scored.append((score, int(static_id[vertex])))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {"removals": tuple(static for _score, static in scored)}


def _run_wrapped(
    network: Any,
    spec: OfficialMethodSpec,
    stop_condition: int,
    network_name: str,
    progress: Callable[[str], None] | None,
    dependency_static_ids: dict[str, tuple[int, ...]] | None = None,
) -> Any:
    function = _import_function(spec)
    generator_args: dict[str, Any] = {
        "network_name": network_name,
        "logger": LOGGER,
        "stop_condition": stop_condition,
    }
    if spec.function.startswith("network_entanglement_"):
        generator_args["executor"] = _SerialExecutor()

    if spec.dependency is not None:
        dependency_spec = OFFICIAL_METHOD_SPECS[spec.dependency]
        if progress is not None:
            progress(f"computing dependency {dependency_spec.name} for {spec.name}")
        dependency_run = _run_wrapped(
            network.copy(),
            dependency_spec,
            stop_condition,
            network_name,
            progress,
            dependency_static_ids,
        )
        dependency_ids = tuple(_static_ids_from_run(dependency_run))
        if dependency_static_ids is not None:
            dependency_static_ids[spec.dependency] = dependency_ids
        generator_args[str(spec.dependency_argument)] = np.asarray(
            dependency_ids,
            dtype=int,
        )

    threshold = stop_condition / max(1, int(network.num_vertices()))
    if spec.name in {"OfficialFINDER_R", "OfficialEGND"}:
        return function(
            network.copy(),
            stop_condition=stop_condition,
            logger=LOGGER,
            generator_args=generator_args,
        )
    if spec.name in {"OfficialGND", "OfficialGNDR"}:
        return function(
            network.copy(),
            stop_condition=stop_condition,
            logger=LOGGER,
            generator_args=generator_args,
            predictor=_external_threshold_isolate_safe_predictor,
        )
    if spec.name in {"OfficialGDM", "OfficialGDMR"}:
        from torch import multiprocessing as torch_multiprocessing

        manager = torch_multiprocessing.Manager()
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                return function(
                    network.copy(),
                    stop_condition=stop_condition,
                    threshold=threshold,
                    logger=LOGGER,
                    executor=executor,
                    pool_size=1,
                    mp_manager=manager,
                )
        finally:
            manager.shutdown()
    if spec.name == "OfficialCoreGDM":
        return function(
            network.copy(),
            stop_condition=stop_condition,
            threshold=threshold,
            logger=LOGGER,
        )
    return function(
        network.copy(),
        stop_condition=stop_condition,
        logger=LOGGER,
        generator_args=generator_args,
    )


def _external_threshold_isolate_safe_predictor(
    network: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, Any]:
    """Keep edge-list-omitted isolates behind every represented vertex.

    The review extension reconstructs its active graph from an edge list, so
    isolated graph-tool vertices are absent from that C++ graph.  GND leaves
    unselected vertices at score zero; stable sorting can otherwise send an
    omitted isolate to the extension before the LCC target is reached.  Moving
    only those isolates to the tail preserves every upstream GND score and the
    relative order of all edge-represented vertices.
    """

    from network_dismantling.dismantler import get_predictions

    predictions, prediction_time = get_predictions(network, **kwargs)
    guarded = np.asarray(predictions, dtype=float).copy()
    isolated_indices = [
        int(vertex)
        for vertex in network.vertices()
        if int(vertex.out_degree()) == 0
    ]
    if isolated_indices:
        guarded[isolated_indices] = -np.inf
    return guarded, prediction_time


def run_official_review_sequence(
    graph: nx.Graph,
    method: str,
    *,
    review_path: str | Path | None = None,
    zip_path: str | Path | None = None,
    extraction_root: str | Path | None = None,
    stop_condition: int = 1,
    brute_force_max_nodes: int = 18,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> OfficialSequenceResult:
    """Run one official method and return its mapped structural sequence."""

    if method not in OFFICIAL_METHOD_SPECS:
        raise ValueError(f"Unknown official review method: {method!r}.")
    if stop_condition < 1:
        raise ValueError("stop_condition must be at least 1.")
    if brute_force_max_nodes < 4:
        raise ValueError("brute_force_max_nodes must be at least 4.")
    equipment = _official_projection_graph(graph)
    applicability = official_method_applicability(
        equipment,
        method,
        brute_force_max_nodes=brute_force_max_nodes,
    )
    if not applicability.applicable:
        raise OfficialMethodNotApplicableError(
            method,
            applicability.code,
            applicability.detail,
        )
    spec = OFFICIAL_METHOD_SPECS[method]
    source_root = ensure_review_source(
        review_path=review_path,
        zip_path=zip_path,
        extraction_root=extraction_root,
        progress=progress,
    )
    source_identity_kind, source_identity = _source_identity(source_root)
    _require_modules(method, spec.required_modules)
    compatibility_patches = _prepare_source(source_root, spec, progress)
    if spec.name in {"OfficialGND", "OfficialGNDR"} and any(
        degree == 0 for _node, degree in equipment.to_undirected().degree()
    ):
        compatibility_patches = (
            *compatibility_patches,
            "external_threshold_isolated_vertex_tail_guard",
        )
    if spec.kind.startswith("heuristic_"):
        compatibility_patches = (
            *compatibility_patches,
            "heuristic_active_graph_lcc_adapter",
        )
    converted, static_to_node = _to_graph_tool(equipment)
    if converted.num_vertices() == 0:
        raise RuntimeError("Official review method received an empty equipment graph.")
    effective_stop = min(int(stop_condition), max(1, int(converted.num_vertices())))
    fingerprint = graph_fingerprint(equipment)
    if progress is not None:
        progress(f"calling {method} on undirected equipment projection")
    started = perf_counter()
    dependency_static_ids: dict[str, tuple[int, ...]] = {}
    with _working_directory(source_root):
        if spec.kind == "bruteforce_target":
            run = _run_bruteforce_target_ranking(
                converted,
                spec,
                effective_stop,
                brute_force_max_nodes,
            )
        elif spec.kind.startswith("heuristic_"):
            run = _run_heuristic(
                converted,
                spec,
                effective_stop,
                fingerprint,
                seed,
            )
        else:
            run = _run_wrapped(
                converted,
                spec,
                effective_stop,
                fingerprint,
                progress,
                dependency_static_ids,
            )
    static_ids = _static_ids_from_run(run)
    unknown_ids = tuple(
        static_id for static_id in static_ids if static_id not in static_to_node
    )
    if unknown_ids:
        raise RuntimeError(
            f"{method} returned out-of-range static node IDs: {unknown_ids}."
        )
    sequence = tuple(static_to_node[static_id] for static_id in static_ids)
    if not sequence:
        raise RuntimeError(f"{method} returned no valid node removals.")
    final_lcc_size = _verified_final_lcc_size(
        equipment.to_undirected(),
        sequence,
        effective_stop,
    )
    compatibility_patches = (
        *compatibility_patches,
        "sequence_lcc_target_replay_verification",
    )
    dependency_method = spec.dependency or ""
    dependency_sequence = tuple(
        static_to_node[static_id]
        for static_id in dependency_static_ids.get(dependency_method, ())
    )
    return OfficialSequenceResult(
        method=method,
        method_group=spec.group,
        sequence=sequence,
        graph_fingerprint=fingerprint,
        node_count=equipment.number_of_nodes(),
        edge_count=equipment.to_undirected().number_of_edges(),
        stop_condition=effective_stop,
        verified_final_lcc_size=final_lcc_size,
        runtime_seconds=perf_counter() - started,
        review_root=str(source_root),
        sequence_semantics=(
            "official_optimal_lcc_target_frequency_ranking"
            if spec.kind == "bruteforce_target"
            else "upstream_static_sorter_python_active_lcc_order"
            if spec.kind == "heuristic_static"
            else "upstream_sorter_active_graph_dynamic_order"
            if spec.kind == "heuristic_dynamic"
            else "official_structural_removal_order"
        ),
        source_identity_kind=source_identity_kind,
        source_identity=source_identity,
        compatibility_patches=compatibility_patches,
        adapter_version=OFFICIAL_ADAPTER_VERSION,
        projection_fingerprint=official_projection_fingerprint(equipment),
        sequence_sha256=sequence_sha256(sequence),
        dependency_method=dependency_method,
        dependency_sequence_sha256=(
            sequence_sha256(dependency_sequence) if dependency_sequence else ""
        ),
    )
