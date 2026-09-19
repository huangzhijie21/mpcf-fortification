"""Canonical identifiers for the MPCF revision experiments.

Three identity levels are used throughout, matching the revised manuscript:

``graph_id``
    ``topology + N + seed``.  In the main experiment the 45 graphs are the
    independent experimental units; ``budget`` is a repeated measure *within*
    one ``graph_id``.

``base_id``
    ``topology + N + seed`` as well.  In the role-composition experiment the
    six compositions and three budgets sharing one ``base_id`` form a single
    repeated-measures cluster and are never treated as independent samples.

``instance_id``
    A globally unique identifier for one frozen graph *view*: the experiment,
    the base graph, and any panel-specific modifier (composition, cost profile,
    correlation profile, scale tier).  One ``instance_id`` maps to exactly one
    frozen graph file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ID_VERSION = "mpcf-ids-v1"
SEPARATOR = "__"


def graph_id(topology: str, node_count: int, seed: int) -> str:
    """Independent experimental unit of the main experiment."""

    return f"{str(topology)}-N{int(node_count)}-s{int(seed)}"


def base_id(topology: str, node_count: int, seed: int) -> str:
    """Repeated-measures cluster identifier.

    Identical in form to :func:`graph_id`; the two names are kept distinct so
    that a reader of the CSV can see which clustering the analysis uses.
    """

    return graph_id(topology, node_count, seed)


def instance_id(
    experiment: str,
    base: str,
    *modifiers: str | None,
) -> str:
    """Globally unique frozen-instance identifier.

    ``modifiers`` are appended in the given order, skipping empty values, so
    ``instance_id("role", "centralized-N42-s11", "4-3-3-4")`` yields
    ``role__centralized-N42-s11__4-3-3-4``.
    """

    parts: list[str] = [str(experiment), str(base)]
    for modifier in modifiers:
        if modifier is None:
            continue
        text = str(modifier).strip()
        if text:
            parts.append(text)
    return SEPARATOR.join(parts)


def composition_label(role_sizes: Sequence[int] | Mapping[str, int]) -> str:
    """``"4-3-3-4"`` style label for one role-size vector."""

    if isinstance(role_sizes, Mapping):
        values = [int(role_sizes[key]) for key in ("S", "C", "L", "E")]
    else:
        values = [int(value) for value in role_sizes]
    return "-".join(str(value) for value in values)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON used for every hash in this revision layer."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def config_hash(payload: Any) -> str:
    """Short stable digest of a configuration mapping."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Files whose contents actually determine a reported number: the solver core,
#: the instance generator, the cost systems, the baseline adapters and the
#: revision harness/panels.  Hashing exactly these keeps the scientific code
#: revision stable while analysis, plotting, gate and documentation edits move
#: independently -- a broad tree hash would report a "different revision" for a
#: comment change, which would be misleading rather than protective.
NUMERICAL_CORE: tuple[str, ...] = (
    "src/rmcd_f/task_path_fortification.py",
    "src/rmcd_f/operational_motif.py",
    "src/rmcd_f/synthetic.py",
    "src/rmcd_f/cost_profiles.py",
    "src/rmcd_f/model.py",
    "src/rmcd_f/protection_baselines.py",
    "src/rmcd_f/mpcf_supplementary.py",
    "src/rmcd_f/baselines.py",
    "src/rmcd_f/keyset.py",
    "src/rmcd_f/official_review.py",
    "src/rmcd_f/rev/panels.py",
    "src/rmcd_f/rev/harness.py",
)


#: Revision identifier recorded on every row of the published archive, under
#: the original **byte-exact** rule.  That rule hashed the raw bytes of
#: :data:`NUMERICAL_CORE` as deployed, and the deployed tree happened to carry
#: mixed line endings (``task_path_fortification.py`` in CRLF,
#: ``rev/panels.py`` in LF), so the value is not reproducible from a fresh
#: checkout on an arbitrary platform.
LEGACY_CODE_REVISION = "core-a8deebd9ddef8772f9768affdf9ee483d1e48193"

#: The same code as :data:`LEGACY_CODE_REVISION`, hashed under the
#: line-ending-normalised rule and before the solver identifiers were renamed
#: from their historical prefix to ``MPCF``.
PRE_RENAME_CODE_REVISION = "core-f018acb54d16e1cfd42e539d86562dcff1577365"

#: This repository, under the normalised rule.  It differs from
#: :data:`PRE_RENAME_CODE_REVISION` only in identifier names and in the
#: ``NUMERICAL_CORE`` module filenames: no formula, bound, cost system or panel
#: definition changed, which was checked by relabelling the published archive
#: and regenerating every statistics and table CSV with identical numbers.
RENAMED_CODE_REVISION = "core-75c2b55cf133a5bf762f4277b252b8c7977da757"


def code_commit(root: Path | str | None = None) -> str:
    """Revision identifier for the code that produced the numbers.

    Uses the git commit when the tree is a checkout, and otherwise a content
    digest over :data:`NUMERICAL_CORE` only.  The revision therefore changes
    when a solver, generator, cost system, baseline adapter or panel definition
    changes, and does not change when statistics, plotting, gates or prose are
    edited.

    Line endings are normalised before hashing.  Hashing raw bytes instead
    would make the identifier depend on the checkout platform -- the same
    source in CRLF and in LF is the same code, and two people reproducing the
    same experiment should not be told they used different revisions.  See
    :data:`LEGACY_CODE_REVISION` for the value the published archive records
    under the older, byte-exact rule.
    """

    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    digest = hashlib.sha256()
    missing: list[str] = []
    for relative in NUMERICAL_CORE:
        path = base / relative
        if not path.is_file():
            missing.append(relative)
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    if missing:
        raise FileNotFoundError(
            f"Cannot fingerprint the numerical core; missing: {missing}"
        )
    return "core-" + digest.hexdigest()[:40]


def freeze_id(*parts: str) -> str:
    """Short digest used for manifest-level freeze identifiers."""

    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))
