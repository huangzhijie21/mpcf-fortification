#!/usr/bin/env python3
"""Verify the bundled official NetworkDismantling and FINDER source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
FINDER_REQUIRED_MODEL_FILES = (
    "nrange_30_50_iter_78000.ckpt.data-00000-of-00001",
    "nrange_30_50_iter_78000.ckpt.index",
    "nrange_30_50_iter_78000.ckpt.meta",
)


def _tree_digest(root: Path, relative_files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_files):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with (root / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify(review_root: Path) -> dict[str, object]:
    review_root = review_root.resolve()
    finder_root = review_root / "network_dismantling" / "FINDER_ND"
    source_root = finder_root / "src" / "lib"
    model_root = finder_root / "models"

    missing_review = tuple(
        relative
        for relative in (
            "README.md",
            "CITATIONS.md",
            "network_dismantling/__init__.py",
            "network_dismantling/FINDER_ND/python_interface.py",
            "network_dismantling/FINDER_ND/setup.py",
            "network_dismantling/FINDER_ND/UPSTREAM_FINDER_LICENSE",
            "network_dismantling/FINDER_ND/UPSTREAM_FINDER_README.md",
        )
        if not (review_root / relative).is_file()
    )
    missing_source = tuple(
        name
        for name in FINDER_REQUIRED_SOURCE_FILES
        if not (source_root / name).is_file()
    )
    empty_source = tuple(
        name
        for name in FINDER_REQUIRED_SOURCE_FILES
        if (source_root / name).is_file()
        and (source_root / name).stat().st_size == 0
    )
    missing_models = tuple(
        name
        for name in FINDER_REQUIRED_MODEL_FILES
        if not (model_root / name).is_file()
    )
    ok = not (missing_review or missing_source or empty_source or missing_models)

    report: dict[str, object] = {
        "ok": ok,
        "review_root": str(review_root),
        "finder_source_file_count": len(FINDER_REQUIRED_SOURCE_FILES),
        "finder_model_file_count": len(FINDER_REQUIRED_MODEL_FILES),
        "missing_review_files": list(missing_review),
        "missing_finder_source_files": list(missing_source),
        "empty_finder_source_files": list(empty_source),
        "missing_finder_model_files": list(missing_models),
    }
    if ok:
        report["finder_source_tree_sha256"] = _tree_digest(
            source_root, FINDER_REQUIRED_SOURCE_FILES
        )
    return report


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-path",
        type=Path,
        default=project_root / "external" / "review-main",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = verify(args.review_path)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    status = "PASS" if report["ok"] else "FAIL"
    print(f"[{status}] official source bundle: {report['review_root']}")
    if report["ok"]:
        print(
            "  FINDER source files: {0}; models: {1}".format(
                report["finder_source_file_count"],
                report["finder_model_file_count"],
            )
        )
        print(
            "  FINDER source tree SHA-256: "
            + str(report["finder_source_tree_sha256"])
        )
        return 0

    for key in (
        "missing_review_files",
        "missing_finder_source_files",
        "empty_finder_source_files",
        "missing_finder_model_files",
    ):
        values = report[key]
        if values:
            print(f"  {key}: {', '.join(values)}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
