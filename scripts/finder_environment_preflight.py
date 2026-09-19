#!/usr/bin/env python
"""Validate the isolated Python 3.7 environment for official FINDER inference."""

from __future__ import print_function

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


EXPECTED_PACKAGES = (
    ("Cython", "0.29.13"),
    ("absl-py", "0.8.1"),
    ("astor", "0.8.1"),
    ("gast", "0.2.2"),
    ("google-pasta", "0.1.8"),
    ("grpcio", "1.32.0"),
    ("h5py", "2.10.0"),
    ("Keras-Applications", "1.0.8"),
    ("Keras-Preprocessing", "1.1.2"),
    ("Markdown", "3.1.1"),
    ("networkx", "2.3"),
    ("numpy", "1.17.3"),
    ("pandas", "0.25.2"),
    ("protobuf", "3.20.3"),
    ("scipy", "1.3.1"),
    ("six", "1.12.0"),
    ("tensorboard", "1.14.0"),
    ("tensorflow", "1.14.0"),
    ("tensorflow-estimator", "1.14.0"),
    ("termcolor", "1.1.0"),
    ("tqdm", "4.36.1"),
    ("Werkzeug", "0.16.1"),
    ("wrapt", "1.11.2"),
)


def _distribution_version(name):
    import pkg_resources

    return pkg_resources.get_distribution(name).version


def _load_exporter():
    path = Path(__file__).resolve().with_name("export_finder_legacy.py")
    spec = importlib.util.spec_from_file_location(
        "rmcd_f_finder_environment_preflight_exporter",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load export_finder_legacy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_report(path, report):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pip_freeze():
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return sorted(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-path", required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    review_root = Path(args.review_path).resolve()
    finder_root = review_root / "network_dismantling" / "FINDER_ND"
    report = {
        "success": False,
        "python_executable": sys.executable,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "review_root": str(review_root),
        "finder_root": str(finder_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "checks": {},
    }
    failures = []

    python_ok = sys.version_info[:2] == (3, 7)
    report["checks"]["python_3_7"] = python_ok
    if not python_ok:
        failures.append("Python 3.7.x is required")

    package_versions = {}
    for name, expected in EXPECTED_PACKAGES:
        try:
            actual = _distribution_version(name)
            package_versions[name] = actual
            if actual != expected:
                failures.append(
                    "{0}=={1} required; found {2}".format(
                        name, expected, actual
                    )
                )
        except Exception as exc:
            package_versions[name] = None
            failures.append("{0} unavailable: {1}".format(name, exc))
    report["package_versions"] = package_versions
    report["checks"]["pinned_packages"] = not any(
        value is None or value != dict(EXPECTED_PACKAGES)[name]
        for name, value in package_versions.items()
    )

    try:
        exporter = _load_exporter()
        source_root = finder_root / "src" / "lib"
        missing_source = [
            name
            for name in exporter.FINDER_REQUIRED_SOURCE_FILES
            if not (source_root / name).is_file()
        ]
        model_prefix = finder_root / "models" / exporter.FINDER_MODEL_NAME
        missing_model = [
            str(path)
            for path in (
                Path(str(model_prefix) + ".data-00000-of-00001"),
                Path(str(model_prefix) + ".index"),
                Path(str(model_prefix) + ".meta"),
            )
            if not path.is_file()
        ]
        if missing_source:
            raise RuntimeError(
                "missing FINDER source files: {0}".format(
                    ", ".join(missing_source)
                )
            )
        if missing_model:
            raise RuntimeError(
                "missing FINDER checkpoint files: {0}".format(
                    ", ".join(missing_model)
                )
            )
        source_hash = exporter._source_hash(review_root)
        report["source_identity"] = source_hash
        report["source_file_count"] = len(
            exporter.FINDER_REQUIRED_SOURCE_FILES
        )
        report["checkpoint_file_count"] = 3
        report["checks"]["source_tree"] = True
    except Exception as exc:
        exporter = None
        failures.append("source-tree validation failed: {0}".format(exc))
        report["checks"]["source_tree"] = False

    if args.build and exporter is not None:
        try:
            exporter._ensure_builds(review_root)
            report["checks"]["cython_extensions"] = True
        except Exception as exc:
            failures.append("extension build failed: {0}".format(exc))
            report["checks"]["cython_extensions"] = False
            report["build_traceback"] = traceback.format_exc()

    try:
        import networkx
        import numpy
        import scipy
        import tensorflow as tf

        report["runtime_imports"] = {
            "networkx": networkx.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "tensorflow": tf.__version__,
            "tensorflow_session_api": bool(hasattr(tf, "Session")),
        }
        if not hasattr(tf, "Session"):
            raise RuntimeError("TensorFlow 1.x Session API is unavailable")
        report["checks"]["runtime_imports"] = True
    except Exception as exc:
        failures.append("runtime imports failed: {0}".format(exc))
        report["checks"]["runtime_imports"] = False
        report["import_traceback"] = traceback.format_exc()

    if args.load_checkpoint and not failures:
        finder_path = str(finder_root)
        if finder_path not in sys.path:
            sys.path.insert(0, finder_path)
        try:
            from FINDER import FINDER

            model_path = finder_root / "models" / exporter.FINDER_MODEL_NAME
            dqn = FINDER()
            try:
                dqn.LoadModel(str(model_path))
            finally:
                session = getattr(dqn, "session", None)
                if session is not None:
                    session.close()
            report["checkpoint"] = str(model_path)
            report["checks"]["checkpoint_restore"] = True
        except Exception as exc:
            failures.append("checkpoint restore failed: {0}".format(exc))
            report["checks"]["checkpoint_restore"] = False
            report["checkpoint_traceback"] = traceback.format_exc()

    try:
        report["pip_freeze"] = _pip_freeze()
    except Exception as exc:
        failures.append("pip freeze failed: {0}".format(exc))

    report["failures"] = failures
    report["success"] = not failures
    _write_report(
        Path(args.json_output).resolve() if args.json_output else None,
        report,
    )

    if failures:
        print("[FAIL] official FINDER environment preflight")
        for failure in failures:
            print("  - " + failure)
        return 3
    print("[PASS] official FINDER environment preflight")
    print("  Python: {0} at {1}".format(report["python_version"], sys.executable))
    print("  Source identity: " + report["source_identity"])
    if args.load_checkpoint:
        print("  Checkpoint restored: " + report["checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
