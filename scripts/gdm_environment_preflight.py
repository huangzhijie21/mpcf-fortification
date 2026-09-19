#!/usr/bin/env python3
"""Read-only environment audit for official GDM, GDMR, and CoreGDM."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA_VERSION = "gdm-environment-preflight-1.2"
EXPECTED_MODEL_FILE_COUNT = 70
EXPECTED_MODEL_TOTAL_BYTES = 73_489_640
EXPECTED_MODEL_TREE_SHA256 = (
    "36571708695dea9d6bfcc1346ff1214090aafb63a2cdaf2d0da8e724d1f9cdcf"
)
MODEL_BASE_RELATIVE_ROOT = Path(
    "network_dismantling/GDM/out/models_newpg"
)
MODEL_RELATIVE_ROOT = (
    MODEL_BASE_RELATIVE_ROOT
    / "synth_train_NEW"
    / "t_0.18"
    / "GAT_Model"
)
REQUIRED_SOURCE_FILES = (
    "network_dismantling/__init__.py",
    "network_dismantling/GDM/python_interface.py",
    "network_dismantling/GDM/dataset_providers.py",
    "network_dismantling/GDM/network_dismantler.py",
    "network_dismantling/GDM/models/GAT.py",
    "network_dismantling/GDM/models/layers/gat_conv.py",
    "network_dismantling/CoreGDM/python_interface.py",
    "network_dismantling/CoreGDM/core_grid.py",
    "network_dismantling/CoreGDM/core_network_dismantler.py",
)
REQUIRED_RUNTIME_MODULES = (
    "dill",
    "graph_tool",
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "torch_geometric",
    "torch_sparse",
    "tqdm",
)


class PrerequisiteError(RuntimeError):
    """A runtime check could not start because an earlier check failed."""


class CheckFailure(RuntimeError):
    """A failed runtime check that still has useful diagnostic details."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _filesystem_path(path: Path) -> Path:
    """Return a Windows extended-length path without changing report paths."""

    resolved = path.resolve()
    if os.name != "nt":
        return resolved
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return resolved
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def _file_size(path: Path) -> int:
    return _filesystem_path(path).stat().st_size


def _validate_models_folder_path(
    actual_models: Path,
    review_root: Path,
) -> dict[str, str]:
    """Validate the official model-tree root and its frozen checkpoint leaf."""

    expected_root = (review_root / MODEL_BASE_RELATIVE_ROOT).resolve()
    expected_leaf = (review_root / MODEL_RELATIVE_ROOT).resolve()
    actual_root = actual_models.resolve()
    if actual_root != expected_root:
        raise RuntimeError(
            "official GDM resolved its model-tree root incorrectly: "
            f"{actual_root} != {expected_root}"
        )
    if not expected_leaf.is_dir():
        raise FileNotFoundError(
            "official GDM checkpoint leaf is absent: "
            f"{expected_leaf}"
        )
    return {
        "models_folder_path": str(actual_root),
        "checkpoint_leaf": str(expected_leaf),
    }


def _tree_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = _file_size(path)
        digest.update(size.to_bytes(8, "big"))
        with _filesystem_path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _inspect_review_tree(review_root: Path) -> dict[str, Any]:
    review_root = review_root.resolve()
    missing_source_files = [
        relative
        for relative in REQUIRED_SOURCE_FILES
        if not (review_root / relative).is_file()
    ]
    model_root = review_root / MODEL_RELATIVE_ROOT
    model_files = sorted(model_root.glob("*.h5")) if model_root.is_dir() else []
    empty_model_files = [
        path.name for path in model_files if _file_size(path) == 0
    ]
    total_model_bytes = sum(_file_size(path) for path in model_files)
    model_tree_sha256 = (
        _tree_digest(model_root, model_files) if model_files else ""
    )
    representative_model = (
        min(model_files, key=_file_size) if model_files else None
    )
    extension_root = (
        review_root
        / "network_dismantling"
        / "common"
        / "external_dismantlers"
    )
    extensions = sorted(extension_root.glob("dismantler*.so"))
    errors: list[str] = []
    if missing_source_files:
        errors.append("required GDM/CoreGDM source files are missing")
    if not model_root.is_dir():
        errors.append("official GDM pretrained-model directory is missing")
    elif not model_files:
        errors.append("official GDM pretrained-model directory contains no .h5 files")
    if empty_model_files:
        errors.append("one or more official GDM pretrained models are empty")
    if len(model_files) != EXPECTED_MODEL_FILE_COUNT:
        errors.append(
            "official GDM pretrained-model count differs from the frozen bundle"
        )
    if total_model_bytes != EXPECTED_MODEL_TOTAL_BYTES:
        errors.append(
            "official GDM pretrained-model byte count differs from the frozen bundle"
        )
    if (
        EXPECTED_MODEL_TREE_SHA256
        and model_tree_sha256 != EXPECTED_MODEL_TREE_SHA256
    ):
        errors.append(
            "official GDM pretrained-model SHA-256 differs from the frozen bundle"
        )
    return {
        "ok": not errors,
        "review_root": str(review_root),
        "required_source_file_count": len(REQUIRED_SOURCE_FILES),
        "missing_source_files": missing_source_files,
        "model_root": str(model_root),
        "model_file_count": len(model_files),
        "model_total_bytes": total_model_bytes,
        "model_tree_sha256": model_tree_sha256,
        "empty_model_files": empty_model_files,
        "representative_model": (
            str(representative_model) if representative_model else ""
        ),
        "external_extension_files": [str(path) for path in extensions],
        "errors": errors,
    }


def _inspect_external_build_environment(
    review_root: Path,
    *,
    conda_prefix: str | None = None,
) -> dict[str, Any]:
    review_root = review_root.resolve()
    extension_root = (
        review_root
        / "network_dismantling"
        / "common"
        / "external_dismantlers"
    )
    extensions = sorted(extension_root.glob("dismantler*.so"))
    prefix_text = conda_prefix if conda_prefix is not None else os.environ.get(
        "CONDA_PREFIX", ""
    )
    prefix = Path(prefix_text).resolve() if prefix_text else None
    interpreter_prefix = Path(sys.prefix).resolve()
    prefix_matches_interpreter = bool(
        prefix and prefix == interpreter_prefix
    )
    path_python = shutil.which("python") or ""
    path_python_matches_interpreter = bool(
        path_python
        and Path(path_python).resolve() == Path(sys.executable).resolve()
    )
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    boost_headers = (
        [prefix / "include" / "boost" / "python.hpp"] if prefix else []
    )
    python_headers = (
        [prefix / "include" / f"python{python_version}" / "Python.h"]
        if prefix
        else []
    )
    boost_libraries = (
        sorted((prefix / "lib").glob("libboost_python*.so*"))
        if prefix and (prefix / "lib").is_dir()
        else []
    )
    python_libraries = (
        sorted((prefix / "lib").glob(f"libpython{python_version}.so*"))
        if prefix and (prefix / "lib").is_dir()
        else []
    )
    tools = {name: shutil.which(name) or "" for name in ("make", "g++")}
    makefile = extension_root / "Makefile"
    build_prerequisites = {
        "conda_prefix": str(prefix) if prefix else "",
        "interpreter_prefix": str(interpreter_prefix),
        "prefix_matches_interpreter": prefix_matches_interpreter,
        "path_python": path_python,
        "path_python_matches_interpreter": path_python_matches_interpreter,
        "makefile": str(makefile) if makefile.is_file() else "",
        "tools": tools,
        "boost_headers": [str(path) for path in boost_headers if path.is_file()],
        "python_headers": [
            str(path) for path in python_headers if path.is_file()
        ],
        "boost_libraries": [str(path) for path in boost_libraries],
        "python_libraries": [str(path) for path in python_libraries],
    }
    can_build = bool(
        prefix
        and prefix_matches_interpreter
        and path_python_matches_interpreter
        and makefile.is_file()
        and all(tools.values())
        and build_prerequisites["boost_headers"]
        and build_prerequisites["python_headers"]
        and boost_libraries
        and python_libraries
    )
    runtime_ready = bool(extensions)
    return {
        "runtime_ready": runtime_ready,
        "extension_present": bool(extensions),
        "extension_files": [str(path) for path in extensions],
        "can_build_in_current_environment": can_build,
        "build_prerequisites": build_prerequisites,
        "errors": (
            []
            if runtime_ready
            else [
                "the common dismantler extension is absent; compile it in the "
                "current GDM environment and rerun this preflight"
            ]
        ),
    }


def _module_record(module: Any) -> dict[str, Any]:
    return {
        "version": str(getattr(module, "__version__", "")),
        "file": str(getattr(module, "__file__", "")),
    }


_CHECKPOINT_PATTERN = re.compile(
    r"^F(?P<features>.+?)"
    r"_CL(?P<conv>.+?)"
    r"_H(?P<heads>.+?)"
    r"_FL(?P<fully_connected>.+?)"
    r"_C(?P<concat>.+?)"
    r"_NS(?P<negative_slope>.+?)"
    r"_D(?P<dropout>.+?)"
    r"_B(?P<bias>.+?)"
    r"_S(?P<seed>-?\d+)"
    r"_L"
)


def _parse_checkpoint_architecture(checkpoint: Path) -> SimpleNamespace:
    match = _CHECKPOINT_PATTERN.match(checkpoint.stem)
    if match is None:
        raise ValueError(
            f"unrecognized official GDM checkpoint name: {checkpoint.name}"
        )
    expected_features = [
        "chi_degree",
        "clustering_coefficient",
        "degree",
        "kcore",
    ]
    if match.group("features") != "_".join(expected_features):
        raise ValueError(
            "representative checkpoint uses an unexpected feature schema: "
            + match.group("features")
        )

    def integers(name: str) -> list[int]:
        return [int(value) for value in match.group(name).split("_")]

    def floats(name: str) -> list[float]:
        return [float(value) for value in match.group(name).split("_")]

    def booleans(name: str) -> list[bool]:
        values = match.group(name).split("_")
        if any(value not in {"True", "False"} for value in values):
            raise ValueError(
                f"invalid Boolean field {name}: {match.group(name)}"
            )
        return [value == "True" for value in values]

    conv_layers = integers("conv")
    heads = integers("heads")
    concat = booleans("concat")
    negative_slope = floats("negative_slope")
    dropout = floats("dropout")
    bias = booleans("bias")
    layer_counts = {
        len(conv_layers),
        len(heads),
        len(concat),
        len(negative_slope),
        len(dropout),
        len(bias),
    }
    if len(layer_counts) != 1:
        raise ValueError(
            "checkpoint convolutional parameter lengths are inconsistent"
        )
    return SimpleNamespace(
        features=expected_features,
        conv_layers=conv_layers,
        heads=heads,
        fc_layers=integers("fully_connected"),
        concat=concat,
        negative_slope=negative_slope,
        dropout=dropout,
        bias=bias,
        seed_train=int(match.group("seed")),
    )


def _execute_check(
    name: str,
    function: Callable[[], Mapping[str, Any] | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        details = function()
        return {
            "name": name,
            "status": "pass",
            "elapsed_seconds": time.perf_counter() - started,
            "details": dict(details or {}),
        }
    except BaseException as exc:
        report = {
            "name": name,
            "status": (
                "blocked" if isinstance(exc, PrerequisiteError) else "fail"
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": _tail(traceback.format_exc()),
        }
        details = getattr(exc, "details", None)
        if isinstance(details, Mapping):
            report["details"] = dict(details)
        return report


def _audit_linux_extension_links(extension: Path) -> dict[str, Any]:
    """Audit direct shared-library resolution and record relocation diagnostics."""

    ldd = shutil.which("ldd")
    if not ldd:
        raise RuntimeError("ldd is unavailable for extension ABI audit")
    completed = subprocess.run(
        [ldd, str(extension)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    ldd_output = completed.stdout + completed.stderr
    if completed.returncode != 0 or "not found" in ldd_output:
        raise RuntimeError(
            "ldd found unresolved external-dismantler dependencies:\n"
            + _tail(ldd_output)
        )
    relevant_lines = [
        line.strip()
        for line in ldd_output.splitlines()
        if "libpython" in line or "boost_python" in line
    ]
    if not any("libpython" in line for line in relevant_lines):
        raise RuntimeError("ldd did not resolve a Python shared library")
    if not any("boost_python" in line for line in relevant_lines):
        raise RuntimeError("ldd did not resolve a Boost.Python shared library")
    current_prefix = str(Path(sys.prefix).resolve())
    wrong_prefix = [
        line for line in relevant_lines if current_prefix not in line
    ]
    if wrong_prefix:
        raise RuntimeError(
            "external dismantler links Python/Boost outside the current "
            "environment: " + "; ".join(wrong_prefix)
        )

    relocation = subprocess.run(
        [ldd, "-r", str(extension)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    relocation_output = relocation.stdout + relocation.stderr
    if "not found" in relocation_output:
        raise RuntimeError(
            "ldd -r found a missing external-dismantler shared library:\n"
            + _tail(relocation_output)
        )
    undefined_symbols = [
        line.strip()
        for line in relocation_output.splitlines()
        if line.lstrip().startswith("undefined symbol:")
    ]
    return {
        "ldd_relevant_lines": relevant_lines,
        "ldd_returncode": completed.returncode,
        "ldd_r_returncode": relocation.returncode,
        "ldd_r_undefined_symbol_count": len(undefined_symbols),
        "ldd_r_undefined_symbol_tail": undefined_symbols[-20:],
    }


def _child_runtime_probe(review_root: Path) -> dict[str, Any]:
    review_root = review_root.resolve()
    os.chdir(review_root)
    if str(review_root) not in sys.path:
        sys.path.insert(0, str(review_root))
    context: dict[str, Any] = {}

    def environment_alignment() -> Mapping[str, Any]:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        path_python = shutil.which("python") or ""
        if not conda_prefix:
            raise RuntimeError("CONDA_PREFIX is not set")
        if Path(conda_prefix).resolve() != Path(sys.prefix).resolve():
            raise RuntimeError(
                "CONDA_PREFIX does not match sys.prefix: "
                f"{conda_prefix} != {sys.prefix}"
            )
        if not path_python:
            raise RuntimeError("python is not available on PATH")
        if Path(path_python).resolve() != Path(sys.executable).resolve():
            raise RuntimeError(
                "PATH resolves python from a different environment: "
                f"{path_python} != {sys.executable}"
            )
        return {
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "conda_prefix": conda_prefix,
            "path_python": path_python,
        }

    def import_dependencies() -> Mapping[str, Any]:
        modules: dict[str, Any] = {}
        failures: dict[str, str] = {}
        for name in REQUIRED_RUNTIME_MODULES:
            try:
                modules[name] = importlib.import_module(name)
            except BaseException as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
        context.update(modules)
        details = {
            name: _module_record(module) for name, module in modules.items()
        }
        torch = modules.get("torch")
        if torch is not None:
            details["cuda"] = {
                "available": bool(torch.cuda.is_available()),
                "torch_cuda_version": str(torch.version.cuda or ""),
                "device_count": int(torch.cuda.device_count()),
                "device_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
            details["torch_cxx11_abi"] = bool(
                getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)
            )
        if "torch_geometric" in modules:
            try:
                pyg_typing = importlib.import_module("torch_geometric.typing")
                details["pyg_with_torch_sparse"] = bool(
                    getattr(pyg_typing, "WITH_TORCH_SPARSE", False)
                )
            except BaseException as exc:
                failures["torch_geometric.typing"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        details["failures"] = dict(failures)
        context["dependency_failures"] = failures
        if failures:
            raise CheckFailure(
                "dependency imports failed: "
                + "; ".join(
                    f"{name} -> {error}"
                    for name, error in failures.items()
                ),
                details=details,
            )
        return details

    def import_entry_points() -> Mapping[str, Any]:
        if context.get("dependency_failures"):
            raise PrerequisiteError(
                "blocked by failed dependency imports: "
                + ", ".join(sorted(context["dependency_failures"]))
            )
        gdm_module = importlib.import_module(
            "network_dismantling.GDM.python_interface"
        )
        core_module = importlib.import_module(
            "network_dismantling.CoreGDM.python_interface"
        )
        actual_models = Path(gdm_module.models_folder_path).resolve()
        model_paths = _validate_models_folder_path(
            actual_models,
            review_root,
        )
        entry_points = {
            "OfficialGDM": getattr(gdm_module, "GDM"),
            "OfficialGDMR": getattr(gdm_module, "GDMR"),
            "OfficialCoreGDM": getattr(core_module, "CoreGDM"),
        }
        not_callable = [
            name for name, function in entry_points.items() if not callable(function)
        ]
        if not_callable:
            raise TypeError(
                "official entry points are not callable: " + ", ".join(not_callable)
            )
        context["gdm_module"] = gdm_module
        models_module = importlib.import_module(
            "network_dismantling.GDM.models"
        )
        model_class = models_module.models_mapping.get("GAT_Model")
        if model_class is None:
            raise RuntimeError(
                "GAT_Model is absent from the official models_mapping"
            )
        context["gdm_model_class"] = model_class
        return {
            "entry_points": sorted(entry_points),
            **model_paths,
            "model_mapping": sorted(models_module.models_mapping),
        }

    def sparse_runtime() -> Mapping[str, Any]:
        if "torch" not in context or "torch_sparse" not in context:
            raise PrerequisiteError(
                "blocked because torch and torch_sparse did not both import"
            )
        torch = context["torch"]
        torch_sparse = context["torch_sparse"]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
        counts: dict[str, int] = {}
        for device in devices:
            sparse = torch_sparse.SparseTensor(
                row=torch.tensor([0, 1], dtype=torch.long, device=device),
                col=torch.tensor([1, 0], dtype=torch.long, device=device),
                sparse_sizes=(2, 2),
            )
            if int(sparse.nnz()) != 2:
                raise RuntimeError(
                    f"torch_sparse returned an invalid nnz value on {device}"
                )
            counts[device] = int(sparse.nnz())
        return {"nnz_by_device": counts}

    def tiny_gat_forward() -> Mapping[str, Any]:
        if "torch" not in context or context.get("dependency_failures"):
            raise PrerequisiteError(
                "blocked because the GDM runtime dependencies did not all import"
            )
        torch = context["torch"]
        layer_module = importlib.import_module(
            "network_dismantling.GDM.models.layers.gat_conv"
        )
        features_cpu = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        edge_index_cpu = torch.tensor(
            [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]],
            dtype=torch.long,
        )
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
        output_shapes: dict[str, list[int]] = {}
        for device in devices:
            layer = layer_module.GATConv(
                in_channels=4,
                out_channels=2,
                heads=1,
                concat=True,
                dropout=0.0,
                bias=True,
                add_self_loops=True,
            ).to(device)
            layer.eval()
            with torch.no_grad():
                output = layer(
                    features_cpu.to(device),
                    edge_index_cpu.to(device),
                )
            if tuple(output.shape) != (4, 2):
                raise RuntimeError(
                    f"unexpected GAT output shape on {device}: "
                    f"{tuple(output.shape)}"
                )
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(
                    f"GAT output contains non-finite values on {device}"
                )
            output_shapes[device] = list(output.shape)
        return {"output_shapes": output_shapes}

    def checkpoint_model_forward() -> Mapping[str, Any]:
        if "torch" not in context:
            raise PrerequisiteError("blocked because torch did not import")
        if "gdm_model_class" not in context:
            raise PrerequisiteError(
                "blocked because GAT_Model was not registered"
            )
        torch = context["torch"]
        models = sorted((review_root / MODEL_RELATIVE_ROOT).glob("*.h5"))
        if not models:
            raise FileNotFoundError("no official GDM checkpoint is available")
        checkpoint = min(models, key=_file_size)
        weights = torch.load(
            str(_filesystem_path(checkpoint)),
            map_location="cpu",
        )
        if not isinstance(weights, Mapping) or not weights:
            raise TypeError("representative GDM checkpoint is not a state mapping")
        tensor_count = sum(
            1 for value in weights.values() if torch.is_tensor(value)
        )
        if tensor_count == 0:
            raise TypeError("representative GDM checkpoint contains no tensors")
        architecture = _parse_checkpoint_architecture(checkpoint)
        model = context["gdm_model_class"](architecture)
        incompatible = model.load_state_dict(weights, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not re.fullmatch(
                r"convolutional_layers\.\d+\.lin_r\.weight",
                key,
            )
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "checkpoint/model state mismatch: "
                f"missing={disallowed_missing}, unexpected={unexpected}"
            )
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]],
            dtype=torch.long,
        )
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
        forward_ranges: dict[str, list[float]] = {}
        for device in devices:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(
                    features.to(device),
                    edge_index.to(device),
                )
            if tuple(output.shape) != (4,):
                raise RuntimeError(
                    f"unexpected loaded-model output shape on {device}: "
                    f"{tuple(output.shape)}"
                )
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(
                    f"loaded-model output is non-finite on {device}"
                )
            forward_ranges[device] = [
                float(output.min().item()),
                float(output.max().item()),
            ]
        return {
            "checkpoint": str(checkpoint),
            "state_entry_count": len(weights),
            "tensor_count": tensor_count,
            "allowed_missing_keys": list(incompatible.missing_keys),
            "forward_ranges": forward_ranges,
        }

    def external_extension_import() -> Mapping[str, Any]:
        extensions = sorted(
            (
                review_root
                / "network_dismantling"
                / "common"
                / "external_dismantlers"
            ).glob("dismantler*.so")
        )
        if not extensions:
            raise PrerequisiteError(
                "external dismantler is not built in the current environment"
            )
        module = importlib.import_module(
            "network_dismantling.common.external_dismantlers.dismantler"
        )
        for name in ("Graph", "lccThresholdDismantler", "thresholdDismantler"):
            if not hasattr(module, name):
                raise AttributeError(f"external dismantler lacks {name}")
        details: dict[str, Any] = {
            "skipped": False,
            "extension": str(getattr(module, "__file__", "")),
        }
        if platform.system() == "Linux":
            details.update(_audit_linux_extension_links(extensions[0]))
        return details

    checks = [
        _execute_check("environment_alignment", environment_alignment),
        _execute_check("dependency_imports", import_dependencies),
        _execute_check("official_entry_points", import_entry_points),
        _execute_check("torch_sparse_runtime", sparse_runtime),
        _execute_check("tiny_gat_forward", tiny_gat_forward),
        _execute_check("checkpoint_model_forward", checkpoint_model_forward),
        _execute_check("external_extension_import", external_extension_import),
    ]
    return {
        "ok": all(check["status"] == "pass" for check in checks),
        "checks": checks,
    }


def _run_runtime_probe(
    review_root: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rmcd-gdm-preflight-") as directory:
        child_output = Path(directory) / "runtime-probe.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-probe",
            "--review-path",
            str(review_root.resolve()),
            "--child-output",
            str(child_output),
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(review_root.resolve()) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=review_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "stdout_tail": _tail(exc.stdout or ""),
                "stderr_tail": _tail(exc.stderr or ""),
                "checks": [],
                "errors": [
                    "isolated GDM runtime probe exceeded its hard timeout"
                ],
            }
        payload: dict[str, Any] = {}
        if child_output.is_file():
            try:
                payload = json.loads(child_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                payload = {"errors": [f"could not read child report: {exc}"]}
        errors = list(payload.get("errors", []))
        if completed.returncode != 0 and not errors:
            errors.append(
                "isolated runtime probe exited with code "
                f"{completed.returncode}"
            )
        return {
            "ok": bool(payload.get("ok")) and completed.returncode == 0,
            "timed_out": timed_out,
            "returncode": completed.returncode,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "checks": payload.get("checks", []),
            "errors": errors,
        }


def audit(
    review_root: Path,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    review_root = review_root.resolve()
    source = _inspect_review_tree(review_root)
    external = _inspect_external_build_environment(review_root)
    runtime = (
        _run_runtime_probe(review_root, timeout_seconds=timeout_seconds)
        if source["ok"]
        else {
            "ok": False,
            "checks": [],
            "errors": ["runtime probe skipped because the source audit failed"],
        }
    )
    warnings: list[str] = []
    dependency_check = next(
        (
            check
            for check in runtime.get("checks", [])
            if check.get("name") == "dependency_imports"
        ),
        None,
    )
    if dependency_check is not None:
        pyg_version = str(
            dependency_check.get("details", {})
            .get("torch_geometric", {})
            .get("version", "")
        )
        if pyg_version and pyg_version != "1.1.2":
            warnings.append(
                "upstream GDM documents PyTorch Geometric 1.1.2 as its exact "
                "paper environment; this integrated source uses a compatibility "
                "layer, so the runtime smoke remains mandatory"
            )
    extension_check = next(
        (
            check
            for check in runtime.get("checks", [])
            if check.get("name") == "external_extension_import"
        ),
        None,
    )
    if extension_check is not None:
        unresolved_count = int(
            extension_check.get("details", {}).get(
                "ldd_r_undefined_symbol_count",
                0,
            )
        )
        if unresolved_count:
            warnings.append(
                "ldd -r reported "
                f"{unresolved_count} Python/Boost relocation symbols, but the "
                "extension imported and exposed its required API in the isolated "
                "Python process; the symbols are retained as diagnostics rather "
                "than treated as a missing-library failure"
            )
    if not external["extension_present"] and external[
        "can_build_in_current_environment"
    ]:
        warnings.append(
            "the current environment has the apparent build prerequisites, but "
            "strict PASS requires compiling and importing the extension first"
        )
    errors = [
        *source.get("errors", []),
        *external.get("errors", []),
        *runtime.get("errors", []),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(
            source["ok"]
            and external["runtime_ready"]
            and runtime["ok"]
        ),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "source": source,
        "external_dismantler": external,
        "runtime_probe": runtime,
        "warnings": warnings,
        "errors": errors,
    }


def _print_report(report: Mapping[str, Any]) -> None:
    status = "PASS" if report["ok"] else "FAIL"
    print(f"[{status}] official GDM environment preflight")
    python = report["python"]
    print(f"  Python: {python['version']} at {python['executable']}")
    source = report["source"]
    print(
        "  GDM models: {0} files, {1} bytes".format(
            source["model_file_count"],
            source["model_total_bytes"],
        )
    )
    external = report["external_dismantler"]
    print(
        "  External dismantler: present={0}; build_ready={1}".format(
            external["extension_present"],
            external["can_build_in_current_environment"],
        )
    )
    for check in report["runtime_probe"].get("checks", []):
        print(f"  {check['name']}: {check['status'].upper()}")
        if check["name"] == "dependency_imports":
            details = check.get("details", {})
            torch_record = details.get("torch", {})
            cuda_record = details.get("cuda", {})
            if torch_record:
                print(
                    "    torch={0}; torch_cuda={1}; cuda_available={2}".format(
                        torch_record.get("version", ""),
                        cuda_record.get("torch_cuda_version", ""),
                        cuda_record.get("available", False),
                    )
                )
        if check["status"] in {"fail", "blocked"}:
            print(
                "    {0}: {1}".format(
                    check.get("error_type", "Error"),
                    check.get("error", ""),
                )
            )
    for warning in report.get("warnings", []):
        print(f"  WARNING: {warning}")
    for error in report.get("errors", []):
        print(f"  ERROR: {error}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-path",
        type=Path,
        default=project_root / "external" / "review-main",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--child-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_probe:
        if args.child_output is None:
            parser.error("--child-output is required with --child-probe")
        payload = _child_runtime_probe(args.review_path)
        args.child_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0 if payload["ok"] else 3

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    report = audit(args.review_path, timeout_seconds=args.timeout_seconds)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print_report(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
