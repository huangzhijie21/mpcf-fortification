"""Automatic capture of the hardware and software environment.

The revised manuscript must state OS, CPU, physical/logical core counts, RAM,
Python version, optimisation solver and version, thread counts, time limit,
MIP gap, feasibility/integrality tolerances, solver seed, graph seeds, code
revision, run timestamp and the runtime definition.  All of it is written to
``metadata/environment.json`` at experiment start instead of being recalled by
hand afterwards.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ids import code_commit

#: What each reported runtime actually includes.  Recorded verbatim so that a
#: reader never has to guess whether modelling or replay time was counted.
RUNTIME_DEFINITION = {
    "runtime_selection_s": (
        "time for the method to produce its protection set; includes problem "
        "construction and solver time for that method, excludes the common "
        "adaptive PathCut evaluation"
    ),
    "runtime_evaluation_s": (
        "time for the single unified exact min-cut (adaptive PathCut) evaluation "
        "of the finished protection set"
    ),
    "runtime_total_s": "runtime_selection_s + runtime_evaluation_s",
    "excluded": "graph generation, instance freezing, CSV writing and plotting",
}


def _cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _physical_cores() -> int | None:
    """Physical core count, falling back to ``lscpu``/``sysfs`` then logical."""

    try:
        import psutil  # optional

        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            pairs = set()
            physical = core = None
            for line in handle:
                if line.startswith("physical id"):
                    physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if physical is not None and core is not None:
                        pairs.add((physical, core))
                    physical = core = None
            if physical is not None and core is not None:
                pairs.add((physical, core))
            if pairs:
                return len(pairs)
    except OSError:
        pass
    return None


def _total_ram_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil  # optional

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _load_average() -> dict[str, float] | None:
    """Host load average.

    On a shared container host this reflects the whole machine, not just this
    container, so it is recorded as a timing-noise indicator rather than as a
    statement about this experiment's own parallelism.
    """

    try:
        one, five, fifteen = os.getloadavg()
        return {
            "load1": round(one, 2),
            "load5": round(five, 2),
            "load15": round(fifteen, 2),
        }
    except (OSError, AttributeError):
        return None


def _cgroup_quota_cores() -> float | None:
    """CPU quota actually granted to this container, when cgroup-limited."""

    for path, scale in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", 100_000.0),
    ):
        try:
            text = open(path, encoding="utf-8").read().split()
        except OSError:
            continue
        if path.endswith("cpu.max") and len(text) == 2:
            if text[0] == "max":
                return None
            try:
                return round(float(text[0]) / float(text[1]), 2)
            except (ValueError, ZeroDivisionError):
                return None
        if scale is not None and text:
            try:
                value = float(text[0])
            except ValueError:
                continue
            if value > 0:
                return round(value / scale, 2)
    return None


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _gurobi_version() -> str | None:
    try:
        import gurobipy  # type: ignore

        return ".".join(str(part) for part in gurobipy.gurobi.version())
    except Exception:
        return None


def _cplex_version() -> str | None:
    try:
        import cplex  # type: ignore

        return getattr(cplex, "__version__", "present")
    except Exception:
        return None


def _git_status(root: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            dirty = [line for line in result.stdout.splitlines() if line.strip()]
            return {"available": True, "dirty": bool(dirty), "dirty_files": len(dirty)}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"available": False, "dirty": None, "dirty_files": None}


def observe_environment(
    *,
    root: Path | str | None = None,
    time_limit_s: float | None = None,
    mip_gap: float | None = None,
    solver_seed: int | None = None,
    graph_seeds: Sequence[int] = (),
    threads: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the environment block for one experiment launch."""

    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    from scipy.optimize import milp  # noqa: F401  (probe availability)

    logical = os.cpu_count()
    payload: dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "distribution": _linux_distribution(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": _cpu_name(),
            "physical_cores": _physical_cores(),
            "logical_cores": logical,
            "cgroup_quota_cores": _cgroup_quota_cores(),
        },
        "host_load_average": _load_average(),
        "ram": {
            "total_bytes": _total_ram_bytes(),
            "total_gib": (
                round(_total_ram_bytes() / (1024**3), 2)
                if _total_ram_bytes() is not None
                else None
            ),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "solvers": {
            "milp_backend": "scipy.optimize.milp",
            "milp_backend_version": _package_version("scipy"),
            "highs_version": _highs_version(),
            "gurobi": _gurobi_version(),
            "cplex": _cplex_version(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "networkx", "pandas", "matplotlib", "highspy")
        },
        "threads": {
            "requested_workers": threads,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "logical_cores": logical,
        },
        "solver_settings": {
            "time_limit_s": time_limit_s,
            "mip_gap": mip_gap,
            "presolve": True,
            "feasibility_tolerance": "solver default (HiGHS)",
            "integrality_tolerance": "solver default (HiGHS)",
            "solver_seed": solver_seed,
        },
        "graph_seeds": list(graph_seeds),
        "code": {
            "commit": code_commit(base),
            "root": str(base),
            "git": _git_status(base),
        },
        "runtime_definition": RUNTIME_DEFINITION,
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def _linux_distribution() -> str | None:
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def _highs_version() -> str | None:
    """Version of the HiGHS library actually linked by SciPy."""

    try:
        import highspy  # type: ignore

        return getattr(highspy, "HIGHS_VERSION", None) or "highspy present"
    except Exception:
        pass
    try:
        from scipy.optimize._highspy import _core  # type: ignore

        return getattr(_core, "HIGHS_VERSION_STRING", None)
    except Exception:
        return None


def write_environment(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def freeze_command_line() -> str:
    return " ".join([shutil.which(sys.argv[0]) or sys.argv[0], *sys.argv[1:]])
