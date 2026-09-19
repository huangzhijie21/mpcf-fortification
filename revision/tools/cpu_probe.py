#!/usr/bin/env python3
"""Portable CPU throughput probe, to quantify host contention.

Runtime is a reported result, so it matters whether a container is actually
getting the cores it advertises.  This runs a fixed single-thread workload and
reports operations per second plus the CPU time it actually received.
"""
from __future__ import annotations

import os
import time

import numpy as np


def cpu_time() -> float:
    times = os.times()
    return times.user + times.system


def busy(seconds: float = 5.0) -> tuple[float, float]:
    """Return (wall seconds, million 64-bit integer adds completed)."""

    start_wall = time.perf_counter()
    start_cpu = cpu_time()
    count = 0
    total = 0
    matrix = np.random.default_rng(0).random((160, 160))
    while time.perf_counter() - start_wall < seconds:
        for _ in range(200):
            matrix @ matrix
        count += 200
    wall = time.perf_counter() - start_wall
    used = cpu_time() - start_cpu
    del total
    return wall, count / wall


def main() -> int:
    wall, rate = busy()
    cores = os.cpu_count()
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = float("nan")
    print(f"logical_cores      : {cores}")
    print(f"wall_seconds       : {wall:.2f}")
    print(f"matmul_rate_per_s  : {rate:.1f}")
    print(f"host_load_1min     : {load1:.1f}")
    print(f"cgroup_quota       : {open('/sys/fs/cgroup/cpu.max').read().strip() if os.path.exists('/sys/fs/cgroup/cpu.max') else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
