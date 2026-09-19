#!/usr/bin/env python3
"""Wait for every server, then download, merge and analyse the whole archive.

This is the single command that turns three running servers into one publishable
result directory:

1. poll each server until its panel is complete;
2. force a final ``--collect-only`` so ``runs_long.csv`` is built from a
   quiescent shard tree rather than from a partially written one;
3. download ``results/`` and ``metadata/`` and the logs;
4. merge the three archives with duplicate and fingerprint checks;
5. run statistics, figures and acceptance gates on the merged archive.
"""

from __future__ import annotations

import os

import argparse
import base64
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import remote_exec  # noqa: E402  (vendored-paramiko helper in the same directory)

CODE = ROOT
DOWNLOADS = CODE / "revision" / "downloads"

SERVERS_EXPECT = {
    "srv0": {"main": 45, "heterogeneity": 180, "seedext": 45},
    "srv1": {"b002": 27, "b005": 27, "b010": 27},
    "srv2": {"role": 270, "s1000_b002": 9, "s1000_b005": 9, "s1000_b010": 9},
}


def _server(index: int) -> dict:
    """Read one run server from the environment.

    Nothing about the deployment is stored in the repository: set
    ``MPCF_SRV<n>_HOST`` / ``MPCF_SRV<n>_PORT`` / ``SSH_PASSWORD_SRV<n>``
    (see ``revision/scripts/servers.env.example``).
    """

    name = f"srv{index}"
    return {
        "host": os.environ.get(f"MPCF_SRV{index}_HOST", f"srv{index}.example.net"),
        "port": int(os.environ.get(f"MPCF_SRV{index}_PORT", "22")),
        "password": os.environ.get(f"SSH_PASSWORD_SRV{index}", ""),
        "expect": SERVERS_EXPECT[name],
    }


SERVERS = {f"srv{i}": _server(i) for i in (1, 2, 3)}

PROBE = (
    "import glob, os, json\n"
    "counts = {}\n"
    "for d in glob.glob('/root/mpcf_rev/results/shards/*'):\n"
    "    if os.path.isdir(d):\n"
    "        counts[os.path.basename(d)] = len(glob.glob(os.path.join(d, '*.csv')))\n"
    "print('COUNTS=' + json.dumps(counts))\n"
)

PROBE_INSTALLED: dict[str, bool] = {}


def _run_probe(client, name: str) -> dict[str, int]:
    """Read per-shard result counts from a server.

    The probe is shipped base64-encoded so that no shell quoting rule can
    corrupt it and byte-identical code runs on every machine.
    """

    if not PROBE_INSTALLED.get(name):
        payload = base64.b64encode(PROBE.encode("utf-8")).decode("ascii")
        _exec(
            client,
            "mkdir -p /root/mpcf_rev && printf %s "
            f"{shlex.quote(payload)} | base64 -d > /root/mpcf_rev/probe_counts.py",
        )
        PROBE_INSTALLED[name] = True
    code, out, err = _exec(
        client,
        "export PATH=/root/miniconda3/bin:$PATH; python /root/mpcf_rev/probe_counts.py",
    )
    for line in out.splitlines():
        if line.startswith("COUNTS="):
            try:
                return json.loads(line[len("COUNTS="):])
            except ValueError:
                return {}
    if err:
        print(f"    probe stderr: {err.strip()[:160]}")
    return {}


def _connect(name: str):
    spec = SERVERS[name]
    args = argparse.Namespace(
        host=spec["host"], port=spec["port"], user="root", timeout=600.0
    )
    import os

    os.environ["SSH_PASSWORD"] = spec["password"]
    return remote_exec._connect(args), args


def _exec(client, command: str, timeout: float = 600.0) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def panel_complete(client, name: str) -> tuple[bool, dict[str, int]]:
    counts = _run_probe(client, name)
    expected = SERVERS[name]["expect"]
    ok = all(counts.get(shard, 0) >= need for shard, need in expected.items())
    return ok, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-hours", type=float, default=9.0)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Snapshot the current state instead of waiting for every panel to finish.",
    )
    args = parser.parse_args(argv)

    deadline = time.time() + args.max_hours * 3600
    clients = {}
    for name in SERVERS:
        clients[name] = _connect(name)[0]
        print(f"connected to {name}", flush=True)

    try:
        while not args.no_wait and time.time() < deadline:
            state = {}
            for name, client in clients.items():
                ok, counts = panel_complete(client, name)
                state[name] = (ok, counts)
                print(f"[{time.strftime('%H:%M:%S')}] {name}: {'DONE' if ok else 'running'} {counts}", flush=True)
            if all(ok for ok, _ in state.values()):
                break
            time.sleep(args.poll_seconds)
        else:
            print("timed out waiting for all panels", flush=True)

        # Final quiescent collection on each server.
        for name, client in clients.items():
            print(f"collecting {name}", flush=True)
            _exec(
                client,
                "export PATH=/root/miniconda3/bin:$PATH; cd /root/mpcf_rev/app && "
                "for e in main role scaling heterogeneity seedext; do "
                "python scripts/rev_run.py --experiment $e --output /root/mpcf_rev/results "
                "--collect-only >>/root/mpcf_rev/logs/collect.log 2>&1; done; echo collected",
                timeout=1800.0,
            )
    finally:
        for client in clients.values():
            client.close()

    # Download.
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for name, spec in SERVERS.items():
        target = DOWNLOADS / name
        target.mkdir(parents=True, exist_ok=True)
        client = _connect(name)[0]
        try:
            for sub in ("results", "metadata", "statistics", "figures", "methods", "tables", "gates"):
                (target / sub).mkdir(parents=True, exist_ok=True)
                code, out, err = _exec(
                    client,
                    f"cd /root/mpcf_rev/results 2>/dev/null && tar -czf /tmp/{name}_{sub}.tgz {sub} 2>/dev/null; "
                    f"echo $?",
                )
                if code != 0 or "0" not in out[:4]:
                    print(f"  {name}/{sub}: nothing to archive")
                    continue
                with client.open_sftp() as sftp:
                    sftp.get(f"/tmp/{name}_{sub}.tgz", str(target / f"{sub}.tgz"))
                subprocess.run(
                    ["tar", "-xzf", str(target / f"{sub}.tgz"), "-C", str(target)],
                    check=False,
                )
                print(f"  {name}/{sub} downloaded", flush=True)
        finally:
            client.close()

    # Merge and analyse.
    inputs = ",".join(str(DOWNLOADS / name) for name in SERVERS)
    merged = CODE / "revision" / ("snapshot_archive" if args.no_wait else "final_archive")
    subprocess.run(
        [sys.executable, str(HERE / "merge_archives.py"), "--inputs", inputs, "--output", str(merged)],
        check=True,
    )
    for script in ("rev_statistics.py", "rev_plots.py", "rev_gates.py"):
        extra = ["--resamples", "10000"] if script == "rev_statistics.py" else []
        result = subprocess.run(
            [sys.executable, str(CODE / "scripts" / script), "--output", str(merged), *extra],
            check=False,
        )
        print(f"{script} exit={result.returncode}", flush=True)
    print(f"\nfinal archive: {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
