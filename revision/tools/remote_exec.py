#!/usr/bin/env python3
"""Non-interactive SSH/SFTP helper for the MPCF revision runs.

Credentials are read from the environment (SSH_PASSWORD) and never written to
disk. Host keys are pinned through remote/known_hosts with a trust-on-first-use
fallback that records any new key.

Usage
-----
    python remote_exec.py exec  --command "uptime"
    python remote_exec.py put   --local a.zip --remote /root/a.zip
    python remote_exec.py get   --remote /root/out.zip --local dl/out.zip
    python remote_exec.py script --local run.sh        # upload + bash it

All operations accept --host/--port/--user overrides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CODE_ROOT = HERE.parents[2]              # .../working_copy/code
VENDOR = CODE_ROOT / "remote" / "vendor"
KNOWN_HOSTS = CODE_ROOT / "remote" / "known_hosts"
sys.path.insert(0, str(VENDOR))

import paramiko  # noqa: E402  (vendored)

DEFAULT_HOST = os.environ.get("MPCF_SSH_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("MPCF_SSH_PORT", "22"))
DEFAULT_USER = "root"


def _connect(args):
    password = os.environ.get("SSH_PASSWORD")
    if not password:
        raise SystemExit("SSH_PASSWORD is not set in the environment")
    client = paramiko.SSHClient()
    if KNOWN_HOSTS.exists():
        client.load_host_keys(str(KNOWN_HOSTS))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=30,
        auth_timeout=30,
        banner_timeout=30,
    )
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.save_host_keys(str(KNOWN_HOSTS))
    except Exception:
        pass
    client.get_transport().set_keepalive(30)
    return client


def _ensure_remote_dir(sftp, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass


def cmd_exec(client, args):
    _, stdout, stderr = client.exec_command(args.command, timeout=args.timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    payload = {"exit_code": code, "stdout": out, "stderr": err}
    print(json.dumps(payload))
    return code


def cmd_put(client, args):
    with client.open_sftp() as sftp:
        _ensure_remote_dir(sftp, str(Path(args.remote).parent).replace("\\", "/"))
        sftp.put(args.local, args.remote)
        size = sftp.stat(args.remote).st_size
    print(json.dumps({"status": "PUT", "remote": args.remote, "bytes": size}))
    return 0


def cmd_get(client, args):
    dest = Path(args.local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.open_sftp() as sftp:
        sftp.get(args.remote, str(dest))
    print(json.dumps({"status": "GET", "local": str(dest), "bytes": dest.stat().st_size}))
    return 0


def cmd_script(client, args):
    local = Path(args.local)
    data = local.read_bytes()
    remote = args.remote or ("/root/_upload/" + local.name)
    remote_dir = str(Path(remote).parent).replace("\\", "/")
    with client.open_sftp() as sftp:
        _ensure_remote_dir(sftp, remote_dir)
        with sftp.open(remote, "wb") as fh:
            fh.write(data)
        sftp.chmod(remote, 0o755)
    command = args.command or f"bash {remote}"
    _, stdout, stderr = client.exec_command(command, timeout=args.timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    print(json.dumps({"exit_code": code, "stdout": out, "stderr": err}))
    return code


def cmd_fingerprint(client, args):
    key = client.get_transport().get_remote_server_key()
    print(json.dumps({
        "status": "CONNECTED",
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "host_key_sha256": hashlib.sha256(key.asbytes()).hexdigest(),
    }))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--user", default=DEFAULT_USER)
    sub = p.add_subparsers(dest="op", required=True)

    e = sub.add_parser("exec")
    e.add_argument("--command", required=True)
    e.add_argument("--timeout", type=float, default=300.0)
    e.set_defaults(func=cmd_exec)

    u = sub.add_parser("put")
    u.add_argument("--local", required=True)
    u.add_argument("--remote", required=True)
    u.set_defaults(func=cmd_put)

    g = sub.add_parser("get")
    g.add_argument("--remote", required=True)
    g.add_argument("--local", required=True)
    g.set_defaults(func=cmd_get)

    s = sub.add_parser("script")
    s.add_argument("--local", required=True)
    s.add_argument("--remote")
    s.add_argument("--command")
    s.add_argument("--timeout", type=float, default=7200.0)
    s.set_defaults(func=cmd_script)

    f = sub.add_parser("fingerprint")
    f.set_defaults(func=cmd_fingerprint)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    client = _connect(args)
    try:
        return args.func(client, args)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
