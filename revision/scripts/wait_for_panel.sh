#!/usr/bin/env bash
# Block until one panel finishes on this server, then print its status.
# Usage: wait_for_panel.sh <marker-name> <pgrep-regex>
set -uo pipefail

MARKER="${1:?marker name}"
PATTERN="${2:?pgrep pattern}"
ROOT=/root/mpcf_rev/results

for _ in $(seq 1 720); do
  if [ -f "${ROOT}/_${MARKER}" ]; then
    echo "marker _${MARKER} present"
    break
  fi
  if ! pgrep -f "${PATTERN}" >/dev/null 2>&1; then
    echo "no process matching ${PATTERN}; treating as finished"
    break
  fi
  sleep 30
done

echo "=== final status ==="
bash /root/mpcf_rev/status.sh 2>/dev/null | sed -n '1,6p;/shards/,/^logs/p' | head -24
