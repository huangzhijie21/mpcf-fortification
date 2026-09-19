#!/usr/bin/env bash
# Block until every shard directory under a prefix holds the expected number of
# result files, then print a status summary.  More reliable than a marker file,
# because a stale marker from an earlier attempt can survive a restart.
#
# Usage: wait_for_shards.sh <shard-prefix> <expected-per-shard>
set -uo pipefail

PREFIX="${1:?shard prefix}"
EXPECTED="${2:?expected files per shard}"
ROOT=/root/mpcf_rev/results

for _ in $(seq 1 960); do
  dirs=$(find "${ROOT}/shards" -maxdepth 1 -type d -name "${PREFIX}*" 2>/dev/null)
  if [ -n "${dirs}" ]; then
    done_all=1
    for d in ${dirs}; do
      n=$(find "${d}" -name '*.csv' 2>/dev/null | wc -l)
      if [ "${n}" -lt "${EXPECTED}" ]; then done_all=0; fi
    done
    if [ "${done_all}" -eq 1 ]; then
      echo "all ${PREFIX}* shards reached ${EXPECTED} files"
      break
    fi
  fi
  sleep 45
done

echo "=== per-shard counts ==="
for d in $(find "${ROOT}/shards" -maxdepth 1 -type d -name "${PREFIX}*" 2>/dev/null); do
  printf '  %-22s %s/%s\n' "$(basename "$d")" "$(find "$d" -name '*.csv' | wc -l)" "${EXPECTED}"
done
echo "=== workers still running: $(pgrep -cf 'rev_run[.]py' || echo 0) ==="
