#!/usr/bin/env bash
# Launch one MPCF revision panel on a dedicated server.
#
# Usage: bash run_panel_server.sh <experiment> [shard-tag:budget ...]
#
# With no shard arguments the panel runs as a single pass.  With shard
# arguments it runs one pass per "tag:budget" pair concurrently, which removes
# the serialisation of the three budgets inside one instance and therefore
# lowers the wall-clock floor from 3 x time_limit to time_limit.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

EXPERIMENT="${1:?experiment required}"
shift || true

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs
WORKERS="${WORKERS:-30}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
# Optional size split, so the scaling panel can be balanced across machines
# that share a CPU model without producing duplicate runs.
EXTRA=()
if [[ -n "${SCALING_SIZES:-}" ]]; then
  EXTRA+=(--scaling-sizes "${SCALING_SIZES}")
fi
if [[ -n "${SCALING_SEEDS:-}" ]]; then
  EXTRA+=(--scaling-seeds "${SCALING_SEEDS}")
fi
mkdir -p "${ROOT}" "${LOGS}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONHASHSEED=0

status() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOGS}/${EXPERIMENT}_server.log"
}

status "START ${EXPERIMENT} on $(hostname) ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs))"

if [[ "$#" -eq 0 ]]; then
  python scripts/rev_run.py --experiment "${EXPERIMENT}" --output "${ROOT}" \
    --workers "${WORKERS}" --time-limit "${TIME_LIMIT}" "${EXTRA[@]}" \
    >>"${LOGS}/${EXPERIMENT}.log" 2>&1
  status "END ${EXPERIMENT} exit=$?"
else
  pids=()
  for spec in "$@"; do
    tag="${spec%%:*}"
    budget="${spec##*:}"
    python scripts/rev_run.py --experiment "${EXPERIMENT}" --output "${ROOT}" \
      --shard-name "${tag}" --budget-fractions "${budget}" \
      --workers "${WORKERS}" --time-limit "${TIME_LIMIT}" "${EXTRA[@]}" \
      >>"${LOGS}/${EXPERIMENT}_${tag}.log" 2>&1 &
    pids+=("$!")
    status "  shard ${tag} budget=${budget} pid=$!"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status "  shard pid=${pid} failed"
  done
  status "END ${EXPERIMENT} shards"
fi

touch "${ROOT}/_DONE_${EXPERIMENT}"
status "PANEL ${EXPERIMENT} COMPLETE on $(hostname)"
