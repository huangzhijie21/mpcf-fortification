#!/usr/bin/env bash
# Orchestrator v2: role panel, then the scaling panel split by budget.
#
# Why split by budget: `run_instance` solves the three budgets of one instance
# serially, and at N=1000 every budget hits the unified limit.  One N=1000
# instance therefore costs 3 x time_limit of *wall clock* no matter how many
# cores exist -- a floor that extra servers cannot lower.  Running one process
# per budget turns that floor into a single time_limit and lets the three parts
# overlap, which is a pure scheduling win with no change to the protocol: every
# run still uses the same instance, the same budget and the same unified limit.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs
WORKERS="${WORKERS:-26}"
MAIN_LIMIT="${MAIN_LIMIT:-600}"
SCALING_LIMIT="${SCALING_LIMIT:-3600}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# Wait for whichever panel is already running so timings are not contaminated.
# Match on the experiment flag alone: the real command line interleaves
# --output between the script name and the experiment name.
while pgrep -f "experim[e]nt heterogeneity" >/dev/null 2>&1; do
  sleep 20
done
status "heterogeneity no longer running"

# Guard against a second copy of this orchestrator stacking panels on one box.
if pgrep -f "experim[e]nt role" >/dev/null 2>&1; then
  status "another role panel is already running; refusing to double-book cores"
  exit 3
fi

status "START role"
python scripts/rev_run.py --experiment role --output "${ROOT}" \
  --workers "${WORKERS}" --time-limit "${MAIN_LIMIT}" \
  >>"${LOGS}/role.log" 2>&1
status "END   role exit=$?"

status "START scaling (3 budget shards in parallel)"
SHARD_WORKERS=$(( WORKERS / 3 ))
[[ "${SHARD_WORKERS}" -lt 2 ]] && SHARD_WORKERS=2
pids=()
for spec in "002:0.02" "005:0.05" "010:0.10"; do
  tag="${spec%%:*}"; budget="${spec##*:}"
  python scripts/rev_run.py --experiment scaling --output "${ROOT}" \
    --shard-name "scaling_b${tag}" --budget-fractions "${budget}" \
    --workers "${SHARD_WORKERS}" --time-limit "${SCALING_LIMIT}" \
    >>"${LOGS}/scaling_b${tag}.log" 2>&1 &
  pids+=("$!")
  status "  launched scaling budget ${budget} pid=$! workers=${SHARD_WORKERS}"
done
for pid in "${pids[@]}"; do
  wait "${pid}"
  status "  scaling shard pid=${pid} exit=$?"
done
status "END   scaling"

status "recomputing statistics"
python scripts/rev_statistics.py --output "${ROOT}" --resamples 10000 \
  >>"${LOGS}/statistics.log" 2>&1
python scripts/rev_plots.py --output "${ROOT}" >>"${LOGS}/plots.log" 2>&1
python scripts/rev_gates.py --output "${ROOT}" >>"${LOGS}/gates.log" 2>&1
status "END   analysis"

touch "${ROOT}/_PANELS_DONE"
status "PANELS COMPLETE"
