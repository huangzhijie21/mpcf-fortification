#!/usr/bin/env bash
# Chained server run for the remaining MPCF revision panels.
#
# Panels run sequentially so that no two panels contend for cores: runtime is a
# reported quantity and must not depend on what else the machine is doing.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs
WORKERS="${WORKERS:-26}"
MAIN_TIME_LIMIT="${MAIN_TIME_LIMIT:-600}"
SCALING_TIME_LIMIT="${SCALING_TIME_LIMIT:-3600}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONHASHSEED=0

mkdir -p "${LOGS}"
status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

run_panel() {
  local name="$1"; shift
  status "START ${name}"
  python scripts/rev_run.py --output "${ROOT}" "$@" >>"${LOGS}/${name}.log" 2>&1
  local code=$?
  status "END   ${name} exit=${code}"
  return 0
}

# 1. the in-package BPD reference baseline, omitted from the first pass
run_panel main_bpd \
  --experiment main --shard-name main_bpd --workers "${WORKERS}" \
  --time-limit "${MAIN_TIME_LIMIT}" --methods BPDReference-Protect

# 2. joint b_v x m_v heterogeneity (Exact + Greedy on 4 profiles) plus the
#    explicit heterogeneous-cost conversion variants (Raw / PerCost /
#    PrefixKnapsack / FullKnapsack), which only differ from one another when
#    protection cost is heterogeneous.
run_panel heterogeneity \
  --experiment heterogeneity --workers "${WORKERS}" \
  --include-conversions \
  --time-limit "${MAIN_TIME_LIMIT}"

# 3. role composition: 45 bases x 6 compositions x 3 budgets
run_panel role \
  --experiment role --workers "${WORKERS}" \
  --time-limit "${MAIN_TIME_LIMIT}"

# 4. scaling with the larger unified limit
run_panel scaling \
  --experiment scaling --workers "${WORKERS}" \
  --time-limit "${SCALING_TIME_LIMIT}"

if [[ "${WITH_UPSTREAM:-0}" == "1" ]]; then
  run_panel main_upstream \
    --experiment main --shard-name main_upstream --workers "${WORKERS}" \
    --time-limit "${MAIN_TIME_LIMIT}" \
    --official-sequence-roots "${UPSTREAM_ROOTS:?set UPSTREAM_ROOTS}" \
    --methods "$(cat /root/mpcf_rev/upstream_methods.txt | paste -sd, -)"
fi

status "ALL PANELS COMPLETE"
touch "${ROOT}/_PIPELINE_DONE"
