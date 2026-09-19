#!/usr/bin/env bash
# Re-run every panel that lives on server 2 under the new revision.
#
# Role composition first (its own algorithm settings), then the heavy half of
# the scaling panel (N=1000) with the unified 3600 s limit.  Both run on the
# same EPYC model, so the scaling panel stays hardware-uniform across servers.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs
WORKERS="${WORKERS:-30}"
ROLE_LIMIT="${ROLE_LIMIT:-600}"
SCALING_LIMIT="${SCALING_LIMIT:-3600}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOGS}/srv2_rerun.log"; }

status "START role"
python scripts/rev_run.py --experiment role --output "${ROOT}" \
  --workers "${WORKERS}" --time-limit "${ROLE_LIMIT}" >>"${LOGS}/role.log" 2>&1
status "END   role exit=$?"

status "START scaling N=1000 (3 budget shards)"
cd /root/mpcf_rev
pids=()
for spec in "s1000_b002:0.02" "s1000_b005:0.05" "s1000_b010:0.10"; do
  tag="${spec%%:*}"; budget="${spec##*:}"
  ( cd /root/mpcf_rev/app && SCALING_SIZES=1000 WORKERS=10 TIME_LIMIT="${SCALING_LIMIT}" \
      python scripts/rev_run.py --experiment scaling --output "${ROOT}" \
      --shard-name "${tag}" --budget-fractions "${budget}" \
      --workers 10 --time-limit "${SCALING_LIMIT}" --scaling-sizes 1000 \
      >>"${LOGS}/scaling_${tag}.log" 2>&1 ) &
  pids+=("$!")
  status "  shard ${tag} pid=$!"
done
for pid in "${pids[@]}"; do wait "${pid}"; done
status "END   scaling N=1000"

python scripts/rev_run.py --experiment role --output "${ROOT}" --collect-only >>"${LOGS}/collect.log" 2>&1
python scripts/rev_run.py --experiment scaling --output "${ROOT}" --collect-only >>"${LOGS}/collect.log" 2>&1

touch "${ROOT}/_SRV2_RERUN_DONE"
status "SERVER 2 RERUN COMPLETE"
