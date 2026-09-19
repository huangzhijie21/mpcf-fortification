#!/usr/bin/env bash
# Re-run every panel that lives on server 0 under the new revision.
#
# The main panel now resolves the upstream baselines directly, so the separate
# "upstream" and "finder" passes are no longer needed: passing both sequence
# matrices to the main run produces all 36 comparison variants plus the two
# certified references in one pass.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs
WORKERS="${WORKERS:-30}"
LIMIT="${LIMIT:-600}"
MATRICES="${ROOT}/official_all_matrix,${ROOT}/official_finder_matrix"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOGS}/srv0_rerun.log"; }

status "START main (38 methods incl. upstream + FINDER)"
python scripts/rev_run.py --experiment main --output "${ROOT}" \
  --workers "${WORKERS}" --time-limit "${LIMIT}" \
  --official-sequence-roots "${MATRICES}" >>"${LOGS}/main.log" 2>&1
status "END   main exit=$?"

status "START heterogeneity"
python scripts/rev_run.py --experiment heterogeneity --output "${ROOT}" \
  --include-conversions --workers "${WORKERS}" --time-limit "${LIMIT}" \
  >>"${LOGS}/heterogeneity.log" 2>&1
status "END   heterogeneity exit=$?"

status "collecting"
python scripts/rev_run.py --experiment main --output "${ROOT}" --collect-only >>"${LOGS}/collect.log" 2>&1
python scripts/rev_run.py --experiment heterogeneity --output "${ROOT}" --collect-only >>"${LOGS}/collect.log" 2>&1

touch "${ROOT}/_SRV0_RERUN_DONE"
status "SERVER 0 RERUN COMPLETE"
