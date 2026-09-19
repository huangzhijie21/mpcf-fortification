#!/usr/bin/env bash
# Server 0 (Intel Xeon 8352V) final pass.
#
# Server 0 hosts the whole main panel, the heterogeneity panel and the upstream
# comparison panel, so all three are produced on one CPU model and their
# runtimes stay comparable.  The upstream sequence matrix lives outside the
# results tree so that wiping the panel shards cannot delete it.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
MATRIX=/root/mpcf_rev/results/official_all_matrix
LOGS=/root/mpcf_rev/logs
REVIEW=/root/mpcf_rev/external/review-main
WORKERS="${WORKERS:-30}"
LIMIT="${LIMIT:-600}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOGS}/srv0_final.log"; }

status "waiting for the official sequence matrix to finish"
while pgrep -f "run_official_review_mat[r]ix" >/dev/null 2>&1; do sleep 30; done
status "matrix finished"

# Replace every shard with a run of the single frozen revision.
rm -rf "${ROOT}/shards" "${ROOT}/results" "${ROOT}/instances" "${ROOT}/metadata" \
       "${ROOT}/statistics" "${ROOT}/tables" "${ROOT}/methods" "${ROOT}/gates" \
       "${ROOT}/figures" "${ROOT}/logs"
mkdir -p "${ROOT}"

status "START main panel"
python scripts/rev_run.py --experiment main --output "${ROOT}" \
  --workers "${WORKERS}" --time-limit "${LIMIT}" >>"${LOGS}/main.log" 2>&1
status "END   main exit=$?"

status "START main BPD reference pass"
python scripts/rev_run.py --experiment main --output "${ROOT}" \
  --shard-name main_bpd --methods BPDReference-Protect \
  --workers "${WORKERS}" --time-limit "${LIMIT}" >>"${LOGS}/main_bpd.log" 2>&1
status "END   main_bpd exit=$?"

status "START heterogeneity panel"
python scripts/rev_run.py --experiment heterogeneity --output "${ROOT}" \
  --include-conversions --workers "${WORKERS}" --time-limit "${LIMIT}" \
  >>"${LOGS}/heterogeneity.log" 2>&1
status "END   heterogeneity exit=$?"

status "START upstream comparison pass"
python - <<'PY' > /root/mpcf_rev/upstream_methods.txt
import sys
sys.path.insert(0, "/root/mpcf_rev/app/src")
from rmcd_f.rev.registry import comparison_variant_names
print(",".join(n for n in comparison_variant_names() if n.endswith("-Protect")))
PY
python scripts/rev_run.py --experiment main --output "${ROOT}" \
  --shard-name main_upstream --workers "${WORKERS}" --time-limit "${LIMIT}" \
  --official-sequence-roots "${MATRIX}" \
  --methods "$(cat /root/mpcf_rev/upstream_methods.txt)" \
  >>"${LOGS}/main_upstream.log" 2>&1
status "END   main_upstream exit=$?"

status "START statistics / figures / gates"
python scripts/rev_statistics.py --output "${ROOT}" --resamples 10000 >>"${LOGS}/statistics.log" 2>&1
python scripts/rev_plots.py --output "${ROOT}" >>"${LOGS}/plots.log" 2>&1
python scripts/rev_gates.py --output "${ROOT}" --allow-pending >>"${LOGS}/gates.log" 2>&1
status "END   analysis exit=$?"

touch "${ROOT}/_SRV0_DONE"
status "SERVER 0 COMPLETE"
