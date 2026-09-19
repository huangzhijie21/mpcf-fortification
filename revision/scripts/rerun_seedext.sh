#!/usr/bin/env bash
# Seed-extension robustness panel on server 0.
#
# Runs after the main and heterogeneity panels so it does not contend with
# them.  The panel adds five prespecified seeds per topology-size cell; the
# original 45 graphs stay untouched as the frozen full-method benchmark.
#
# The extension carries one upstream dismantling transfer (GND), which needs
# frozen removal sequences for the *new* graphs, so the official export matrix
# runs on the extension instances first.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs
REVIEW=/root/mpcf_rev/external/review-main
MATRIX="${ROOT}/official_seedext_matrix"
WORKERS="${WORKERS:-30}"
LIMIT="${LIMIT:-600}"

export PYTHONPATH=/root/mpcf_rev/app/src:${REVIEW}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0

status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOGS}/seedext.log"; }

status "waiting for main + heterogeneity to finish"
while pgrep -f "experim[e]nt main" >/dev/null 2>&1 || pgrep -f "experim[e]nt heterogeneity" >/dev/null 2>&1; do
  sleep 60
done
status "main and heterogeneity are done"

status "STEP 1/4 prepare the 45 prespecified extension graphs"
python scripts/rev_run.py --experiment seedext --output "${ROOT}" --prepare-only \
  >>"${LOGS}/seedext_prepare.log" 2>&1
status "prepared exit=$?"

status "STEP 2/4 export upstream GND sequences for the extension graphs"
python scripts/run_official_review_matrix.py \
  --instance-manifest "${ROOT}/metadata/instance_manifest_seedext.csv" \
  --methods OfficialGND \
  --review-path "${REVIEW}" \
  --dismantling-python /root/autodl-tmp/envs/dismantling/bin/python \
  --gdm-python /root/autodl-tmp/envs/dismantling/bin/python \
  --finder-python /root/mpcf_rev/finder_pinned.sh \
  --stop-condition 1 \
  --workers 8 --gdm-slots 2 --finder-slots 4 --eigenvector-slots 2 --native-threads 1 \
  --general-timeout-seconds 900 --gdm-timeout-seconds 900 --finder-timeout-seconds 900 \
  --output "${MATRIX}" --log-root "${LOGS}/official_seedext" \
  >>"${LOGS}/seedext_matrix.log" 2>&1
status "upstream export exit=$?"

status "STEP 3/4 run the extension panel (9 methods)"
python scripts/rev_run.py --experiment seedext --output "${ROOT}" \
  --workers "${WORKERS}" --time-limit "${LIMIT}" \
  --official-sequence-roots "${MATRIX}" \
  >>"${LOGS}/seedext_run.log" 2>&1
status "extension panel exit=$?"

status "STEP 4/4 collect"
python scripts/rev_run.py --experiment seedext --output "${ROOT}" --collect-only \
  >>"${LOGS}/seedext_collect.log" 2>&1

touch "${ROOT}/_SEEDEXT_DONE"
status "SEED-EXTENSION COMPLETE"
