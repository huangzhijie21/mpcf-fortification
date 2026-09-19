#!/usr/bin/env bash
# Add the transferred upstream dismantling baselines to an existing main panel.
#
# Runs only the official *-Protect variants, so the MPCF and local rows already
# in results/shards/main are never duplicated.  Methods whose upstream export
# did not reach full coverage on the frozen 45-graph panel are dropped by
# resolve_methods and reported in metadata/panel_summary.csv and
# methods/variant_coverage.csv rather than silently disappearing.
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/mpcf_rev/app

ROOT=/root/mpcf_rev/results
MATRIX=/root/mpcf_rev/results/official_all_matrix
LOGS=/root/mpcf_rev/logs
WORKERS="${WORKERS:-26}"
LIMIT="${LIMIT:-600}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# Every comparison variant plus the two certified references, so the panel is
# complete; resolve_methods keeps only those with a frozen sequence.
python - <<'PY' > /root/mpcf_rev/upstream_methods.txt
import sys
sys.path.insert(0, "/root/mpcf_rev/app/src")
from rmcd_f.rev.registry import comparison_variant_names
names = [n for n in comparison_variant_names() if n.endswith("-Protect")]
print(",".join(names))
PY

status "upstream methods requested: $(tr ',' '\n' < /root/mpcf_rev/upstream_methods.txt | wc -l)"

python scripts/rev_run.py \
  --output "${ROOT}" \
  --experiment main \
  --shard-name main_upstream \
  --workers "${WORKERS}" \
  --time-limit "${LIMIT}" \
  --official-sequence-roots "${MATRIX}" \
  --methods "$(cat /root/mpcf_rev/upstream_methods.txt)" \
  >>"${LOGS}/main_upstream.log" 2>&1
status "END main_upstream exit=$?"

status "recomputing statistics"
python scripts/rev_statistics.py --output "${ROOT}" --resamples 10000 \
  >>"${LOGS}/statistics.log" 2>&1
status "END statistics exit=$?"

python scripts/rev_plots.py --output "${ROOT}" >>"${LOGS}/plots.log" 2>&1
status "END plots exit=$?"

python scripts/rev_gates.py --output "${ROOT}" >>"${LOGS}/gates.log" 2>&1
status "END gates exit=$?"

touch "${ROOT}/_UPSTREAM_DONE"
status "UPSTREAM PASS COMPLETE"
