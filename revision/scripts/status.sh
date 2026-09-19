#!/usr/bin/env bash
# One-line-per-panel status for an MPCF revision server.
set -uo pipefail

ROOT=/root/mpcf_rev/results
LOGS=/root/mpcf_rev/logs

echo "host     : $(hostname)"
echo "cpu      : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "load     : $(cut -d' ' -f1-3 /proc/loadavg)"
echo "rev      : $(cd /root/mpcf_rev/app && PATH=/root/miniconda3/bin:$PATH python -c 'from rmcd_f.rev.ids import code_commit; print(code_commit())' 2>/dev/null || echo unknown)"
echo "procs    : $(pgrep -cf 'rev_run[.]py' || echo 0) rev_run, $(pgrep -cf 'run_official_review_mat[r]ix' || echo 0) matrix"
echo "shards   :"
for dir in "${ROOT}"/shards/*/; do
  [ -d "$dir" ] || continue
  printf '   %-18s %s\n' "$(basename "$dir")" "$(find "$dir" -name '*.csv' | wc -l)"
done
echo "results  :"
for f in "${ROOT}"/results/*.csv; do
  [ -f "$f" ] || continue
  printf '   %-28s %s rows\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
done
echo "logs     :"
for f in "${LOGS}"/*.log; do
  [ -f "$f" ] || continue
  last=$(grep -v Warning "$f" 2>/dev/null | tail -1 | cut -c1-110)
  printf '   %-30s %s\n' "$(basename "$f")" "$last"
done
echo "markers  : $(ls "${ROOT}"/_* 2>/dev/null | xargs -n1 basename 2>/dev/null | paste -sd' ' -)"
