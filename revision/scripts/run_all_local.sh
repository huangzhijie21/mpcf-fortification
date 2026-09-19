#!/usr/bin/env bash
#
# Run every MPCF panel on THIS machine.
#
# Nothing here needs a cluster: each panel is solved by a local process pool
# (`rev_run.py --workers N`), and the four analysis stages read plain CSV.  The
# published archive was produced on three containers only because that was the
# hardware available; the same panels run unchanged on a laptop or a workstation,
# just for longer.
#
# Measured cost of the full published panel set (single-threaded core hours,
# summed over every solver call in the archive):
#
#     main            5.3      role          14.3
#     heterogeneity  23.5      seedext        5.4
#     scaling       140.2      ----------------------
#     total         188.7 core hours
#
# So roughly 24 h of wall clock on 8 cores, 12 h on 16, 6 h on 32.  The scaling
# panel dominates because N=1000 with a 3600 s per-method limit is genuinely
# expensive; run `--smoke` first, and consider `--skip-scaling` for a quick pass.
#
# Usage
# -----
#   bash revision/scripts/run_all_local.sh                 # everything
#   bash revision/scripts/run_all_local.sh --smoke         # 1 graph/panel, ~5 min
#   WORKERS=16 bash revision/scripts/run_all_local.sh      # pin the pool size
#   bash revision/scripts/run_all_local.sh --skip-scaling  # all but the heavy panel
#   bash revision/scripts/run_all_local.sh --smoke --with-scaling   # include it
#
# `--smoke` shortens the per-method limits to 20 s and skips the scaling panel,
# because one graph still runs every method: 38 methods at a 600 s limit is a
# 6-hour worst case for a single panel.
#
set -euo pipefail

OUT="${OUT:-results}"
SMOKE=0
SKIP_SCALING=0
SKIP_UPSTREAM=0
WITH_SCALING=0

for arg in "$@"; do
  case "$arg" in
    --smoke)        SMOKE=1 ;;
    --skip-scaling) SKIP_SCALING=1 ;;
    --with-scaling) WITH_SCALING=1 ;;
    --skip-upstream) SKIP_UPSTREAM=1 ;;
    -h|--help)      sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Default the pool to the machine's usable core count, capped so a big host is
# not oversubscribed by the MILP's own threads.
if [[ -z "${WORKERS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    CORES="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  else
    CORES="${NUMBER_OF_PROCESSORS:-4}"
  fi
  WORKERS="$CORES"
  [[ "$WORKERS" -gt 16 ]] && WORKERS=16
fi

# Remember whether the caller pinned the limits, so --smoke can lower them.
[[ -n "${MAIN_LIMIT:-}" ]] && MAIN_LIMIT_SET=1
[[ -n "${SCALING_LIMIT:-}" ]] && SCALING_LIMIT_SET=1

MAIN_LIMIT="${MAIN_LIMIT:-600}"      # seconds, PER METHOD TOTAL
SCALING_LIMIT="${SCALING_LIMIT:-3600}"

SMOKE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  # A smoke run has to actually be quick.  Note that `--limit-instances 1` only
  # reduces the number of *graphs*, not the number of methods: the surviving
  # graph still runs all 38 of them, so a 600 s per-method limit would put the
  # worst case at 38 x 600 s for one panel.  Cap the limits hard, and leave the
  # scaling panel (N=1000) out unless it is asked for explicitly.
  SMOKE_ARGS=(--limit-instances 1)
  [[ -z "${MAIN_LIMIT_SET:-}" ]] && MAIN_LIMIT="${MAIN_LIMIT_SMOKE:-20}"
  [[ -z "${SCALING_LIMIT_SET:-}" ]] && SCALING_LIMIT="${SCALING_LIMIT_SMOKE:-20}"
  [[ "$WITH_SCALING" == "0" ]] && SKIP_SCALING=1
fi

cd "$(dirname "$0")/../.."
echo "== MPCF local run =="
echo "   output      : $OUT"
echo "   workers     : $WORKERS"
echo "   main limit  : ${MAIN_LIMIT}s   scaling limit: ${SCALING_LIMIT}s"
[[ "$SMOKE" == "1" ]] && echo "   mode        : SMOKE (1 instance per panel)"
echo

run_panel () {
  local name="$1"; shift
  echo "--- $name ---"
  python scripts/rev_run.py --experiment "$name" --output "$OUT" \
      --workers "$WORKERS" "$@" "${SMOKE_ARGS[@]}"
}

# Panels that do not need any third-party code.
run_panel main          --time-limit "$MAIN_LIMIT"
run_panel role          --time-limit "$MAIN_LIMIT"
run_panel heterogeneity --time-limit "$MAIN_LIMIT" --include-conversions
run_panel seedext       --time-limit "$MAIN_LIMIT"

if [[ "$SKIP_SCALING" == "0" ]]; then
  run_panel scaling --time-limit "$SCALING_LIMIT"
else
  echo "--- scaling: skipped (--skip-scaling) ---"
fi

if [[ "$SKIP_UPSTREAM" == "0" && -d external/review-main ]]; then
  echo "--- upstream transfer variants (third-party tree present) ---"
  python scripts/run_official_review_matrix.py \
      --instance-manifest "$OUT/metadata/instance_manifest_main.csv" \
      --review-path external/review-main \
      --output "$OUT/official_matrix" || \
    echo "    upstream matrix failed; the affected variants stay 'not evaluated'"
  python scripts/rev_run.py --experiment main --output "$OUT" \
      --shard-name main_upstream --workers "$WORKERS" --time-limit "$MAIN_LIMIT" \
      --official-sequence-roots "$OUT/official_matrix" || true
else
  echo "--- upstream variants: skipped (no external/review-main) ---"
  echo "    the registry will report them as 'not evaluated'; nothing is faked"
fi

# Merge every shard into one runs_long.csv, then analyse.
echo "--- collect ---"
python scripts/rev_run.py --experiment all --output "$OUT" --collect-only

echo "--- statistics ---"
python scripts/rev_statistics.py --output "$OUT" --resamples 10000

echo "--- figures ---"
python scripts/rev_plots.py --output "$OUT"

echo "--- acceptance gates ---"
python scripts/rev_gates.py --output "$OUT" --allow-pending

echo
echo "done.  archive: $OUT/"
echo "  results/runs_long.csv      every run, 36 columns"
echo "  statistics/                graph-clustered tests"
echo "  figures/                   PDF + 600 dpi PNG"
echo "  gates/acceptance_gates.csv which invariants held"
