#!/usr/bin/env bash
# CPU-pinned launcher for the isolated FINDER environment.
#
# Why this exists
# ---------------
# TensorFlow 1.14 sizes its intra-op thread pool from the number of CPUs the
# process can *schedulably* see (sched_getaffinity), not from OMP_NUM_THREADS.
# A bare `python export_finder_legacy.py` therefore grabs every core it can see,
# and running several FINDER instances concurrently makes them fight over the
# same cores instead of going faster.
#
# This wrapper pins each process to a small, rotating slice of the cores the
# container is actually allowed to use.  TF then sees only that slice, sizes its
# pool accordingly, and N concurrent instances occupy disjoint cores.  Results
# are unaffected: thread count changes scheduling only, and the adapter already
# forces CPU-only inference.
#
# Note: containers here are granted a non-zero-based CPU set (e.g. 64-95), so
# the allowed list is read from /proc rather than assumed to start at 0.
#
# Usage: point --finder-python at this file.
set -uo pipefail

REAL_PYTHON="${FINDER_REAL_PYTHON:-/root/autodl-tmp/envs/finder/bin/python}"
CORES_PER_TASK="${FINDER_CORES_PER_TASK:-2}"

if [[ ! -x "${REAL_PYTHON}" ]]; then
  printf 'FINDER python not found: %s\n' "${REAL_PYTHON}" >&2
  exit 2
fi

# Expand the container's allowed CPU list ("64-95,100-101") into an array.
ALLOWED="$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"
CPUS=()
IFS=',' read -ra PARTS <<< "${ALLOWED}"
for part in "${PARTS[@]}"; do
  if [[ "${part}" == *-* ]]; then
    start="${part%-*}"
    end="${part#*-}"
    for ((cpu = start; cpu <= end; cpu++)); do CPUS+=("${cpu}"); done
  else
    CPUS+=("${part}")
  fi
done
TOTAL="${#CPUS[@]}"
if [[ "${TOTAL}" -lt 1 ]]; then
  printf 'Could not read an allowed CPU list; running unpinned.\n' >&2
  exec "${REAL_PYTHON}" "$@"
fi

LANES=$(( TOTAL / CORES_PER_TASK ))
[[ "${LANES}" -lt 1 ]] && LANES=1

# Rotate the slice per process so concurrent instances do not overlap.
LANE=$(( $$ % LANES ))
FIRST="${CPUS[$(( LANE * CORES_PER_TASK ))]}"
LAST_INDEX=$(( LANE * CORES_PER_TASK + CORES_PER_TASK - 1 ))
[[ "${LAST_INDEX}" -ge "${TOTAL}" ]] && LAST_INDEX=$(( TOTAL - 1 ))
LAST="${CPUS[${LAST_INDEX}]}"

export OMP_NUM_THREADS="${CORES_PER_TASK}"
export OMP_THREAD_LIMIT="${CORES_PER_TASK}"
export OPENBLAS_NUM_THREADS="${CORES_PER_TASK}"
export MKL_NUM_THREADS="${CORES_PER_TASK}"
export NUMEXPR_NUM_THREADS="${CORES_PER_TASK}"
export BLIS_NUM_THREADS="${CORES_PER_TASK}"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONHASHSEED=0

exec taskset -c "${FIRST}-${LAST}" "${REAL_PYTHON}" "$@"
