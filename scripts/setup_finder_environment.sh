#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
ENV_PREFIX="${FINDER_ENV_PREFIX:-/root/autodl-tmp/envs/finder}"
REVIEW_PATH="${REVIEW_PATH:-external/review-main}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-outputs/finder_environment_preflight.json}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda-pkgs}"
TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp/finder-build}"
export CONDA_PKGS_DIRS TMPDIR
export PIP_NO_CACHE_DIR=1
export CUDA_VISIBLE_DEVICES=-1
export PYTHONHASHSEED=0

mkdir -p -- "${CONDA_PKGS_DIRS}" "${TMPDIR}" "$(dirname -- "${PREFLIGHT_JSON}")"

if [[ ! -x "${CONDA_EXE}" ]]; then
  printf 'Conda executable is unavailable: %s\n' "${CONDA_EXE}" >&2
  exit 2
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  printf '[finder-setup] creating isolated Python 3.7 environment at %s\n' \
    "${ENV_PREFIX}"
  "${CONDA_EXE}" create -y -p "${ENV_PREFIX}" -c conda-forge \
    python=3.7 pip=23.1.2 setuptools=59.8.0 wheel=0.38.4
else
  printf '[finder-setup] reusing existing environment at %s\n' "${ENV_PREFIX}"
fi

FINDER_PYTHON="${ENV_PREFIX}/bin/python"
PYTHON_VERSION="$(${FINDER_PYTHON} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${PYTHON_VERSION}" != "3.7" ]]; then
  printf 'Existing FINDER environment uses Python %s; expected 3.7.\n' \
    "${PYTHON_VERSION}" >&2
  printf 'Move that environment aside or choose another FINDER_ENV_PREFIX.\n' >&2
  exit 3
fi

printf '[finder-setup] installing pinned official FINDER dependencies\n'
"${FINDER_PYTHON}" -m pip install \
  --prefer-binary \
  --requirement requirements-finder-py37.txt
"${FINDER_PYTHON}" -m pip check

printf '[finder-setup] building extensions and restoring the official checkpoint\n'
"${FINDER_PYTHON}" scripts/finder_environment_preflight.py \
  --review-path "${REVIEW_PATH}" \
  --build \
  --load-checkpoint \
  --json-output "${PREFLIGHT_JSON}"

printf '[finder-setup] PASS\n'
printf '  Python: %s\n' "${FINDER_PYTHON}"
printf '  Preflight: %s\n' "${PREFLIGHT_JSON}"
