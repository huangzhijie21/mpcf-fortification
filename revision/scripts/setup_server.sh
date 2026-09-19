#!/usr/bin/env bash
# Prepare one AutoDL container for an MPCF revision panel.
#
# Every server runs the same frozen code and regenerates the frozen instances
# from their seeds, so instance fingerprints are identical across machines and
# only the hardware differs -- which is exactly why the CPU model is recorded.
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH

echo "=== host ==="
hostname
grep -m1 "model name" /proc/cpuinfo || true
echo "logical cores: $(nproc)"
free -g | head -2 | tail -1

echo "=== base packages ==="
python -V
for module in numpy scipy pandas networkx matplotlib pytest yaml highspy; do
  if python -c "import ${module}" >/dev/null 2>&1; then
    echo "  ${module}: present"
  else
    echo "  ${module}: MISSING"
  fi
done

echo "=== install missing ==="
python -m pip install --no-input --upgrade \
  "scipy>=1.11" "pandas>=2.0" "pytest>=7.4" "pyyaml>=6.0" highspy 2>&1 | tail -3

echo "=== verify ==="
python - <<'PY'
import numpy, scipy, pandas, networkx, matplotlib, pytest, yaml, highspy
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("networkx", networkx.__version__)
print("matplotlib", matplotlib.__version__)
print("pytest", pytest.__version__)
PY

mkdir -p /root/mpcf_rev/logs /root/mpcf_rev/results
echo "=== READY ==="
