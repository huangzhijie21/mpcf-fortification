#!/usr/bin/env bash
# After the role panel finishes on this EPYC server, take the heavy half of the
# scaling panel (N=500 and N=1000).  srv1 handles N=200/300 on the same CPU
# model, so the two halves are hardware-identical and stay comparable.
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
LOGS=/root/mpcf_rev/logs
status() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOGS}/srv2_chain.log"; }

while pgrep -f "experim[e]nt role" >/dev/null 2>&1; do sleep 30; done
status "role finished"

status "START statistics for the role panel"
cd /root/mpcf_rev/app
python scripts/rev_run.py --experiment role --output /root/mpcf_rev/results --collect-only >>"${LOGS}/role_collect.log" 2>&1
status "role rows collected exit=$?"

status "START scaling sizes 500,1000 with 3 budget shards"
cd /root/mpcf_rev
sed -i 's/\r$//' run_panel_server.sh
SCALING_SIZES=500,1000 WORKERS=10 TIME_LIMIT=3600 \
  bash run_panel_server.sh scaling s500_1000_b002:0.02 s500_1000_b005:0.05 s500_1000_b010:0.10 \
  >>"${LOGS}/srv2_chain.log" 2>&1
status "END scaling sizes 500,1000"
touch /root/mpcf_rev/results/_SRV2_DONE
status "SERVER 2 COMPLETE"