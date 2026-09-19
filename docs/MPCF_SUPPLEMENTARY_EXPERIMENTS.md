# MPCF supplementary validation protocol

This protocol adds five bounded experiments without changing the MPCF objective,
solver, frozen graph panel, or primary comparison protocol.

## Evidence questions

1. **Finite correctness.** On graphs with at most 20 nodes, does exhaustive
   defense search agree with MPCF-Exact and MPCF-CG? On the smaller capacity
   panel, does exhaustive attack-set search agree with the independent capacity
   response model?
2. **General protection budgets.** Under pre-outcome, role-balanced protection
   costs in `{1,2,4}`, how much is lost by raw structural ranking, and how much is
   recovered by score-per-cost, prefix knapsack, MPCF-Greedy, and MPCF-Exact?
3. **Source of improvement.** What is contributed by task-node filtering,
   directed mission-path semantics, and joint set optimization, in that order?
4. **Operational interpretation.** How do relative gaps vary by topology, scale,
   and budget; which S/C/L/E roles are selected; and how do an initial cut, an
   equal-cost alternative cut, Greedy protection, and Exact protection differ in
   one predeclared representative case?
5. **Decision flexibility.** Which nodes are certified mandatory or possible in
   the lexicographically normalized optimal family, and which alternatives are
   observed in bounded no-good enumeration?

## Scope controls

- All medium-scale comparisons reuse the existing 45 frozen instances and their
  graph fingerprints.
- The heterogeneous protection-cost overlay is assigned before optimization and
  is topology-, attack-cost-, and outcome-oblivious.
- The information ladder contains exactly one predeclared method at each level;
  it is a contribution decomposition, not an expanded tuning grid.
- Mandatory and possible membership are certified by forced-in/forced-out MILP
  solves. Node frequencies are labelled exact only when enumeration is complete;
  otherwise they are descriptive frequencies from a bounded sample.
- Paired differences are summarized by their median, Wilcoxon signed-rank test,
  and rank-biserial effect size. Win/tie/loss counts are secondary diagnostics.

## Server entry point

Run `scripts/run_mpcf_supplementary_server.sh`. It writes five independent stage
directories and a final `supplementary_evidence_gate.csv`. A root `_COMPLETE`
marker is created only after every stage and evidence-scope gate passes.

From the existing `rmcd_mpcf_v0.8.5_server` directory:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/rmcd-f-py310

python -m pip install --no-deps --force-reinstall \
  dist/rmcd_f-0.8.5-py3-none-any.whl

FULL_MANIFEST=$(find /root/autodl-tmp/chap5 \
  -type f -path "*/mpcf_full_v1/evaluation/instance_manifest.csv" \
  -print -quit)
FULL_ROOT=$(dirname "$FULL_MANIFEST")
FROZEN_ROOT="$(dirname "$FULL_ROOT")/prepared"

mkdir -p logs
nohup env \
  CONTROL_PYTHON=/root/autodl-tmp/envs/rmcd-f-py310/bin/python \
  FULL_ROOT="$FULL_ROOT" \
  FROZEN_ROOT="$FROZEN_ROOT" \
  OUTPUT_ROOT=outputs/mpcf_supplementary_v1 \
  WORKERS=20 \
  EXHAUSTIVE_WORKERS=12 \
  FAMILY_WORKERS=6 \
  RESUME=1 \
  bash scripts/run_mpcf_supplementary_server.sh \
  > logs/mpcf_supplementary_v1.log 2>&1 &

echo $! > logs/mpcf_supplementary_v1.pid
tail -f logs/mpcf_supplementary_v1.log
```

After the root `_COMPLETE` marker appears:

```bash
tar -czf mpcf_supplementary_v1_results.tar.gz \
  outputs/mpcf_supplementary_v1 \
  logs/mpcf_supplementary_v1.log
sha256sum mpcf_supplementary_v1_results.tar.gz \
  > mpcf_supplementary_v1_results.tar.gz.sha256
```
