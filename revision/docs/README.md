# MPCF Revision Layer

This directory and `src/rmcd_f/rev/` implement the revision plan for the MPCF
paper. The MPCF solver core is **not** rewritten: `MPCF-Exact`,
`MPCF-CG` and `MPCF-Greedy` keep their formulations, their certificates and
their objective. What changed is the surrounding experiment organisation,
statistics, logging, registry, heterogeneity design and reproducibility.

## What changed, by revision item

| # | Item | Where |
|---|---|---|
| 1 | Unified `graph_id` / `base_id` / `instance_id`; `C_INF` replaces `M`; `num_edges` replaces `edge_count`; role output uses `P` | `src/rmcd_f/rev/ids.py`, `src/rmcd_f/task_path_fortification.py` |
| 2 | Statistics rewritten around graph- and base-ID clusters | `src/rmcd_f/rev/stats.py`, `scripts/rev_statistics.py` |
| 3 | Explicit 36-variant baseline registry | `src/rmcd_f/rev/registry.py` |
| 4 | Scaling to `N=1000` with full solver logging under a unified time limit | `src/rmcd_f/rev/panels.py`, `src/rmcd_f/task_path_fortification.py` |
| 5 | Joint `b_v` x `m_v = delta_v / a_v` heterogeneity with measured Spearman correlation | `src/rmcd_f/rev/panels.py` |
| 6 | `MPCF-Greedy` unit tests, including the three-node counterexample | `tests/rev/test_rev_greedy_properties.py` |
| 7 | Plotting and statistics decoupled from the solvers; CSV-driven figures | `scripts/rev_plots.py`, `scripts/rev_statistics.py` |
| — | Seed-extension robustness analysis against the "are five seeds enough?" objection | `src/rmcd_f/rev/seedext.py`, `src/rmcd_f/rev/robustness.py` |
| — | `environment.json` captured automatically | `src/rmcd_f/rev/env.py` |
| — | Acceptance gates executed before publication | `scripts/rev_gates.py` |

## The unified run-level format

Every algorithm in every experiment writes into one `results/runs_long.csv`
with exactly these 36 columns. Fields a method does not produce are the literal
`NA`, never dropped, so one reader can consume every experiment.

```
experiment  instance_id  graph_id  base_id  topology  N  num_edges  seed
composition  budget_ratio  budget_abs  method  variant
kappa  kappa_opt  relative_gap  actual_cost  selected_count  selected_nodes
runtime_selection_s  runtime_evaluation_s  runtime_total_s  status
incumbent  best_bound  final_gap  bb_nodes  replay_kappa  certificate_mode
cg_L  cg_U  cg_iterations  cg_cuts  time_limit_s
config_hash  code_commit
```

Solver-logging semantics:

* **Exact** — `incumbent`, `best_bound`, `final_gap`, `bb_nodes`, `status`, and
  `replay_kappa` from an independent max-flow/min-cut replay of the frozen
  protection set.
* **CG** — `cg_L`, `cg_U`, `cg_iterations`, `cg_cuts`, `certificate_mode`
  (`cg_closed` when the master and the separation oracle close).
* **Greedy** — objective, realised cost, selected count, iteration count and
  runtime, with `certificate_mode = open` because no optimality is claimed.

`relative_gap = (kappa_opt - kappa) / kappa_opt`, where `kappa_opt` is the
cross-solver certified optimum of that `(instance, budget)` cell. Two
independently structured exact solvers must agree or the cell is flagged
`certificate_conflict`.

## Experiment identity

* `graph_id = topology + N + seed` — the independent unit of the main panel.
* `budget_ratio` — a repeated measure **inside** a `graph_id`, never a sample.
* `base_id = topology + N + seed` — the repeated-measures cluster of the
  role-composition panel; its six compositions and three budgets move together.
* `instance_id` — the frozen graph view, unique per experiment and modifier.

## Statistics

Primary analysis, exactly as pre-specified:

1. average the relative gap over the three budgets inside each `(graph, method)`;
2. Friedman omnibus over the 45 graphs, one block per graph, reporting the
   statistic, `df` and `p`;
3. pairwise Wilcoxon signed-rank against `MPCF-Exact` on the 45 graph-level
   paired observations;
4. Holm-adjust the pairwise family, and report rank-biserial, mean and median
   paired effects and a clustered percentile bootstrap 95% CI.

The bootstrap draws **whole graphs** and carries every one of a graph's budget
observations in with it (10,000 resamples). The role-composition analysis draws
whole `base_id` clusters. Nothing treats 135 graph-budget pairs or 810
role-composition records as independent samples.

Secondary analyses repeat the same tests inside each budget (`n = 45`).

## Directory layout of a published archive

```
results/     runs_long.csv, instance_budget_cells.csv, <experiment>_runs.csv
statistics/  statistics_graph_level.csv, pairwise_graph_level.csv,
             statistics_by_budget.csv, role_repeated_statistics.csv,
             bootstrap_summary.csv, zero_budget_audit.csv
methods/     method_variant_registry.csv, variant_reconciliation.json
metadata/    environment.json, experiment_config.{json,yaml},
             graph_manifest.csv, instance_manifest.csv, panel_summary.csv
figures/     fig1, fig3..fig9 as vector PDF and 600 dpi PNG
tables/      table_main.csv, table_statistics.csv, table_scaling.csv,
             table_heterogeneity.csv, table_role.csv
gates/       acceptance_gates.csv
logs/        per-panel launcher logs
```

## Running

```bash
# 1. freeze panels and solve
python scripts/rev_run.py --experiment main --output results --workers 26 --time-limit 600
python scripts/rev_run.py --experiment role  --output results --workers 26 --time-limit 600
python scripts/rev_run.py --experiment heterogeneity --output results --workers 26 --time-limit 600
python scripts/rev_run.py --experiment scaling --output results --workers 26 --time-limit 3600

# 2. statistics (raw CSV in, statistics CSV out; no solver is imported)
python scripts/rev_statistics.py --output results --resamples 10000

# 3. figures (CSV in, PDF/PNG out)
python scripts/rev_plots.py --output results

# 4. acceptance gates
python scripts/rev_gates.py --output results
```

A later pass can add methods to a panel already on disk without re-solving the
earlier ones, because each pass writes its own shard directory and
`runs_long.csv` concatenates all shards:

```bash
python scripts/rev_run.py --experiment main --output results \
  --shard-name main_upstream --methods OfficialGND-Protect,... \
  --official-sequence-roots <matrix root>
```

## Notes on two design decisions

**`budget_abs` is the exact budget the solver used, not a rounded integer.**
The main panel has `b_v = 1` for every node, so `budget_abs = ratio * N` and
`N=42` at `2%` gives `0.84` — below the cheapest protection, i.e. a
zero-effective-budget cell. `results/instance_budget_cells.csv` records both
`budget_abs` and `budget_abs_floor`, and `statistics/zero_budget_audit.csv`
reports the descriptive results for all cells and for the effective-budget
subset separately. Rounding `budget_abs` would hide that degeneracy.

**The unified time limit is a per-method total, not a per-internal-solve limit.**
`MPCF-Exact`, `MPCF-CG` and `MPCF-Greedy` always share the same limit within a
panel. The cut-generation solver can run hundreds of separation rounds inside a
single "solve" (326 were observed), so a per-solve limit would let CG do dozens
of times more work than Exact in the same wall clock and the comparison would be
meaningless. The main and role panels use 600 s and the scaling panel uses
3600 s, because that is where instances actually hit the boundary.
`time_limit_s` is stored on every row so the rule is auditable, and the
`unified_time_limit` gate checks it as a method total with a 5 % + 5 s tolerance
(a single indivisible separation-oracle call, ~40-50 s, can overrun by about
1.4 % at 3600 s).

## Code revision identifiers

`code_commit` reports the git commit when the tree is a checkout (which is the
case for a cloned release), and otherwise a content digest over the twelve files
in `NUMERICAL_CORE`. The digest normalises line endings, so the same source hashes
identically whether it was checked out in CRLF or LF.

| value | rule | what it identifies |
|---|---|---|
| `core-a8deebd9ddef8772f9768affdf9ee483d1e48193` | byte-exact, as deployed | recorded on every row of the published archive |
| `core-f018acb54d16e1cfd42e539d86562dcff1577365` | line-ending normalised | the same code, before identifiers were renamed to `MPCF` |
| `core-75c2b55cf133a5bf762f4277b252b8c7977da757` | line-ending normalised | the code in this repository |

The first two describe the same code: the deployed tree carried mixed line endings
(`task_path_fortification.py` in CRLF, `rev/panels.py` in LF), which is the only
reason they differ; under the normalised rule the *content* is identical across
the deployed tree, the pre-rename release and a fresh clone.

The third differs from the second only in identifier names — the `MPCF` rename and
the corresponding `NUMERICAL_CORE` filenames such as `mpcf_supplementary.py`. No
formula, bound, cost system or panel definition changed, which was checked by
relabelling the published archive's 3 834 method cells and regenerating every
downstream file: all 34 statistics and table CSVs stayed numerically identical.
