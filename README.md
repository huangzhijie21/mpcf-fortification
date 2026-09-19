# MPCF — Finite-Budget Mission-Path Fortification

Reference implementation of **MPCF** (Mission-Path Cut Fortification): a
certified solver for placing a *finite protection budget* on the nodes of a
directed equipment network so that the worst-case **adaptive, cost-weighted
mission-path cut** an attacker can achieve is minimised.

---

## 1. What the method does

An attacker removes nodes from a directed mission network subject to a per-node
attack cost. The defender first spends a **finite budget** `B` protecting
nodes; protecting node `v` changes its attack cost from `a_v` to
`δ_v = m·a_v`. The quantity being maximised by the defender is

```
κ(P) = min over adaptive attacker strategies of  (cost of the cheapest surviving mission path)
```

computed exactly as a weighted node-split minimum cut on the frozen protection
set `P`. The optimisation is

```
maximise  κ(P)
s.t.      Σ_{v∈P} b_v  ≤  B          (b_v = protection cost of node v)
          P ⊆ V_R                    (removable nodes only)
```

Three solvers are provided. **They are not three independent contributions of
the same status** — two of them are exact and one is an ablation:

| solver | method | certificate |
|---|---|---|
| `MPCF-Exact` | compact MILP (HiGHS via `scipy.optimize.milp`) | `solver_closed` — proven optimal |
| `MPCF-CG` | cut-generation master + exact PathCut separation oracle | `cg_closed` when `L = U` |
| `MPCF-Greedy` | one-step exact-marginal ablation | **none** — `certificate_mode = open` |

`MPCF-Greedy` deliberately claims no optimality. It exists to quantify how much
the exactness is worth, and the paper reports its gap against the certified
optimum. It is *not* a proposed method.

Every run is independently re-verified: the frozen protection set is replayed
through a max-flow/min-cut computation and the result is compared with the
solver's own objective (`replay_kappa`). A successful run without that
independent replay fails the `baseline_fairness` gate.

---

## 2. Repository layout

```
src/rmcd_f/                 the package
  task_path_fortification.py   the three solvers, SolverLog, certificates
  operational_motif.py         path-closed system construction, node splitting
  model.py, flow.py            network model and max-flow / min-cut
  synthetic.py                 the 4:3:3:4 topology generator
  cost_profiles.py             attack-cost and protection-cost profiles
  protection_baselines.py      local baselines (degree, betweenness, ...)
  official_review.py           adapters for external/upstream baselines
  cli.py                       graph I/O and the `rmcd-f` command
  rev/                         the revision layer (see below)
scripts/
  rev_run.py                   run a panel; writes results/runs_long.csv
  rev_statistics.py            raw CSV in -> statistics CSV out (no solver)
  rev_plots.py                 statistics CSV in -> PDF/PNG out (no solver)
  rev_gates.py                 acceptance gates; non-zero exit on FAIL
  run_official_review_matrix.py, run_official_review_sequences.py
                               upstream baselines (need the third-party tree)
  export_finder_legacy.py      FINDER export shim (Python 3.7 env)
tests/                      88 tests; `pytest -q` from the repo root
revision/
  scripts/run_all_local.sh   run every panel on this machine (start here)
  tools/                     merge and inventory helpers
  scripts/                   shell drivers
  docs/                      changes, delivery summary, figure/table inventory
docs/                        algorithm specs and protocol documents
```

### The `src/rmcd_f/rev/` layer

`rev/` is *not* part of the mathematics. It is the experiment-organisation layer
added during revision, and the numerical core is untouched by it:

| module | role |
|---|---|
| `ids.py` | `graph_id` / `base_id` / `instance_id`; `code_commit`, `config_hash` |
| `schema.py` | the 36-column `runs_long.csv` contract |
| `panels.py` | deterministic panel freezing (main, scaling, role, heterogeneity) |
| `seedext.py` | the 5 prespecified extension seeds |
| `registry.py` | the explicit 36-variant baseline registry |
| `harness.py` | one instance × one budget → one set of run records |
| `stats.py` | graph-clustered repeated-measures statistics |
| `robustness.py` | the seed-extension stability analysis |
| `env.py` | automatic `environment.json` capture |

---

## 3. Install

Requires **Python ≥ 3.10**.

```bash
git clone <this-repo> mpcf-fortification
cd mpcf-fortification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[test,experiments]"
```

That installs `networkx`, `numpy`, `scipy` (the MILP backend is the HiGHS
bundled with SciPy), plus `pytest` and `matplotlib`.

Verify the install:

```bash
pytest -q                          # expect: 88 passed
rmcd-f --help
```

> Run `pytest` **from the repository root**. Some tests import the `scripts/`
> package, which is resolved relative to the working directory.

### Optional: the upstream baselines

26 of the 36 comparison variants are **adapters around third-party
implementations** (GND, GDM, CoreHD, CI, EI, FINDER, MinSum, network
entanglement, vertex entanglement, …). That upstream code is **not** included
here — it is a separate project under its own licence. To reproduce those rows
you must obtain it yourself and point the tooling at it:

```bash
git clone <upstream-network-dismantling-repo> external/review-main
python scripts/official_source_preflight.py --review-path external/review-main
```

Two of the upstream families additionally need their own isolated environments,
because they require TensorFlow 1.14 / Python 3.7:

```bash
bash scripts/setup_finder_environment.sh        # FINDER  (Python 3.7, TF 1.14)
python scripts/gdm_environment_preflight.py     # GDM     (torch + torch_scatter)
```

Without those, the registry marks those variants as not evaluated and the
coverage table says so explicitly — nothing is silently skipped.

---

## 4. Quick start

```bash
# 1. freeze a small panel and solve it
python scripts/rev_run.py --experiment main --output results \
    --workers 8 --time-limit 600 --limit-instances 2

# 2. statistics (raw runs only; no solver is imported)
python scripts/rev_statistics.py --output results --resamples 10000

# 3. figures
python scripts/rev_plots.py --output results

# 4. acceptance gates (non-zero exit if any gate fails)
python scripts/rev_gates.py --output results
```

`--limit-instances 2` is a smoke-test hook; drop it for a real run.

---

## 5. Reproducing the published experiments

**Everything runs on one machine.** Each panel is solved by a local process pool
(`rev_run.py --workers N`); the four analysis stages read plain CSV. The
published archive happened to be produced on three containers because that was
the hardware on hand — no part of this repository requires a cluster, a
scheduler, or any remote host.

The one-command path:

```bash
# smoke test first: one graph per panel, every method still runs (about 8 min
# on 4 cores; the scaling panel is skipped here)
bash revision/scripts/run_all_local.sh --smoke

# then the real thing (auto-detects cores; override with WORKERS=16)
bash revision/scripts/run_all_local.sh
```

`--smoke` shortens the per-method limit to 20 s and skips the scaling panel,
because reducing the graph count does not reduce the method count: one graph
still runs all 38 methods, so a 600 s limit would make the worst case 38 × 600 s
for a single panel. Add `--with-scaling` if you want that panel in the smoke run.

Measured cost of the full published panel set, summed over every solver call in
the archive:

| panel | core-hours | 8 cores | 16 cores | 32 cores |
|---|---:|---:|---:|---:|
| `main` | 5.3 | 0.7 h | 0.3 h | 0.2 h |
| `role` | 14.3 | 1.8 h | 0.9 h | 0.4 h |
| `heterogeneity` | 23.5 | 2.9 h | 1.5 h | 0.7 h |
| `seedext` | 5.4 | 0.7 h | 0.3 h | 0.2 h |
| `scaling` | 140.2 | 17.5 h | 8.8 h | 4.4 h |
| **total** | **188.7** | **23.6 h** | **11.8 h** | **5.9 h** |

The `scaling` panel dominates because `N=1000` under a 3600 s per-method limit is
genuinely expensive. `--skip-scaling` gives a full pass in about 4 core-hours on
8 cores. These figures come from x86 server CPUs (Intel Xeon 8352V / AMD EPYC
9654); a laptop will be slower, so start with `--smoke`.

### Doing it panel by panel

Always pass the **same** time limit to every method inside a panel.

```bash
# main panel: 45 graphs x 3 budgets x 38 methods
python scripts/rev_run.py --experiment main --output results \
    --workers 8 --time-limit 600

# role-composition panel: 45 base graphs x 6 compositions
python scripts/rev_run.py --experiment role --output results \
    --workers 8 --time-limit 600

# joint heterogeneity panel: b_v x m_v coupling profiles
python scripts/rev_run.py --experiment heterogeneity --output results \
    --workers 8 --time-limit 600 --include-conversions

# seed-extension panel: the 5 prespecified extra seeds, 9 key methods
python scripts/rev_run.py --experiment seedext --output results \
    --workers 8 --time-limit 600

# scaling panel: N in {200,300,500,1000}, 9 instances per cell
python scripts/rev_run.py --experiment scaling --output results \
    --workers 8 --time-limit 3600

# merge every shard into results/runs_long.csv
python scripts/rev_run.py --experiment all --output results --collect-only

# analysis
python scripts/rev_statistics.py --output results --resamples 10000
python scripts/rev_plots.py --output results
python scripts/rev_gates.py --output results
```

Instances are frozen deterministically from `(topology, N, seed)`, and re-freezing
is idempotent: an identical freeze reuses the existing file, and a divergent one
is refused. This is what makes the panel reproducible across machines.

### The unified time limit

`--time-limit` is a **per-method total**, not a per-internal-solve limit. The
cut-generation solver can run hundreds of separation rounds inside one "solve";
if the limit were per-solve, CG could do dozens of times more work than Exact
within the same wall clock and the comparison would be meaningless. All three
solvers share the same overall budget, and `time_limit_s` is stored on every
row so the rule is auditable. One indivisible separation-oracle call
(~40–50 s) can overrun the limit by ≈1.4 % at 3600 s; the gate tolerance is
5 % + 5 s.

---

## 6. Output contract

Every algorithm in every panel writes into **one** `results/runs_long.csv` with
exactly these 36 columns. A field a method does not produce is the literal `NA`
— never dropped, so a single reader can consume every experiment:

```
experiment  instance_id  graph_id  base_id  topology  N  num_edges  seed
composition  budget_ratio  budget_abs  method  variant
kappa  kappa_opt  relative_gap  actual_cost  selected_count  selected_nodes
runtime_selection_s  runtime_evaluation_s  runtime_total_s  status
incumbent  best_bound  final_gap  bb_nodes  replay_kappa  certificate_mode
cg_L  cg_U  cg_iterations  cg_cuts  time_limit_s
config_hash  code_commit
```

`relative_gap = (kappa_opt − kappa) / kappa_opt`, where `kappa_opt` is the
cross-solver certified optimum of that `(instance, budget)` cell. The two exact
solvers must agree; a disagreement is flagged as `certificate_conflict` and the
`exact_vs_cg` gate fails.

### `budget_abs` is exact, not rounded

The main panel has `b_v = 1` for every node, so `budget_abs = ratio × N` and
`N = 42` at `2 %` gives **0.84** — below the cheapest single protection, i.e. a
*zero-effective-budget* cell. `results/instance_budget_cells.csv` records both
`budget_abs` and `budget_abs_floor`, and `statistics/zero_budget_audit.csv`
reports the all-cells and effective-budget subsets separately. Rounding would
hide that degeneracy.

### Inferential units

| id | definition | role |
|---|---|---|
| `graph_id` | `topology + N + seed` | the **independent unit** of the main panel |
| `budget_ratio` | 0.02 / 0.05 / 0.10 | a **repeated measure inside** a graph |
| `base_id` | `topology + N + seed` | the cluster of the role-composition panel |

Statistics therefore average the three budgets **inside** each `(graph, method)`
before testing, run Friedman over 45 graph blocks, use 45 graph-level paired
Wilcoxon tests, Holm-adjust the pairwise family, and bootstrap **whole graphs**
(10 000 resamples, carrying all of a graph's budgets together). Nothing treats
135 graph-budget pairs or 810 role records as independent samples.

### Revision identifiers

`code_commit` records the git commit when the tree is a checkout, and otherwise
a content digest over the twelve files of `NUMERICAL_CORE`, with line endings
normalised so the same source hashes identically on every platform.

| value | rule | what it identifies |
|---|---|---|
| `core-a8deebd9ddef8772f9768affdf9ee483d1e48193` | byte-exact, as deployed | recorded on every row of the published archive |
| `core-f018acb54d16e1cfd42e539d86562dcff1577365` | line-ending normalised | the same code, before identifiers were renamed to `MPCF` |
| `core-75c2b55cf133a5bf762f4277b252b8c7977da757` | line-ending normalised | the code in this repository |

The first two describe the same code: the deployed tree happened to carry mixed
line endings (`task_path_fortification.py` in CRLF, `rev/panels.py` in LF), which
is the only reason they differ. Hashing the deployed tree, the pre-rename release
and a fresh clone under the normalised rule gives one value, which is how the
reproducibility claim was checked.

The third differs from the second **only in identifier names** — the `MPCF`
rename, and the corresponding `NUMERICAL_CORE` filenames such as
`mpcf_supplementary.py`. No formula, bound, cost system or panel definition
changed. That was verified the strong way: relabelling the published archive's
3 834 method cells to the `MPCF` names and regenerating every downstream file
left all 34 statistics and table CSVs numerically identical.

So a clone reports a git SHA and a non-checkout copy reports `core-75c2b55c…`;
either way `code_revision_consistency` holds, because every row in one run
carries the same value. When comparing a fresh run against the published CSV,
compare the *numbers*, not this identifier.

---

## 7. Optional: splitting a run across machines

**You do not need this to reproduce anything.** The panels are the same job
whether they run on one host or several; see §5 for the single-machine path,
which is what a reader should use.

It is documented only because the published archive was produced that way: three
32-CPU containers, with each panel assigned to one machine. If you have a cluster
and want the same layout, `revision/tools/` and `revision/scripts/` drive it:

```bash
cp revision/scripts/servers.env.example revision/scripts/servers.env
# edit it, then:
source revision/scripts/servers.env

python revision/tools/remote_exec.py exec --command "uptime"   # smoke test
python revision/tools/finalize.py --poll-seconds 300 --max-hours 7
```

`finalize.py` waits for every panel, forces a final `--collect-only`, downloads
each server's results, merges them with duplicate and fingerprint checks, and
then runs statistics, plots and gates. `merge_archives.py` does the merge alone
if you already have per-host result trees.

**If you do distribute, do not mix CPU models inside one panel.** Wall-clock time
is a reported result here, so the `hardware_uniformity` gate records the CPU model
per experiment and fails the build if one experiment spans two models. In the
published run: main + heterogeneity + seedext + upstream on an Intel Xeon 8352V,
role + scaling on two AMD EPYC 9654 hosts (single-core throughput differed by
about 1.7×, which is exactly why the gate exists).

**Credentials and hosts are never stored in this repository.** The server table
is read from `MPCF_SRV<n>_HOST`, `MPCF_SRV<n>_PORT` and `SSH_PASSWORD_SRV<n>`.

---

## 8. Acceptance gates

`scripts/rev_gates.py` writes `gates/acceptance_gates.csv` and exits non-zero if
any gate fails. The published archive passes all 17:

```
exact_replay                objective matches an independent min-cut replay
exact_vs_cg                 two independent exact solvers agree everywhere
cg_bounds                   L <= U on every cut-generation run
greedy_feasibility          every Greedy solution respects the budget
baseline_budget_feasibility every method returns a budget-feasible set
no_method_beats_optimum     no method exceeds a certified optimum
greedy_monotonicity         trace-level monotonicity (unit-tested)
graph_dependence            primary unit is graph_id; budgets averaged inside
role_dependence             role panel clustered on base_id
baseline_fairness           every successful run carries an independent replay
scaling_fairness            one time limit and one method set per scaling cell
reproducibility             no row lacks seed / config hash / code revision
code_revision_consistency   every row from a single code revision
unified_time_limit          the declared limit is respected as a method total
seed_extension              extension seeds prespecified and fully covered
hardware_uniformity         one CPU model per experiment
plotting                    every figure rebuilds from exported CSV tables
```

Two properties worth knowing if you extend this:

* **`MPCF-Exact` and `MPCF-CG` are fully deterministic.** Re-running the
  frozen revision reproduced 1 080/1 080 Exact and CG rows bit-for-bit.
* **`MPCF-Greedy` is not, once it hits its time limit** — its incumbent then
  depends on wall-clock. Such rows are marked `status = time_limit` with
  `certificate_mode = open`. Do not treat two Greedy runs at the limit as a
  reproducibility failure; treat the marker as the signal it is.

---

## 9. Statistics and figures are decoupled from the solvers

`rev_statistics.py` and `rev_plots.py` import **no solver**. They read
`runs_long.csv` and the exported CSV tables only, so a figure can be re-styled
or a test re-run on a laptop without a solver installation, and a reviewer can
recompute every number from the published CSVs.

Seeded analyses (bootstrap, seed-extension) take `--seed` and are reproducible.

---

## 10. Documentation

| document | content |
|---|---|
| `docs/MPCF_ALGORITHM_SPEC.md` | the algorithm in full |
| `docs/THEORY_AND_PROOFS.md` | the certificates and their proofs |
| `docs/OFFICIAL_DISMANTLING_BASELINES.md` | every upstream baseline and its adapter |
| `docs/MPCF_BASELINE_PROTOCOL.md` | baseline fairness rules |
| `revision/docs/README.md` | what the revision layer changed, item by item |
| `revision/docs/CHANGES_ZH.md` | the same, in Chinese, with code locations |
| `revision/docs/DELIVERY_SUMMARY_ZH.md` | the published numbers |
| `revision/docs/REVIEWER2_SEED_ROBUSTNESS_ZH.md` | the seed-extension response, in Chinese |
| `revision/docs/SERVER_BACKUP_ZH.md` | how the published run was deployed and archived |

Four of the five documents under `revision/docs/` are in Chinese;
`revision/docs/README.md`, and everything else in this repository, is in English.

---

## 11. Citation

If you use this code, please cite the paper it accompanies:

> Huang, Z., Li, X., Wang, T., & Bai, L. (2026). *Identifying Critical
> Fortification Node Sets for Kill-Web Mission Paths via Minimum Node-Cut
> Optimization*. Preprint. https://doi.org/10.20944/preprints202609.0576.v1

```bibtex
@misc{huang2026mpcf,
  title  = {Identifying Critical Fortification Node Sets for Kill-Web Mission
            Paths via Minimum Node-Cut Optimization},
  author = {Huang, Zhijie and Li, Xiaobo and Wang, Tao and Bai, Liang},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.20944/preprints202609.0576.v1},
  url    = {https://doi.org/10.20944/preprints202609.0576.v1}
}
```

The same metadata is in `CITATION.cff`, which GitHub reads to offer a
"Cite this repository" button. It is a preprint; when the paper is published,
update both the BibTeX above and the `preferred-citation` block in `CITATION.cff`
to the version of record.

## 12. Licence

No licence file is included yet; the author must choose one. Until then the
default applies and **all rights are reserved**. If you intend this to be
reusable, MIT or Apache-2.0 is the usual choice for research code.
