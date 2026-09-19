# MPCF 返修代码修改说明（中文）

本文件对应返修任务书的七个部分，逐条说明**改了什么、改在哪里、跑出来长什么样**。
核心原则没有变：`MPCF-Exact / MPCF-CG / MPCF-Greedy` 的数学模型、目标函数、
有限提升攻击代价、精确自适应 PathCut 攻击者全部保持原样。

---

## 一、统一实验 ID 与数据结构

**新增** `src/rmcd_f/rev/ids.py`、`src/rmcd_f/rev/schema.py`

| 概念 | 定义 | 用途 |
|---|---|---|
| `graph_id` | `topology + N + seed` | 主实验的**独立实验单位**（45 个） |
| `base_id` | `topology + N + seed` | Role 实验的**重复测量簇**（同一 base 下的 6 个 composition × 3 个 budget 同进同出） |
| `instance_id` | 实验 + base + 修饰符 | 一个冻结图视图，全局唯一 |
| `budget_ratio` | 0.02 / 0.05 / 0.10 | 图内重复测量，**不是**独立样本 |

**变量改名**（`src/rmcd_f/task_path_fortification.py`）：
大容量常数 `M` → `C_INF`；列名 `edge_count` → `num_edges`。
`C_INF` 现在是一个带名字的常量并附说明"有限容量代替无穷"。

**统一输出格式** `results/runs_long.csv`，固定 36 列（缺的填 `NA`，不删列）：

```
experiment instance_id graph_id base_id topology N num_edges seed
composition budget_ratio budget_abs method variant
kappa kappa_opt relative_gap actual_cost selected_count selected_nodes
runtime_selection_s runtime_evaluation_s runtime_total_s status
incumbent best_bound final_gap bb_nodes replay_kappa certificate_mode
cg_L cg_U cg_iterations cg_cuts time_limit_s config_hash code_commit
```

> **一处需要你确认的取舍**：`budget_abs` 存的是**求解器实际使用的精确预算**，不是取整
> 后的整数。主实验 `b_v ≡ 1`，所以 `N=42 @ 2%` 的 `budget_abs = 0.84`，**低于最便宜的
> 单点防护代价 1**，是一个"零有效预算"退化格。取整会把这个退化**藏起来**。
> 整数下限另存在 `results/instance_budget_cells.csv` 的 `budget_abs_floor` 列，
> 并在 `statistics/zero_budget_audit.csv` 中单独报告。

## 二、重写统计模块

**新增** `src/rmcd_f/rev/stats.py`、`scripts/rev_statistics.py`

主分析（预先指定）：

1. 先在每个 `(graph, method)` 内把三个 budget 的 relative gap 求平均：
   $\bar g_{gm}=\frac{1}{3}\sum_b g_{gbm}$；
2. 以 **45 个 graph 为 block** 做 Friedman 检验；
3. 与 `MPCF-Exact` 做 **45 个 graph-level 配对**的 Wilcoxon 符号秩检验；
4. Holm 校正 + matched-pairs rank-biserial + **整簇 bootstrap 95% CI**。

关键实现细节：bootstrap 抽的是**整张图**，一次抽中就把它的 3 个 budget 全部带进来。
实测 `n_resampled_observations = 135`、`n_clusters = 45`，**不会**把 135 个
graph-budget 对当 135 个独立样本。

输出文件（全部机器可读）：

| 文件 | 内容 |
|---|---|
| `statistics/statistics_graph_level.csv` | Friedman 统计量、df、n=45、p |
| `statistics/pairwise_graph_level.csv` | comparator、n、Wilcoxon 统计量、`p_raw`、`p_holm`、rank-biserial、均值/中位数效应、CI 上下限 |
| `statistics/statistics_by_budget.csv` | 2%/5%/10% 各自 n=45 的完整统计 |
| `statistics/role_repeated_statistics.csv` | 以 `base_id` 为簇的重复测量分析 |
| `statistics/bootstrap_summary.csv` | metric、method、estimate、CI、bootstrap unit |
| `statistics/zero_budget_audit.csv` | all 135 与 nonzero-effective 120 的描述性结果 |
| `statistics/seed_effect_stability.csv` | 5 种子 vs 10 种子的效应估计漂移（见第十节） |
| `statistics/seed_paired_intervals.csv` | 5 种子 vs 10 种子的配对效应与整簇 95% CI |
| `statistics/seed_convergence.csv` | 全部 C(10,k) 子集上的收敛曲线 |
| `statistics/seed_stratified_bootstrap.csv` | 按 topology×N 分层的 bootstrap 复核 |
| `statistics/seed_robustness_manifest.json` | 扩展种子的**预指定**声明（机器可读） |

## 三、显式 variant registry

**新增** `src/rmcd_f/rev/registry.py` → `methods/method_variant_registry.csv`

数字对账（`methods/variant_reconciliation.json`，代码断言，不是手写）：

```
declared_methods            45
certified_reference_methods  2   (MPCF-Exact / MPCF-CG，是参照不是对照)
main_panel_variants         38
excluded_from_main_panel     7   (连通性/特征向量周期性/上游 n<=18 限制/覆盖不足)
comparison_variants         36   ← 45 − 2 − 7
heterogeneous_cost_conversion_variants  8
```

每个 variant 一行，含 `variant_id / base_method / reference / code_source /
native_graph / directed_handling / node_weight_handling / execution_mode /
cost_conversion / parameters / coverage_n / runtime_definition`，
并额外记录 `predeclared_panel / in_comparison_set / dependency_environment /
experimental_upstream / exclusion_reason`。

`Raw / PerCost / PrefixKnapsack / FullKnapsack` 现在是**显式的 conversion 字段**
（`cost_conversion` 列），不是藏在代码分支里；它们作为 8 个 variant 实际跑在
heterogeneity 面板上（只有在防护代价异质时才有区别）。

另出 `methods/variant_coverage.csv`：每个 variant 的**实际覆盖数**，
缺失的方法不会静默消失。

## 四、扩展 scaling + 求解器日志

`src/rmcd_f/task_path_fortification.py`：

* 新增 `SolverLog`：`incumbent / best_bound / final_gap / bb_nodes /
  replay_kappa / certificate_mode / cg_L / cg_U / cg_iterations / cg_cuts /
  time_limit_s / replay_matches`。
* Exact 从 HiGHS 取 `mip_node_count`（B&B 节点数）、`mip_dual_bound`、`mip_gap`，
  并用**独立重算的 max-flow/min-cut** 填 `replay_kappa`。
* CG 记录 `L / U / U−L / iterations / generated cuts / certificate mode`。
* **Greedy 新增 `time_limit` 参数**，使三种 MPCF 方法可以在**完全相同的实例和
  统一时限**下比较。
* 所有 MILP 固定 `threads=1`、`random_seed=0`，避免多 worker 超订并保证计时可比。

Scaling 面板：`N ∈ {200, 300, 500, 1000}`，固定 4:3:3:4 组成与 `3N` 条有向接口边，
统一 3600 s 时限，**每格 9 个实例**。聚合表 `tables/table_scaling.csv` 自动从 raw runs 生成：
`N × budget × method → median/max runtime、solved count、median/max gap、
bb_nodes 中位数、final_gap 最大值、status 分布`。

> **统一时限的口径在本轮被明确为「每方法总量」而不是「每次内部求解」。**
> 起因是实测发现 CG 的 cut-generation 在一次「求解」内会生成数百轮 cut
>（最多观测到 326 轮），若按单次内部求解计时，它可以在总时限内开出几十倍的工作量，
> 三方法就不可比了。现在 `overall_time_limit` 由三个方法共同遵守，
> 由 `unified_time_limit` 验收门按「方法总量、容差 5% + 5 s」逐行核对。
> 一个不可分割的 separation oracle 调用（约 40–50 s）在 3600 s 上最多造成约 1.4% 超出，
> 在容差内。

## 五、联合异质性实验

`src/rmcd_f/rev/panels.py::apply_joint_heterogeneity`

* $b_v \in \{1,2,4\}$（防护代价），$m_v=\delta_v/a_v \in \{0.5,1,2\}$（相对提升倍数）；
* 三个 profile：`independent` / `positive_corr` / `negative_corr`，外加 `homogeneous` 对照；
* 每个 role 内部保持多重集不变，**只改变 b 与 m 的耦合方式**；
* **实测 Spearman 相关系数写进 manifest 并画进图**，不靠名字声称：
  最终实测得到 independent = **+0.044**（范围 −0.214…+0.321）、
  positive_corr = **+1.000**、negative_corr = **−0.978**（范围 −0.959…−1.000）；
* `b_v`、`m_v`、每点 uplift 都写进图文件节点属性，进程重启也能精确复现。

输出 `tables/table_heterogeneity.csv`、`tables/table_heterogeneity_greedy_gap.csv`
以及 `runs_long.csv` 中对应的行。

## 六、MPCF-Greedy 单元测试

`tests/rev/test_rev_greedy_properties.py`（14 个测试）：

1. **每一步满足预算**；2. **κ(P) 单调不下降**；3. **最多选 |V_R| 个节点后终止**；
4. **三节点反例**：搜索程序 `revision/tools/find_greedy_counterexample.py`
   在"2 个 C 节点 + 1 个 L 节点、边界固定不可移除"的实例族中**精确命中你论文的数值**：

   ```
   Greedy objective = 3,  Exact objective = 4   (budget = 2)
   c0: a=1, δ=1     c1: a=1, δ=2     l0: a=3, δ=1
   Greedy 选 {c0, l0} → κ=3   Exact 选 {c1, l0} → κ=4
   ```

`tests/rev/test_rev_robustness.py`（11 个测试，本轮新增）另行保证第十节的
种子扩展分析在数学上正确，其中 2 个是针对「面板拿到了错误的种子」这一 bug 的回归测试
（该 bug 曾让扩展面板重复了主面板的图）。

测试总数：**55 个，全部通过**（schema 16 + statistics 14 + greedy 14 + robustness 11）。

## 七、绘图与统计彻底解耦

* `scripts/rev_statistics.py`：只读 `runs_long.csv` → 统计 CSV，**不 import 任何求解器**。
* `scripts/rev_plots.py`：只读 CSV → 图，**不 import 任何求解器**。
* 每张图同时输出**矢量 PDF** 和 **600 dpi PNG**。
* 期刊排版：双栏 7.2 in 宽、8–9 pt 字号、legend 移到数据之外、混尺度的面板拆开。
* Figure 1 留足左边 margin；Figure 5（运行时间）与 Figure 6（解质量）按你的编号**拆成两张**。

## 八、环境自动记录

`src/rmcd_f/rev/env.py` → `metadata/environment.json`（每次启动自动写）：
OS/发行版/内核、CPU 完整型号、物理核/逻辑核/cgroup 配额、总内存、
Python 版本、MILP 后端与 HiGHS 版本、Gurobi/CPLEX 可用性、各依赖包版本、
线程设置、统一时限、MIP gap、容差、solver seed、图 seed 列表、
代码修订号、运行时间戳、**runtime 定义**（selection / evaluation / total 各含什么）。

## 九、验收门（自动执行）

`scripts/rev_gates.py` → `gates/acceptance_gates.csv`，任一 FAIL 则退出码非 0。
当前 **17 / 17 全部 PASS**（无 PENDING、无 SKIPPED）：

`exact_replay`、`exact_vs_cg`、`cg_bounds`、`greedy_feasibility`、
`baseline_budget_feasibility`、`no_method_beats_optimum`、`greedy_monotonicity`、
`graph_dependence`、`role_dependence`、`baseline_fairness`、`scaling_fairness`、
`reproducibility`、`code_revision_consistency`、**`unified_time_limit`**、
**`seed_extension`**、**`hardware_uniformity`**、`plotting`。

其中两个是本轮**新增**的：

* `unified_time_limit`：逐行核对每个 MPCF 运行是否遵守它声明的**方法总量**时限
  （容差 5% + 5 s）。实测最差超出 **0.0 s**。
* `seed_extension`：核对扩展种子确实是**预指定**的、覆盖 45 张新图 × 9 方法、
  且 9 个 topology×N 格各自都带满 10 个种子。

`hardware_uniformity` 强制**同一个实验绝不能混用不同 CPU 型号**，
因为运行时间本身就是要报告的实验结果。本轮的分配是
主实验 + 异质性 + 种子扩展 + 上游对照在 Intel Xeon 8352V，
role 与 scaling 在 AMD EPYC 9654。

## 十、种子扩展稳健性（应对「5 个种子够不够」）

**新增** `src/rmcd_f/rev/seedext.py`、`src/rmcd_f/rev/robustness.py`、
`scripts/rev_plots.py::figS1`

设计上刻意**不动**原有的 45 图冻结基准——它仍然是全方法主分析的面板。
扩展是**加法**：另加 5 个**预指定**种子 `(66, 77, 88, 99, 110)`
（与主面板的 `(11, 22, 33, 44, 55)` 无交集，任何一个都不与它们相同），
只跑 9 个关键方法，得到 45 张新图 × 3 budget × 9 method = **1 215 条记录**，
合并后共 **90 张图**、9 个 topology×N 格各 10 个种子。

三个问题各对应一个输出：

1. **效应估计稳不稳** → `seed_effect_stability.csv`：同时报告 5 种子与 10 种子的
   均值/中位数及其差值。实测 8 个对照的 |Δ| ≤ **0.0034**，效应方向 8/8 保持。
2. **区间稳不稳、会不会随样本变窄** → `seed_paired_intervals.csv`：
   对 `MPCF-Exact` 的图级配对效应与整簇 95% CI。实测 7/7 个非零宽度区间**全部收窄**
   （Greedy 0.0130 → 0.0099，NoProtection 0.0539 → 0.0387，整体收窄 21%–35%），
   且 `rank_biserial` 在 5 种子与 10 种子下**都是 −1.000**。
3. **估计在哪一步稳定** → `seed_convergence.csv` + Figure S1：
   对每个 k 枚举**全部** `C(10,k)` 个子集（k≥2 共 **1 013** 个），
   而不是某一种累加顺序，因此曲线与「先加哪个种子」无关。

关键的诚实性处理：

* 扩展方法的名单与种子都写死在 `seedext.py` 里，**在任何结果产生之前就固定**，
  并由 `seed_robustness_manifest.json` 记录 `prespecified: true`；
* `seedext.py` 与 `robustness.py` **刻意放在数值核心之外**
  （不进 `ids.NUMERICAL_CORE` 的 12 文件哈希清单），
  这样新增分析不会改变已冻结归档的 `code_commit`，新旧结果仍可对账；
* 收敛曲线中，子集均值对 k **恒等不变**（数学性质：每个种子出现在相同数量的子集中），
  随 k 变化的只有离散带宽。这一点在清单里被明确写出，
  以免读者误以为「均值稳定」是抽样巧合。
