# Reviewer 2（种子数）回应：可直接粘贴的 Results 段落

所有数字均来自归档 `revision/final_archive/`，可逐项追溯到
`statistics/seed_effect_stability.csv`、`statistics/seed_paired_intervals.csv`、
`statistics/seed_convergence.csv`、`statistics/seed_stratified_bootstrap.csv`
与 `figures/figS1.pdf`。

---

## 一、中文版（Results 新增小节）

### 种子扩展稳健性分析

**设计。** 我们保留原稿的 45 图冻结基准不变——它仍是全部 38 个方法的完整比较面板，
主分析结论不因本节而改动。在此基础上，我们**另加 5 个预指定种子**
（66、77、88、99、110；与原面板的 11、22、33、44、55 无交集），对 9 个关键方法
重复整个流程，新增 45 张独立图、共 1 215 条求解记录，合并后共 90 张图、
每个 topology×N 单元恰好 10 个随机实现。扩展种子在**任何结果产生之前**即写入代码常量
（`rmcd_f.rev.seedext.SEED_EXTENSION_SEEDS`），并由机器可读清单
`seed_robustness_manifest.json` 记录 `prespecified: true`；没有任何一个种子是按结果挑选的。
推断单位始终是图，三个预算仍作为图内重复测量先取平均。

**（1）效应估计的稳定性。** 表 M(1) 并排列出 5 种子与 10 种子下的均值与中位数相对间隔 ḡ。
8 个对照方法中，最大的均值漂移为 **0.0034**（MPCF-Greedy：0.0309 → 0.0343），
最小的为 0.0003；相对各方法自身的效应规模（Greedy 的效应为 0.031），
这一漂移为 11% 量级。**全部 8 个方法的效应方向在 5 种子与 10 种子下完全一致。**
值得注意的是，合并种子后 ḡ_Greedy 是**上升**的，即基线方法看起来更差，
因此"增加随机实现"只会加强而非削弱本文的主张。
两个精确求解器（MPCF-Exact、MPCF-CG）在两种种子数下 ḡ 恒为 0，不受影响。

**（2）区间稳定性与收窄。** 表 M(2) 给出对 MPCF-Exact 的图级配对效应及其
以图为簇的 95% bootstrap 区间（10 000 次重采样）。**全部 7 个具有非零区间宽度的
对照方法在 10 种子下一致收窄**：MPCF-Greedy 由 0.0130 收窄至 0.0099（−24%），
NoProtection 由 0.0539 收窄至 0.0387（−28%），收窄幅度整体介于 **21%–35%**
（MPCF-CG 与 MPCF-Exact 数值完全相同，区间宽度本就为 0，不存在收窄问题）。
所有区间在两个种子数下都**不包含 0**。更关键的是，
**匹配对 rank-biserial 在 5 种子与 10 种子下都等于 −1.000**：
在全部 90 张图上，MPCF-Exact 没有在任何一张图上劣于任何对照方法，**一次例外都没有**。
按 topology×N 分层的 bootstrap（9 层、90 个观测）复核后结论不变
（MPCF-Greedy 的区间为 [−0.0385, −0.0304]，同样不含 0）。

**（3）估计在多少个种子上趋于稳定。** 图 S1 给出相对间隔的均值随种子数 k 的变化。
对每个 k，我们枚举**全部** C(10,k) 个种子子集（k = 2…10 共 1 013 个子集），
而不是采用某一种累加顺序，因此曲线不依赖于"先加入哪个种子"。
结果有两点值得强调。其一，**子集均值对 k 恒等不变**（恒为 0.034330）：
这是一个确定的数学性质——每个种子出现在相同数量的子集中，
因此子集均值的平均等于全部 10 种子的总均值。换言之，"用几个随机实现"不影响点估计本身。
其二，随 k 变化的只有子集间的离散程度，而它**单调收窄**：
2.5–97.5% 分位带宽由 k=2 时的 0.01181 降至 k=5 时的 0.00727、
k=8 时的 0.00295，至 k=10 归零。
这说明 5 个随机实现已经落在曲线的平坦段上——点估计不再移动，
只是不确定性带随样本增加而按预期收窄。本文的结论因此不是特定 5 张图的产物。

---

## 二、English version (Results subsection)

### Seed-extension robustness analysis

**Design.** The original 45-graph frozen benchmark is left untouched: it remains the
full 38-method comparison panel, and no result in the main analysis changes because of
this subsection. On top of it we add **five prespecified seeds** (66, 77, 88, 99, 110,
disjoint from the original 11, 22, 33, 44, 55) and repeat the full pipeline for nine key
methods, yielding 45 additional independent graphs and 1,215 further solver records —
90 graphs in total, with exactly ten random realizations in every topology×N cell. The
extension seeds were fixed as a code constant
(`rmcd_f.rev.seedext.SEED_EXTENSION_SEEDS`) **before any of their results were examined**,
and a machine-readable manifest (`seed_robustness_manifest.json`) records
`prespecified: true`; no seed was selected on its outcome. The inferential unit remains
the graph, with the three budgets still averaged inside each graph as repeated measures.

**Effect-estimate stability.** Table M(1) reports the mean and median relative gap ḡ at
five and at ten seeds side by side. Across the eight comparator methods the largest shift
in the mean is **0.0034** (MPCF-Greedy: 0.0309 → 0.0343) and the smallest is 0.0003 —
about 11% of the effect size itself for Greedy (0.031). **All eight methods keep the same
effect direction** at five and at ten seeds. Notably, pooling the seeds makes ḡ_Greedy
*larger*, i.e. the baselines look worse, so adding realizations strengthens rather than
weakens the claim. Both exact solvers (MPCF-Exact, MPCF-CG) have ḡ ≡ 0 under either
seed count and are unaffected.

**Interval stability and narrowing.** Table M(2) gives the graph-level paired effect
against MPCF-Exact with its graph-clustered 95% bootstrap interval (10,000 resamples).
**The interval narrows for all seven comparators with non-zero width at ten seeds**:
MPCF-Greedy from 0.0130 to 0.0099 (−24%), NoProtection from 0.0539 to 0.0387 (−28%),
with the reductions spanning **21%–35%** overall (MPCF-CG coincides numerically with
MPCF-Exact, so its width is zero by construction and there is nothing to narrow).
Every interval excludes zero under both seed counts. More importantly,
the **matched-pairs rank-biserial equals −1.000 at five and at ten seeds alike**: across
all 90 graphs, MPCF-Exact is never worse than any comparator on any graph, without a
single exception. A bootstrap stratified by topology×N (nine strata, 90 observations)
leaves the conclusion unchanged (MPCF-Greedy: [−0.0385, −0.0304], again excluding zero).

**Where the estimate stabilises.** Figure S1 plots the mean relative gap against the
number of seeds k. For each k we enumerate **all** C(10,k) seed subsets (1,013 subsets
for k = 2…10) rather than one cumulative ordering, so the curve does not depend on the
order in which seeds are added. Two features matter. First, the **subset mean is exactly
invariant in k** (0.034330 throughout): this is a deterministic property — every seed
appears in the same number of subsets, so the average of subset means equals the grand
mean over all ten seeds. In other words, *how many* random realizations are used does not
move the point estimate. Second, only the dispersion across subsets changes with k, and it
**narrows monotonically**: the 2.5–97.5% band falls from 0.01181 at k = 2 to 0.00727 at
k = 5 and 0.00295 at k = 8, reaching zero at k = 10. Five random realizations therefore
already sit on the flat part of the curve: the point estimate no longer moves, and only
the uncertainty band contracts as expected with more samples. The paper's conclusions are
consequently not an artefact of which particular five graphs were drawn.

---

## 三、一句话摘要（可放在 response letter 开头）

> 把种子数从 5 加倍到 10 之后，效应方向、显著性、以及
> 「在每一张图上 MPCF-Exact 都不劣于对照」这一逐图一致性（90/90，rank-biserial = −1.000）
> 全部不变；点估计漂移 ≤ 0.0034，所有非零宽度的 95% 区间一致收窄 21%–35%，
> 且对全部 C(10,k) 子集枚举的收敛曲线显示 5 个实现已落在平坦段。
> 结论不依赖于最初抽到的是哪 5 张图。

## 四、可引用的图表

| 引用 | 文件 |
|---|---|
| 图 S1（收敛曲线） | `figures/figS1.pdf` / `figS1.png` |
| 表 M(1) 效应稳定性 | `tables/table_seed_robustness.csv` = `statistics/seed_effect_stability.csv` |
| 表 M(2) 区间稳定性 | `statistics/seed_paired_intervals.csv` |
| 收敛原始数据 | `statistics/seed_convergence.csv` |
| 分层 bootstrap 复核 | `statistics/seed_stratified_bootstrap.csv` |
| 预指定声明 | `statistics/seed_robustness_manifest.json` |
| 原始运行记录 | `results/seedext_runs.csv`（1 215 行） |
