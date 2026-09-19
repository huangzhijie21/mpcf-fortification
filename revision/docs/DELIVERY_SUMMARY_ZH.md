# MPCF 返修实验结果交付摘要

代码修订：`core-a8deebd9ddef8772f9768affdf9ee483d1e48193`（全部 20 979 行同一修订号）
归档：`revision/final_archive/`　验收门：**17 / 17 PASS**（无 PENDING、无 SKIPPED）

---

## 一、运行规模

| experiment | 冻结实例 | 运行行数 | 说明 |
|---|---:|---:|---|
| `main` | 45 | 5 130 | 45 graph × 3 budget × **38 method**（36 对照 variant + MPCF-Exact/CG） |
| `role` | 270 | 8 910 | 45 base × 6 composition × 3 budget × 11 method |
| `heterogeneity` | 180 | 5 400 | 4 profile × 3 budget × 10 method（含 8 个 conversion variant） |
| `scaling` | 36 | 324 | N ∈ {200,300,500,1000} × 3 budget × 3 method，**每格满 9 实例** |
| `seedext` | 90 | 1 215 | 45 图 × 3 budget × 9 method（新增的 5 个预指定种子） |
| **合计** | **576** | **20 979** | 45 个不同 method；1 728 个 budget 格；0 冲突行、0 重复行 |

## 二、三台机器的分配（硬件同质性）

| 服务器 | CPU | 承担的 experiment |
|---|---|---|
| srv0 | Intel Xeon 8352V @2.10 GHz | `main`（含 upstream 对照）、`heterogeneity`、`seedext` |
| srv1 | AMD EPYC 9654 | `scaling` N=200/300 |
| srv2 | AMD EPYC 9654 | `role`、`scaling` N=500/1000 |

实测单核吞吐：Xeon 3294 /s，EPYC 5668–5834 /s。**同一 experiment 绝不跨 CPU 型号**，
该约束由 `hardware_uniformity` 验收门逐实验核对 CPU 型号并强制执行通过。
论文中主表计时（Xeon）与 scaling 计时（EPYC）需分别注明机器，不可直接做绝对比较。

## 三、主统计（Reviewer 2 要求的图级聚类）

**主分析**（每个 graph 内先把 3 个 budget 的 relative gap 求平均，再检验；推断单位 = graph）：

```
Friedman:  chi2(37) = 911.785,  n = 45 graphs,  p = 2.398e-167
```

**两两比较**：37 个对照（36 个对照 variant + `MPCF-CG`），全部对 `MPCF-Exact`，
每个都是 **45 个图级配对**（不是 135 条记录），聚类 bootstrap 每次重采样带入
135 个观测、45 个簇，Holm 逐步校正：

| 统计量 | 结果 |
|---|---|
| rank-biserial = **−1.000** 的对照数 | **36 / 37** |
| 其中 **45/45 张图全部判 Exact 更优**（零平局）的 | **35 / 36** |
| Holm 校正后最大 p（rb = −1 的对照） | **2.278e-07**（即 `MPCF-Greedy`） |
| 与 Exact 完全一致的对照 | `MPCF-CG`（rb = 0, p = 1, 45/45 平局） |

| comparator | p_raw | p_holm | rank-biserial | mean effect | 95% CI（整簇） | 胜/平/负 |
|---|---:|---:|---:|---:|---|---|
| NoProtection | 5.16e-09 | 1.70e-07 | −1.000 | −0.1929 | [−0.2212, −0.1672] | 0/0/45 |
| OfficialRandomStatic-Protect | 5.16e-09 | 1.70e-07 | −1.000 | −0.1734 | [−0.2004, −0.1489] | 0/0/45 |
| RandomProtect | 5.18e-09 | 1.70e-07 | −1.000 | −0.1710 | [−0.1996, −0.1446] | 0/0/45 |
| OfficialEI_S1-Protect | 5.15e-09 | 1.70e-07 | −1.000 | −0.1636 | [−0.1878, −0.1422] | 0/0/45 |
| …（共 36 项，全部 rb = −1.000，p_holm ≤ 2.28e-07） | | | | | | |
| 最接近的对照：PathFrequencyProtect | 5.68e-14 | 2.10e-12 | −1.000 | −0.0950 | [−0.1079, −0.0833] | 0/0/45 |
| MPCF-Greedy | 1.14e-07 | **2.28e-07** | −1.000 | −0.0309 | [−0.0376, −0.0246] | 0/8/37 |
| MPCF-CG | 1.000 | 1.000 | 0.000 | 0.0000 | [0, 0] | 0/45/0 |

> **36 个对照全部 rank-biserial = −1.000**：`MPCF-Exact` 在**每一张图**上都不劣于对照，
> 且其中 35 个对照是 **45/45 全部严格更优**；只有 `MPCF-Greedy` 出现 8 张图的平局
>（Greedy 在 8 张图上恰好也取到了最优值），**零张图**上被任何对照击败。
> `MPCF-CG` 与 Exact 完全一致（45/45 平局），是两个独立精确求解器的互证。

**分预算副分析**（每个 budget 各自 n = 45 graphs，budget 不再被当作独立样本）：

| budget | Friedman | df | p | n |
|---|---:|---:|---:|---:|
| 2% | 417.798 | 37 | 5.46e-67 | 45 |
| 5% | 734.870 | 37 | 1.45e-131 | 45 |
| 10% | 960.300 | 37 | 1.73e-179 | 45 |

**Role-composition 重复测量分析**（簇单位 = `base_id`，45 个 base，composition 与 budget 在簇内重采样）：

```
Friedman:  chi2(10) = 432.259,  n = 45 base_id,  p = 1.267e-86
10 个两两比较 p_holm 全部 = 5.684e-13,  rank-biserial 全部 = -1.000
```

**零有效预算审计**：`all_cells` 135 条 → `nonzero_effective_budget` 120 条
（N=42 @ 2% 的预算 0.84 < 最小单点防护代价 1，是退化格，单独报告，不混入主分析）。

## 四、Scaling：统一 3600 s 时限（**每方法总量**）下的三方法对比

**每格 9 个实例**，`solved_count` = 该方法**自己关掉了证书**（证明最优性）的实例数。

| N | budget | Exact 中位/最大 (s) | Exact 关证 | CG 中位/最大 (s) | CG 关证 | Greedy 中位/最大 (s) | Greedy 中位 gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 2% | 1.8 / 3.8 | **9/9** | 6.2 / 19.9 | 9/9 | 580 / 986 | 0.000 |
| 200 | 5% | 2.6 / 14.8 | **9/9** | 16.8 / 42.1 | 9/9 | 1 422 / 2 424 | 0.028 |
| 200 | 10% | 5.3 / 218.0 | **9/9** | 38.3 / 132.8 | 9/9 | 2 751 / 3 601 | 0.075 |
| 300 | 2% | 2.7 / 11.4 | **9/9** | 15.9 / 223.2 | 9/9 | 3 263 / 3 603 | 0.027 |
| 300 | 5% | 6.3 / 21.4 | **9/9** | 54.7 / 3 600.0 | 8/9 | 3 600 / 3 602 | 0.101 |
| 300 | 10% | 101.9 / 3 602.8 | 8/9 | 173.9 / 3 600.0 | 7/9 | 3 601 / 3 603 | 0.188 |
| 500 | 2% | 14.1 / 30.8 | **9/9** | 392.6 / 683.5 | 9/9 | 3 602 / 3 611 | 0.103 |
| 500 | 5% | 59.5 / 1 131.9 | **9/9** | 2 271.6 / 3 600.1 | 6/9 | 3 602 / 3 608 | 0.182 |
| 500 | 10% | 1 060.5 / 3 607.9 | 5/9 | 3 600.0 / 3 600.1 | 3/9 | 3 601 / 3 608 | 0.233 |
| 1000 | 2% | 79.8 / 215.6 | **9/9** | 3 603.9 / 3 621.4 | 3/9 | 3 607 / 3 640 | 0.108 |
| 1000 | 5% | 125.8 / 3 636.4 | 8/9 | 3 600.1 / 3 633.0 | 3/9 | 3 607 / 3 666 | 0.205 |
| 1000 | 10% | 3 632.8 / 3 643.0 | 4/9 | 3 600.0 / 3 600.0 | 3/9 | 3 605 / 3 627 | 0.500 |

**结论（如实报告，不美化）**：

* `MPCF-Exact` 在 **N=1000 @ 2% 上仍以中位 79.8 s 关证 9/9**，在 N=500 @ 2%/5% 上也是 9/9。
* 但 **`MPCF-Exact` 在 N=500 @ 10% 只关掉 5/9、N=1000 @ 10% 只关掉 4/9**——
  统一 3600 s 规则下确实存在它无法关证的格。这些格在表中以 `solved_count` 与
  `status_counts = {"optimal": k, "time_limit": 9-k}` 如实呈现，**没有用趋势外推填补**。
* `MPCF-CG` 的 cut-generation 在 N=1000 上反而更早撞限（3/9），
  说明大规模下 master 问题本身变重，不再是小规模时那个「比 Exact 略慢」的形态。
* `MPCF-Greedy` 的 `solved_count` 恒为 0 是**设计使然**（它不声称最优性，
  `certificate_mode = open`），其 `status` 为 `feasible`；真正说明它不可扩展的是
  **中位 relative gap 随 N 与预算单调上升**（N=1000 @ 10% 中位 gap = **0.500**），
  以及 N≥500 后几乎每格都撞上时限。

> **对早前临时数字的更正**：scaling 面板此前只有部分覆盖（N=1000 每格仅 3 个实例），
> 当时读出的「N=1000 Exact 约 10 s 关闭」是小样本偏乐观的读数。
> 补齐全部 9 个实例后，N=1000 @ 10% 的真实中位是 3 632.8 s、关证 4/9。
> **这正是坚持补满而不是外推的原因**——临时数字已作废，以本表为准。

## 五、联合异质性（b_v × m_v）

**实测 Spearman 相关系数**（写进 manifest 与 Figure 7b，不靠名称声称）：

| profile | n | 均值 ρ(b_v, m_v) | 范围 |
|---|---:|---:|---|
| `independent` | 45 | **+0.044** | −0.214 … +0.321 |
| `positive_corr` | 45 | **+1.000** | +1.000 … +1.000 |
| `negative_corr` | 45 | **−0.978** | −0.959 … −1.000 |

**结果**：`MPCF-Exact` 在全部 4 个 profile × 3 个 budget 上 relative gap 恒为 **0.0000**；
`MPCF-Greedy` 的 gap 随预算增大而增大（如 negative_corr @10% 为 0.1267，
positive_corr @2% 仅 0.0064），说明**代价—效能耦合方式确实改变启发式难度，但不改变精确解的最优性**。

8 个显式 conversion variant（`Degree`/`Betweenness` × `Raw`/`PerCost`/`PrefixKnapsack`/`FullKnapsack`）
全部在异质防护代价下实际运行；在 `b_v ≡ 1` 的 homogeneous 对照下四种策略结果相同，
在 `b_v ∈ {1,2,4}` 下出现分离，验证了 conversion 字段的必要性。

## 六、种子扩展稳健性（Reviewer 2：5 个种子够不够）

**设计**：保留原 45 图冻结基准不动（仍是全方法主分析），另加 **5 个预指定种子
（66/77/88/99/110，与主面板的 11/22/33/44/55 无交集）**，只跑 9 个关键方法，
共 45 张新图、1 215 条记录。种子在跑之前就写死在 `rmcd_f.rev.seedext.SEED_EXTENSION_SEEDS`
里，没有任何一个种子是按结果挑的。推断单位始终是 graph，budget 仍在图内先平均。

**(1) 效应估计是否稳定**（原始 5 种子 45 图 vs 合并 10 种子 90 图）：

| method | 5 种子 ḡ | 10 种子 ḡ | Δ | 效应方向保持 |
|---|---:|---:|---:|---:|
| MPCF-Greedy | 0.0309 | **0.0343** | **+0.0034** | ✅ |
| InitialPathCutProtect | 0.0972 | 0.0939 | −0.0032 | ✅ |
| BPDReference-Protect | 0.1274 | 0.1243 | −0.0031 | ✅ |
| PathFrequencyProtect | 0.0950 | 0.0931 | −0.0019 | ✅ |
| OfficialGND-Protect | 0.1118 | 0.1131 | +0.0013 | ✅ |
| NoProtection | 0.1929 | 0.1920 | −0.0009 | ✅ |
| BetweennessProtect | 0.1054 | 0.1051 | −0.0003 | ✅ |
| MPCF-Exact / MPCF-CG | 0 | 0 | 0 | ✅ |

* 全部 8 个对照的 **|Δ| ≤ 0.0034**，相对效应本身（如 Greedy 0.031）是 11% 量级的扰动。
* 合并不但没削弱结论，反而让 **ḡ_Greedy 变大**（基线看起来更差）。

**(2) 区间是否稳定、是否收窄**（对 `MPCF-Exact` 的图级配对效应，10 000 次整簇 bootstrap）：

| method | 效应(5) | 95% CI(5) | 宽度(5) | 效应(10) | 95% CI(10) | 宽度(10) | 收窄 | rb(5) | rb(10) |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| MPCF-Greedy | −0.0309 | [−0.0376, −0.0246] | 0.0130 | −0.0343 | [−0.0393, −0.0294] | **0.0099** | ✅ | −1 | −1 |
| NoProtection | −0.1929 | [−0.2212, −0.1672] | 0.0539 | −0.1920 | [−0.2120, −0.1733] | **0.0387** | ✅ | −1 | −1 |
| BPDReference-Protect | −0.1274 | [−0.1473, −0.1097] | 0.0376 | −0.1243 | [−0.1373, −0.1122] | **0.0250** | ✅ | −1 | −1 |
| BetweennessProtect | −0.1054 | [−0.1229, −0.0888] | 0.0341 | −0.1051 | [−0.1165, −0.0943] | **0.0222** | ✅ | −1 | −1 |
| OfficialGND-Protect | −0.1118 | [−0.1275, −0.0986] | 0.0289 | −0.1131 | [−0.1240, −0.1030] | **0.0210** | ✅ | −1 | −1 |
| PathFrequencyProtect | −0.0950 | [−0.1079, −0.0833] | 0.0246 | −0.0931 | [−0.1014, −0.0855] | **0.0159** | ✅ | −1 | −1 |
| InitialPathCutProtect | −0.0972 | [−0.1071, −0.0886] | 0.0185 | −0.0939 | [−0.1014, −0.0868] | **0.0146** | ✅ | −1 | −1 |
| MPCF-CG | 0 | [0, 0] | 0 | 0 | [0, 0] | 0 | ✅ | 0 | 0 |

* **7/7 个具有非零区间宽度的对照，其 95% 区间在 10 种子下一致收窄**
  （收窄幅度 21%–35%；Greedy 收窄 24%，NoProtection 收窄 28%。
  `MPCF-CG` 与 Exact 数值完全相同，宽度本就为 0，不存在收窄问题）。
* **全部区间都不含 0**，且 `rank-biserial` 在 5 种子与 10 种子下**都是 −1.000**：
  `MPCF-Exact` 在 **90 张图中的每一张**上都不劣于任何对照，**一次例外都没有**。
* 分层 bootstrap（按 topology×N 分 9 层、90 个观测）复核结论不变：
  Greedy CI = [−0.0385, −0.0304]，同样不含 0。

**(3) 估计在哪一步稳定**（Figure S1，枚举**全部** C(10,k) 个子集，不依赖累加顺序）：

| k | 子集数 | 子集均值 | 2.5%–97.5% 带 | 带宽 |
|---:|---:|---:|---|---:|
| 2 | 45 | 0.034330 | [0.028658, 0.040466] | 0.01181 |
| 3 | 120 | 0.034330 | [0.029076, 0.040481] | 0.01141 |
| 4 | 210 | 0.034330 | [0.029994, 0.038957] | 0.00896 |
| 5 | 252 | 0.034330 | [0.030696, 0.037963] | 0.00727 |
| 6 | 210 | 0.034330 | [0.031245, 0.037220] | 0.00598 |
| 8 | 45 | 0.034330 | [0.032795, 0.035748] | 0.00295 |
| 10 | 1 | 0.034330 | [0.034330, 0.034330] | 0 |

* 子集均值对 k **恒等不变**——因为每个种子出现在相同数量的子集中，
  所以「用几个种子」不影响点估计；随 k 变化的只有离散带宽，且**单调收窄至 0**。
  这条曲线是面板的确定性性质，不是某一次抽样的运气。

> **给 Reviewer 2 的一句话结论**：把种子数从 5 加倍到 10 之后，
> 效应方向、显著性、效应方向一致性（90/90 张图 rb = −1）全部不变，
> 点估计漂移 ≤ 0.0034，而所有 95% 区间一致收窄。
> 结论不依赖最初那 5 张图具体是哪 5 张。

## 七、36-variant 注册表对账

```
methods/variant_reconciliation.json
  declared_methods                       45
  certified_reference_methods             2   (MPCF-Exact / MPCF-CG)
  main_panel_variants                    38
  excluded_from_main_panel                7
  comparison_variants                    36
  heterogeneous_cost_conversion_variants   8
  registry_rows_total                    53
```

**实际覆盖：36 / 36 对照 variant 全部具备完整 45 图覆盖，且全部已评价。**
`OfficialFINDER_R-Protect` 已在隔离的 Python 3.7 + TensorFlow 1.14 环境中成功运行
（45 图 × 3 预算，全流程 **43 s**，每实例 6.3–7.4 s），并经 3 个独立进程的重复性门
验证序列 sha256 完全一致（`3618b397ccb24e1b`）。上游默认 `stepRatio=0.01` 未做任何改动。

主实验面板因此从原稿 Table 6 的 22 个方法扩展到 **38 个实际评价的方法**
（3 个 MPCF + 9 个本地基线 + 26 个上游迁移基线，与注册表声明的 36 对照 + 2 参照完全一致）。
7 个 `excluded_from_main_panel` 的方法（含 `OfficialEGND`，协议中本就标注为 conditional）
在 `variant_coverage.csv` 中写明 `exclusion_reason`，**不是静默丢弃**。

## 八、验收门（17 / 17 全部通过）

```
exact_replay                  PASS   worst |objective - replay_kappa| = 9.75e-07
exact_vs_cg                   PASS   两个独立精确求解器零冲突
cg_bounds                     PASS   所有 CG 运行满足 L <= U (max L-U = 5.68e-14)
greedy_feasibility            PASS   所有 Greedy 解满足预算
baseline_budget_feasibility   PASS   所有方法返回预算可行集
no_method_beats_optimum       PASS   无方法超过已认证最优值
greedy_monotonicity           PASS   轨迹级单调性由单元测试保证
graph_dependence              PASS   主统计单位为 graph_id，n=45，记录数 5130
role_dependence               PASS   簇单位为 base_id，配对 n=45
baseline_fairness             PASS   20 979 次运行全部经过同一精确 min-cut 评价
scaling_fairness              PASS   108 个 scaling 格，时限零不一致，方法零缺失
reproducibility               PASS   无缺失 seed/config/commit；19 份 environment
code_revision_consistency     PASS   全部行来自同一代码修订
unified_time_limit            PASS   每方法总量口径，最差超出 0.0 s
seed_extension                PASS   5+5 预指定种子；45 张扩展图 × 9 方法
hardware_uniformity           PASS   每个 experiment 单一 CPU 型号
plotting                      PASS   9 张图全部可由 18 张 CSV 表重建
```

## 九、可复现性验证（一次免费交叉检验）

重构后的主实验与重构前旧代码逐格对比：

* **1 618 / 1 620 行逐位相同**（κ、实际代价、选中节点集合全部一致）
* 仅 2 行不同，且都是 Greedy 在 N=112/10% 撞上 600 s 统一时限的格
  （`status=time_limit`、`runtime_selection_s=600.0`）
* **全部 1 080 行 Exact/CG 结果完全一致** → 重构未改变任何数学结果

同时暴露一个应当在论文中明说的性质：**Exact 与 CG 完全确定性；
Greedy 一旦触及时限，其 incumbent 依赖墙钟**。schema 中的
`status=time_limit` 与 `certificate_mode=open` 正是为此设计的可审计标记。

## 十、数据完整性声明

* 归档中**没有任何一个数字来自趋势外推、插值或占位**。
  未关证的格以 `solved_count` / `status_counts` / `n_instances` 如实标注。
* `budget_abs` 记录的是**实际使用的精确预算**而非四舍五入值；
  `N=42 @ 2% → 0.84 < 1`（最小防护代价）的格由 `zero_budget_audit.csv` 单列，
  同时给出 all-cells 与 effective-cells 两套描述性结果。
* 统计与绘图已与求解器解耦：`rev_plots.py` 只读 CSV，不再调用任何求解器。
* **manifest 更正记录**：`metadata/graph_manifest.csv` 曾虚报 621 个冻结实例，
  其中 45 行是种子扩展第一次尝试用主面板种子（s11–s55）冻结、
  接线 bug 修好后从未被求解的 `seedext__*` 图。
  `merge_archives.py` 现已只保留确实产生过运行的实例（**576**），
  被丢弃的 instance_id 逐个记录在 `merge_summary.json` 的
  `manifest_rows_dropped_no_runs` 与 `dropped_instance_ids` 中。
  该更正**只影响元数据**：`runs_long.csv`、`instance_budget_cells.csv`
  与全部 21 个 statistics/tables 文件在更正前后**逐字节相同**，
  17 项验收门依旧全绿 —— 没有任何一个已发表数字因此改变。
* 已发布归档可由 `revision/server_backup/` 单独重建并逐字节复现
  （见 `SERVER_BACKUP_ZH.md`）。
