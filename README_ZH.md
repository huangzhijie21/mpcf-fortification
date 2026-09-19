# MPCF —— 有限预算下的任务路径加固

[English README](README.md)

本仓库是论文方法的参考实现。给定一个有向装备网络，攻击者按每个节点的
攻击代价移除节点；防御者在**有限预算** `B` 下先加固若干节点，把节点 `v`
的攻击代价从 `a_v` 抬升到 `δ_v = m·a_v`。防御者要最大化的量是

```
κ(P) = 在冻结加固集 P 上，自适应的、按代价加权的最优路径割的最小代价
```

它在实现中被精确地算成一个**加权节点分裂最小割**。优化问题是

```
max  κ(P)
s.t. Σ_{v∈P} b_v ≤ B        （b_v 为节点 v 的加固代价）
     P ⊆ V_R                （只能加固可移除节点）
```

> **代码修订**：`core-a8deebd9ddef8772f9768affdf9ee483d1e48193`。
> 本仓库是该修订的干净子集，去掉了 1.7 GB 历史输出、第三方上游代码树
> 与服务器备份。已发表归档的全部数字均由这份代码产出。

---

## 一、三个求解器

**它们不是三个同等地位的贡献** —— 两个是精确求解器，一个是消融方法：

| 求解器 | 方法 | 最优性证书 |
|---|---|---|
| `MPCF-Exact` | 紧凑 MILP（SciPy 自带的 HiGHS） | `solver_closed`，已证最优 |
| `MPCF-CG` | 割生成主问题 + 精确 PathCut 分离 oracle | `cg_closed`（当 `L = U`） |
| `MPCF-Greedy` | 一步精确边际消融 | **无**，`certificate_mode = open` |

`MPCF-Greedy` **刻意不声称最优性**。它的存在是为了量化"精确性值多少钱"，
论文报告的是它与已认证最优值之间的间隔（gap）。它不是一个被提出的方法。

每次运行都会被独立复核：把冻结的加固集重新做一次最大流/最小割，
与求解器自己报的目标值比对（`replay_kappa`）。一个成功但缺乏该独立复核的
运行会让 `baseline_fairness` 门直接失败。

---

## 二、目录结构

```
src/rmcd_f/                 包本体
  task_path_fortification.py   三个求解器、SolverLog、证书
  operational_motif.py         路径闭包系统构造、节点分裂
  model.py, flow.py            网络模型与最大流/最小割
  synthetic.py                 4:3:3:4 拓扑生成器
  cost_profiles.py             攻击代价与加固代价 profile
  protection_baselines.py      本地基线（度、介数…）
  official_review.py           上游基线适配器
  cli.py                       图读写与 rmcd-f 命令
  rev/                         返修实验层（见下）
scripts/
  rev_run.py                   跑一个面板，写出 results/runs_long.csv
  rev_statistics.py            原始 CSV 进 → 统计 CSV 出（不 import 求解器）
  rev_plots.py                 统计 CSV 进 → PDF/PNG 出（不 import 求解器）
  rev_gates.py                 验收门；任一失败则退出码非 0
  run_official_review_matrix.py / _sequences.py   上游基线（需第三方树）
tests/                       88 个测试；在仓库根目录跑 pytest -q
revision/
  scripts/run_all_local.sh     在本机跑完所有面板（从这里开始）
  tools/                       合并与清单工具
  scripts/                     shell 驱动脚本
  docs/                        修改说明、交付摘要、图表清单
docs/                        算法规范与协议文档
```

`src/rmcd_f/rev/` **不属于数学部分**，它是返修时新增的实验组织层，
数值核心没有被它改动：`ids.py` 定义实例标识，`schema.py` 定义 36 列契约，
`panels.py` 做确定性面板冻结，`stats.py` 做图级聚类统计，
`robustness.py` 做种子扩展稳定性分析。

---

## 三、安装

需要 **Python ≥ 3.10**。

```bash
git clone <本仓库> mpcf-fortification
cd mpcf-fortification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[test,experiments]"
```

这会装上 `networkx`、`numpy`、`scipy`（MILP 后端是 SciPy 内置的 HiGHS），
以及 `pytest` 与 `matplotlib`。验证：

```bash
pytest -q          # 期望：88 passed
rmcd-f --help
```

> **必须在仓库根目录运行 `pytest`**：部分测试会 import `scripts/` 包，
> 它按工作目录解析。

### 可选的：上游基线

36 个对照 variant 里有 **26 个是第三方实现的适配器**
（GND、GDM、CoreHD、CI、EI、FINDER、MinSum、network/vertex entanglement 等）。
那部分上游代码**不在本仓库内** —— 它是另一个项目，有自己的许可证。
要复现这些行，需要你自己获取并指路：

```bash
git clone <上游 network_dismantling 仓库> external/review-main
python scripts/official_source_preflight.py --review-path external/review-main
```

其中两个家族还需要各自隔离的环境（依赖 TensorFlow 1.14 / Python 3.7）：

```bash
bash scripts/setup_finder_environment.sh        # FINDER（Python 3.7 + TF 1.14）
python scripts/gdm_environment_preflight.py     # GDM（torch + torch_scatter）
```

没有这些环境时，注册表会把这些 variant 标为"未评价"，覆盖表会**明说**，
不会有任何静默跳过。

---

## 四、快速上手

```bash
# 1. 冻结一个小面板并求解
python scripts/rev_run.py --experiment main --output results \
    --workers 8 --time-limit 600 --limit-instances 2

# 2. 统计（只读原始 runs，不 import 任何求解器）
python scripts/rev_statistics.py --output results --resamples 10000

# 3. 出图
python scripts/rev_plots.py --output results

# 4. 验收门（任一失败则退出码非 0）
python scripts/rev_gates.py --output results
```

`--limit-instances 2` 是冒烟测试开关；正式跑请去掉。

---

## 五、复现已发表实验

**全部实验都在一台机器上跑。** 每个面板由本地进程池求解
（`rev_run.py --workers N`），之后四个分析阶段只读 CSV。
已发表归档当初跑在三台容器上，只是因为当时手头是那套硬件 ——
本仓库**没有任何一处**需要集群、调度器或远程主机。

一条命令跑完：

```bash
# 先冒烟：每个面板 1 张图，但所有方法仍会跑一遍
# （4 核约 8 分钟；此模式下跳过 scaling 面板）
bash revision/scripts/run_all_local.sh --smoke

# 再跑正式的（自动识别核数，可用 WORKERS=16 覆盖）
bash revision/scripts/run_all_local.sh
```

`--smoke` 把每方法时限压到 20 s 并跳过 scaling 面板。原因是**减少图数并不减少方法数**：
剩下那 1 张图仍要跑全部 38 个方法，600 s 时限下单个面板的最坏情况是 38 × 600 s。
想让 scaling 也进冒烟，加 `--with-scaling`。

下表是全部面板的实测代价（对归档中每次求解调用的运行时间求和）：

| 面板 | 核心小时 | 8 核 | 16 核 | 32 核 |
|---|---:|---:|---:|---:|
| `main` | 5.3 | 0.7 h | 0.3 h | 0.2 h |
| `role` | 14.3 | 1.8 h | 0.9 h | 0.4 h |
| `heterogeneity` | 23.5 | 2.9 h | 1.5 h | 0.7 h |
| `seedext` | 5.4 | 0.7 h | 0.3 h | 0.2 h |
| `scaling` | 140.2 | 17.5 h | 8.8 h | 4.4 h |
| **合计** | **188.7** | **23.6 h** | **11.8 h** | **5.9 h** |

`scaling` 面板占了大头：`N=1000` 配 3600 s/方法的时限本来就贵。
加 `--skip-scaling` 可在 8 核上约 4 核心小时内跑完全部其余面板。
以上数字来自服务器级 x86 CPU（Intel Xeon 8352V / AMD EPYC 9654），
笔记本会更慢，所以**先用 `--smoke` 试一次**。

### 逐面板手动运行

同一面板内必须给所有方法**相同的时限**。

```bash
# 主面板：45 图 × 3 预算 × 38 方法
python scripts/rev_run.py --experiment main --output results \
    --workers 8 --time-limit 600

# role 组成面板：45 张 base 图 × 6 种组成
python scripts/rev_run.py --experiment role --output results \
    --workers 8 --time-limit 600

# 联合异质性面板：b_v × m_v 耦合 profile
python scripts/rev_run.py --experiment heterogeneity --output results \
    --workers 8 --time-limit 600 --include-conversions

# 种子扩展面板：5 个预指定新种子 × 9 个关键方法
python scripts/rev_run.py --experiment seedext --output results \
    --workers 8 --time-limit 600

# scaling 面板：N ∈ {200,300,500,1000}，每格 9 个实例
python scripts/rev_run.py --experiment scaling --output results \
    --workers 8 --time-limit 3600

# 把所有 shard 合并成 results/runs_long.csv
python scripts/rev_run.py --experiment all --output results --collect-only

# 分析
python scripts/rev_statistics.py --output results --resamples 10000
python scripts/rev_plots.py --output results
python scripts/rev_gates.py --output results
```

实例由 `(topology, N, seed)` **确定性冻结**，且重复冻结是幂等的：
完全相同的冻结会复用已有文件，出现分歧则拒绝写入。
这正是面板能跨机器复现的原因。

### 统一时限的口径

`--time-limit` 是**每方法总量**，不是"每次内部求解"。割生成求解器在一次
"solve" 内部可能跑数百轮分离（实测最多 326 轮）；若按单次内部求解计时，
CG 能在相同时钟内做几十倍于 Exact 的工作量，比较就失去意义。
现在三个方法共享同一个总预算，且 `time_limit_s` 写在每一行上，规则可审计。
一个不可分割的分离 oracle 调用（约 40–50 s）在 3600 s 上最多造成约 1.4% 超出；
门的容差是 5% + 5 s。

---

## 六、输出契约

所有面板、所有算法都写进**同一个** `results/runs_long.csv`，固定 36 列。
方法不产生的字段写字面量 `NA`，**绝不删列**，因此一个读取器能吃下全部实验：

```
experiment  instance_id  graph_id  base_id  topology  N  num_edges  seed
composition  budget_ratio  budget_abs  method  variant
kappa  kappa_opt  relative_gap  actual_cost  selected_count  selected_nodes
runtime_selection_s  runtime_evaluation_s  runtime_total_s  status
incumbent  best_bound  final_gap  bb_nodes  replay_kappa  certificate_mode
cg_L  cg_U  cg_iterations  cg_cuts  time_limit_s
config_hash  code_commit
```

其中 `relative_gap = (kappa_opt − kappa) / kappa_opt`，`kappa_opt` 是该
`(instance, budget)` 格的**跨求解器认证最优值**。两个精确求解器必须一致，
不一致会被标为 `certificate_conflict` 并让 `exact_vs_cg` 门失败。

### `budget_abs` 是精确值，不是取整值

主面板 `b_v ≡ 1`，所以 `budget_abs = ratio × N`，`N = 42` 在 `2%` 下等于
**0.84** —— 低于最便宜的单点加固代价，是一个**零有效预算**退化格。
`instance_budget_cells.csv` 同时记录 `budget_abs` 与 `budget_abs_floor`，
`statistics/zero_budget_audit.csv` 分别报告 all-cells 与有效预算子集。
取整会把这个退化**藏起来**。

### 推断单位

| 标识 | 定义 | 角色 |
|---|---|---|
| `graph_id` | `topology + N + seed` | 主面板的**独立单位** |
| `budget_ratio` | 0.02 / 0.05 / 0.10 | 图**内**的重复测量 |
| `base_id` | `topology + N + seed` | role 组成面板的重复测量簇 |

因此统计先在每个 `(graph, method)` 内把三个预算**求平均**，再做
45 个图 block 的 Friedman、45 个图级配对的 Wilcoxon、Holm 校正，
bootstrap 抽的是**整张图**（10 000 次重采样，图内所有预算一起进出）。
任何地方都不会把 135 个 graph-budget 对或 810 条 role 记录当独立样本。

### 代码修订标识

`code_commit` 在 git 检出时记录 git commit，否则对 `NUMERICAL_CORE` 的 12 个文件
做内容摘要；摘要**先归一化行尾**，所以同一份源码在任何平台上哈希一致。

| 取值 | 规则 | 出现位置 |
|---|---|---|
| `core-a8deebd9ddef8772f9768affdf9ee483d1e48193` | 逐字节，部署时的原样 | 已发表归档的每一行 |
| `core-f018acb54d16e1cfd42e539d86562dcff1577365` | 归一化行尾 | 本仓库在非 git 上下文计算得到 |

两者描述的是同一份代码：部署时的树恰好是混行尾
（`task_path_fortification.py` 为 CRLF、`rev/panels.py` 为 LF），这是唯一差异来源。
已用归一化规则分别对**部署树、本发布树、全新 clone** 求哈希，三者完全一致。
因此 clone 得到的是 git SHA，非检出副本得到 `core-f018acb5…`；
两种情况 `code_revision_consistency` 门都成立，因为一次运行内所有行取值相同。

---

## 七、可选：把一次运行拆到多台机器

**复现任何东西都不需要这一节。** 面板本身是同一份作业，跑在一台还是多台上
没有区别；请用第五节的单机路径。这里写出来只是因为已发表归档当初是那样跑的
（三台 32 核容器，每个面板固定分配到一台机器）。如果你有集群、想用同样布局，
`revision/tools/` 与 `revision/scripts/` 就是那套工具：

```bash
cp revision/scripts/servers.env.example revision/scripts/servers.env
# 编辑后：
source revision/scripts/servers.env

python revision/tools/remote_exec.py exec --command "uptime"   # 连通性测试
python revision/tools/finalize.py --poll-seconds 300 --max-hours 7
```

`finalize.py` 会等待所有面板、强制最后一次 `--collect-only`、下载各机结果、
做重复与指纹校验后合并，然后依次跑统计、出图、验收门。
如果你手上已经有各机的结果目录，只用 `merge_archives.py` 合并即可。

**如果确实要分布式，同一个面板绝不能混用不同 CPU 型号。** 运行时间本身就是要
报告的结果，`hardware_uniformity` 门会逐实验记录 CPU 型号，跨型号即失败。
已发表的分配是：主面板 + 异质性 + 种子扩展 + 上游对照在 Intel Xeon 8352V，
role 与 scaling 在两台 AMD EPYC 9654 上。实测单核吞吐：
Xeon 3294 /s，EPYC 5668–5834 /s，**Xeon 约慢 1.7 倍** —— 这正是那道门存在的理由。

**凭据与主机名从不写入本仓库。** 服务器表从环境变量读取：
`MPCF_SRV<n>_HOST`、`MPCF_SRV<n>_PORT`、`SSH_PASSWORD_SRV<n>`。

---

## 八、验收门

`scripts/rev_gates.py` 写出 `gates/acceptance_gates.csv`，
任一失败即退出码非 0。已发表归档 17 项全过：

```
exact_replay                目标值与独立最小割复核一致
exact_vs_cg                 两个独立精确求解器处处一致
cg_bounds                   每次割生成的 L <= U
greedy_feasibility          每个 Greedy 解都满足预算
baseline_budget_feasibility 所有方法都返回预算可行集
no_method_beats_optimum     没有方法超过已认证最优值
greedy_monotonicity         轨迹级单调性（有单元测试）
graph_dependence            主单位为 graph_id，预算在图内平均
role_dependence             role 面板以 base_id 为簇
baseline_fairness           每个成功运行都带独立复核
scaling_fairness            scaling 每格时限与方法集唯一
reproducibility             无行缺失 seed / config / revision
code_revision_consistency   所有行来自同一代码修订
unified_time_limit          按"每方法总量"口径遵守时限
seed_extension              扩展种子预指定且覆盖完整
hardware_uniformity         每个实验单一 CPU 型号
plotting                    每张图都能从导出的 CSV 表重建
```

两条扩展时要知道的性质：

* **`MPCF-Exact` 与 `MPCF-CG` 完全确定性。** 用冻结修订重跑，
  1 080/1 080 行 Exact 与 CG 结果逐位相同。
* **`MPCF-Greedy` 一旦触及时限就不再确定** —— 其 incumbent 依赖墙钟。
  这类行会被标为 `status = time_limit`、`certificate_mode = open`。
  不要把两次触及上限的 Greedy 运行当成复现失败；那个标记本身就是信号。

---

## 九、统计与绘图与求解器彻底解耦

`rev_statistics.py` 与 `rev_plots.py` **不 import 任何求解器**，
只读 `runs_long.csv` 与导出的 CSV 表。因此可以在没有求解器环境的笔记本上
重画某张图或重跑某个检验，审稿人也可以只用已发表的 CSV 复算每一个数字。

含随机的分析（bootstrap、种子扩展）都接受 `--seed`，可复现。

---

## 十、文档索引

| 文档 | 内容 |
|---|---|
| `docs/MPCF_ALGORITHM_SPEC.md` | 算法完整规范 |
| `docs/THEORY_AND_PROOFS.md` | 证书与证明 |
| `docs/OFFICIAL_DISMANTLING_BASELINES.md` | 每个上游基线及其适配器 |
| `docs/MPCF_BASELINE_PROTOCOL.md` | 基线公平性规则 |
| `revision/docs/README.md` | 返修层逐条改了什么（英文） |
| `revision/docs/CHANGES_ZH.md` | 同上，中文，带代码位置 |
| `revision/docs/DELIVERY_SUMMARY_ZH.md` | 已发表数字汇总 |
| `revision/docs/FIGURE_TABLE_INVENTORY_ZH.md` | 可直接写 caption 的图表清单 |

---

## 十一、引用与许可

引用信息见 `CITATION.cff`。

**本仓库目前没有 LICENSE 文件**，需要作者自行选择；在此之前适用默认规则，
即**保留所有权利**。若希望他人可复用，研究代码通常选 MIT 或 Apache-2.0。
