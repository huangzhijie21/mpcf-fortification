# NetworkDismantling/review 官方基线接入说明

## 1. 比较口径

本项目调用
[`NetworkDismantling/review`](https://github.com/NetworkDismantling/review)
中的官方实现生成结构瓦解序列，但不采用其 Largest Connected Component
(`LCC`) 作为本章性能终点。所有序列都映射回原始物理装备节点，并只取第一个
经本项目独立验证满足

```text
Omega_R(G-D) < K
```

的前缀。因而所有方法比较的是同一角色模体容量边界下的集合代价、集合规模和
运行时间。

官方实现接收的是简单无向图，所以输入是原有向装备网的简单无向物理投影：

- 节点仅包括物理装备节点；
- 保留节点身份，去除边方向、平行边和自环；
- 不把 task、requirement 或虚拟节点加入投影；
- 不向官方算法泄露 `Omega_R`、角色标签或实验结果；
- 官方排序结束后才在原有向角色网中逐前缀复核 `Omega_R<K`。

因此这些方法是严格的**结构瓦解迁移基线**，不是 RMCD 目标的同构求解器。

## 2. 已接入方法

共注册 32 个官方序列方法/变体和 1 个基于官方 Brute Force 输出的派生诊断。

| 类别 | 方法 |
|---|---|
| Exhaustive LCC 派生诊断 | `ReviewBruteForceFrequencyRank` |
| CI | `OfficialCI_L1`, `OfficialCI_L2`, `OfficialCI_L3` |
| Core/GND | `OfficialCoreHD`, `OfficialGND`, `OfficialGNDR` |
| EI | `OfficialEI_S1`, `OfficialEI_S2` |
| Min-Sum | `OfficialMinSum`, `OfficialMinSumR` |
| Network Entanglement | Small/Mid/Large 及各自 reinsertion，共 6 项 |
| Vertex Entanglement | 原方法及 reinsertion，共 2 项 |
| GDM | `OfficialGDM`, `OfficialGDMR` |
| CoreGDM | `OfficialCoreGDM`，保留上游 `almost ready` 实验性标记 |
| EGND | `OfficialEGND` |
| FINDER | `OfficialFINDER_R`（上游公共函数实际启用 reinsertion） |
| 官方节点启发式 | Degree、Betweenness、Eigenvector、PageRank 的 static/dynamic 版本，以及 RandomStatic |

`ReviewBruteForceFrequencyRank` 是特殊派生诊断项。上游函数输出“最优 LCC
瓦解集合中的节点出现频率”，而不是移除序列。本适配层按频率降序、静态节点
编号升序形成确定性排序；该排序不是上游返回的某个精确瓦解集合，因此不使用
`Official` 前缀，也不进入“官方序列”性能声明。由于上游枚举节点组合，默认
只允许 `n<=18`，仅用于小规模补充诊断。

`OfficialCoreGDM` 虽可尝试运行，但官方仓库明确标注为 “almost ready,
needs some fixes”。结果表必须保留 `experimental_upstream=1`，不能与稳定方法
混为一谈。

官方启发式的评分函数保持上游实现不变。适配器 `1.4` 仅修复序列执行与
复现性记录层：

- static 方法在初始活动图上调用一次官方评分函数；
- dynamic 方法在每次删除后的活动节点 `GraphView` 上重算官方评分；
- 已删除节点不再作为零度孤立点参与下一次 `argmax`；
- LCC 停止条件仍按上游定义逐步复核；
- 状态文件记录 `heuristic_active_graph_lcc_adapter`，不得把该兼容处理隐去。
- 源码树身份排除编译产物、profiling 文件、EGND 运行配置和 FINDER 的
  Cython 派生文件，同时保留 `FINDER_ND/src/lib` 原始源码；
- 隔离启动器为成功、超时、信号退出和普通失败统一写入
  `isolated_method_status.csv`。

该处理修复上游 dynamic generator 重复返回同一静态节点，以及 external
dismantler 将 `network_name` 误传给无此参数的 sorter 的问题；它不融合角色、
模体容量或 RMCD 信息。

因此，static 状态行使用
`upstream_static_sorter_python_active_lcc_order`，dynamic 状态行使用
`upstream_sorter_active_graph_dynamic_order`。后者是“上游评分函数 +
活动图执行适配”的序列，不表述为上游代码逐行原生序列。状态行同时记录并列
规则、活动节点过滤和 `uses_upstream_function_defaults=0`。

## 3. 为什么分两阶段

官方仓库明确要求多套依赖环境：通用 `dismantling` 环境、GDM 环境和旧版
FINDER 环境不能可靠合并。为避免环境切换改变图实例，本项目采用：

1. RMCD 环境生成实例与内容指纹；
2. 各官方环境分别读取同一实例，导出节点序列 CSV；
3. RMCD 环境按图指纹和方法名严格合并；
4. 所有方法在原角色装备网上复核共同阈值。

任何缺失序列、失败状态、冲突序列或图指纹不一致都会使正式实验在启动前
退出，不允许静默删除失败方法。

每个序列还记录实际源码的 Git commit 或稳定源码树 SHA-256、适配器版本、
兼容性补丁清单、有效重插入状态、随机性状态和参数 JSON。稳定源码树身份只
覆盖可执行算法源码，不吸收 `gmon.out`、编译目录、二进制和 EGND 的
`config.h` 等运行状态。`--resume` 只有在实例清单哈希、源码身份、停止条件、
适配器版本和算法种子全部一致时才复用结果。

## 4. 服务器运行

以下命令均在 RMCD 项目根目录执行。完整服务器源码包已包含审计后的
`external/review-main`，以及从 FINDER 原作者仓库补齐的
`FINDER_ND/src/lib`；不再需要另行上传 `review-main.zip` 或
`FINDER-master.zip`。Python wheel 本身不内嵌这些第三方源码。

解压后先验证第三方源码清单：

```bash
python scripts/official_source_preflight.py
```

该检查只验证源码、预训练模型和来源文件是否完整，不要求当前环境已经安装
graph-tool、TensorFlow 或 PyTorch。

### 4.1 生成冻结实例

```bash
conda activate rmcd-f-py310

python scripts/run_critical_set_study.py \
  --study-id official_smoke_n12 \
  --topologies centralized \
  --seeds 11 \
  --role-sizes 3,3,3,3 \
  --interface-density 0.30 \
  --capacity-levels 1,2,3 \
  --remaining-fractions 0.25,0.50,0.75 \
  --methods rmcd-pcd,rmcd-exact,official-all \
  --prepare-only \
  --workers 1 \
  --output outputs/official_smoke_n12
```

成功标志为：

```bash
test -f outputs/official_smoke_n12/_PREPARED
```

### 4.2 通用官方环境

按官方 README 创建并激活含 `graph-tool`、Boost、编译器的环境，再安装当前
RMCD 源码供适配脚本读取实例：

```bash
conda activate dismantling
python -m pip install --no-deps -e .

python scripts/run_official_review_sequences.py \
  --instance-manifest outputs/official_smoke_n12/instance_manifest.csv \
  --methods core,heuristic,OfficialEGND \
  --review-path external/review-main \
  --stop-condition 1 \
  --output outputs/official_smoke_n12/seq_generic
```

正式批量建议用独立进程封装：

```bash
INSTANCE_MANIFEST=outputs/official_smoke_n12/instance_manifest.csv \
METHODS=core,heuristic,OfficialEGND \
REVIEW_PATH=external/review-main \
OUTPUT_ROOT=outputs/official_smoke_n12/seq_generic_isolated \
LOG_ROOT=logs/official_smoke_n12_generic \
METHOD_TIMEOUT_SECONDS=3600 \
bash scripts/run_official_review_isolated.sh
```

隔离启动器无论单个方法成功还是失败，都会在
`$OUTPUT_ROOT/isolated_method_status.csv` 中记录退出码、开始/结束时间、硬
超时、日志和输出目录。退出码 `124` 明确记为 `TIMEOUT`，`128+signal`
记为 `SIGNAL`；因此被硬超时杀死、来不及生成
`official_method_status.csv` 的方法仍有机器可读审计记录。

小规模 Brute Force 单独运行：

```bash
python scripts/run_official_review_sequences.py \
  --instance-manifest outputs/official_smoke_n12/instance_manifest.csv \
  --methods ReviewBruteForceFrequencyRank \
  --brute-force-max-n 18 \
  --review-path external/review-main \
  --stop-condition 1 \
  --output outputs/official_smoke_n12/seq_bruteforce
```

### 4.3 GDM 环境

先激活候选 GDM 环境，并在该环境中编译公共 dismantler 扩展。`make clean`
用于避免把其他 Conda 环境留下的 ABI 不兼容 `.so` 误当成当前环境产物：

```bash
conda activate gdm
make -C external/review-main/network_dismantling/common/external_dismantlers clean
make -C external/review-main/network_dismantling/common/external_dismantlers
```

随后运行只读预检。该脚本在隔离子进程中验证
`graph-tool`、PyTorch、PyTorch Geometric、`torch_sparse` 的实际导入，
在 CPU 和实际启用的 CUDA 设备上执行最小 `SparseTensor` 与 GAT 前向传播，
把官方权重装载到对应 GAT 结构后再次前向计算，并核对
GDM/GDMR/CoreGDM 入口。它通过基础 `ldd` 严格确认公共扩展没有缺失共享库，
并且 Python 与 Boost.Python 均解析到当前环境。扩展实际导入成功后，
`ldd -r` 报告的 Python/Boost 延迟解析符号只作为诊断记录；这是因为
`ldd -r` 在 Python 解释器之外检查扩展时不能完整复现解释器的全局符号环境。
任何 `not found` 共享库、错误环境链接或实际导入失败仍然是硬失败。
脚本不会安装依赖、编译扩展或改写官方源码：

预检分别核对入口暴露的 `models_newpg` 模型树根目录，以及运行时按
`synth_train_NEW/t_0.18/GAT_Model` 解析的 checkpoint 叶目录；二者不是同一
路径层级。

```bash
/root/autodl-tmp/envs/gdm/bin/python \
  scripts/gdm_environment_preflight.py \
  --review-path external/review-main \
  --json-output outputs/gdm_environment_preflight.json
```

预检 `PASS` 后再执行序列导出。若实际运行前扩展被清理，
GDM/GDMR/CoreGDM 均会在目标 GDM 环境中按需重建；不能复用另一个
Python/Conda 环境中来源不明的扩展文件作为依赖通过证据。
适配器 `1.4` 还会在导出成功前回放每条节点序列，只有最终结构最大连通
分量达到声明的 `stop_condition` 才写入 `SUCCESS`；验证值记录在
`verified_final_lcc_size`。

```bash
conda activate gdm
python -m pip install --no-deps -e .

python scripts/run_official_review_sequences.py \
  --instance-manifest outputs/official_smoke_n12/instance_manifest.csv \
  --methods OfficialGDM,OfficialGDMR,OfficialCoreGDM \
  --review-path external/review-main \
  --stop-condition 1 \
  --output outputs/official_smoke_n12/seq_gdm
```

### 4.4 FINDER 环境

FINDER 使用原作者发布时的 Python 3.7 / TensorFlow 1.14 依赖栈，并与
`dismantling`、`rmcd-f-py310` 环境隔离。首次运行先执行：

```bash
bash scripts/setup_finder_environment.sh

/root/autodl-tmp/envs/finder/bin/python \
  scripts/finder_environment_preflight.py \
  --review-path external/review-main \
  --build \
  --load-checkpoint \
  --json-output outputs/finder_environment_preflight.json
```

安装脚本默认把 Conda 包缓存、临时编译文件和环境本体放在
`/root/autodl-tmp`，不占用 30 GB 系统盘。预检只有在固定版本、全部 Cython
扩展以及官方 checkpoint 恢复均通过后才返回成功。

随后用该环境的绝对 Python 路径导出序列：

```bash
export CUDA_VISIBLE_DEVICES=-1

/root/autodl-tmp/envs/finder/bin/python scripts/export_finder_legacy.py \
  --instance-manifest outputs/official_smoke_n12/instance_manifest.csv \
  --review-path external/review-main \
  --stop-condition 1 \
  --output outputs/official_smoke_n12/seq_finder
```

单次成功仅证明环境和推理路径成立。在进入统一阈值评价前，应在同一冻结实例上
执行进程间重复性门槛：

```bash
INSTANCE_MANIFEST=outputs/official_smoke_n12_connected/instance_manifest.csv \
OUTPUT=outputs/official_smoke_n12_connected/finder_repeatability_v1 \
REPEATS=3 \
bash scripts/run_finder_repeatability_smoke.sh
```

该审计每次启动独立 Python 3.7 进程，并逐实例比较完整节点序列、公共前缀、
源码身份和适配器版本。只有全部完整序列及元数据一致时才写 `_SUCCESS`；任何
失败、超时、覆盖缺失或排序冲突均写 `_FAILED`。输出中的 `frozen_inputs/`
同时保存清单和实例 JSON，保证重复性结果可独立核验。

完整服务器源码包已经从 FINDER 原作者仓库的
`code/FINDER_ND/src/lib` 补齐全部 24 个 C++/头文件，并保留 review 的统一
接口、预训练模型和上游许可。来源与输入归档哈希见
`FINDER_SOURCE_PROVENANCE.md`。适配层仍逐文件检查完整性，不能用近似实现
替代后标为 `OfficialFINDER_R`。该命名中的 `R` 是必要的，因为上游
`FINDER_ND()` 虽把元数据写成 `includes_reinsertion=False`，实际却调用
`_finder_nd(..., reinsertion=True)`。

`export_finder_legacy.py` 不导入 `rmcd_f` 或整个
`network_dismantling` 包，而是直接调用上游 `FINDER` 类、官方 checkpoint、
`EvaluateRealData` 和 `EvaluateSol`。这绕过了 review 外层包装器附加但 FINDER
本体不需要的 `graph-tool` 依赖；NetworkX 投影、模型参数和排序逻辑均采用
上游实现。适配器只消费冻结实例 JSON 和指纹清单，并输出与主实验兼容的中立
CSV，兼容性记录为
`direct_networkx_export_without_graph_tool_wrapper`。为保持原作者的独立扩展目录
构建语义，Cython 扩展在不含 review 所加 `__init__.py` 的临时 staging 目录
中由原 `setup.py` 编译，随后仅回拷当前 Python ABI 的 `.so`；该构建兼容层记为
`standalone_upstream_cython_build_staging`，不修改模型、权重与排序逻辑。

### 4.5 共同阈值评价

所有导出目录都应有 `_SUCCESS`。随后回到 RMCD 环境：

```bash
conda activate rmcd-f-py310

SEQ_FILES=$(
  find outputs/official_smoke_n12 \
    -mindepth 2 -type f -name official_sequences.csv \
    -print | paste -sd, -
)

python scripts/run_critical_set_study.py \
  --study-id official_smoke_n12 \
  --topologies centralized \
  --seeds 11 \
  --role-sizes 3,3,3,3 \
  --interface-density 0.30 \
  --capacity-levels 1,2,3 \
  --remaining-fractions 0.25,0.50,0.75 \
  --methods rmcd-pcd,rmcd-exact,official-all \
  --official-sequence-files "$SEQ_FILES" \
  --workers 1 \
  --runtime-context controlled-single-worker \
  --output outputs/official_smoke_n12
```

只有 `_SUCCESS` 表示共同阈值评价完成。

### 4.6 依赖分组 smoke 的统一启动器

Brute Force、GDM 和 FINDER 必须在各自依赖环境中运行，但应消费同一份冻结
实例清单。控制流程在 RMCD Python 3.10 环境中启动，并通过目标环境的绝对
Python 路径串行执行每个方法：

```bash
CONTROL_PYTHON=/root/autodl-tmp/envs/rmcd-f-py310/bin/python \
DISMANTLING_PYTHON=/root/autodl-tmp/envs/dismantling/bin/python \
GDM_PYTHON=/root/autodl-tmp/envs/gdm/bin/python \
FINDER_PYTHON=/root/autodl-tmp/envs/finder/bin/python \
INSTANCE_MANIFEST=outputs/official_smoke_n12_connected/instance_manifest.csv \
REVIEW_PATH=external/review-main \
OUTPUT_ROOT=outputs/official_smoke_n12_connected/seq_dependency_v14 \
LOG_ROOT=logs/official_smoke_n12_connected_dependency_v14 \
bash scripts/run_official_dependency_smoke.sh
```

启动器按以下固定顺序执行：

1. `ReviewBruteForceFrequencyRank`；
2. `OfficialGDM`、`OfficialGDMR`、实验性 `OfficialCoreGDM`；
3. `OfficialFINDER_R`。

方法不会共享 Python 进程，也不会并发改写官方源码目录。统一台账写入
`$OUTPUT_ROOT/isolated_method_status.csv`；任一方法失败或超时，批次只写
`_INCOMPLETE`，成功方法的结果仍保留。若某个依赖环境尚未建立，可以将对应
的 `RUN_GDM` 或 `RUN_FINDER` 设为 `0`，先完成其余门槛。例如仅验证小规模
穷举诊断：

```bash
RUN_GDM=0 RUN_FINDER=0 \
CONTROL_PYTHON=/root/autodl-tmp/envs/rmcd-f-py310/bin/python \
DISMANTLING_PYTHON=/root/autodl-tmp/envs/dismantling/bin/python \
INSTANCE_MANIFEST=outputs/official_smoke_n12_connected/instance_manifest.csv \
REVIEW_PATH=external/review-main \
OUTPUT_ROOT=outputs/official_smoke_n12_connected/seq_bruteforce_v13 \
LOG_ROOT=logs/official_smoke_n12_connected_bruteforce_v13 \
bash scripts/run_official_dependency_smoke.sh
```

FINDER 的旧环境导出适配器版本为 `legacy-finder-direct-1.3`。其源码身份过滤规则与
主适配器 v1.4 对齐，排除编译目录、Cython 派生 C/C++ 文件、扩展模块和
运行时配置，同时保留
`FINDER_ND/src/lib` 中的真实上游源文件。

### 4.7 全部经典与学习基线的统一实验入口

完整方法不在同一个 Conda 环境中执行，但必须消费同一份冻结实例并在同一
`Omega_R < K` 下评价。统一入口如下：

```bash
CONTROL_PYTHON=/root/autodl-tmp/envs/rmcd-f-py310/bin/python \
DISMANTLING_PYTHON=/root/autodl-tmp/envs/dismantling/bin/python \
GDM_PYTHON=/root/autodl-tmp/envs/dismantling/bin/python \
FINDER_PYTHON=/root/autodl-tmp/envs/finder/bin/python \
STUDY_ID=official_unified_n12_v1 \
TOPOLOGIES=centralized,modular,distributed \
SEEDS=11,22,33 \
ROLE_SIZES=3,3,3,3 \
INTERFACE_DENSITY=0.30 \
EVAL_WORKERS=12 \
bash scripts/run_unified_baseline_validation.sh
```

该入口依次完成：

1. 生成未按结果筛选的冻结实例；
2. 在通用环境隔离运行全部经典、中心性和 EGND 方法；
3. 在独立依赖环境运行 Brute Force、GDM/GDMR/CoreGDM 和 FINDER；
4. 由 `audit_official_sequence_coverage.py` 核对每个
   `graph_fingerprint × method`，拒绝节点序列冲突及源码/适配器冲突；
5. 只把覆盖全部冻结实例的方法合并为
   `official_sequences_complete.csv`，再与 RMCD 和本地基线进行共同阈值评价。

覆盖不足的方法不会从记录中消失：

- `official_sequence_coverage.csv`：逐实例状态；
- `official_method_coverage_summary.csv`：逐方法覆盖率；
- `complete_methods.txt`：进入平衡比较的方法；
- `incomplete_methods.txt`：失败、超时或缺失方法；
- `_PARTIAL`：平衡评价已完成，但官方全集并未全部覆盖；
- `_SUCCESS`：全部请求方法覆盖全部实例。

主文应展示预注册的代表方法，完整方法矩阵和失败覆盖审计放入补充材料。
不同阈值是同一图上的重复测量，统计独立样本量仍为 seed 数，而不是
`seed × threshold` 行数。

## 5. 正式实验组织

- `official-all` 包含 32 个官方序列项和 1 个派生诊断，只适用于 `n<=18`
  的小规模完整覆盖实验。
- `official-stable` / `official-all_scalable` 排除派生 Brute Force 诊断和上游
  实验性 CoreGDM，作为正式中大规模实验的方法全集。
- `official-experimental` 单独运行 CoreGDM；成功与失败均进入方法状态审计，
  不进入默认主性能汇总。
- GDM、FINDER、EGND 等耗时方法宜逐方法导出并保留独立日志。
- 通用/GDM 环境可用 `scripts/run_official_review_isolated.sh` 逐方法启动
  独立进程，并通过 `METHOD_TIMEOUT_SECONDS` 设置硬超时，避免一个方法挂起
  阻断全部基线。
- `OfficialEigenvectorDynamic` 在连通的 12 节点控制图上仍超过 300 秒，
  不进入主对比集合；失败状态保留在补充审计表中。
- `OfficialEigenvectorStatic`、`OfficialEI_S2` 和 `OfficialEGND` 已在稠密
  连通控制图上通过，但在含孤立节点的稀疏图上失败或超时，只能进入满足其
  输入适用域的连通实例补充比较，不能通过删除失败实例形成主表。
- 同一方法若会编译或改写自身配置文件，不要在同一源码目录并发运行多个实例。
- 吞吐实验可并行运行互不共享源码目录的环境；算法运行时间比较必须使用单
  worker、相同硬件和相同依赖环境。
- 正式表同时报告 `sequence_generation_seconds` 和
  `threshold_evaluation_seconds`，不能把跨环境总时间伪装成同环境计时。

主文可选取预注册的代表方法展示，33 项完整结果放入补充材料，但所有运行状态
必须在 `official_method_status.csv` 中公开。`official_method_catalog.csv`
记录模块、函数、依赖环境和上游实验性状态。

## 6. 结果与引用边界

需要同时保存：

- `official_sequence_manifest.json`
- `official_method_catalog.csv`
- `official_method_status.csv`
- `isolated_method_status.csv`
- `official_sequences.csv`
- `study_manifest.json`
- `instance_manifest.csv`
- `critical_set_results.csv`
- `selected_nodes.csv`
- `method_summary.csv`

官方仓库包装代码采用 GPLv3，但各算法目录可能具有不同许可证。发布包不
重新分发 `review-main` 源码。论文除引用 NetworkDismantling/review 对应的
Nature Reviews Physics 综述外，还应按其 `CITATIONS.md` 引用每个实际使用
算法的原始论文。
