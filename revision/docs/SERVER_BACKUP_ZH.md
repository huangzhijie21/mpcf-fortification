# 服务器完整备份清单（可以安全清空服务器）

备份时间：2026-09-19
备份工具：`revision/tools/backup_servers.py`
备份位置：`revision/server_backup/`
压缩总量：**92.33 MB**（对应服务器上约 250 MB 原始数据）

---

## 一、结论

**现在可以清空并关闭三台服务器了。** 本备份已通过最强验证：

> 用备份里的三个 `results.tgz` 重新跑一遍 `merge_archives.py`，
> 生成的 `runs_long.csv`、`instance_budget_cells.csv`、`graph_manifest.csv`
> 与已发布的 `revision/final_archive/` **逐字节完全相同**
> （sha256 一致；见第六节）。

也就是说：备份足以完整重建已发表的归档，不含任何缺口。

## 一之二、顺带修掉的一个归档缺陷

做这次比对时发现 `metadata/graph_manifest.csv` 有 **621 行，而实际只有 576 个实例产生过运行**。
多出的 45 行是种子扩展第一次尝试时用**主面板种子**（s11/s22/s33/s44/s55）冻结的
`seedext__*` 图 —— 那次接线 bug 修好后重跑，这 45 个图从未被求解，
但它们仍留在 manifest 里，把"冻结实例数"虚报成 621。

已在 `merge_archives.py` 中修正：**只保留确实产生过运行的实例**，
并把被丢弃的 instance_id 逐个列进 `merge_summary.json`
（`manifest_rows_dropped_no_runs` 与 `dropped_instance_ids`），不静默删除。

影响范围已核实为**纯元数据**：`runs_long.csv` 与 `instance_budget_cells.csv`
未变，重新生成的 21 个 statistics/tables 文件与修正前**逐字节相同**，
17 项验收门依旧全绿。也就是说没有任何一个已发表数字因此改变，
改变的只是 manifest 描述的对象数（621 → 576）。

## 二、此前遗漏了什么（为什么必须做这次备份）

`finalize.py` 当初只下载了 `/root/mpcf_rev/results/` 下的 7 个子目录
（results、metadata、statistics、figures、methods、tables、gates），
**没有下载**服务器上这些仍然只存在于服务器的东西：

| 只存在于服务器的内容 | 大小 | 为什么重要 |
|---|---:|---|
| `results/instances/` | 21 MB | **576 个冻结图 JSON**，是"实例冻结"这一步的物证 |
| `results/shards/` | 6.7 MB | 原始 per-worker 产物，`runs_long.csv` 的审计底稿 |
| `results/official_*_matrix/` | 37 MB | 上游基线 sequence 导出，是主面板的**输入**而非输出 |
| `results/logs/` | 4.5 MB | exact/cg/greedy/baselines 求解器日志 |
| `/root/mpcf_rev/logs/` | 23 MB | 环境安装、preflight、panel、official matrix 全过程日志 |
| `external/review-main/` | 127 MB | **已打补丁且已编译**的上游代码树 |
| `app/` + `app_old/` | 5.6 MB | 部署的源码修订 + 重构前旧树（复现性交叉验证用） |
| conda 环境 | 5.0 GB | `dismantling`(py3.11) 与 `finder`(py3.7) |

## 三、备份内容

每台机器一个目录，`revision/server_backup/<srv>/`：

| tar | srv0 | srv1 | srv2 | 内容 |
|---|---:|---:|---:|---|
| `results.tgz` | 4.00 MB | 1.33 MB | 2.20 MB | 整个 `results/` 树 |
| `logs.tgz` | 1.56 MB | 0.003 MB | 0.005 MB | 服务器级日志 |
| `code.tgz` | 1.83 MB | 1.14 MB | 1.14 MB | `app/` + `app_old/` |
| `scripts.tgz` | 0.01 MB | 0.003 MB | 0.003 MB | 顶层驱动脚本 |
| `env.tgz` | 0.004 MB | 0.004 MB | 0.004 MB | 解释器版本 + `pip freeze` |
| `external.tgz` | 81.46 MB | — | — | 编译后的上游代码树 |
| `scratch.tgz` | 2.14 MB | — | — | scratch / verify / smoke / FINDER 重复性探针 |

srv0 另存 `env_conda.tgz`（`conda list` + `--explicit` URL 清单）
与 `mpcf_revision_server.deployed.zip`（服务器上实际部署的 bundle）。

所有 tar 都经过**两端字节数比对**，sha256 记录在
`revision/server_backup/backup_manifest.json`。

## 四、`external.tgz` 为什么不能只靠 `review-main.zip`

本地早已有上游 `review-main.zip`（sha256 `b1023d6b…`，与服务器上的
`/root/mpcf_rev/review-main.zip` **完全一致**）。但解压后的树是 **127 MB / 573 文件**，
而 zip 只有 **303 文件**。多出来的 207 个文件是：

* **9 个编译好的扩展**（`.so`）：`FINDER.cpython-37m-…so`、
  `nstep_replay_mem_prioritized.…so`、`PrepareBatchGraph.…so`、
  `mvc_env.…so`、`dismantler.so`、EI 的 `exploimmun.o` 等 —— 这些是
  为 Python 3.7 单独编译的，重建成本高；
* **70 个 GDM 模型 `.h5`** 与 `reinsertion` 可执行文件；
* 63 个 `__pycache__` 文件。

所以 zip 只是"源码备份"，`external.tgz` 才是"可运行状态备份"，两者都需要。

## 五、conda 环境：存的是配方，不是字节

`/root/autodl-tmp` 共 5.0 GB（其中 `conda-pkgs` 1.5 GB 只是包缓存，
`envs` 3.6 GB 是两个环境）。备份保存的是**可重建的完整配方**：

* `environment_spec.txt` — 解释器版本 + 两个环境的 `pip freeze`
* `conda_spec.txt` — `conda list` + `conda list --explicit`（含下载 URL）

关键版本已核对：

| 环境 | 关键包 |
|---|---|
| `finder` (Python 3.7.12) | tensorflow 1.14.0、Cython 0.29.13 |
| `dismantling` (Python 3.11.16) | torch 2.14.0+cpu、torch_scatter 2.1.2、torch_sparse 0.6.18、torch_geometric 2.8.0.post1、libboost-devel 1.90.0（conda-forge）|

配合 `revision/scripts/setup_server.sh`（已备份）即可重建。
**注**：`libboost-devel` 是 conda 包，不在 `pip freeze` 里，
只在 `conda_spec.txt` 中 —— 当初编译 Boost.Python 扩展就靠它。

## 六、复核方式

```bash
# 1. 解压
revision/server_backup/srv{0,1,2}/*.tgz  ->  <srv>/extracted/

# 2. 从备份重新合并
python revision/tools/merge_archives.py \
  --inputs srv0/extracted/results,srv1/extracted/results,srv2/extracted/results \
  --output revision/server_backup/verify_merge

# 3. 与已发布归档比对（应逐字节相同）
#    results/runs_long.csv              8 979 672 B
#    results/instance_budget_cells.csv    140 132 B
#    metadata/graph_manifest.csv          290 445 B  (576 行)
```

该复核已经执行并通过，产物保留在 `revision/server_backup/verify_merge/`。

## 七、每台机器的验证数字

| 项目 | srv0 | srv1 | srv2 | 合计 |
|---|---:|---:|---:|---:|
| `runs_long.csv` 行数 | 11 745 | 243 | 8 991 | **20 979** |
| 冻结图 JSON | 576 | 531 | 531 | （去重后 **576**）|
| shard 原始文件 | 540 | 162 | 594 | 1 296 |
| 代码修订 | `core-a8deebd9…` | 同 | 同 | 单一修订 |

> 每台服务器都生成了全部 576 个实例（按种子确定性重放），但只运行自己那份面板。
> 这种跨机重复正是 `merge_archives.py` 能校验 `graph_fingerprint` 一致性的基础 ——
> 指纹不一致时它会拒绝合并。
## 八、清空前建议保留的本地目录

```
revision/final_archive/     已发表归档（图表、统计、门禁）
revision/server_backup/     本次完整备份（92 MB）
revision/dist/              review-main.zip + 最新 server bundle
revision/server_scripts/    服务器私有脚本（seedext_resume.sh 等）
revision/scripts/           驱动脚本
revision/tools/             合并、统计、门禁、备份工具
src/, scripts/, tests/      代码本体
```

`revision/snapshot_archive/` 是中途快照（scaling 未补满时的版本），
已被 `final_archive/` 取代，可删；`revision/downloads/` 是旧的
不完整下载（只含 7 个子目录），保留作对照或删除均可。
