# RMCD-F 算法与实验冻结规范 v1.1

## 0. 文档地位

本文件取代 v1.0 中与下列内容冲突的规定：

1. 单位容量、单位成本情形的学术用途；
2. 精确 MILP 与论文主算法的关系；
3. 参数化角色割分解算法的认证语义；
4. 合成装备网的生成与审计规则。

未被本文件修改的研究边界继续沿用 v1.0。研究对象仍然只有固定角色模体

\[
\mathcal R:\quad S\rightarrow C\rightarrow L\rightarrow E,
\]

物理装备节点仍是唯一可攻击和可保护对象。不引入边攻击、级联、恢复、角色转换、概率攻击、学习模块、多模体或任务价值优化。

---

## 1. 唯一研究问题

本文研究：在短时吸收窗口内，哪些物理装备节点的集体失效会以最小代价把可并发形成的角色模体服务能力压低到任务需求以下，以及有限的事前保护资源应如何配置，才能提高攻击方自适应重优化后的最小击穿代价。

角色模体容量是结构状态量，不等于任务成功率。结构到功能的关系只通过独立的匹配删除实验验证，不进入节点选择目标。

### 1.1 军事语义与时间边界

- (S\to C) 表示在既定体系接口下能够把有效感知报告交付给指控节点的装备关系；该关系已经包含其必要的数据接入条件，不再单独展开上行通信子网。
- (C\to L\to E) 表示指控决策经通信/中继装备送达执行装备的下行关系。
- 若研究场景必须显式建模双向通信、反馈闭环或多级指挥，则不属于本文固定角色模体，不能通过临时增加边类型修补。
- 给定固定吸收窗口 (\Delta t)，(u_v) 是折算为“完整角色模体服务单元”的最大并发角色槽数。该参数由想定或预注册容量等级外生给定，不由 RMCD 结果反推。
- 本文的防护决策发生在扰动之前，研究的是“面向吸收能力的事前加固”；扰动发生后只评价静态窗口内的剩余结构能力，不宣称在线恢复或实时自适应控制。

---

## 2. 三类计算对象

### 2.1 `RMCD-Exact`

`RMCD-Exact` 是基于紧凑 MILP 的精确基准求解器。它用于：

- 小中规模最优值和证书；
- 检验专用算法的上下界与最优性差距；
- 枚举最优攻击集并分析核心、并集和排除代价；
- 作为 RMCD-F 外层约束生成的精确攻击 oracle。

紧凑 MILP、最大流阻断和通用求解器调用不作为原创算法贡献。

### 2.2 `RMCD-PCD`

`RMCD-PCD` 是角色专用的有界分解求解流程，含义为 Parametric Cut Decomposition。它利用固定四层角色图的割结构，通过参数化加权最小割生成候选角色割，在每个割内求解容量覆盖，并同时维护合法下界和可行上界。其拉格朗日最小割和割枚举骨架与既有最大流阻断研究存在实质重合，因此在完成与 Royset--Wood (2007) 及 Bentoumi 等 (2025) 的逐项差异证明之前，不将 PCD 宣称为新的通用网络瓦解算法。

### 2.3 `RMCD-F`

`RMCD-F` 是事前自适应加固扩展。外层保护问题调用 RMCD 攻击 oracle，并允许攻击方在观察保护集合后重新优化。约束生成框架本身不宣称为原创；论文只研究它在角色模体容量目标下产生的保护结构和适应性价值。

---

## 3. 角色模体容量

装备图为

\[
G_E=(V,\mathcal E,\tau),
\]

其中 \(\tau(v)\in\{S,C,L,E\}\)，合法边仅为 \(S\to C\)、\(C\to L\) 和 \(L\to E\)。节点容量 \(u_v\in\mathbb Z_+\) 表示节点在吸收窗口内可同时提供的角色服务单元数。

给定删除集 \(D\)，角色模体容量为

\[
\Omega_{\mathcal R}(D)=
\max \sum_{m\in\mathcal M}y_m
\]

满足

\[
\sum_{m:v\in m}y_m\le u_v\mathbf 1(v\notin D),
\qquad y_m\in\mathbb Z_+.
\]

通过节点拆分构造角色扩展图后，有

\[
\Omega_{\mathcal R}(D)=\operatorname{MaxFlow}(H_{\mathcal R}(D)).
\]

该等价关系必须以定理和独立证书报告。

---

## 4. 单位同质退化定理

### 定理 1

假设：

1. 所有节点均可攻击；
2. \(u_v=1\)；
3. 全部节点具有相同正常数攻击成本 \(c_v^A=c>0\)；
4. 初始容量为 \(\Omega_0\)；
5. \(1\le K\le\Omega_0\)。

则使 \(\Omega_{\mathcal R}(D)\le K-1\) 的最小攻击成本满足

\[
C^*(K)=c(\Omega_0-K+1).
\]

### 证明要点

删除 \(|D|\) 个单位容量节点最多破坏初始整数最大流分解中的 \(|D|\) 个节点不交路径，因此

\[
\Omega_{\mathcal R}(D)\ge\Omega_0-|D|.
\]

可行攻击必须满足 \(|D|\ge\Omega_0-K+1\)。另一方面，从任意初始最小节点割中删除 \(\Omega_0-K+1\) 个节点后，该割剩余容量为 \(K-1\)，故下界可达。

### 推论

- 单位同质成本容量前沿为 \(\Phi(k)=\Omega_0-k\)；
- 当 \(K>1\) 时，同一最小割存在多个等成本子集，交集核心不能作为一般稳定现象；
- 单位同质情形只能用于理论说明、软件正确性和运行时间基准；
- 禁止用该情形支持“RMCD 在最小瓦解成本上优于基线”的经验结论。

实现必须检测该条件，并允许直接返回基于初始最小割的闭式最优解。

---

## 5. 非平凡主问题设置

主实验采用预先给定的整数并发容量 \(u_v\)。其物理解释限定为吸收窗口内的并发角色服务槽，例如感知航迹处理槽、指控任务槽、通信中继信道或执行通道。

主实验原则：

- 攻击成本默认 \(c_v^A=1\)，避免通过人为成本制造优势；
- 保护成本默认 \(c_v^P=1\)；
- 节点容量采用预注册的小整数等级，并在生成网络前确定；
- 容量多重集不得根据 RMCD 结果、功能结果或预期关键节点反向调整；
- 异质攻击成本只进入单独的敏感性实验；
- 如果容量等级无法由场景假设独立解释，则论文降级为方法模型研究，不作军事能力推广。

---

## 6. RMCD-PCD 数学分解

令 \(\mathscr C_{st}\) 为角色扩展图的源汇割集合。对固定割 \(C\)，定义割内容量覆盖问题

\[
\phi_K(C)=
\min_{D\subseteq C}\sum_{v\in D}c_v^A
\]

满足

\[
\sum_{v\in D}u_v
\ge
\sum_{v\in C}u_v-(K-1).
\]

则

\[
C^*(K)=\min_{C\in\mathscr C_{st}}\phi_K(C).
\]

固定割问题使用整数动态规划精确求解；单位容量时退化为选择割内攻击成本最低的相应节点。

对阈值约束引入 \(\lambda\ge0\)，定义

\[
w_v(\lambda)=
\min\{c_v^A,\lambda u_v\}
\]

并计算加权最小角色割。得到合法拉格朗日下界

\[
g(\lambda)=
-\lambda(K-1)
+\min_{C\in\mathscr C_{st}}
\sum_{v\in C}w_v(\lambda).
\]

对受保护或禁止攻击节点，权重只能取 \(\lambda u_v\)，不能取攻击成本截断。

每个加权最小割同时产生：

- 一个合法下界 \(g(\lambda)\)；
- 一个固定割容量覆盖候选；
- 一个可独立验证的可行攻击上界。

### 命题 2：有限对偶搜索区间

给定任一已验证可行攻击及其成本上界 (UB)，若节点容量为整数且攻击成本为正常数，则拉格朗日对偶至少存在一个最大化乘子满足

\[
\lambda^*\in[0,UB].
\]

证明要点如下。可行攻击对应至少一条截距不超过 (UB)、斜率不大于 0 的配置线，因此 (g(\lambda)\le UB)。任一斜率为正的配置线具有非负截距和至少为 1 的整数斜率；当 (\lambda>UB) 时其值严格大于 (UB)，不可能成为下包络的活动线。因此 (g) 在 (UB) 之后不存在正的活动斜率，至少一个对偶最大点落在该有限区间内。

### 命题 3：有限线池收敛

角色扩展图的割集合和每个割内的攻击子集均有限，因此 (g(\lambda)) 是有限条仿射配置线的下包络。受限线池的下包络是 (g) 的上近似。若主问题给出的乘子尚未闭合对偶上、下界，则加权割 oracle 必然返回至少一条此前未发现的活动配置线；否则该点的受限下包络已经等于真实 (g)，与“未闭合”矛盾。因此在不设置 oracle 次数上限且采用精确有理数运算时，对偶线生成有限终止。该结论只保证对偶最大化完成，不保证整数原问题与拉格朗日对偶之间无间隙。

---

## 7. RMCD-PCD 算法

```text
Algorithm RMCD-PCD(G_E, tau, u, cA, K, tolerance, max_oracle_calls)
1. Validate the fixed role semantics and build H_R.
2. Compute Omega_0 and check whether the instance is breakable.
3. If every node is attackable, all node capacities are one, and all attack
   costs share the same positive value:
     return the closed-form minimum-cut solution.
4. Obtain a verified feasible attack UB and set lambda_max=UB. With integer
   node capacities this interval contains a dual maximizer.
5. Initialize the exact-rational affine-line master on [0, lambda_max].
   At each master maximizer lambda, call the weighted role-cut oracle and add
   every returned active affine configuration line.
6. For every new cut C:
     a. solve the exact within-cut capacity cover phi_K(C);
     b. verify the resulting attack on the original flow network;
     c. update UB and the incumbent attack.
7. Update LB with the true oracle value g(lambda). The restricted line-master
   value is only an upper bound on the dual optimum and is never reported as
   a primal lower bound.
8. If the exact bounds satisfy UB-LB <= tolerance:
     return the globally certified attack.
9. If the dual master closes while UB>LB, report a closed-dual incumbent gap.
   This interval must not be called the true integer duality gap unless an
   exact oracle separately proves UB=C*(K).
10. If the oracle limit is reached:
     return the incumbent, LB, UB and explicit optimality gap.
11. Optionally invoke RMCD-Exact when an exact result is required.
```

### 认证纪律

- 有限参数搜索本身不等于精确最大化拉格朗日对偶；
- 只有存在可行攻击且 \(UB-LB\) 在容差内时，PCD 才能标记全局最优；
- 对偶搜索已闭合但原始上下界未闭合时标记 `DUAL_CLOSED_GAP`；只有实际达到 oracle 调用上限时才标记 `LIMIT_REACHED`；
- 调用 `RMCD-Exact` 后获得的最优性必须明确标记为 exact fallback，不能归功于 PCD 下界闭合；
- 所有候选攻击都必须重新计算原图残余角色模体容量和割证书。

---

## 8. 受控合成装备网

由于不存在可公开获得且同时包含四类军事装备角色、合法关系、并发容量和攻防成本的数据，主实验使用受控合成网络，不伪称真实数据。

生成器只包含三类理想化拓扑原型：

1. `centralized`：各角色接口存在明显的关系集中；
2. `modular`：关系优先受模块边界约束，极稀疏设置允许形成分离模块；
3. `distributed`：合法关系在同角色节点间较均匀分布。

这三者是组织结构的理想化极端，不是只改变单一统计量的因果处理。不得把原型间差异直接解释为某一个拓扑机制的独立因果效应。

三类拓扑在同一实验组内必须匹配：

- 四类角色节点数量；
- 三个角色接口的边数；
- 每一角色内部的节点容量多重集；
- 攻击成本和保护成本多重集；
- 随机种子集合。

禁止事项：

- 不得根据 RMCD、LCC、任务功能或任何目标结果筛选生成实例；
- 不得删除“结果不好”的种子；
- 不得为某一算法单独调整拓扑参数；
- 不得把手工演示图作为性能证据；
- 不得把合成场景称为真实装备网或实战数据。

生成器必须在任何优化之前由构造规则保证至少一条合法完整角色链，不得先计算 RMCD 再删除零容量实例。元数据必须记录全部参数、随机种子、角色规模、接口边数、逐角色容量分布、预置角色链和生成失败次数。

---

## 9. 实验轨道

### 9.1 理论与正确性

- 单位同质退化定理；
- 小图枚举与 RMCD-Exact 一致性；
- PCD 上下界合法性；
- PCD 闭合时与精确最优值一致；
- 未闭合时不得误报最优。

### 9.2 算法性能

在同一异质容量实例上比较：

- RMCD-PCD；
- RMCD-Exact；
- Maximum Flow Blocker 的公开精确模型或实现；
- Random、Degree、Betweenness、CI、CoreHD、GND；
- Motif-Greedy。

基线序列统一按首次使 \(\Omega_{\mathcal R}\le K-1\) 的攻击成本评估，同时报告其原生 LCC 破坏结果，避免只在 RMCD 指标上形成单边比较。

### 9.3 拓扑原型分层分析

在匹配角色规模、接口边数、逐角色容量和成本后，按理想化拓扑原型分层观察：

- 最小击穿成本；
- 最优集合的替代性；
- 角色割位置；
- PCD 最优性差距；
- RMCD-F 相对静态保护的增益。

该轨道只检验算法是否跨不同结构原型保持有效，不声明单一拓扑统计量的因果作用。若论文需要机制因果解释，必须另行采用接口内度序列保持的重连零模型；该零模型不进入当前算法定义。

### 9.4 独立结构功能验证

完整任务闭合指标 \(Q\) 与 RMCD 代码物理隔离。匹配删除成本、节点数和普通连通性损失后，检验角色模体容量损失是否仍能解释额外的 \(Q\) 下降。该检验失败时，删除结构功能桥接主张，不增加新目标修补算法。

---

## 10. 允许的论文贡献

只有证据支持时，允许表述：

1. 定义面向固定军事角色协同结构的并发角色模体容量；
2. 揭示单位同质设置下阈值阻断的退化性质；
3. 将集体关键装备识别形式化为角色模体容量 blocker 问题；
4. 构建并验证适用于角色扩展图的有证书参数化割分解实现 RMCD-PCD，同时明确其与既有流阻断算法的继承关系；
5. 在攻击方自适应重优化下分析角色模体容量的事前强化；
6. 通过独立匹配删除实验检验结构容量损失与任务闭合下降的条件性关系。

禁止表述：

- 首次提出最大流 blocker、网络 interdiction、紧凑 MILP 或 CCG；
- 单位同质实验说明 RMCD 的最优成本优势；
- PCD 有限扫描等于全局最优；
- 合成装备网等于真实无人集群；
- 角色模体容量等于作战效能；
- RMCD-F 是适用于所有复杂网络的通用瓦解算法。

---

## 11. 开发门控

正式扩大实验前必须全部通过：

- [ ] 单位同质退化定理测试；
- [ ] `RMCD-Exact` 明确别名与向后兼容；
- [ ] RMCD-PCD 的 LB/UB、gap 和 fallback 来源可追踪；
- [ ] PCD 在随机小图上不产生非法下界；
- [ ] 合成生成器不依赖任何求解结果；
- [ ] 三类拓扑的角色数、接口边数和容量多重集匹配；
- [ ] 受控生成参数写入元数据；
- [ ] 与 2025 Maximum Flow Blocker 模型完成逐项差异表；
- [ ] 结构功能验证代码与选择算法物理隔离。

未通过这些门控前，服务器运行只属于软件验证，不属于论文正式实验。
