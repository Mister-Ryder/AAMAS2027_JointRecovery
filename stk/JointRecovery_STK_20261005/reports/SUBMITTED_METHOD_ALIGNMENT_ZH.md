# 拟投稿完整方法与新 STK 实验的对齐审计

审计范围：只读第一篇 `main_v4.tex` 所引用的方法、实验、结果与旧最终交付报告，以及本轮 P1 拟合、固定调用回放、actual-policy 程序和已签收的开发结果。不修改检查点、冻结协议或实验，不训练，不启动云任务。证据截止本地已签收的 [P1 开发报告](P1_DEVELOPMENT_SIGNOFF_ZH.md)；本报告没有取得 P2 独立测试完成结论。

**结论：新 P1/P2 保留了拟投稿方法的核心预测模型与真实可行返回，但当前 actual-policy 使用简化的在线分配器，尚不能逐项代表旧稿完整程序。冻结 P2 应继续完成；其结果应明确称为“新 STK 最小学习分配器的独立验证/测试”，而不能替代完整在线控制器与公开发表方法的比较。**

## 1. 已完成的结果各自回答什么

| 证据 | 实际测试对象 | 能回答的问题 | 不能据此回答的问题 |
|---|---|---|---|
| P0 实际恢复探针 | 同一个真实 snapshot 的条件 CHILS 请求，含完整可行成员、两次重复和失败/零收益 | 原始 STK 图中是否存在共同 warm 与当前 incumbent 之外的实际恢复；不同 native 档位的响应 | 学习模型是否选对请求；完整策略是否按 D 返回；是否优于公开方法 |
| P1 固定 K 回放 | 已完成 cold-clone 标签上的请求选择，K=1/2/4 | 同档、同 K 的学习优先级是否比同接口控制选到更好的请求 | 多轮状态反馈的收益；真实在线计时胜利；完整图求解器比较 |
| P1 72 次实际开发校准 | 新 `p1_actual_policy.py` 的六个策略，完整调用者计时 | 冻结共同 D 所需的开发成本分布 | 独立测试优势；旧稿完整分配器性能 |
| 冻结 P2 计划 | r006/r007 验证、r008–r011 测试，六策略、三 D、全部六配置 | **当前简化程序**在未用于拟合/成本校准的物理来源上的完整调度收益与按时率 | 被省略机制的贡献；旧稿完整程序；公开发表 whole-graph 方法比较 |

固定调用回放不是仅消融：它比较了两个模型与两个无模型优先级。但 Greedy/P1/CheapSummary 都是同一恢复内核上的内部控制，不能把该表写成已发表先进算法的性能表。

已签收开发报告中，200 ms、K=1 的 Capacity 两种子平均净收益为 703.445 秒，Greedy 为 632.404 秒，差 71.041 秒；K=4 的差只剩 3.865 秒。Capacity-29 在 r005 的 K=1 低于 Greedy，主要均值提升来自 r004。这里的单位是**当前 snapshot 之外净增加的原始 contact 时长**，不是完整调度提升百分比；两开发来源不能代替独立测试。

## 2. 核心方法逐项对齐与缺失

旧稿来源：[方法正文](E:/01-Joycecyq/2026-AAMAS/第一篇/paper/v4_sections/method_residual.tex)、[在线控制器](E:/01-Joycecyq/2026-AAMAS/第一篇/src/joint_recovery/v4_budgeted_recovery.py:279)、[协调恢复封装](E:/01-Joycecyq/2026-AAMAS/第一篇/src/joint_recovery/v4_residual_controller_coordination.py)、[模型与缓存回调](E:/01-Joycecyq/2026-AAMAS/第一篇/src/joint_recovery/v4_residual_model.py:211)。

| 方法项 | 新程序的可证实现 | 对齐判断 |
|---|---|---|
| action-conditioned B、位移、R；R 上真实冲突与合法 clique factors | P0 scope 与 P1 重建严格绑定原图、ordered R、q 与成员；不把全天同资源 contacts 当完全团 | 已保留核心语义 |
| 真实三共同贪心 warm，L=w(W)、clique-partition U，已知 q 单独加入 | 复用真实执行 warm 与原始 `ResidualCapacity` / `ResidualCheapSummary` 模型 | 已保留；共同前缀收益必须单列 |
| width32、两层图编码、fractional capacity、warm-gated residual、membership 辅助监督 | 新 Capacity 复用实际模型，辅助系数 .05；CheapSummary 为无图编码控制 | 核心预测机制已保留；没有新增结构消融证据 |
| 同真实 state 的 alternative executions；标签不进输入 | 重建冻结 snapshot、warm 与完整返回 T；同请求两重复取均值、occupancy 均值 | 有执行对齐监督，但不是旧稿完全相同的目标 |
| 旧稿的 admitted target 由 snapshot 的完整 deadline 决定；late/failed 分类处理 | 新 P0 标签没有完整 caller D；新拟合以实际条件返回值及超过 snapshot g* 的边际值监督 | **没有学习完整 D 下的 admission/late-response 规律**；不能称其 deadline-conditioned 标签 |
| remaining-time request head | 新拟合与实际推理的 `ControllerState.remaining_seconds` 均固定为 0 | **没有学到 remaining-head 适应**；真实剩余时间仅用于外部分配准入 |
| 原稿 `max(0, predicted_gain−g*) / expected_cost`，并以绝对预测值/成本作次级排序 | 新 actual 第 81–82 行只按一次 `raw_gain` 排序；成本仅用于准入，不进入优先级 | **目标分配规则不同**，不能把“准入检查”当“成本敏感优先级” |
| 执行后更新同 action 的 actual warm，随后重新预测并按新 g*、spent、remaining 选下个请求 | 新 actual 的全部请求保留同一个 snapshot warm，排序冻结；父进程只更新最终可返回 incumbent | **没有在线多轮 warm/预测/排序反馈**。连续执行最多四请求仍不是该闭环 |
| 相同 ordered R、factors、warm、scale、graph/model version 时重用编码；warm 改变重新编码 | 新 actual 一次 pack/inference 使用静态 packing cache，没有传入 `DecisionEmbeddingCache`；之后不再编码 | **未测试部署编码复用及 warm 失效重编码**。一次 batch 内的共同表示处理不能替代跨轮缓存测量 |
| 原稿 s_G=max(1,全图平均权重)；开发 conditional regret 选检查点 | 新拟合用仅 TRAIN 冻结的 duration-median scale，四 fit 使用最终 epoch40；两 seed17/29 | 新版本有明确单位/冻结修正，**非逐项原稿复现**；两 seed 都必须保留 |
| 原稿最多八次、10/50/200 ms；新 D 的统一冻结 | 新程序最多四请求、10/50/200/1000 ms；D=3.898923968/1.949461984/0.974730992 秒 | 工作协议已改变；不得把旧/新收益数值直接横比 |
| 完整原图校验、原始 reward 重算、准备/推理/执行/parse/IPC/fullcheck/返回全计 D | 新 actual 父进程持有已验证 incumbent，逐候选完整校验；caller 超 D 时严格收益归零；resident setup 单列 | 已有真实端到端测试接口；**不能只因控制器简化就称其没有实际在线实验** |

具体新代码：[标签与输入重建](../code/p1_fit_and_allocate.py:77)、[尺度冻结](../code/p1_fit_and_allocate.py:178)、[训练 loss](../code/p1_fit_and_allocate.py:211)、[固定调用回放](../code/p1_fit_and_allocate.py:250)、[实际预测/冻结排序/请求执行](../code/p1_actual_policy.py:46)、[父进程独立返回](../code/p1_actual_policy.py:169)、[真实 caller 计时](../code/p1_actual_policy.py:228)。

还有一个比较语义需要注明：旧稿的 AnytimeGreedyPortfolio 是“共同前缀后停止”；本轮 Greedy 则按 q+w(W) 排序后继续调用同一 native 执行器。它更接近无模型请求分配控制，不能把二者当成同一个基线跨版本比较。

## 3. 能否回答“改善仍不及公开发表方法”

**现有新 STK 证据不能回答这个问题。原因是缺少同图、同原始 weighted objective、同完整 D 下的直接方法比较，而不是已有证据证明其胜出或落后。**

旧稿 [最终交付报告](E:/01-Joycecyq/2026-AAMAS/第一篇/docs/research_v4/FINAL_DELIVERY_ZH.md) 与 [Results](E:/01-Joycecyq/2026-AAMAS/第一篇/paper/v4_sections/results_residual.tex) 已明确：40 个原始公共 unit-weight 图上的 17 配置/2,040 runs 是 published solver profiles，**没有同期完整 JointRecovery policy 结果**。这些表不能给本方法与 CHILS、m²wis 等的同预算差距排序。

旧完整在线开发结果确实有不利事实：D=0.556 秒时 Capacity 严格收益57.542%，共同贪心前缀234.266%，Capacity迟到83.333%；D=2.221 秒时 Capacity236.334%，CheapSummary238.355%。它支持“旧开发设定没有建立完整学习策略优势”，但不能扩展成“已证实不及所有公开方法”。旧数据与新自然时长 STK 图也不是同一问题分布。

新 P2 的六策略仅为 Capacity 两 seed、CheapSummary 两 seed、P1、Greedy。CHILS 在这些策略中是**共享恢复子问题执行器**，不是独立 whole-graph 竞争方法。即使 Capacity 胜过 P1，也不能写成胜过 SEA 2025 CHILS。

旧正式 bibliography 已提供可直接追溯的 weighted-MWIS 基线来源：CHILS（SEA 2025）、m²wis（JGAA 2024）、Struction（ALENEX 2021）、WeightedBR（ALENEX 2019）。它们适合进入新图的直接比较；实际可执行性、warm-start 支持、整数范围和总成本必须逐实现说明。DIFUSCO/COExpander 的已释放 unit-MIS recipe 不可直接称作原始时长 weighted-MWIS 基线。

## 4. 最短必要的完整测试路线

以下是补齐交付证据所需的路线，本审计未执行任何一步，也不触发重训/探索。

1. **按原冻结协议完成当前 P2。** r006/r007 验证和 r008–r011 测试必须保留全部 cell、两个 seed、失败/迟到/零值，仍用既定三 D。该结果用于回答当前最小 STK 模型的真实测试表现，不因本审计重跑或改模型。
2. **另设完整闭环 STK 迁移版本。** 使用已封存的新 Capacity/summary 检查点；接回旧程序的多轮 priority 回调、timely 同 action warm 更新、g*/spent 更新、成本敏感排序、编码复用及 warm 变化失效机制，同时保留新 caller-owned 校验/返回边界。全部相同控制采用同一闭环，复用冻结 D、menu、源划分；准备、每轮重编码/推理、请求、校验/返回全计费。补测程序按照旧稿最多八请求，FullCapacity/FullCheapSummary/FullGreedy/FullP1 全部同八请求；必须与原 P2 的最多四请求表分列。先在验证来源确认接口可运行并冻结程序，再整块执行测试。**新 checkpoint 的 remaining 输入必须继续固定0，不能临时喂入未训练的非零值；论文因此应把这一版明确写为“完整在线控制框架＋固定 remaining-head 的 STK 迁移版本”。** 其性能可代表这个完整新版本，但仍非旧稿逐项同配置复现。
3. **若交付声称的是旧稿逐项完整方法：补旧封存 checkpoint 的零样本 STK 迁移测试。** 必须保留旧回调/动态 remaining/head 输入与旧归一化、工作点和调用协议，使用准确的原始时长/native 数值适配、共同 incumbent 与同完整 D，独立封存结果。这条路线不需要重训，但它测试的是“旧完整方法迁移”，不能冒称新 STK 训练的模型。对新 checkpoint 重新训练动态 remaining/head 属于另一版本，不能在冻结测试后悄悄替换。
4. **加入实际已发表 whole-graph MWIS 方法。** 优先已可用的单线程官方 CHILS，继而选择具有合适数值接口的 m²wis/Struction/WeightedBR。它们应允许在完整原图上搜索，不能被限制到我方256节点 action menu以削弱基线。共享原图、原始时长 reward、起始可行 S 与完整 D；数据准备、启动/搜索、parse、原图校验与caller返回按同一 resident/cold 边界计费。原始微秒整数导出必须检查溢出与舍入；不能为适配32位实现悄悄降低权重精度。没有可靠 warm-start 或可行中间输出的实现应如实注明配置与失败/N/A。

第二步与第三步应分别命名、分别列结果；只有第三步能回答“旧稿逐项完整程序”的迁移表现，只有第二步能回答“本轮新 STK fit 接入完整在线框架”的表现。无需先铺开消融或超参数搜索即可补齐这两种含义。

## 5. 最终交付必须出现的结果

必须给出**独立测试上的完整 feasible schedule 性能表**：完整方法版本、Greedy/P1/summary 控制、实际 published whole-graph 方法及正式引用；列原始目标值↑、严格按时增量↑、共同 paid prefix 外净增量↑、实际 caller 秒↓、按时率↑、真实 native 调用数，注明 D、线程、setup边界。应同时给出各物理来源结果、两个模型 seed 和源等权汇总，不能把配置/状态当新增独立物理样本。

最终回答必须明确三件事：完整方法在测试上是否改善；改善是否主要来自共同贪心；相较最强实际已发表基线是否仍落后、领先或条件不同而不可比。机会探针、内部回放和训练 loss 可以解释这些结果，但不能代替该性能表。当前审计发现的是**证据覆盖范围不足**，没有证明新完整方法会胜出，也没有证明其必然落后。

审计后的代码交付：隔离 [full_joint_recovery_deadline.py](../code/full_joint_recovery_deadline.py) 已实现第二步的动态闭环、父校验后确认的 warm/g* 更新与精确编码缓存。仅完成一次语法及小型 fake-executor/缓存 guard，未运行云端实际测试；本报告仍没有完整 STK 方法或公开基线的性能结论。详见 [补测契约](../code/FULL_CONTROLLER_CONTRACT.md)。
