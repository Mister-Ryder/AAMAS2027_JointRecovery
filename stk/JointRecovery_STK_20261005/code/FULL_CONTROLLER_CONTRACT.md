# 完整在线恢复框架补测接口

此独立程序不修改 P1、当前冻结 P2、旧稿控制器或模块 globals；不训练。四个 new-STK epoch40 检查点、fit protocol、归一化和 TRAIN 条件成本均逐项绑定封存收据。**remaining-head 仍固定0，不能声称已验证剩余时间学习。** 当前交付是完整在线框架的 STK 迁移版本，不是旧稿逐项相同工作协议。

```text
python code/full_joint_recovery_deadline.py
  --runtime-root CAPSULE/runtime
  --fit-root FIT/out
  --graph GRAPH.npz
  --out FRESH
  --chils CHILS
  --chils-source CHILS_SOURCE
  --policy FullCapacity
  --fit-seed 17
  --action-seed 17
  --deadline-seconds FROZEN_D
  --budgets-ms 10 50 200 1000
  --max-calls 8
  --cpu-index 0
```

实际调用须在已提供的 Linux CPU 环境。policy 可为 FullCapacity、FullCheapSummary、FullGreedy、FullP1；学习模型 seed17/29 均须保留。`--cpu-index` 从当前允许的 affinity 中选择一个 CPU；模型/native 都只使用这一 CPU。max8 是 executor 请求数，空 scope 请求单列、不冒记 native call。菜单和动作 seed 必须完全匹配上述值。

每个决策从与 P2 相同的原图和全图三贪心 incumbent 开始，resident 图/模型/初态 setup 单列。D 内重新构造11动作、R256、共同三贪心 warm、特征/界限。每轮仅考虑未 spent 且 native amount/共同完整条件成本可容纳的请求；按 `[pred−g*]+/cost`、`[pred]+/cost`、固定请求顺序排序。 paid 推理后重新检查准入，低排名的短请求仍可使用未消耗额度。

native 返回由子进程原图核验，再经父进程完整核验。父返回 ACK 后，才更新同 action 的 warm、g*，spent 请求不免费重试。下一轮重新预测/排序；原始 `DecisionEmbeddingCache` 用实际 graph/model parameter versions 绑定，structural key 严格等于 ordered R、factors、warm、scale；warm 变化必须重新编码。缓存 packing/lookups/编码/IPC/ACK 全在 D 内。

父始终持有已验证原始 S/更好候选；截止后搜索清理异步，实际 caller return 迟到时严格收益为0。没有物理硬实时保证。模型和 native 失败、空恢复、零收益、不足预算、未观察到完成的请求均保留；不能把请求已开始推断为 native 已完成。

输出 `full_joint_recovery.json`：

| 字段 | 含义 |
|---|---|
| schema | `joint_recovery_stk_full_actual_v1` |
| status | `FULL_JOINT_RECOVERY_ACTUAL_COMPLETE`；失败/截止仍为完整已测 cell，另看 stop_reason |
| graph.graph_id/source_group/npz_sha256 | 输入绑定，与 P2 loader 相同 |
| policy、fit_seed、action_seed、deadline_seconds | 实际冻结设置 |
| initial_mask_sha256 | n 长度连续 int8 mask 的原始 C-order bytes SHA256；供整图基线确认起点 |
| initial_value_seconds、returned_value_seconds、actual_gain_seconds | 原始 contact duration 目标与可行增量 |
| strict_on_time_gain_seconds | 完整 caller 未迟到时的 actual gain，迟到为0 |
| paid_shared_prefix_gain_seconds | 父已在 D 内核验的共同前缀增量 |
| actual_gain_beyond_paid_shared_prefix_seconds | 已保存候选比共同前缀多的增量，不含学习归因承诺 |
| strict_on_time_gain_beyond_paid_shared_prefix_seconds | 上项在实际 caller 未迟到时有效，否则0 |
| external_caller_observed_return_seconds、missed_return_sample | 与当前 P2 一致的真实返回边界 |
| stop_reason | finished、failed、deadline_reached；细分 controller_stop_reason 在 finished event |
| executed_requests、completed_requests | 父观察到的开始/返回事件计数；八额度含 empty scope |
| actual_native_calls | 完整 finished 收据中的实际数量；未取得 finished 时为 null |
| observed_completed_native_calls | 父已收到完成行里 native_called 的真实数量 |
| requests_without_observed_completed_response | 开始但未取得完成成员响应的数量，不能填成成功0增益 |
| events | 每次开始/候选/父核验/完整 finished 中的动态轮次、warm 与缓存成本记录 |
| deployment_resident_setup_seconds | 给定原图/模型/初始 S 的独立 setup，不能重复加 global greedy 成本 |
| predictor、fit_protocol_sha256、source hashes | sealed checkpoint、protocol、数值 backend 与程序版本 |

一次必要的小 guard：`python code/full_joint_recovery_guard.py`。已实际通过语法、fake clock/executor 的成本排序/动态 g*/warm/spent/预算，以及实际 cache 类的 exact warm/R/factors/scale 和 graph/model-version 失效检查；无 native、拟合或云任务。完整真实 tensor inference/native/端到端性能仍需根任务按阶段执行，不能把该 guard 解释为性能优势。
