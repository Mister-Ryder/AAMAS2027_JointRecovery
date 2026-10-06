# P0 一次分析的口径

输入为真实 `execution/*/out/JR-DUAL-*/summary.json`、同目录的去重 `snapshot_*.json`、原始全图 `initial_global_greedy.json` 及实际 `trajectory_*.json`，并使用本次冻结的 `graphs/*.npz` 计算候选域被完整固定 base 阻挡的比例。分析器不改变动作、域、cap、executor、标签或来源分组。

执行命令：`python code/analyze_p0.py`。若存在同一图多套执行收据，必须以 `--execution-root` 明确选择一套，避免混合协议或重复计数。

输出保存在 `reports/p0_analysis/`：

- `analysis.json`：来源数、完成范围、每图 G1–G4、实际域结构与建议。
- `graph_acceptance.csv` 和 `source_group_summary.csv`：按配置图报告，然后在物理来源内等图平均。状态和请求不算额外独立物理来源。
- `state_summary.csv`：逐控制状态的真实额外机会与重复预算差异。
- `all_requests.csv`：完整保留失败、零增益、已花不可分配请求及分项成本。局部恢复百分比、完整目标增加秒数、完整目标增加百分比分列。
- `summary_similar_request_pairs.csv`：同状态、同 workpoint、近似相似 q/L/U/P1 与规模的请求结果差异。这里只识别值得后续检验的关系信号，不能证明图编码必要或优于 CheapSummary。
- `paired_200_1000ms_results.csv`：相同 frozen scope/warm/seed/repeat 的两个执行预算配对；至少两次同方向非零差异才计入 G3。
- `P0_ANALYSIS_ZH.md`：简洁中文验收结论与下一步限制。

G1 的 30% 和 G3 的约 20% 是开发投入参考，不能作为删除测试样本的门槛。共同三贪心、实际 history 和 native 求解器带来的收益均不归给学习。没有完整 caller deadline 收据时，严格迟到率明确为未测；native 进程超过搜索 workpoint 不等于完整策略迟到。

只有 P0 出现实际额外机会并且完整范围已完成后，才讨论小模型闭环。机会不足时最多建议计划允许的单因素 cap512 或单壳对照，并列明截断、耦合和完整 base 封锁的已测证据；分析器不会自动生成更有利样本或启动训练。
