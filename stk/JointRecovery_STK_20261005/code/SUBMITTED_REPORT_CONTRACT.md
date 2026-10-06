# 完整方法/整图基线结果报告接口

本脚本只读已完成的本地镜像，不连接云端、不执行算法、不改paper/results、不绘图、不发布。

```text
python code/submitted_comparison_report.py
  --comparison-root execution/submitted_comparison/out
  --original-p2-root execution/p2_confirmation/out
  --budgets-file execution/p1_followup/out/calibration/frozen_budgets.json
  --out reports/SUBMITTED_METHOD_RESULTS
```

上述路径可传绝对路径；所有相对输入/输出路径按dataset-root解析，无需改变工作目录。`--dataset-root` 默认脚本所在dataset根；`--plan` 默认protocol/SUBMITTED_COMPARISON_EXTENSION_PLAN_20261005.json；`--p1-root` 默认execution/p1_followup/out。输出限制在该dataset的reports内。

扩展每块必须先有 `summary.status=SUBMITTED_COMPARISON_MATRIX_COMPLETE`、expected=received=288/576 且全量rows存在，才会打开 `actual_run_rows.json`。否则该块WAITING，不读取live rows。原P2同样要求 `ACTUAL_SERIAL_MATRIX_COMPLETE`、216/432全量rows；P1固定调用回放只读已完成summary，均不进入完整方法主结果。

主输入行沿现有comparison driver字段：source/graph_id/station_view/ground_gap_seconds/satellite_gap_seconds、policy_label、budget_id/deadline_seconds、graph_sha256_from_metadata、report_usable、failure_or_unknown_outcome、initial_mask_sha256、initial/returned/actual_gain、strict_delivered_gain_including_failed_runs_seconds、paid_shared_prefix_gain_seconds、actual_gain_beyond_paid_shared_prefix_seconds、caller/missed_return/stop_reason。所有原行字段完整保留，未知JSON值为null，CSV明确写为`null`。

严格失败交付0取driver已写数值，不把缺失数值自行补0。提交缺失/不可用与已按时返回prefix的native failed-stop分开。严格总目标/百分比在同原图、同已核对初态下按共同S计算；原始reported未知字段不被peer锚点覆盖。输入/初态/精确D有差异时保留数据并标记不可比。

汇总：每图两个学习seed等权（逐seed仍单列）；每母源六配置等权；TEST四母源r008–r011等权。两个seed范围不作CI，配置/运行数不作独立物理样本数。比较Capacity逐seed及均值对FullGreedy、FullP1、CheapSummary逐seed/均值、CHILS-p1、HiGHS；StrongCheapControl仅为FullGreedy与CheapSummary均值的描述性较强参照，不据结果挑seed或修改部署策略。

输出：

- `SUBMITTED_METHOD_RESULTS_ZH.md`：中文结论、带↑↓的完整目标/增量/成本/按时率表、失败和迟到、逐seed直接回答及方法边界。
- `main_source_equal.csv`：完整方法主汇总（含逐seed和均值）。
- `main_per_physical_source.csv`、`main_per_graph_configuration.csv`：母源/配置追溯，供后续曲线。
- `all_completed_actual_run_rows.csv`：原行与明确命名的衍生字段；仅完整block。
- `capacity_comparator_differences.csv`：signed差值、四母源正/负/平，不声称显著性。
- `original_p2_static_internal_diagnostic.csv`、`p1_fixed_call_internal_diagnostic.csv`：旧静态max4/P1回放，明确不作完整方法主结果。
- `REPORT_MANIFEST.json`：输入hash、完成状态、异常、remaining-head0、协议时间与未知值规则。

报告明确：remaining-head0未验证；wholegraph与cap256搜索范围不同；扩展在原P2已开始后冻结、PLAN记载未查看VAL/TEST结果，不暗称更早预注册；CHILS引用SEA2025正式论文，HiGHS仅按官方软件文档作通用MILP参照，没有SciPy1.10.1 solver内x0 warm start。
# 结果用途与方法可见名称更新

本工具产出三档短预算的效率与压力诊断；完整求解质量主结果使用 `quality_comparison_report.py`，位于 `reports/JOINTRECOVERY_QUALITY_RESULTS`。本文方法 `FullCapacity` 显示 JointRecovery（JR）；P1 / FullP1 为独立替代估计对照。raw keys 与统计保持不变，CSV 增 method_display / method_role 列，旁边 README_METHODS_ZH.md 给完整对照。以下原接口契约仍用于这些短预算诊断。
