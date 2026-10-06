# JointRecovery 完整求解质量报告契约

```text
python code/quality_comparison_report.py --comparison-root execution/quality_comparison/out --out reports/JOINTRECOVERY_QUALITY_RESULTS
```

路径相对数据集根目录解析。可用 `--allowance-file` 指定其他本地已封存 `selected_allowance.json`。不连接云、不训练、不调用算法，不覆盖原始实验结果。

先读质量余量协议和各块 `summary.json`。质量协议 schema 为 `joint_recovery_stk_quality_allowance_v1`，status 为 `QUALITY_ALLOWANCE_PROTOCOL`，stage 为 `frozen_from_development_completion`，development_sources 为 `[4]`，selection_uses_gain/test_outcomes_used 都必须为 false。selectedD 从 DEV 完成性 60→120→300 秒阶梯选定，实际曲线为 `evaluation_deadline_seconds=sorted(unique([10,30,60,120,selectedD]))`。

各块 summary 必须标记 `QUALITY_COMPARISON_MATRIX_COMPLETE`，本地实际行数与声明相同才读 `actual_run_rows.json`。验证集 r006/r007，测试集 r008–r011，每来源六配置、八原始策略，每曲线预算一个独立真实运行。通常 VAL384 / TEST768；selectedD=300 时 VAL480 / TEST960。未完整块 WAITING，不读 live 行。

输出：

- `JOINTRECOVERY_QUALITY_RESULTS_ZH.md`：主表只用 DEV 完成性冻结的 selectedD；原始完整目标秒↑、相对 S 增益秒↑及百分比↑，实际 caller 秒↓另表，严格按时收益为次要项。
- `quality_source_equal.csv`：所有真实曲线点，先图内两 fit 等权（均值行）、每来源六配置等权，再物理来源等权。
- `quality_per_physical_source.csv`、`quality_per_graph_configuration.csv`：同样全部预算，保留来源和图。
- `all_completed_quality_run_rows.csv`：所有封存行及衍生列，失败、无 native、迟到、零增益保留。
- `jointrecovery_quality_comparator_differences.csv`：逐预算 JointRecovery seed17/29/均值对全部对照完整目标差；缺少任一完整质量则 N/A。
- `QUALITY_REPORT_MANIFEST.json` 和 `README_METHODS_ZH.md`：状态、输入哈希、方法原键与显示名对照。

CSV 原始 `policy_label` 不改变。可见列为 `method_display`、`method_display_en`、`method_role`；差值行还有 `target_method_display` 和 `comparator_method_display`。本文方法原键 `FullCapacity`，显示 **JointRecovery（JR）**；旧 `Capacity` 仅静态 max4 内部诊断，`FullP1` / `P1` 是 Independent-replacement+CHILS 对照。

绘图主 CSV 的分组键：`phase`、`deadline_seconds`、`policy_label`；`budget_id` 作为报告归类，不用其拼法区分时间，`is_primary_quality_allowance` 标记主表点。指标：`initial_S_seconds`、`complete_objective_seconds`、`complete_gain_seconds`、`complete_gain_percent`、`caller_seconds`、`complete_result_rate`、`native_output_rate`、`external_fallback_only_rate`、`legit_no_native_required_rate`、`on_time_return_rate`、`observed_late_rate`、`strict_gain_seconds_secondary`。各完整质量和 caller 指标都有 `_fit_seed_min` / `_fit_seed_max`（两种子范围，非 CI）。状态/计数为 `grid_complete`、`declared_cells`、`received_cells`、`complete_quality_cells`、`incomplete_quality_cells`、`no_native_output_cells`、`observed_late_cells`、`submission_failure_cells`。

实际完整质量使用父进程收到的原图完整可行向量，须程序正常 finished、身份/完整 mask 有效、实际按时返回。公开算法须实际原生完整输出；Full 方法须实际父验证 native 响应，或者 `legit_no_native_required=true` 的全空/无请求恢复域合法收据。HiGHS 只有外部 S 保底时不能算求解成功。原始 `returned_value_seconds` 保留诊断，不能填 `complete_objective_seconds`。任何预定质量缺失都会使对应整体均值 N/A，不删失败仅对成功样本求均值。曲线 N/A 可以留断点，不能补0或让整个图消失。

`native_output_rate` 是已验证原生响应率（Full 使用 `observed_verified_native_responses>0`，公开使用 `native_complete_output_produced`），不是尝试调用次数。`legit_no_native_required_rate` 另列；没有原生响应且非合法无需 native 的可用父返回记 `external_fallback_only_rate`，属于保底交付诊断，不能当 native 求解成功或本文学习优势。

图输入完成条件：manifest `status=COMPLETED_QUALITY_TEST_REPORT`、`completed_block_statuses.test.status=COMPLETED`；即封存矩阵和身份协议完整，不要求每行算法成功。完整质量值为 `null` 时保留缺点；失败率/无输出率/迟到率仍可绘真实分母。没有启动任何实验，仅做必要语法与字段检查。
