# 新STK数据与实验公开发布内容建议

盘点只读取本地文件大小、少量完成签收及源码包成员目录，未重跑实验、未连接云服务器、未删除或改动原始内容。Git目标为 `Mister-Ryder/AAMAS2027_JointRecovery`。文件大小是未压缩值；Release实际压缩大小需打包后确定。

当前快照共有 **21636 个文件，10.312 GB**。12份母源的原窗口与可编辑STK场景均在本地，72份NPZ图及72份元数据均在本地；四个最终checkpoint合计 **120860 bytes**。

| 一级目录 | 文件数 | 未压缩MB | 最大文件 |
|---|---:|---:|---|
| [ROOT_FILES] | 1 | 0.02 | README_zh.md (0.02 MB) |
| code | 72 | 1.64 | code/__pycache__/p1_results_report.cpython-313.pyc (0.06 MB) |
| execution | 16717 | 6999.38 | execution/p1_fit/out/feature_index.json (683.08 MB) |
| graphs | 144 | 193.60 | graphs/JR-DUAL-r004-R12-g0680.npz (3.43 MB) |
| protocol | 12 | 0.94 | protocol/JointRecovery_parameters.json (0.86 MB) |
| raw_geometry | 4599 | 3016.02 | raw_geometry/JR-DUAL-r010/access_raw.csv (6.50 MB) |
| reports | 91 | 100.68 | reports/P1_LABEL_CHARACTERISTICS/request_recorded_scalars.csv (44.48 MB) |

## Git直接提交

- 根README、`code/`（去除字节码）、`protocol/`、`reports/`和`graphs/`；72图总计约193.6 MB，最大图约3.43 MB，可直接提交，便于专家拿到实际算法输入。
- 四个 `execution/p1_fit/out/<variant-seed>/final.pt`、拟合protocol/completion和完整训练配置；原始checkpoint字节与签收中的SHA保持原义。
- 各阶段汇总与原行索引、collection/protocol/completion/selected_allowance、关键执行/来源绑定收据。完整trace另放Release，Git说明其原目录与资产对应关系，不能只给汇总表。
- 大量重复逐调用日志或临时native输入不必全部进入Git，但完整阶段Release仍保留可复核的原行、members、日志及trace。

## Release分包

按本次任务的限制，保守以Git单文件100,000,000 bytes、Release单资产2,000,000,000 bytes为守卫，建议资产目标不超过1.5 GB。不能把整个10.3 GB目录做成单资产；不得仅因放到Release而遗漏原始证据。

| 建议资产 | 未压缩MB | 内容 |
|---|---:|---|
| stk-geometry-r000.zip | 249.79 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r001.zip | 251.44 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r002.zip | 251.50 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r003.zip | 251.49 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r004.zip | 251.48 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r005.zip | 251.42 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r006.zip | 251.49 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r007.zip | 251.49 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r008.zip | 251.50 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r009.zip | 251.41 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r010.zip | 251.51 | original_windows_editable_STK_scene_all_dependencies_manifest |
| stk-geometry-r011.zip | 251.50 | original_windows_editable_STK_scene_all_dependencies_manifest |
| training-labels-r000.zip | 254.57 | all_four_workpoint_training_labels_original_masks_and_receipts |
| training-labels-r001.zip | 253.46 | all_four_workpoint_training_labels_original_masks_and_receipts |
| training-labels-r002.zip | 273.78 | all_four_workpoint_training_labels_original_masks_and_receipts |
| training-labels-r003.zip | 275.22 | all_four_workpoint_training_labels_original_masks_and_receipts |
| training-labels-r004.zip | 274.80 | all_four_workpoint_training_labels_original_masks_and_receipts |
| training-labels-r005.zip | 274.51 | all_four_workpoint_training_labels_original_masks_and_receipts |
| p0-pilot-four-workpoints.zip | 508.08 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| p1-fit.zip | 1549.31 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| p1-followup.zip | 60.58 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| p2-confirmation.zip | 329.91 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| submitted-comparison.zip | 773.24 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| quality-development.zip | 58.94 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| quality-comparison.zip | 1766.52 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| chils-wholegraph-gate.zip | 2.43 | full_stage_raw_rows_trace_logs_protocol_checkpoints_if_applicable |
| p0-original-probe-jobs.zip | 333.07 | original_P0_execution_results_including_failed_smoke_if_present |

12个 `stk-geometry-rNNN.zip` 每个约250 MB，包含该母源的contacts/boundary/access_raw/raw_access_reports、manifest、实际设置与完整 `scene/` 依赖；不能只上传`.sc`主文件。训练标签按r000–r005分6包，保留四工作点、所有重复执行与原始members/失败记录。P1 fit、P0、P1 followup、P2、短时限完整方法比较、质量DEV及主质量/曲线结果各自分包。若阶段压缩资产仍超过2 GB，则按物理source或原子子目录进一步拆包，不裁掉失败、未产出或迟到结果。

**唯一超过Git单文件保守上限的文件**是 `execution/p1_fit/out/feature_index.json`，683,078,812 bytes。它属于训练输入/特征索引，必须保留在fit Release或独立资产中；不能直接Git提交，不能当无关缓存丢弃。当前没有单文件达到2 GB，但阶段资产总和仍需分包。

## 科学源码闭合与可排除项

`ROOT/runtime/`尚未展开。最新候选源码包为 `execution/c7b484879387e6b5c99ac808795577a54e3bb956f0d31debd8bc0255b351609d.tar.gz`；它确实含 `runtime/src/joint_recovery/` 和 `runtime/experiments/`，并包含当前quality pipeline与source orchestrator。发布器应把机器清单中的科学成员按原成员路径展开：`runtime/src/...`仍放在该路径，`runtime/experiments/...`亦同。

P0直接依赖core、v4_budgeted_recovery、v4_factors、v4_neighborhoods与v4_residual_common；学习模型还依赖v4_residual_model、v4_factorized_model、v4_model、v4_model_fast及v4_factors_fast。保留runtime/src整体和列出的experiment科学模块可保证这层源码闭合；NumPy/PyTorch/SciPy1.10.1和实际CHILS源码/二进制身份另按环境与gate收据提供。

**12个capsule不能仅按“运输包”一概排除。** 可不发布tar容器本身，但必须先展开每个不同冻结阶段的科学source版本，生成“原始路径 → 实際字节SHA → 公开路径/版本”的映射；相同内容可以共享，原source_sha字段不得被改写成当前源码的SHA。此次不做全树hash，因此未宣称这些包字节重复。归档中不被当前科学闭合导入的旧论文analysis/figure/report脚本可不重复发布。

`__pycache__`与`.pyc/.pyo`可不发布（本地26文件、868993 bytes）；全部本地保留，不执行删除。不能用相同文件大小、名字或阶段数据交叉复用来推断原实验记录重复并删除。

## 当前结果覆盖与发布签收

- `p1_fit`: `ACTUAL_P1_FIT_COMPLETE`。
- `p1_followup`: `ALL_CORE_P1_DEVELOPMENT_COMPARISONS_COMPLETE`。
- `p2_confirmation`: `complete`，原行648/None。
- `submitted_comparison`: `complete`，原行864/864。
- `quality_development`: `QUALITY_DEVELOPMENT_COMPLETION_THRESHOLD_COMPLETE`。
- `quality_comparison`: `QUALITY_COMPARISON_MATRIX_COMPLETE`，原行1440/1440。
- `chils_wholegraph_gate`: `WHOLEGRAPH_NUMERIC_GUARD_PASS`。

当前质量比较已完整记录1,440格（VAL480、TEST960，10/30/60/120/300秒），其中算法未产出、失败、迟到或N/A仍须原样公开；矩阵收集完整不等于每个求解器都有完整解。发表内容应同时给主质量表、完整曲线、原始行与这些状态。

发布资产完成后，应生成相对路径清单、每资产实际压缩bytes及SHA，并从Git README链接Release资产，说明按原相对路径解包。此份盘点不代替上传签收；没有上传或发布操作。

## 本地原 CHILS 执行文件与依赖闭合补充

已在 `第一篇/tools/CHILS-v3/CHILS` 找到本次实验原 Linux ELF 执行文件（49,176 字节），SHA256 为 `19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0`，与冻结门控所用二进制一致。对应本地官方源码 commit 为 `515952724cd3dcc6c4365a340ecf0f1da782119a`。建议随可复现实验 Release 附带该执行文件、同目录官方 `LICENSE`、官方源 commit 收据及构建环境说明；压缩包应恢复 Linux 可执行权限。`chils_public_release` 中 48,392 字节的 CHILS 执行文件 SHA 不同，不能替代本次原二进制。

`REFERENCE_FILES` 是 `code/p0_recovery_probes.py` 中的路径元组，并非 tar 内的同名文件。其 5 个文件在 canonical capsule 中均按 `runtime/` 前缀存在，详见 JSON 的 `reference_files_from_p0_module`。展开时保留所有列出的文件、完整 `runtime/src` 与依赖的 `runtime/experiments`、模型导入；不要只复制入口脚本。
