# 新 STK 科学材料：阅读与复现指南

本指南说明既有公开仓库 [Mister-Ryder/AAMAS2027_JointRecovery](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery) 新增目录 `stk/JointRecovery_STK_20261005` 的本次公开材料：**新 STK 数据、冻结科学代码、模型与实验报告**，没有修改旧论文。对应 Release tag 为 `stk-jointrecovery-20261006`；实际上传状态、可用资产和校验和以发布收据及 [该 Release 页面](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/tag/stk-jointrecovery-20261006) 为准。本文档说明材料组织与使用方法，本身不宣称上传已完成。

## 先读结果，再按需要复现

首读 [最终质量简报](JOINTRECOVERY_QUALITY_RESULTS/FINAL_QUALITY_SIGNOFF_ZH.md) 和 [完整质量报告](JOINTRECOVERY_QUALITY_RESULTS/JOINTRECOVERY_QUALITY_RESULTS_ZH.md)，再看 [五点质量/时间曲线 PDF](JOINTRECOVERY_QUALITY_FIGURES/test/quality_curves_test.pdf)。数据定义见 [目录字典](DATASET_CATALOG_ZH.md)、[数据交付说明](DATASET_DELIVERY_ZH.md) 和 [入口 README](../README_zh.md)。

当前已完成 12 个物理来源、72 张配置图；主质量验证 480、测试 960 个真实单元，共 1440 个。允许时间 10/30/60/120/300 秒的每个点分别实际运行，主表为开发完成性规则选定的 300 秒。**JointRecovery（JR，原键 FullCapacity）没有超过公开 CHILS-p1。** HiGHS 无完整产出的单元保留 N/A，不能由此宣称 JR 获胜。remaining-head 固定 0，未验证该学习分量；8 个 raw 配置包含拟合种子、消融和内部对照，不是 8 个独立发表算法。

原静态 max4/P1 回放只作内部诊断；[短预算报告](SUBMITTED_METHOD_RESULTS/SUBMITTED_METHOD_RESULTS_ZH.md) 及其 [简报](SUBMITTED_METHOD_RESULTS/SHORT_EFFICIENCY_SIGNOFF_ZH.md) 保留效率压力结果，不替代主质量结论。所有图的 min–max 是两个 fit seed 的观测范围，不是置信区间；来源内配置、请求和状态不是额外独立物理样本。详细方法名称与预算修订见 [协议说明](METHOD_NAMES_AND_QUALITY_EVALUATION_ZH.md)。

## Release 解压位置

克隆仓库后，把 `stk/JointRecovery_STK_20261005` 记为 **ROOT**。Git 直接提供 code、protocol、reports、72张图、contacts、四个 final checkpoint、冻结科学运行时与 vendor CHILS 源码；Release 补充较大的原始几何和全量执行证据。只下载 Git 源码压缩包，不包含这些大型 Release 资产。

下载该 tag 的十个 ZIP，按发布清单校验 SHA256。**所有 ZIP 内路径均相对于 dataset 根，全部直接在 ROOT 解压并合并目录树；不得再创建一层同名 dataset 目录。** 原始记录文件保留发布内容，确切大小以发布清单为准：

| ZIP | 内容 |
|---|---|
| `01-dataset-r000-r003.zip` | r000–r003 数据来源 |
| `02-dataset-r004-r007.zip` | r004–r007 数据来源 |
| `03-dataset-r008-r011.zip` | r008–r011 数据来源 |
| `04-p0-results.zip` | P0 探针结果与证据 |
| `05-train-labels.zip` | 冻结训练/开发标签 |
| `06-fit-models-and-features.zip` | 模型、训练记录与特征 |
| `07-p1-followup-p2-results.zip` | P1 follow-up 与原 P2 |
| `08-short-comparison-and-quality-development.zip` | 短预算比较与质量开发 |
| `09-quality-comparison-results.zip` | 质量验证/测试完整执行结果 |
| `10-code-protocol-reports-frozen-sources.zip` | 代码、协议、报告与冻结源码 |

恢复后的关键结构应为：

```text
ROOT/
  README_zh.md
  code/                         # 报告、绘图、STK与执行入口
  protocol/                     # 原参数与冻结计划
  graphs/                       # 72组 NPZ + JSON
  raw_geometry/JR-DUAL-rNNN/     # 完整contacts、原报告、整个scene目录
  execution/
    p1_fit/out/                  # 四个final.pt、history、protocol/completion
    p1_followup/out/fixed_calls/
    p1_followup/out/calibration/
    p2_confirmation/out/validation/ 及 test/
    submitted_comparison/out/validation/ 及 test/
    quality_development/out/
    quality_comparison/out/      # selected_allowance、validation/test、shards
    chils_wholegraph_gate/out/   # 原版CHILS完整图数值收据
  reports/
```

原运行 JSON 中的绝对路径是历史追溯信息，不能要求读者拥有原机器目录，也不要为迁移而修改冻结 JSON、权重、checkpoint 或 SHA。下列报告命令显式指定当前本地根；模型程序按当前 `fit-root/<variant>-seedNN/final.pt` 找文件并核对原 SHA。保留完整相对树即可重新读取现有结果。

## 仅重生成已有报告与图

下面命令从 ROOT 运行。它们读取已封存的本地结果，不启动 STK、训练或 native 求解。主要依赖 Python、NumPy、Matplotlib；无需为这一步安装 PyTorch、STK 或 CHILS。生成时间戳、本地路径和字体/渲染版本可能不同，不能要求新 PNG 与旧 PNG 字节完全一致；实验输入与数值由 manifest 追溯。

P1 工具的输出固定为 `reports/P1_RESULTS`，没有 `--out`。如需原发布报告保持原样，先在 ROOT 的副本上重生成。**须显式给五个输入根**，避免自动发现把其它阶段混入：一个 fit、一个 fixed-call allocation、三个实际 benchmark 根。

```text
python code/p1_results_report.py --dataset-root . --fit-root execution/p1_fit/out --allocation-out execution/p1_followup/out/fixed_calls --benchmark-out execution/p1_followup/out/calibration --benchmark-out execution/p2_confirmation/out/validation --benchmark-out execution/p2_confirmation/out/test
```

原短预算报告必须绑定原 P2 和原校准预算 `frozen_budgets.json`，不能替换为新的 selected_allowance/300 秒协议。下例写入新报告目录：

```text
python code/submitted_comparison_report.py --dataset-root . --comparison-root execution/submitted_comparison/out --original-p2-root execution/p2_confirmation/out --p1-root execution/p1_followup/out --budgets-file execution/p1_followup/out/calibration/frozen_budgets.json --out reports/REGENERATED_SUBMITTED_METHOD_RESULTS
python code/submitted_comparison_figures.py --results-root reports/REGENERATED_SUBMITTED_METHOD_RESULTS
```

主质量报告及独立 TEST 图：

```text
python code/quality_comparison_report.py --dataset-root . --comparison-root execution/quality_comparison/out --out reports/REGENERATED_QUALITY_RESULTS
python code/quality_comparison_figures.py --report-root reports/REGENERATED_QUALITY_RESULTS --out reports/REGENERATED_QUALITY_FIGURES/test --phase test
```

质量工具默认读取 comparison 根内 `selected_allowance.json`；若另有封存副本，可显式使用 `--allowance-file <本地文件>`。绘图可改为 `--phase validation`，或使用 `--metric gain_seconds` / `--metric objective_seconds`；默认是相对共同初态的 `gain_percent`。输出 PDF/PNG、图用数据 JSON、输入/生成器/输出 SHA 和说明。不完整输入返回 WAITING；缺 native 完整质量保持断点，不填零，不只对成功样本求均值。

接口详情见 [P1契约](../code/P1_CONTRACT.md)、[短预算报告契约](../code/SUBMITTED_REPORT_CONTRACT.md)、[主质量报告契约](../code/QUALITY_REPORT_CONTRACT.md)。原图修复与统一风格收据见 [绘图 QA](FIGURE_LAYOUT_STYLE_QA_ZH.md)。

## Windows STK 11 数据构建

读现有 NPZ/CSV 和重生成报告无需 STK。若重做物理几何，最低依赖为：安装并可正常授权使用的 **STK 11（原记录为 11.2）**、已注册 `STK11.Application` COM、匹配的 Windows Python 与 `pywin32`。`stk_geometry.py` 实际使用 `win32com.client`/`gencache`，不使用 STK12 API。完整 STK 安装应包含脚本引用的默认安装目录 `C:\Program Files\AGI\STK 11\` 下 ObjectModel XML 与 Earth 中央天体文件。建图另需 NumPy。

新建独立的 REBUILD_ROOT，先复制本发布的 `code/` 与 `protocol/`，不要在已发布原几何目录中覆盖。从 REBUILD_ROOT 运行完整 12 个来源：

```text
python code/stk_geometry.py --parameters protocol/JointRecovery_parameters.json --output-root . --replicates r000 r001 r002 r003 r004 r005 r006 r007 r008 r009 r010 r011
python code/build_graphs.py --dataset-root . --replicates r000 r001 r002 r003 r004 r005 r006 r007 r008 r009 r010 r011
```

生成器需要输出根已存在。STK 场景须保留整个 `scene/`，不只取 `.sc`；中文路径的原生保存/重载使用脚本现有 ASCII 技术桥接。参数、实际轨道、站点、Access 设置及原始 provider 报告均随几何归档。权重为完整窗口原时长，owner 为卫星身份，资源成员不是全天 clique。原始相对端点是计奖/建边依据；求解器微秒整数化不代表物理精度。

若只想从已发布 `raw_geometry` 重建冲突图，可跳过 STK 命令，在完整 ROOT 副本上调用 `build_graphs.py`。不同 STK build、事件求解和环境的物理重算不保证与冻结 CSV 位级相同，报告原版本与新增执行收据，不把新数据覆盖成旧原始结果。

## Linux 重新执行科学算法

重新训练/求解与上面的“读旧结果”不同，需要完整冻结运行时（含 `src/joint_recovery`、`experiments`）、checkpoint、NumPy/PyTorch，以及公开求解器依赖。`--runtime-root` 指向包含上述目录的运行时根，不是 STK 数据 ROOT，也不能随意用已修改的代码代替；源码 SHA 与 fit protocol 会核对。

已经记录的版本：

| 项目 | 实际记录 | 收据位置 |
|---|---|---|
| Python | 3.8.10，Linux公开基线执行 | `execution/quality_comparison/out/shards/r011/test/runs/JR-DUAL-r011-R12-g0170__quality-10s__HiGHS-MILP/actual_policy.json` 的 `python_version` |
| NumPy | 1.22.4，公开基线执行 | 同文件 `solver_source_and_binary.numpy_version` |
| SciPy | 1.10.1；现 HiGHS 接口明确要求此版本 | 同文件 `solver_source_and_binary.scipy_version`，同时保留 native HiGHS 库 SHA |
| PyTorch | 1.11.0+cu113；原拟合使用 CUDA | `execution/p1_fit/out/environment.json`；完整评价加载 checkpoint 到 CPU并使用单CPU |
| Matplotlib | 按实际图生成收据取值 | `reports/JOINTRECOVERY_QUALITY_FIGURES/test/QUALITY_FIGURE_MANIFEST.json` 的 `matplotlib_version`；不将其当训练环境版本 |

上述版本来自已经封存的相应阶段，不证明所有机器/阶段的全部包相同。未在收据中出现的系统库、编译器及其它包版本应注明未知或由 Release 的环境清单提供，不能用当前电脑的版本补造历史环境。

CHILS 必须使用[官方实现](https://github.com/KarlsruheMIS/CHILS) commit **`515952724cd3dcc6c4365a340ecf0f1da782119a`**，正式方法出处为 [SEA 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.22)。需要 Linux、C/OpenMP 工具链与原版 `-p1 -c1 -r17`。原执行二进制 SHA 为 `19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0`；原收据绑定完整 src/include 文件集合、二进制、ABI、NPZ 和数值域。

**仅检出同一 commit 不保证换编译器后得到相同二进制。** 若 Release 资产包含已记录原生二进制，按资产清单取得并核对；若没有，复现实验执行者需取得合法原版构建与相应身份记录，不能把任意 fresh build 宣称通过既有 gate，或删除 SHA 检查。完整图数值收据位于 `execution/chils_wholegraph_gate/out/{source_proof,guard,completion}.json` 及 `gates/*.json`；没有凭旧 cap256 证书放行整图。SciPy/HiGHS 是通用 MILP 外部参照，其 API 没有内部 x0 warm start；共同 S 仅作为外部保底。

四个 final epoch40 checkpoint 位于：

```text
execution/p1_fit/out/ResidualCapacity-seed17/final.pt
execution/p1_fit/out/ResidualCapacity-seed29/final.pt
execution/p1_fit/out/ResidualCheapSummary-seed17/final.pt
execution/p1_fit/out/ResidualCheapSummary-seed29/final.pt
```

`history.json`、fit `protocol.json`、`completion.json` 与检查点一并保留；不能只上传 pt 后丢掉模型/归一化/训练来源绑定。完整框架接口见 [FULL_CONTROLLER_CONTRACT.md](../code/FULL_CONTROLLER_CONTRACT.md)。例如另建输出目录的一次完整方法执行接口如下，尖括号路径必须换成当前完整环境：

```text
python code/full_joint_recovery_deadline.py --runtime-root <FROZEN_RUNTIME_ROOT> --fit-root execution/p1_fit/out --graph graphs/JR-DUAL-r008-R12-g0170.npz --chils <ORIGINAL_CHILS_BINARY> --chils-source <ORIGINAL_CHILS_SOURCE> --policy FullCapacity --fit-seed 17 --action-seed 17 --budgets-ms 10 50 200 1000 --max-calls 8 --deadline-seconds 300 --cpu-index 0 --out execution/NEW_REPRODUCTION/full_one
```

`--cpu-index` 是当前允许 affinity 集合内的下标，并非直接 OS CPU编号。新执行必须用全新输出、单方法单CPU，记录自己的硬件/affinity与完整 caller 成本；结果可能受 native soft-time 和机器速度影响，不能复制旧收益作为新跑值。完整多时间点入口为 `quality_method_comparison_pipeline.py dev|evaluate`，其真实必需路径参数包括 `--dataset-root --runtime-root --fit-root --numeric-root --quality-plan --chils --chils-source --out`；evaluate 另须 `--original-comparison-root --development-root`。使用冻结协议、所有固定图/方法/时间点与相同初态，不根据胜负删数据，详见代码和 [质量计划](../protocol/QUALITY_EVALUATION_PLAN_20261006.json)。本指南编写期间没有启动这些计算。

## 结果、曲线与元数据追溯

主质量原矩阵在 `execution/quality_comparison/out/{validation,test}/actual_run_rows.json/.csv`，分来源执行材料在 `shards/`，整个完成状态见 `summary.json`/`completion.json`，统一允许时间与曲线档位见 `selected_allowance.json`。完整报告的 `quality_source_equal.csv`、`quality_per_physical_source.csv`、`quality_per_graph_configuration.csv` 和 `jointrecovery_quality_comparator_differences.csv` 保留各预算、种子、来源及 signed 差值。

质量曲线的图用 CSV 来源和 SHA 在 `QUALITY_FIGURE_MANIFEST.json`，逐点可追溯数据在 `QUALITY_FIGURE_DATA.json`；短预算对应 `figures/FIGURE_MANIFEST.json`。原时长目标 ↑、增量 ↑、完整 caller 成本 ↓、完成率 ↑、失败/迟到/无输出 ↓ 分别解释，允许上限不等于实际耗时。晚到 raw 可行质量、外部 fallback S、合法无需 native 的空域完成均有独立字段，不能混同 native 求解成功。

本发布便于专家审阅真实新数据、代码、模型和当前性能限制，不承诺论文录用，也不把负面结果移出全部可追溯记录。
