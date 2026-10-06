# JointRecovery 定制STK数据集与实验入口

本目录保存依据 `JointRecovery_STK_Priority_Plan.docx` 新建的星地链路仿真数据。原始几何、完整窗口、可编辑STK场景、冲突图、来源划分和后续实际执行记录均归档在这里。数据来自本次独立STK运行，没有使用旧接触CSV填补或平移窗口。

**12个母机会库及72张派生图已全部完成；共有250,782条完整规划接触、868条跨界接触。** TRAIN/VALIDATION/TEST分别为6/2/4个物理来源。统计是母库层面去重后的窗口数量，不能把6张派生图中的重复窗口当新增物理观察。

TRAIN的6个母库含125,427条完整接触和36张图；10/50/200/1000毫秒四档恢复探针标签、四次40轮拟合、开发端固定调用对照及72项成本校准均已完成。原静态max4验证216/测试432单元已全部下载。原三档短预算保留为压力诊断；新的主结果使用充裕求解时间，并独立实测多个时间点。

原 `p1_fit → p1_followup → p2_confirmation` 流程已完成，记录见[执行状态](execution/PIPELINE_ORCHESTRATION.json)。完整框架短预算比较 `submitted_comparison` 已完成864/864并完整下载，补充报告和图已生成；新的 `quality_development` 与 `quality_comparison` 均已完成并完整下载；主质量矩阵验证480、独立测试960，共1440个真实单元全部封存。保持现有检查点、图和算法参数，不重训、不重新采标签、不铺P3。

先读[数据交付说明](reports/DATASET_DELIVERY_ZH.md)。各母库路径、图节点/边数、冻结SHA和字段定义见[数据目录与数据字典](reports/DATASET_CATALOG_ZH.md)。几何、建图、开发、独立验证和测试均已完成；数据准备和实际算法评价分别保留完成记录。

## 按用途选择文件

| 需要做什么 | 文件入口 | 用法 |
|---|---|---|
| 核对新方案和完整参数 | `protocol/` | 原Word、`JointRecovery_parameters.json`与预先固定的契约 |
| 使用完整物理机会 | `raw_geometry/JR-DUAL-rNNN/contacts.csv` | 每行一个完整规划窗口；按原始相对端点计奖 |
| 查跨界和原始报告 | 同目录`boundary_contacts.csv`、`access_raw.csv`、`raw_access_reports.jsonl` | 跨界窗口单列保留；padding内原始报告用于追溯，不直接当完整规划数据 |
| 在STK编辑或重算 | 同目录`scene/JR-DUAL-rNNN.sc`及整个`scene/` | 一并保留全部卫星、站点和其他依赖，不只拷贝`.sc` |
| 直接运行图算法 | `graphs/JR-DUAL-rNNN-R12-g0170.npz`等 | `.npz`是数组；同名`.json`是统计、来源和输入SHA |
| 查实际轨道与设置 | 各母库`actual_orbit_elements.json`、`actual_station_settings.json`、`access_settings_actual.json`、`manifest.json` | 以实际应用和成功运行收据为准 |
| 查看探针/训练/分配结果 | `execution/`及`reports/` | 执行记录与分析报告分开；失败、无增益和迟到保留 |
| 复用工具 | `code/` | STK生成、建图、探针、阶段推进、目录更新及结果报告 |

命名示例：`JR-DUAL-r004-R8-g0340`表示双壳层物理实现r004、固定8站子网、地面间隔340秒。每母库派生R8/R12 × gap170/340/680秒的6张图。

## 固定物理模型

168颗卫星分属两壳层：A壳84颗，名义高度550km、倾角53°、7面×12星、Walker f=2；B壳84颗，名义高度1100km、倾角70°、7面×12星、f=4。各来源的实际168组初始元素已在参数JSON预先列明，分别重新传播；轨道历元与72小时规划起点一致。

R12为纬度28°/32°与东经88°/94°/100°/106°/112°/118°的笛卡尔积，WGS84椭球高0米。每站1副天线，每星1个同时通信通道。R8固定保留GS01、GS03、GS04、GS06、GS07、GS09、GS10、GS12，是同母库资源供给对照，不是额外独立物理来源。

STK11.2采用J2传播、初始J2000、最低仰角15°、地球遮挡与几何可见性，关闭地形和光行时。规划72小时，两端各额外传播1小时。Access精确事件开启、最大采样步长30秒、时间收敛0.001秒；每母库固定12对使用15秒步长复查。未加入天气、RF链路预算、业务任务或人为最小时长筛选。

目录名中的20261005是本次生成日期。r000规划历元为2026-11-01 UTC，后续每个来源递增7天，r011为2027-01-17 UTC；各自规划72小时。

## 权重、边和资源语义

每个顶点是一条完整可见区间`[s,e)`，选择即占用整个窗口，唯一收益`w=e−s`。不切片、不平移部分窗口、不统一时长、不添加业务权重。只有完整落入规划时域的窗口进入图；跨界记录保留原端点并另存，不裁短。

将u/v按起点排序。同一天线在`sv < eu + τg`时冲突，同一卫星在`sv < eu + τs`时冲突，两类取并集。包含、同起点以及实际重叠均冲突；恰好满足间隔等号时兼容。`τs=150秒`，`τg∈{170,340,680}秒`。

`owner`/`agents`是卫星身份，不能额外规定每颗卫星全天只能选一个窗口。`vertex_factors`和资源CSR记录天线/卫星归属，成员列表不是全天互斥团；实际互斥以冲突边为准。

CSV相对秒是权威端点，保存9位小数；原始STK Double报告单独保留。`weights`单位仍为秒，图的整数纳秒端点用于原精度建边，微秒tick供求解器序列化。打印精度和tick不代表物理精度。UTC字段是辅助毫秒显示，不能代替相对秒建边或计奖。

## 划分与相关性

| 来源 | 协议划分 | 最小P1中的用途 |
|---|---|---|
| r000–r003 | TRAIN | 拟合；含r000/r001已经曝光的P0开发来源 |
| r004–r005 | TRAIN | 同物理族内部开发holdout和完整调用成本校准，不能称最终validation |
| r006–r007 | VALIDATION | 冻结验证，不用来选择有利场景或事后增加deadline |
| r008–r011 | TEST | 预先固定的新来源确认，不按机会或胜负筛选 |

同一`JR-SOURCE-rNNN`的站网、gap、初态、恢复域和请求始终同组进入训练/验证/测试。不要随机切分窗口或把R8放训练、R12放测试。统计以物理来源为单位；更多配置、状态、重复或拟合种子不会增加独立物理来源数。不同日期和偏置也不自动证明来源之间完全独立。

## 最短使用示例

直接读取已完成图，无需先打开STK：

```python
from pathlib import Path
import numpy as np

root = Path(r"E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005")
with np.load(root / "graphs/JR-DUAL-r000-R12-g0170.npz", allow_pickle=False) as graph:
    weights = graph["weights"]          # 完整接触时长，秒
    edges = graph["edges"]              # 去重无向冲突边，u < v
    owner = graph["owner"]              # 卫星索引，不是全天团
    contact_ids = graph["contact_id"]    # 回溯完整原窗口
```

最终调度应以同一原图顶点编号返回，验证独立集后按原`weights`重新计分。局部恢复还必须和完整固定base联合可行；局部解可行不等于完整调度可行。

在STK打开场景时，完整复制某母库的`scene/`。若STK11自动化接口无法处理中文路径，可整体复制到ASCII路径后打开`.sc`。本次使用独占ASCII技术桥接及重载验证，最终场景仍完整保存在本目录。

## 数据与实验报告分开阅读

[P1开发阶段签收](reports/P1_DEVELOPMENT_SIGNOFF_ZH.md)解释各档、调用数、种子及来源差异，并保留不利结果；其结论仅限内部开发，不能替代P2独立确认。

[原P1/P2结果与图表](reports/P1_RESULTS/P1_RESULTS_ZH.md)已包含四次拟合、固定调用数对照、开发成本标定及原静态max4独立验证/测试。该内部诊断报告不能代表完整JointRecovery的投稿主结果。

[四档标签特征](reports/P1_LABEL_CHARACTERISTICS/P1_LABEL_CHARACTERISTICS_ZH.md)汇总已完成的6个TRAIN母源、432个状态和38,016条实际请求；它描述cold-clone标签，不代替学习模型或实际完整D评价。

[P0签收](reports/P0_PHASE_SIGNOFF_ZH.md)、[P0完整分析](reports/p0_analysis/P0_ANALYSIS_ZH.md)、[真实结构案例](reports/P0_STRUCTURAL_CASES/README_ZH.md)和`reports/P0_FIGURES/`保存恢复机会与原始发现。P1固定调用数标签回放、真实完整D调用及模型拟合分别留有独立记录，不能相互替代，也不能把共同贪心收益归给学习策略。代码存在不代表相应实验已执行，以阶段完成收据和结果报告为准。

P0签收是保留原hash的阶段历史记录，其中“当时r006–r011尚未生成”等文字描述签收时点，不是当前状态。原签收正文不回写；当前12/72完成情况以本入口、最新目录和各母库manifest为准。

刷新目录可运行`python code/dataset_catalog.py --dataset-root <本目录绝对路径>`。工具只索引小型manifest和图JSON，复用冻结SHA，不启动STK、不重算Access、不重复扫描大文件。已完成母库和图不需为更新说明重复生成。

## 2026-10-06：当前方法名称与质量主实验

**本文方法是 JointRecovery（JR），原始执行键 `FullCapacity`。** `FullCheapSummary` 是简化表示消融，`FullGreedy` 是贪心排序内部对照，`FullP1` 是独立替代估计内部对照；P1不是本文方法名称。CHILS-p1（SEA 2025）是公开整图MWIS方法，HiGHS-MILP是通用MILP求解器。两个学习种子分别报告并汇总，不能称八个独立算法。详见[方法名称和时间实验说明](reports/METHOD_NAMES_AND_QUALITY_EVALUATION_ZH.md)。

原静态max4结果已下载至 `execution/p2_confirmation/out`，作为内部机制诊断保存。完整框架的三档短预算 `submitted_comparison` 已完成验证288/测试576，共864个实际单元，完整下载至 `execution/submitted_comparison/out`；[短预算报告](reports/SUBMITTED_METHOD_RESULTS/SUBMITTED_METHOD_RESULTS_ZH.md)和 `reports/SUBMITTED_METHOD_RESULTS/figures/submitted_comparison_2x2.png` 已生成，作为效率与压力补充分析。不能以这些短预算结果替代新的主300秒质量结论。已完成的[短预算中文简报](reports/SUBMITTED_METHOD_RESULTS/SHORT_EFFICIENCY_SIGNOFF_ZH.md)明确：JR三档严格交付收益均低于Greedy-rank+CHILS、Independent-replacement+CHILS与公开CHILS；仅wide档高于JR-CheapSummary。此压力结果没有解决旧性能差距，保留迟到、失败与共享前缀/native收益的归因限制，其压力结果与已经完成的300秒主质量结果分开报告。

新的开发任务 `quality_development` 已完成并下载（CPU42）：固定r004全部六图×八配置的48个基础单元，共52次尝试；46个60秒完成、两个HiGHS配置在300秒仍未准时产生要求的有效原生输出。按完成性规则冻结主表D=300秒，曲线为10/30/60/120/300秒，验证480、测试960单元。全部未完成/N/A与尝试保留，不据此判定JR获胜；详见[开发完成性签收](reports/QUALITY_DEVELOPMENT_COMPLETION_ZH.md)。remaining-head仍固定0，不能声称该学习分量已经得到验证。

验证和测试时间曲线使用独立实跑的10、30、60、120、300秒五个点，主表300秒。验证12图×8配置×5点=480单元，测试24图×8配置×5点=960单元；各点都记录完整原图目标值、相对共同初态增益、实际caller时间、完成率、无输出与迟到情况。外部原S保底不能冒充公开求解器新解，质量缺失保持N/A。

为使用服务器加速，独立执行绑定允许不同物理来源在40/44/48/52独立CPU并发；每个来源内的所有图、方法与时间点仍串行，每方法固定单CPU，先完成两个验证来源再开启四个测试来源。主矩阵任务 `quality_comparison` 已在原短预算任务真实结束且DEV冻结后自动接续（见 `execution/LAUNCH_quality_comparison.json`），两个验证来源r006/r007已于2026-10-05 20:04 UTC全部完成480单元；四个独立TEST来源r008–r011在CPU40/44/48/52完成全部960单元，主矩阵于2026-10-05 21:49 UTC结束。完整结果已下载至 `execution/quality_comparison/out`，1440行均已封存后才用于主报告。此调度修订单独记录在 `protocol/QUALITY_SOURCE_EXECUTION_BINDING_20261006.json`，不回写已经冻结的科学方案。

**独立TEST主结论：JointRecovery仍未解决旧版不及公开CHILS的性能差距。** 在统一300秒允许预算下，JR两种子平均相对同一初态S增加2667.282秒（按四物理来源等权，平均相对增益0.259%），JR-CheapSummary为2656.301秒、Greedy-rank+CHILS为2655.973秒、Independent-replacement+CHILS为1823.578秒，公开CHILS-p1为89277.224秒。JR平均仅比强贪心多11.308秒、比简化表示多10.981秒；seed17与强贪心持平，seed29多22.617秒，不能宣称稳定或显著优势。

JR两seed各24/24项完整产出，平均实际caller耗时3.255秒；CHILS为24/24项、210.183秒。允许上限和实际耗时分开，JR受最多8请求和R256局部恢复域限制，延长上限没有让它自动扩大搜索。HiGHS为15/24项完整产出，其总体质量按全预定矩阵标N/A，不能用剩余失败填零，也不能据此宣称JR击败HiGHS。共同prefix和native收益不归为学习独有，remaining-head固定0的限制仍保留。

首读[最终中文结果简报](reports/JOINTRECOVERY_QUALITY_RESULTS/FINAL_QUALITY_SIGNOFF_ZH.md)，完整表格与比较见[JointRecovery主质量报告](reports/JOINTRECOVERY_QUALITY_RESULTS/JOINTRECOVERY_QUALITY_RESULTS_ZH.md)。所有五个独立实际时间点（10/30/60/120/300秒）见[统一质量与时间曲线PDF](reports/JOINTRECOVERY_QUALITY_FIGURES/test/quality_curves_test.pdf)、[PNG](reports/JOINTRECOVERY_QUALITY_FIGURES/test/quality_curves_test.png)和[图示说明](reports/JOINTRECOVERY_QUALITY_FIGURES/test/QUALITY_FIGURES_ZH.md)。完整汇总见[逐预算/种子表](reports/JOINTRECOVERY_QUALITY_RESULTS/quality_source_equal.csv)、[逐物理来源表](reports/JOINTRECOVERY_QUALITY_RESULTS/quality_per_physical_source.csv)、[逐图配置表](reports/JOINTRECOVERY_QUALITY_RESULTS/quality_per_graph_configuration.csv)；1440个原始评价行在同目录CSV与 `execution/quality_comparison/out` 留存。原P1/P2和短预算补充均继续保留。

CHILS新整图数值门真实48/48通过，3高总权重fixture+1开发图native检查已下载至 `execution/chils_wholegraph_gate/out`，详见[数值执行范围](reports/CHILS_WHOLEGRAPH_NUMERIC_SCOPE_ZH.md)。数值安全检查不代表算法性能优势。


最终交付记录：主TEST表格与结论的一次必要检查、主质量图与短预算图的实际可读性检查均已完成；文字重叠/裁切已修复，两组图的六方法配色与标记一致，JR使用暖橙色突出，输入与统计未改。见[图表修复与统一风格收据](reports/FIGURE_LAYOUT_STYLE_QA_ZH.md)。主报告、曲线、所有原内部和短预算补充均已本地归档，`jointrecovery`自动跟进现已按完成后停止要求暂停。没有追加实验或修改旧论文。
