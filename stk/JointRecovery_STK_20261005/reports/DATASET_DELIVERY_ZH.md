# JointRecovery 新STK数据交付说明

本说明面向数据使用者与专家审阅，说明文件入口、问题契约及来源分组。它是数据交付文档；恢复探针、拟合和算法优势结论分别保存在各阶段实验报告。

根目录：`E:/01-Joycecyq/2026-AAMAS/data/两篇论文数据定制化构建/JointRecovery_STK_20261005`。

## 已完成范围

**r000–r011共12个母机会库和72张派生图已全部完成。** 去重母库共有250,782条完整规划窗口、868条跨规划边界窗口。每条物理接触按母库计一次，不能在6张派生图中重复累加成新增观测。

各来源实际状态、路径、节点/边数和冻结SHA见[数据目录与数据字典](DATASET_CATALOG_ZH.md)及各母库成功`manifest.json`。12个来源均已实际重新传播；它们不再只是参数JSON中的计划数量。

| 来源块 | 已完成母库 / 图 | 用途 |
|---|---:|---|
| TRAIN r000–r005 | 6 / 36 | r000–r003拟合；r004/r005为TRAIN族内开发及成本校准 |
| VALIDATION r006–r007 | 2 / 12 | 冻结验证，不进入拟合 |
| TEST r008–r011 | 4 / 24 | 预先固定的新来源确认 |

数据来自本次STK实际运行的理想化几何输出，不是历史CSV变换或在轨业务观测。几何、建图和数据传输完成不代表所有VALIDATION/TEST算法评价已完成。`execution/HOLDOUT_DATA_PIPELINE.json`的`ALL_SIX_FIXED_HOLDOUT_SOURCES_AND_36_GRAPHS_READY_NOT_EVALUATED`明确区分准备就绪与评价完毕；真正评价状态以各阶段完成收据为准。

TRAIN的6个母库共有125,427条完整窗口、36张图，既定控制状态的10/50/200/1000毫秒四档实际探针标签已完整采齐。Capacity/CheapSummary各种子17/29、共4次P1拟合正在云端运行；这不是完成训练或证明策略优势的声明。

## 文件用途与追溯链

| 层级 | 文件 | 主要用途 |
|---|---|---|
| 预先固定协议 | `protocol/JointRecovery_parameters.json`、原Word与阶段契约 | 12个来源、各168组初始元素、站点、固定子网、时域及图规则 |
| 完整机会 | `raw_geometry/JR-DUAL-rNNN/contacts.csv` | 规划72小时内完整窗口，是建图的接触输入 |
| 跨界机会 | 同目录`boundary_contacts.csv` | 保留跨规划边界的原始端点，不裁短后加入图 |
| 原始Access | 同目录`access_raw.csv`、`raw_access_reports.jsonl` | padding内原始窗口、STK Double报告和真实设置；并非全部属于规划可选机会 |
| 实际应用设置 | 同目录`actual_orbit_elements.json`、`actual_station_settings.json`、`access_settings_actual.json` | 实际轨道、约束与Access设置，不用缺省推测填补 |
| 可编辑物理场景 | 同目录`scene/JR-DUAL-rNNN.sc`与整个`scene/` | STK11编辑及重新传播所需全部卫星、站点和场景依赖 |
| 图数组 | `graphs/JR-DUAL-rNNN-R8-g0340.npz`等 | 权重、冲突边、owner、资源因子、原接触映射与分组 |
| 数值/来源收据 | 同名图`.json`、各母库`manifest.json` | 图统计、来源、数值契约、输入/输出SHA及真实运行状态 |
| 实际执行 | `execution/`及阶段结果目录 | 状态、动作、完整base/R/warm/T、预算、失败与返回记录 |
| 分析和说明 | `reports/` | 字段定义、P0发现、结构案例、诊断图和后续模型/分配结果 |

一条算法结果可沿`执行收据 → graph_id/原顶点成员 → 图NPZ与JSON → mother_contact_index/contact_id → contacts.csv → 原始Access报告与STK场景`回溯。目录中的SHA复用已冻结收据，未为写文档重新扫描全部大文件。

## 物理与数值配置

每个来源有168颗卫星和12站，重新传播72小时。A壳84星：a=6928.137km、e=0、i=53°、7面×12星、f=2；B壳84星：a=7478.137km、e=0、i=70°、7面×12星、f=4。名义高度是相对于6378.137km参考半径的550km和1100km。RAAN与沿轨偏置在参数JSON中预先确定，与算法成绩无关。

R12是纬度28°/32°与东经88°/94°/100°/106°/112°/118°的笛卡尔积，按纬度/经度升序编号GS01–GS12，WGS84椭球高0米。R8固定保留GS01、GS03、GS04、GS06、GS07、GS09、GS10、GS12。每站单天线、每星单同时通信通道。

STK11.2.0采用J2传播和初始J2000，轨道epoch与规划起点相同，前后额外传播各1小时。最低仰角15°，保留地球遮挡，只做几何可见性；关闭地形和光行时，无RF、天气、光照、业务任务或人为最小时长筛选。实际传播参考系枚举、轨道和设置见每母库实际文件。

Access精确事件开启，最大步长30秒，时间收敛0.001秒。每母库12个预先固定对象对使用15秒最大步长复查，区间数与端点差的检查结果逐项保存。CSV9位相对秒、纳秒整数和微秒tick均是数值序列化，不能据此声称物理准确到纳秒或微秒。UTC仅保留STK默认毫秒显示；相对秒才是权威建边与计奖端点。

STK11中文保存路径限制通过独占ASCII临时路径解决，完整依赖最终复制到本data目录并在第二位置重载。`scene_save_bridge.json`记载此步骤。使用者如遇同类接口限制，可将整个`scene/`复制到ASCII路径后打开；不需因此改变轨道、窗口或图。

## 权重、冲突与owner

每个顶点对应完整可见区间`[s,e)`，选择即占用整个窗口，收益严格`w=e−s`。只有完整进入规划时域的窗口纳入图；跨界窗口另存原端点，不截短。没有切片、平移、统一时长、业务权重重赋或虚构替代组合。

u/v按起点排序，冲突为：

```text
same_antenna(u,v) and start(v) < end(u) + ground_gap
OR
same_satellite(u,v) and start(v) < end(u) + satellite_gap
```

地面gap为170/340/680秒，卫星gap固定150秒，两类边取并集。实际重叠、包含和同起点均冲突；恰好达到间隔等号则兼容。`edge_types`位1代表地面天线，位2代表卫星通道，3代表两类同时满足。

`owner`/`agents`对应卫星身份，不规定卫星全天只能选择一次。资源因子和CSR成员表示时间资源归属，不能直接扩成全天团。最终完整调度须在原图验证独立性后按原`weights`计奖；局部恢复还必须与整个固定base联合可行。

## 来源划分与相关性

`JR-SOURCE-rNNN`是来源组。该母库全部6图、站网视图、gap、控制状态、动作和恢复请求始终属于同一split。r000–r005为TRAIN，r006/r007为VALIDATION，r008–r011为TEST。最小P1只拟合r000–r003；r004/r005是TRAIN族内部开发holdout和统一完整返回预算校准，不能称最终validation。

R8/R12、不同gap和局部状态都是同源派生观察。不同日期与随机偏置本身也不是统计独立性的证明。不要随机拆分接触或请求，不要跨来源组泄漏母几何。汇总先对同来源的图/状态求均值，再对来源等权平均；更多状态、执行重复和拟合种子不增加独立物理来源数。四个TEST来源是小型确认规模，来源相关性与泛化范围应在报告中说明。

## 使用方式

算法研究可直接读取`graphs/*.npz`并读取同名JSON获取来源和SHA，不必安装STK。物理重算需要STK11、完整`scene/`与实际设置。复用时保留contact_id、mother_contact_index、source_group和split，不只导出权重/边而丢失物理来源。

例如：

```python
from pathlib import Path
import numpy as np

root = Path(r"E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005")
with np.load(root / "graphs/JR-DUAL-r000-R12-g0170.npz", allow_pickle=False) as graph:
    weights = graph["weights"]
    edges = graph["edges"]
    owner = graph["owner"]
    contact_ids = graph["contact_id"]
```

专家审阅先看本说明、[目录与数据字典](DATASET_CATALOG_ZH.md)、[根入口](../README_zh.md)，再按需查看CSV、图、场景与P0/P1结果。P0结构案例和共同warm之外的净机会是机制证据；模型是否利用关系改善选择、同总时间内是否收到更多收益，仍须由相应匹配对照回答。固定调用数标签回放与实际完整D执行不能互相替代。

已签收的P0文档保持当时内容与hash，其中尚未生成holdout等文字是历史时点的状态。本次只在入口与交付说明更新当前进度，不改写历史阶段签收正文。

仅更新目录时运行`code/dataset_catalog.py`，无需重新生成已完成STK场景、图或实验。本文档更新只整理说明，没有修改协议、物理数据、COM配置、图及已有结果。
