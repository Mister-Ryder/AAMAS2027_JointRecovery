# JointRecovery 新建STK数据集目录与数据字典

更新：2026-10-05 21:21:15（北京时间）。根目录：`E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005`。

**实际完成 12 / 12 个计划母机会库，72 / 72 张计划派生图。** 成功母库共有 250,782 条完整规划接触与 868 条跨界接触。

计划数量与已完成数量分别统计。不同站网/gap重复使用同母窗口，不能把派生图节点相加当成新增物理接触或独立样本。进行中、失败及尚未生成的母库不进入完成计数。本目录不推断模型已训练或学习方法有优势。

## 来源组与阶段

| 来源组 | 划分 | 规划起点UTC | 几何状态 | 完整窗口 | 跨界窗口 | 完成派生图 |
|---|---|---|---|---:|---:|---:|
| JR-SOURCE-r000 | TRAIN | 2026-11-01T00:00:00Z | 已完成并冻结 | 20918 | 71 | 6/6 |
| JR-SOURCE-r001 | TRAIN | 2026-11-08T00:00:00Z | 已完成并冻结 | 20899 | 75 | 6/6 |
| JR-SOURCE-r002 | TRAIN | 2026-11-15T00:00:00Z | 已完成并冻结 | 20892 | 84 | 6/6 |
| JR-SOURCE-r003 | TRAIN | 2026-11-22T00:00:00Z | 已完成并冻结 | 20890 | 74 | 6/6 |
| JR-SOURCE-r004 | TRAIN | 2026-11-29T00:00:00Z | 已完成并冻结 | 20919 | 72 | 6/6 |
| JR-SOURCE-r005 | TRAIN | 2026-12-06T00:00:00Z | 已完成并冻结 | 20909 | 74 | 6/6 |
| JR-SOURCE-r006 | VALIDATION | 2026-12-13T00:00:00Z | 已完成并冻结 | 20893 | 70 | 6/6 |
| JR-SOURCE-r007 | VALIDATION | 2026-12-20T00:00:00Z | 已完成并冻结 | 20892 | 71 | 6/6 |
| JR-SOURCE-r008 | TEST | 2026-12-27T00:00:00Z | 已完成并冻结 | 20892 | 65 | 6/6 |
| JR-SOURCE-r009 | TEST | 2027-01-03T00:00:00Z | 已完成并冻结 | 20884 | 77 | 6/6 |
| JR-SOURCE-r010 | TEST | 2027-01-10T00:00:00Z | 已完成并冻结 | 20910 | 66 | 6/6 |
| JR-SOURCE-r011 | TEST | 2027-01-17T00:00:00Z | 已完成并冻结 | 20884 | 69 | 6/6 |

TRAIN r000–r005（含P0已曝光开发来源）；VALIDATION r006–r007；TEST r008–r011。同一来源的R8/R12、全部gap、初态及恢复域必须始终绑定同一split；日期不同不自动证明统计独立。

## 已完成派生图

| 图 | 来源组 | 站网 | τg / τs（秒） | 顶点 | 冲突边 | 平均度 | 跨owner边占比 |
|---|---|---|---|---:|---:|---:|---:|
| [JR-DUAL-r000-R12-g0170](../graphs/JR-DUAL-r000-R12-g0170.json) | JR-SOURCE-r000 | R12 | 170 / 150 | 20,918 | 160,565 | 15.352 | 49.76% |
| [JR-DUAL-r000-R12-g0340](../graphs/JR-DUAL-r000-R12-g0340.json) | JR-SOURCE-r000 | R12 | 340 / 150 | 20,918 | 183,416 | 17.537 | 56.02% |
| [JR-DUAL-r000-R12-g0680](../graphs/JR-DUAL-r000-R12-g0680.json) | JR-SOURCE-r000 | R12 | 680 / 150 | 20,918 | 232,565 | 22.236 | 65.32% |
| [JR-DUAL-r000-R8-g0170](../graphs/JR-DUAL-r000-R8-g0170.json) | JR-SOURCE-r000 | R8 | 170 / 150 | 13,949 | 86,214 | 12.361 | 61.78% |
| [JR-DUAL-r000-R8-g0340](../graphs/JR-DUAL-r000-R8-g0340.json) | JR-SOURCE-r000 | R8 | 340 / 150 | 13,949 | 101,437 | 14.544 | 67.51% |
| [JR-DUAL-r000-R8-g0680](../graphs/JR-DUAL-r000-R8-g0680.json) | JR-SOURCE-r000 | R8 | 680 / 150 | 13,949 | 134,213 | 19.243 | 75.45% |
| [JR-DUAL-r001-R12-g0170](../graphs/JR-DUAL-r001-R12-g0170.json) | JR-SOURCE-r001 | R12 | 170 / 150 | 20,899 | 160,550 | 15.364 | 49.75% |
| [JR-DUAL-r001-R12-g0340](../graphs/JR-DUAL-r001-R12-g0340.json) | JR-SOURCE-r001 | R12 | 340 / 150 | 20,899 | 183,315 | 17.543 | 55.99% |
| [JR-DUAL-r001-R12-g0680](../graphs/JR-DUAL-r001-R12-g0680.json) | JR-SOURCE-r001 | R12 | 680 / 150 | 20,899 | 232,402 | 22.240 | 65.28% |
| [JR-DUAL-r001-R8-g0170](../graphs/JR-DUAL-r001-R8-g0170.json) | JR-SOURCE-r001 | R8 | 170 / 150 | 13,934 | 86,173 | 12.369 | 61.78% |
| [JR-DUAL-r001-R8-g0340](../graphs/JR-DUAL-r001-R8-g0340.json) | JR-SOURCE-r001 | R8 | 340 / 150 | 13,934 | 101,363 | 14.549 | 67.50% |
| [JR-DUAL-r001-R8-g0680](../graphs/JR-DUAL-r001-R8-g0680.json) | JR-SOURCE-r001 | R8 | 680 / 150 | 13,934 | 134,067 | 19.243 | 75.43% |
| [JR-DUAL-r002-R12-g0170](../graphs/JR-DUAL-r002-R12-g0170.json) | JR-SOURCE-r002 | R12 | 170 / 150 | 20,892 | 160,481 | 15.363 | 49.73% |
| [JR-DUAL-r002-R12-g0340](../graphs/JR-DUAL-r002-R12-g0340.json) | JR-SOURCE-r002 | R12 | 340 / 150 | 20,892 | 183,259 | 17.543 | 55.98% |
| [JR-DUAL-r002-R12-g0680](../graphs/JR-DUAL-r002-R12-g0680.json) | JR-SOURCE-r002 | R12 | 680 / 150 | 20,892 | 232,319 | 22.240 | 65.27% |
| [JR-DUAL-r002-R8-g0170](../graphs/JR-DUAL-r002-R8-g0170.json) | JR-SOURCE-r002 | R8 | 170 / 150 | 13,932 | 86,163 | 12.369 | 61.74% |
| [JR-DUAL-r002-R8-g0340](../graphs/JR-DUAL-r002-R8-g0340.json) | JR-SOURCE-r002 | R8 | 340 / 150 | 13,932 | 101,324 | 14.546 | 67.47% |
| [JR-DUAL-r002-R8-g0680](../graphs/JR-DUAL-r002-R8-g0680.json) | JR-SOURCE-r002 | R8 | 680 / 150 | 13,932 | 134,066 | 19.246 | 75.41% |
| [JR-DUAL-r003-R12-g0170](../graphs/JR-DUAL-r003-R12-g0170.json) | JR-SOURCE-r003 | R12 | 170 / 150 | 20,890 | 160,334 | 15.350 | 49.78% |
| [JR-DUAL-r003-R12-g0340](../graphs/JR-DUAL-r003-R12-g0340.json) | JR-SOURCE-r003 | R12 | 340 / 150 | 20,890 | 183,069 | 17.527 | 56.02% |
| [JR-DUAL-r003-R12-g0680](../graphs/JR-DUAL-r003-R12-g0680.json) | JR-SOURCE-r003 | R12 | 680 / 150 | 20,890 | 232,104 | 22.222 | 65.31% |
| [JR-DUAL-r003-R8-g0170](../graphs/JR-DUAL-r003-R8-g0170.json) | JR-SOURCE-r003 | R8 | 170 / 150 | 13,927 | 86,076 | 12.361 | 61.81% |
| [JR-DUAL-r003-R8-g0340](../graphs/JR-DUAL-r003-R8-g0340.json) | JR-SOURCE-r003 | R8 | 340 / 150 | 13,927 | 101,200 | 14.533 | 67.52% |
| [JR-DUAL-r003-R8-g0680](../graphs/JR-DUAL-r003-R8-g0680.json) | JR-SOURCE-r003 | R8 | 680 / 150 | 13,927 | 133,909 | 19.230 | 75.45% |
| [JR-DUAL-r004-R12-g0170](../graphs/JR-DUAL-r004-R12-g0170.json) | JR-SOURCE-r004 | R12 | 170 / 150 | 20,919 | 160,598 | 15.354 | 49.72% |
| [JR-DUAL-r004-R12-g0340](../graphs/JR-DUAL-r004-R12-g0340.json) | JR-SOURCE-r004 | R12 | 340 / 150 | 20,919 | 183,475 | 17.541 | 55.99% |
| [JR-DUAL-r004-R12-g0680](../graphs/JR-DUAL-r004-R12-g0680.json) | JR-SOURCE-r004 | R12 | 680 / 150 | 20,919 | 232,636 | 22.242 | 65.29% |
| [JR-DUAL-r004-R8-g0170](../graphs/JR-DUAL-r004-R8-g0170.json) | JR-SOURCE-r004 | R8 | 170 / 150 | 13,942 | 86,153 | 12.359 | 61.75% |
| [JR-DUAL-r004-R8-g0340](../graphs/JR-DUAL-r004-R8-g0340.json) | JR-SOURCE-r004 | R8 | 340 / 150 | 13,942 | 101,397 | 14.546 | 67.50% |
| [JR-DUAL-r004-R8-g0680](../graphs/JR-DUAL-r004-R8-g0680.json) | JR-SOURCE-r004 | R8 | 680 / 150 | 13,942 | 134,094 | 19.236 | 75.42% |
| [JR-DUAL-r005-R12-g0170](../graphs/JR-DUAL-r005-R12-g0170.json) | JR-SOURCE-r005 | R12 | 170 / 150 | 20,909 | 160,494 | 15.352 | 49.78% |
| [JR-DUAL-r005-R12-g0340](../graphs/JR-DUAL-r005-R12-g0340.json) | JR-SOURCE-r005 | R12 | 340 / 150 | 20,909 | 183,330 | 17.536 | 56.04% |
| [JR-DUAL-r005-R12-g0680](../graphs/JR-DUAL-r005-R12-g0680.json) | JR-SOURCE-r005 | R12 | 680 / 150 | 20,909 | 232,395 | 22.229 | 65.32% |
| [JR-DUAL-r005-R8-g0170](../graphs/JR-DUAL-r005-R8-g0170.json) | JR-SOURCE-r005 | R8 | 170 / 150 | 13,938 | 86,148 | 12.362 | 61.81% |
| [JR-DUAL-r005-R8-g0340](../graphs/JR-DUAL-r005-R8-g0340.json) | JR-SOURCE-r005 | R8 | 340 / 150 | 13,938 | 101,347 | 14.543 | 67.53% |
| [JR-DUAL-r005-R8-g0680](../graphs/JR-DUAL-r005-R8-g0680.json) | JR-SOURCE-r005 | R8 | 680 / 150 | 13,938 | 134,059 | 19.236 | 75.46% |
| [JR-DUAL-r006-R12-g0170](../graphs/JR-DUAL-r006-R12-g0170.json) | JR-SOURCE-r006 | R12 | 170 / 150 | 20,893 | 160,252 | 15.340 | 49.77% |
| [JR-DUAL-r006-R12-g0340](../graphs/JR-DUAL-r006-R12-g0340.json) | JR-SOURCE-r006 | R12 | 340 / 150 | 20,893 | 183,068 | 17.524 | 56.03% |
| [JR-DUAL-r006-R12-g0680](../graphs/JR-DUAL-r006-R12-g0680.json) | JR-SOURCE-r006 | R12 | 680 / 150 | 20,893 | 232,066 | 22.215 | 65.32% |
| [JR-DUAL-r006-R8-g0170](../graphs/JR-DUAL-r006-R8-g0170.json) | JR-SOURCE-r006 | R8 | 170 / 150 | 13,927 | 86,030 | 12.354 | 61.81% |
| [JR-DUAL-r006-R8-g0340](../graphs/JR-DUAL-r006-R8-g0340.json) | JR-SOURCE-r006 | R8 | 340 / 150 | 13,927 | 101,240 | 14.539 | 67.55% |
| [JR-DUAL-r006-R8-g0680](../graphs/JR-DUAL-r006-R8-g0680.json) | JR-SOURCE-r006 | R8 | 680 / 150 | 13,927 | 133,893 | 19.228 | 75.46% |
| [JR-DUAL-r007-R12-g0170](../graphs/JR-DUAL-r007-R12-g0170.json) | JR-SOURCE-r007 | R12 | 170 / 150 | 20,892 | 160,190 | 15.335 | 49.83% |
| [JR-DUAL-r007-R12-g0340](../graphs/JR-DUAL-r007-R12-g0340.json) | JR-SOURCE-r007 | R12 | 340 / 150 | 20,892 | 182,944 | 17.513 | 56.07% |
| [JR-DUAL-r007-R12-g0680](../graphs/JR-DUAL-r007-R12-g0680.json) | JR-SOURCE-r007 | R12 | 680 / 150 | 20,892 | 231,986 | 22.208 | 65.36% |
| [JR-DUAL-r007-R8-g0170](../graphs/JR-DUAL-r007-R8-g0170.json) | JR-SOURCE-r007 | R8 | 170 / 150 | 13,927 | 85,986 | 12.348 | 61.85% |
| [JR-DUAL-r007-R8-g0340](../graphs/JR-DUAL-r007-R8-g0340.json) | JR-SOURCE-r007 | R8 | 340 / 150 | 13,927 | 101,118 | 14.521 | 67.56% |
| [JR-DUAL-r007-R8-g0680](../graphs/JR-DUAL-r007-R8-g0680.json) | JR-SOURCE-r007 | R8 | 680 / 150 | 13,927 | 133,858 | 19.223 | 75.49% |
| [JR-DUAL-r008-R12-g0170](../graphs/JR-DUAL-r008-R12-g0170.json) | JR-SOURCE-r008 | R12 | 170 / 150 | 20,892 | 160,096 | 15.326 | 49.81% |
| [JR-DUAL-r008-R12-g0340](../graphs/JR-DUAL-r008-R12-g0340.json) | JR-SOURCE-r008 | R12 | 340 / 150 | 20,892 | 182,927 | 17.512 | 56.08% |
| [JR-DUAL-r008-R12-g0680](../graphs/JR-DUAL-r008-R12-g0680.json) | JR-SOURCE-r008 | R12 | 680 / 150 | 20,892 | 231,942 | 22.204 | 65.36% |
| [JR-DUAL-r008-R8-g0170](../graphs/JR-DUAL-r008-R8-g0170.json) | JR-SOURCE-r008 | R8 | 170 / 150 | 13,931 | 85,959 | 12.341 | 61.83% |
| [JR-DUAL-r008-R8-g0340](../graphs/JR-DUAL-r008-R8-g0340.json) | JR-SOURCE-r008 | R8 | 340 / 150 | 13,931 | 101,165 | 14.524 | 67.57% |
| [JR-DUAL-r008-R8-g0680](../graphs/JR-DUAL-r008-R8-g0680.json) | JR-SOURCE-r008 | R8 | 680 / 150 | 13,931 | 133,895 | 19.223 | 75.50% |
| [JR-DUAL-r009-R12-g0170](../graphs/JR-DUAL-r009-R12-g0170.json) | JR-SOURCE-r009 | R12 | 170 / 150 | 20,884 | 160,263 | 15.348 | 49.77% |
| [JR-DUAL-r009-R12-g0340](../graphs/JR-DUAL-r009-R12-g0340.json) | JR-SOURCE-r009 | R12 | 340 / 150 | 20,884 | 182,998 | 17.525 | 56.01% |
| [JR-DUAL-r009-R12-g0680](../graphs/JR-DUAL-r009-R12-g0680.json) | JR-SOURCE-r009 | R12 | 680 / 150 | 20,884 | 231,973 | 22.215 | 65.30% |
| [JR-DUAL-r009-R8-g0170](../graphs/JR-DUAL-r009-R8-g0170.json) | JR-SOURCE-r009 | R8 | 170 / 150 | 13,919 | 86,000 | 12.357 | 61.80% |
| [JR-DUAL-r009-R8-g0340](../graphs/JR-DUAL-r009-R8-g0340.json) | JR-SOURCE-r009 | R8 | 340 / 150 | 13,919 | 101,128 | 14.531 | 67.51% |
| [JR-DUAL-r009-R8-g0680](../graphs/JR-DUAL-r009-R8-g0680.json) | JR-SOURCE-r009 | R8 | 680 / 150 | 13,919 | 133,800 | 19.226 | 75.44% |
| [JR-DUAL-r010-R12-g0170](../graphs/JR-DUAL-r010-R12-g0170.json) | JR-SOURCE-r010 | R12 | 170 / 150 | 20,910 | 160,270 | 15.330 | 49.78% |
| [JR-DUAL-r010-R12-g0340](../graphs/JR-DUAL-r010-R12-g0340.json) | JR-SOURCE-r010 | R12 | 340 / 150 | 20,910 | 183,103 | 17.513 | 56.04% |
| [JR-DUAL-r010-R12-g0680](../graphs/JR-DUAL-r010-R12-g0680.json) | JR-SOURCE-r010 | R12 | 680 / 150 | 20,910 | 232,251 | 22.214 | 65.34% |
| [JR-DUAL-r010-R8-g0170](../graphs/JR-DUAL-r010-R8-g0170.json) | JR-SOURCE-r010 | R8 | 170 / 150 | 13,944 | 86,090 | 12.348 | 61.80% |
| [JR-DUAL-r010-R8-g0340](../graphs/JR-DUAL-r010-R8-g0340.json) | JR-SOURCE-r010 | R8 | 340 / 150 | 13,944 | 101,307 | 14.531 | 67.54% |
| [JR-DUAL-r010-R8-g0680](../graphs/JR-DUAL-r010-R8-g0680.json) | JR-SOURCE-r010 | R8 | 680 / 150 | 13,944 | 134,128 | 19.238 | 75.48% |
| [JR-DUAL-r011-R12-g0170](../graphs/JR-DUAL-r011-R12-g0170.json) | JR-SOURCE-r011 | R12 | 170 / 150 | 20,884 | 160,060 | 15.328 | 49.79% |
| [JR-DUAL-r011-R12-g0340](../graphs/JR-DUAL-r011-R12-g0340.json) | JR-SOURCE-r011 | R12 | 340 / 150 | 20,884 | 182,835 | 17.510 | 56.05% |
| [JR-DUAL-r011-R12-g0680](../graphs/JR-DUAL-r011-R12-g0680.json) | JR-SOURCE-r011 | R12 | 680 / 150 | 20,884 | 231,856 | 22.204 | 65.34% |
| [JR-DUAL-r011-R8-g0170](../graphs/JR-DUAL-r011-R8-g0170.json) | JR-SOURCE-r011 | R8 | 170 / 150 | 13,927 | 86,030 | 12.354 | 61.84% |
| [JR-DUAL-r011-R8-g0340](../graphs/JR-DUAL-r011-R8-g0340.json) | JR-SOURCE-r011 | R8 | 340 / 150 | 13,927 | 101,196 | 14.532 | 67.56% |
| [JR-DUAL-r011-R8-g0680](../graphs/JR-DUAL-r011-R8-g0680.json) | JR-SOURCE-r011 | R8 | 680 / 150 | 13,927 | 133,917 | 19.231 | 75.49% |

各图的数组位于同名`.npz`，统计与SHA位于同名`.json`。R8固定为GS01、GS03、GS04、GS06、GS07、GS09、GS10、GS12；只删除不在子网内的整条接触，保留窗口和收益完全不变。

## 原始机会库与可编辑场景

### JR-DUAL-r000

来源：`JR-SOURCE-r000`，划分：TRAIN。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r000/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r000/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r000/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r000/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r000/scene/JR-DUAL-r000.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r000/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `5337fefd75068fcff1689e7ecf4c248a75f82afcdd4de965b0d678df8b3a6646` |
| `boundary_contacts.csv` | `b8f3ce406c7bf425468e63b731b97731cd7ff62d598e34aced6df6ad7cbd76f1` |
| `access_raw.csv` | `4b3bf63b02198930ce75497410dcbca40b3a589e3b6d66a34fe4031fda29604b` |
| `raw_access_reports.jsonl` | `4dc546078d3b6c491ecbb4521ab4dc4cd5fcc9a31ede10e19c5eb0ac14534032` |
| `scene/JR-DUAL-r000.sc` | `ea6f1c039471f96f9397cd7be2a14c3ff1cdeb1675c5cd34571fb7721969bf50` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `9d975ebe45f6454e8b85a3b6b63e9c5800f46e5331aab4556d21c4914c4cd7d4` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r001

来源：`JR-SOURCE-r001`，划分：TRAIN。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r001/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r001/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r001/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r001/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r001/scene/JR-DUAL-r001.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r001/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `4d4b0347a9fef60f8d43799a53cfb19615c7765bd0d543b21d33d77b7df943c3` |
| `boundary_contacts.csv` | `91c03f8293e0b5b103c9b75463646c488570f76452b4a204f3967a2386a03d46` |
| `access_raw.csv` | `3b015b8b00447d59f756eb98509e98cd2ff023bdaa34a044b0982730b1c93756` |
| `raw_access_reports.jsonl` | `335669e8b7de41168323e4b7a2960d4f0f2c4d320c3960a6584a6b1955b8d39f` |
| `scene/JR-DUAL-r001.sc` | `8c8c530aa30e7572d156d392861036597db13698c53032703eb6603d89712800` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r002

来源：`JR-SOURCE-r002`，划分：TRAIN。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r002/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r002/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r002/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r002/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r002/scene/JR-DUAL-r002.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r002/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `5632056a44ffb3f47979aff1e644a9404f34bd5ba03d99b7546bc82d19a4f3ba` |
| `boundary_contacts.csv` | `ffe886b6344a80e5627f43b14e804405f920a342dff2e6c0837eeb12f4265543` |
| `access_raw.csv` | `cde22099d124a335242f623f49cd67641f3e5cb992d503a2cde4063099015b1e` |
| `raw_access_reports.jsonl` | `ff795389cb566b9fb283117bf3fbb4526fba7b76c08d644d1598afa6129fc841` |
| `scene/JR-DUAL-r002.sc` | `38fa7dc5d9cce3bcf5d92f3881b812e03bbc5f3333163035f0443eaccc796723` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r003

来源：`JR-SOURCE-r003`，划分：TRAIN。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r003/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r003/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r003/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r003/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r003/scene/JR-DUAL-r003.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r003/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `b09924d683b7217e17f476dea757f9fca3124e2d32cfc6f1b625373d0931cbdd` |
| `boundary_contacts.csv` | `3b2fd88978863f5975b3823efc95327831c798d03a87206c93fdc074a3515145` |
| `access_raw.csv` | `4351a5fbb2c0440264293d2b1838c5d6dbb7c16683e23e61106c4c5029412760` |
| `raw_access_reports.jsonl` | `a326c24715232ea84c1456ed9e88ec3cb7177b2bc336f386d821c9077f370a24` |
| `scene/JR-DUAL-r003.sc` | `b0f4da3ca519501038ba2f97c1792f8df255580d809048b7af72e8144ad98a4b` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r004

来源：`JR-SOURCE-r004`，划分：TRAIN。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r004/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r004/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r004/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r004/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r004/scene/JR-DUAL-r004.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r004/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `4d5f82973b0bd34a4def15829923e9a817807b873aa7666c47852c732fa0e2b6` |
| `boundary_contacts.csv` | `777d87ea0ddd533c7e46c8d6815622e40a00cbdaa5eee163a2797e3ba46b2e76` |
| `access_raw.csv` | `b54d98c2fe0d0fad4828842123eba52c05c84291a4adcda671e1080c0cf50dbb` |
| `raw_access_reports.jsonl` | `fedf475b293bfee7145fd0d56e465938c1129cd7fc3936c85e029becaa536445` |
| `scene/JR-DUAL-r004.sc` | `8c63794dc8be9170275daaaeca701f6ba684b25dc7d8cc0d4064c8b89c197353` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r005

来源：`JR-SOURCE-r005`，划分：TRAIN。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r005/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r005/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r005/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r005/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r005/scene/JR-DUAL-r005.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r005/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `999f39c8aae3026dc304d0945a981eea22cb5cbfe516c4f93e83685eecddb05a` |
| `boundary_contacts.csv` | `e291208ef946e806af7f8e078170baf4cf4fbe4c419df6734cd09cea6842a30e` |
| `access_raw.csv` | `3b603788693616c8bd5f36243f7ff792c01fd5df9c2654a4f0905f5d40c44f2b` |
| `raw_access_reports.jsonl` | `272ed9b78ac395d2946aafb9b21c922ae3e8a91126ef9e3ed6c1c392ca2a6390` |
| `scene/JR-DUAL-r005.sc` | `a798f2e1ce6fea4c1f71aa5440e43d46fdfc1e5ef0fb82426a01fb95dfa18f63` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r006

来源：`JR-SOURCE-r006`，划分：VALIDATION。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r006/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r006/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r006/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r006/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r006/scene/JR-DUAL-r006.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r006/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `1c16bc8a11faaa4c6ed27045b3048373c37066779cef780da9688f1034214447` |
| `boundary_contacts.csv` | `b76fd08e7c870b65c3cc8e69d8d145105c2a046f06e23063385f160b90a94a0d` |
| `access_raw.csv` | `59fd7d4d21aaf7c763aa5741692fc2e5529c3891e0350093b7cf96f515dc64fd` |
| `raw_access_reports.jsonl` | `63cfe24d8916d450297b1b6c5aa1b84140b1e9aa209eeebce1ab9faef6433572` |
| `scene/JR-DUAL-r006.sc` | `8e218131d765b4484a2738087adcdc8420add55f3780a741b6f0b1ffb1226826` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r007

来源：`JR-SOURCE-r007`，划分：VALIDATION。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r007/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r007/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r007/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r007/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r007/scene/JR-DUAL-r007.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r007/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `b8aec10ac2a48486a31b2a261f1eccde94f8ed2bc1680b09656219da759ee10e` |
| `boundary_contacts.csv` | `d9db6222708df13d7b07048ed9b1d87c6500fa5f5766fa6d79077e35b4142b10` |
| `access_raw.csv` | `07610f4be23dc4bbaacc966d0e77957ad7390080c72f8b337076be381e645362` |
| `raw_access_reports.jsonl` | `a5430ccd01b371aa3ed299c730d99bc69dc35121cd94b18113502649faafc46c` |
| `scene/JR-DUAL-r007.sc` | `74fa8460b5970cfb0005253977e03f4f46eb9f927755410cc3d30b34ae725cd3` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r008

来源：`JR-SOURCE-r008`，划分：TEST。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r008/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r008/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r008/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r008/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r008/scene/JR-DUAL-r008.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r008/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `571e432a15780a75375dfeeb3efcb9d31de38fa758b6af12880a283f3d90f339` |
| `boundary_contacts.csv` | `3c730a37dce78970786610dc2f9e4bbde527369601de6a0cb72206fd56b98cbe` |
| `access_raw.csv` | `619345e19df9170164b2efebc25c7851f81533b5d39eb000feaf952730902387` |
| `raw_access_reports.jsonl` | `3808506b178124582e9a3b535cfa5c39e7421efa20468505b788d714b45c7991` |
| `scene/JR-DUAL-r008.sc` | `4c3532f3f2fa03787f27c6012aecd882a4c6d000d00562891955da0a3d873dc0` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r009

来源：`JR-SOURCE-r009`，划分：TEST。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r009/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r009/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r009/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r009/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r009/scene/JR-DUAL-r009.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r009/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `ba53b5944a57b558a26c6af09fb03a58164403dbd09a40251c9bbd9300725147` |
| `boundary_contacts.csv` | `a67ab30074fa739aeff4868570d832a02a5cdb348a729eef78c585f161989c53` |
| `access_raw.csv` | `ac047b663c4308065e1e02bc8138f4de6f2f1f78ef7504534767bf2576833369` |
| `raw_access_reports.jsonl` | `dc5f7b7d777a78de4a0232d8585e170fff12ca934195fb0ff30c44a8215c88c4` |
| `scene/JR-DUAL-r009.sc` | `0bb72cd214a43403d33000419c362113c6a0129c80066b953bee169b3edb839c` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r010

来源：`JR-SOURCE-r010`，划分：TEST。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r010/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r010/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r010/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r010/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r010/scene/JR-DUAL-r010.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r010/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `a18216f2fe739392ba43fa6a008ff742856d7edd5276967f2dc2a763bcc179b0` |
| `boundary_contacts.csv` | `bf6020108ed06e994a9223e8e908275c5394f5d985d519b0042f86d277ce3d96` |
| `access_raw.csv` | `1c39c20d8b2575231a4d1a284d0fbfd1304f02f471b8075d59763264b2eed384` |
| `raw_access_reports.jsonl` | `a22130215392cbab01cbd1e285544ea82290688ad2bf76927c85943c41cc2fa4` |
| `scene/JR-DUAL-r010.sc` | `1dab5c8e12a6ab44e88106bf6e72cda82f1a57c02b8ce5188a391cf9e5fcd28e` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

### JR-DUAL-r011

来源：`JR-SOURCE-r011`，划分：TEST。

- [完整规划接触 contacts.csv](../raw_geometry/JR-DUAL-r011/contacts.csv)：72小时内完整落入的窗口。
- [跨规划边界 boundary_contacts.csv](../raw_geometry/JR-DUAL-r011/boundary_contacts.csv)：单列保留，不裁短，不进入图。
- [含padding的 access_raw.csv](../raw_geometry/JR-DUAL-r011/access_raw.csv)与[原始Access报告 JSONL](../raw_geometry/JR-DUAL-r011/raw_access_reports.jsonl)：原始端点与padding边界标记。
- [可编辑STK场景](../raw_geometry/JR-DUAL-r011/scene/JR-DUAL-r011.sc)及同目录全部卫星/站点依赖：不能只复制`.sc`。
- [执行manifest](../raw_geometry/JR-DUAL-r011/manifest.json)：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。

| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |
|---|---|
| `contacts.csv` | `051c716e754475c38828d32cc92aca41b9af92c0a8b4a17614ca9c2a30de6486` |
| `boundary_contacts.csv` | `1e95cdada9ecdde81016486b7c83ffa8c91db54d5879bec7bdb56c65d1c1cc8e` |
| `access_raw.csv` | `d7c2155c4511d25f4b4d8d318b90d7b35ae201637762d82b9c1c84b928e18832` |
| `raw_access_reports.jsonl` | `ae88cd82637384c08e36a2b1e0630df7a1e155dcf83247e089133678bc23c239` |
| `scene/JR-DUAL-r011.sc` | `7c4ebd85765484ef34ad09a910d3e9a4540384201baf9dd9d92abd2ebd8f45ab` |
| 参数JSON | `f2d553b0c468279425db7c4e0071a917ccf1fa6eb62a01afce64e71cb6f4ffca` |
| 原几何生成代码 | `63409c164f8febde28e43cdb8db5261abe6ca1b79e86dbad7f75fb4de702c682` |

固定抽样对：12；15秒与30秒区间数一致：True；端点最大差≤5ms：True。

## 固定问题契约

1. 顶点是一次完整几何可见区间`[s,e)`；收益`w=e−s`，目标为静态星地链路总时长。没有业务任务、窗口切片、部分移动、统一时长或人为重赋权。
2. 每站1副天线、每星1个同时通信通道；`owner=卫星`。资源因子是时间归属，不是全天只能选一次的静态团。
3. 起点排序u/v：同天线且`sv < eu + τg`，或同卫星且`sv < eu + τs`，则冲突；取并集。包含和同起点也冲突；恰好满足gap等号时兼容。τs=150秒，τg=170/340/680秒。
4. 168星双壳层、WGS84理想化站网、最低仰角15°、J2传播、初始J2000；规划72小时，两端各额外传播1小时，轨道epoch仍为规划起点。完整端点不裁切；跨界和padding外窗口另存。
5. Access精确事件开启、最大步长30秒、收敛0.001秒、光行时关闭；12固定对用15秒复查。CSV9位小数、求解器微秒tick均是数值序列化，不是物理精度声明。
6. 原始UTC是辅助毫秒显示，相对秒端点是建边及计奖依据。STK11中文路径保存限制通过独占ASCII临时路径和全依赖复制解决，实际数据仍在本目录。
7. 同母库的站网、gap、局部状态和恢复请求共用来源组，不能跨训练/验证/测试。零机会、失败、无增益和迟到请求须保留于后续独立执行记录。

## 接触CSV数据字典

| 字段 | 类型 / 单位 | 含义 |
|---|---|---|
| `source_group` | string | 来源组；同母库所有站网、gap、状态和恢复请求必须同组划分 |
| `geometry_id / replicate_id` | string | 物理族标识 / 重新传播的物理实现标识 |
| `epoch_utc` | ISO UTC | 规划起点及轨道历元；relative seconds的零点 |
| `contact_id / pass_id` | string | 完整接触唯一标识 / 卫星—站点流的原始过境序号 |
| `satellite_id` | string | 卫星实体；同时通信容量为1；也是owner |
| `site_id / antenna_id` | string | 站点 / 单副天线资源标识 |
| `start_rel_seconds / end_rel_seconds` | decimal seconds | 权威端点；9位小数；不裁切、不整数秒化；区间[s,e) |
| `duration_seconds` | decimal seconds | 严格等于冻结CSV的end−start；唯一收益，无业务权重 |
| `start_utc / end_utc` | STK UTCG | 辅助人工显示；STK默认毫秒显示，不能代替权威相对端点 |
| `boundary_crossing` | string | contacts.csv恒为none；边界表left/right/both；原始表另有outside_before/after |
| `access_settings_hash` | SHA-256 | 对应access_settings_actual.json中的真实Access设置 |
| `padding_boundary_clipped` | boolean | 仅raw/boundary表：传播padding边界上可能被截断；不得作为完整规划接触 |

## 图NPZ数据字典

| 字段 | 类型 / 单位 | 含义 |
|---|---|---|
| `start_decimal_seconds / end_decimal_seconds / duration_decimal_seconds` | string[N] | 冻结CSV原小数；收益为端点差 |
| `start_ns / end_ns` | int64[N], ns | 9位小数的整数表示；原始冲突边在此精度建立 |
| `start_ticks / end_ticks` | int64[N], us | 求解器微秒序列化；不是物理精度声明 |
| `weights` | float64[N], seconds | 完整接触时长；无重新赋权 |
| `owner / agents / satellite_id_mapping` | integer[N] / string[168] | owner=卫星的索引映射；不构成全天团 |
| `edges / edge_u / edge_v` | integer[E,2] / integer[E] | 去重无向冲突边，u<v |
| `edge_types` | uint8[E] | 位1=同天线冲突，位2=同卫星冲突；3=同时满足两类 |
| `mother_contact_index / contact_id` | integer[N] / string[N] | R8/R12顶点回溯到同母contacts.csv；保留原窗口 |
| `vertex_factors / factor_indptr / factor_vertices` | CSR-style arrays | 时间资源归属：每个顶点关联一副地面天线和一个卫星通道；成员并非静态团 |
| `factor_ids / factor_kind / factor_capacity / factor_gap_seconds` | arrays | 天线/卫星资源类型，容量恒1，各自gap |
| `source_group / split / graph_id / contacts_sha256 / parameters_sha256` | scalar strings | 来源分组、派生图和冻结输入证据 |
| `duration_greedy_witness_vertices` | integer[K] | 建图验收的可行调度见证；不是学习算法结果或最优证书 |

## 更新方式

在每个几何/建图阶段完成后运行：

```text
python code/dataset_catalog.py --dataset-root <本数据集绝对目录>
```

工具只读取小型manifest和图JSON，复用已冻结SHA；不会启动STK、加载NPZ、重算场景、重新校验全部窗口或重复大规模实验。程序仅更新`reports/DATASET_CATALOG_ZH.md`和`reports/dataset_catalog.json`。
