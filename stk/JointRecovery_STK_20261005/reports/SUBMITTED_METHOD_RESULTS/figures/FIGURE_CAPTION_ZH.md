**用途：效率与短预算压力诊断；充裕预算求解质量另见 JOINTRECOVERY_QUALITY_RESULTS。本文方法为 JointRecovery（JR），原键 FullCapacity；P1 是独立替代对照。**

# 四面板科学结果图说明

**完整在线 STK 迁移框架的独立 TEST 比较。** A：三组预先冻结的实际完整决策预算 D 下，各方法的严格按时交付增量（原始 contact 完整时长秒，↑）。JointRecovery（JR）、JR-CheapSummary 为 seed17/29 均值；阴影只表示这两个fit的min–max观测范围，**不是置信区间**。B：JointRecovery 两seed均值减去 JR-CheapSummary均值、Greedy-rank+CHILS、CHILS-p1、HiGHS 的有符号严格增量差，颜色和标记对应被减去的参照，0线区分胜负。C：所有已声明cell的实际caller按时返回率（%，↑），失败/未知提交不计按时，迟到仍保留。D：在**预定wide D=3.898923968秒**，四个物理来源的 JointRecovery均值减去同一更强公共算法/软件参照 `CHILS-p1`；wide不是事后选择对我方最有利的D。

D参照在 CHILS-p1 与 HiGHS-MILP 中，依据预定wide D的四母源等权严格增量均值取较强者，所有来源使用同一个参照，未逐源挑不同算法。CHILS-p1是SEA2025正式发表的MWIS方法；HiGHS只是通用MILP软件参照，**不能称其为另一篇专门MWIS发表方法**。D的负差值使用空心斜线柱；正负均展示，不删不利来源。A/B/C始终显示全部六类方法。

三个D的精确值来自已完成报告CSV：0.97473099198192337, 1.9494619839638467, 3.8989239679276935。连线与阴影边界只连接三个实际观测点，不做平滑、不额外生成预算观测。重合曲线表示实际相同或接近的观测；不为分离曲线抖动D或修改数值。

先对每母源六配置等权，再对r008–r011四母源等权；模型两seed并非两个新物理来源。所有失败、零增量和迟到均由已完成报告的driver严格交付统计保留。总收益包含共同三贪心前缀；不能将全部native/prefix收益归给学习，匹配学习分配差异主要应看B中与同框架JR-CheapSummary/Greedy-rank+CHILS的比较。

remaining-head始终0，未验证其学习适应；完整闭环不等于旧稿全部组件验证。Full方法cap256、11动作、最多8请求；整图CHILS/HiGHS搜索范围不同，但原图、原S、原时长目标和完整caller D一致。HiGHS的SciPy1.10.1接口没有solver内x0 warm start。新增比较在原P2开始后、PLAN记载查看原VAL/TEST结果前冻结，不宣称更早预注册。

正式依据：[CHILS，SEA2025，DOI10.4230/LIPIcs.SEA.2025.22](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.22)；[SciPy1.10.1 milp官方文档](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.optimize.milp.html)、[HiGHS官方软件站](https://highs.dev/)。

PNG用于浏览，PDF为可缩放矢量图、嵌入TrueType字体；轴与图内文字使用英文，便于论文复用。统一字体DejaVu Sans、固定方法颜色、不同标记/线型、统一线宽和字号。没有改论文或运行新实验。
