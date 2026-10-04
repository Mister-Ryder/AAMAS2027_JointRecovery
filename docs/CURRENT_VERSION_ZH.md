# 第一篇：当前冻结版本交付说明

依据2026-10-04最新要求，停止继续探索新结构，优先交付当前版本。没有为改善图表
而改变已经完成的协议、删掉失败实例、换分母或把未执行项目写成结果。

## 本版增加的实质内容

- 方法明确为同一已执行贪心响应之上的联合恢复预测，保留实际warm会员、冲突因子、
  可行下界/团划分上界、请求预算和已知立即增益的独立偏置。原图检查负责可行性。
- 完成96图条件恢复采集、全部5×3次40轮训练，以及完整1512次实际在线比较。
- 完成40个原始公开图上的2040次已发表算法运行，表中保留17配置及其正式引用、
  CPU线程/GPU成本边界和失败分母。它是公开求解器比较，尚不是本模型的公开迁移。
- 主文展示问题示意、技术流程、条件机会分析和真实预算曲线；结果小图分别保存，
  由LaTeX控制布局。收益↑、耗时/迟到↓、有效返回↑，粗体标记主表最优均值。
- 各章节由独立agent撰写后统一整合；开发、修复、逐次运行和后续方案保存在MD，
  不作为论文正文。官方模板未改页边距、字号或类文件。

## 必须明确告知的局限

当前版本的学习分配器**没有稳定超过强贪心对照**。主时限0.556s下Capacity的严格
返回增益为57.542%，Greedy为234.266%；Capacity迟到率83.333%。这些百分比除以
原始初态，超过100%不是最优性差距。主表完整保留比较，discussion解释其意义。

2.221s下Capacity增益236.334%，比共同前缀多2.068个百分点；P1为236.218%，
CheapSummary为238.355%。因此存在有限额外恢复，但不能声称图结构组件已建立
总体性能优势。条件数据的20.494%最佳单次机会不是学习策略增益，更不是在线上限。

全部15次训练中11次的条件选择后悔值精确平坦；参数和辅助损失仍改变。这不足以
证明整网梯度崩塌。尺度/预算信号的诊断是一项后续假设，不是已证明有效的修复。

尚未执行本残差模型的公开7560项迁移、完整C3/W3策略比较或封存确认集。
原始大图、原权重的早期求解器记录另存报告，不能与当前模型混作优势。

新版比上一版更完整：技术定义、正式基线、公共数据和可审查证据均有所加强。
**完整性改善不等于性能优势已经建立，也不能保证录用。** 当前最重要的录用风险
仍是学习方法的实际优势与独立泛化证据不足；用户决定基于本版继续讨论后续研究。

## 文件与复现入口

- 最新正文：`paper/main_v4.tex` 与 `paper/main_v4.pdf`；交付时同步至 `paper/main.tex/pdf`。
- 科学补充：`paper/supplementary_v4.tex/pdf`，含特征、容量性质、全部fit选择、成本口径和AI辅助披露。
- 真实在线聚合：`results/v4_residual_online/independent_analysis_01/residual_online_table.csv`。
- 真实公开比较：`results/v4_unit_public_original/independent_audit_01/published_original_table.csv`。
- 原始标签：`results/v4_residual_labels/bulk_03`；全部fit：`results/v4_residual_fit/bulk_01`。
- 原始在线收据/会员：`results/v4_residual_online/bulk_01`；原始公开输出：`results/v4_unit_public_original/bulk_01`。
- 运行参数：`RESIDUAL_ACTUAL_FIT_CLOUD_ARGS.json` / `RESIDUAL_ACTUAL_ONLINE_CLOUD_ARGS.json`。
- 实际模型源码：`src/joint_recovery/v4_residual_model.py`，SHA256
  `d6c974b3e57b1880bbb0d24c6dceca8f343d303c47e74736f3864c8da054a750`。
- 实际训练入口：`experiments/v4_residual_fit.py`，SHA256
  `803f86f680da51261c1726aef302e5b42011f405b4d7f9858165d3d77af7b9b6`。

后续归一化候选及三项定向测试保留在 `research_candidates/normalization_untrained`。
主源码恢复为实际实验版本，候选没有训练或验证，不能用于重现本版结果。
所有原始来源胶囊及结果保留，不重做已通过的全量会员/源码检查。

本次只进行最后的集成科学/版面检查。页数、版本号和交付包大小以
`release/current_version/FINAL_CHECK.json` 与包内清单为准。
