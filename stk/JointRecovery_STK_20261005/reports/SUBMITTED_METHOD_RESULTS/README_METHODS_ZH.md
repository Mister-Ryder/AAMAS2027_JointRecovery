# 方法名称与结果用途

**本文方法是 JointRecovery（JR），原始实现键为 FullCapacity。P1 / FullP1 是独立替代估计对照，不是本文主方法。**

本目录用途：效率与短预算压力诊断（不是充裕求解质量主结果）。原始键永久保留以便回溯封存协议；CSV 的 method_display、method_role 列提供可见名称。

| 原始键（含 seed / mean 变体） | 可见名称 | 实验角色 |
|---|---|---|
| `FullCapacity` | JointRecovery（本文方法，JR；完整框架 STK 迁移） | PROPOSED_FULL_FRAMEWORK |
| `FullCheapSummary` | JR-CheapSummary（简化表示消融） | SIMPLIFIED_REPRESENTATION_ABLATION |
| `FullGreedy` | Greedy-rank+CHILS（贪心请求排序内部对照） | INTERNAL_GREEDY_REQUEST_RANKING |
| `FullP1` | Independent-replacement+CHILS（独立替代估计对照） | INTERNAL_INDEPENDENT_REPLACEMENT_CONTROL |
| `Capacity` | JointRecovery（静态 max4 内部诊断） | INTERNAL_STATIC_MAX4_DIAGNOSTIC |
| `CheapSummary` | JR-CheapSummary（静态简化表示诊断） | INTERNAL_STATIC_SUMMARY_DIAGNOSTIC |
| `Greedy` | Greedy-rank+CHILS（静态贪心排序对照） | INTERNAL_STATIC_GREEDY_CONTROL |
| `P1` | Independent-replacement+CHILS（静态独立替代对照） | INTERNAL_STATIC_INDEPENDENT_CONTROL |
| `CHILS-p1` | CHILS-p1（SEA 2025 公开 MWIS 算法） | PUBLISHED_MWIS_BASELINE |
| `HiGHS-MILP` | HiGHS-MILP（通用 MILP 软件参照） | GENERIC_MILP_SOFTWARE_REFERENCE |
| `StrongCheapControl` | 较强内部对照（描述性参照） | DESCRIPTIVE_CONTROL_REFERENCE |

JointRecovery 完整框架使用动态 g*/warm/spent、联合恢复估计和收益/成本请求排序。当前 STK 迁移版本 remaining-head 固定0，尚未验证该组件。名称统一不改变模型、算法、结果数值或冻结输入。

充裕预算主结果另见 reports/JOINTRECOVERY_QUALITY_RESULTS；旧 reports/SUBMITTED_METHOD_RESULTS 为效率/压力诊断，reports/P1_RESULTS 为静态 max4 / 固定调用等内部诊断。
