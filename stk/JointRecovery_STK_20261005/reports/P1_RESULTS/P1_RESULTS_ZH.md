# JointRecovery：静态 max4 与固定调用内部诊断

**本文完整框架方法为 JointRecovery（JR，FullCapacity）；本目录 Capacity 为其静态 max4 内部诊断。P1 是 Independent-replacement+CHILS 对照，不是本文主方法。** 详见 README_METHODS_ZH.md。充裕求解质量主结果另见 reports/JOINTRECOVERY_QUALITY_RESULTS。

本文件是实验报告，不是论文正文。它只读取真实结果，不执行训练、native搜索或样本筛选；尚无结果的阶段明确为WAITING，不补零，也不输出优势结论。

三种证据分别报告：固定单workpoint的冷快照标签回放；含不同native成本的pooled回放；重新支付动作、scope、共同warm、推理、搜索、完整验证和caller返回的实际完整D执行。前两种不是完整时间优势证据。

按同一物理来源内等图、每图内等实际状态，最后对来源等权。两个拟合种子逐项保留；图中的浅带是两seed的[min,max]范围，不是置信区间。配置图、控制状态及两个seed不是新的物理来源。

## 拟合与来源

状态：ACTUAL_FOUR_FITS_COMPLETE。拟合来源r000–r003；开发来源r004–r005；验证r006–r007；测试r008–r011。

| 模型 | 状态 | 实际最终epoch | checkpoint已镜像 |
|---|---|---:|---|
| JointRecovery（静态 max4 内部诊断）；seed17 | ACTUAL_FIT_COMPLETE | 40 | True |
| JointRecovery（静态 max4 内部诊断）；seed29 | ACTUAL_FIT_COMPLETE | 40 | True |
| JR-CheapSummary（静态简化表示诊断）；seed17 | ACTUAL_FIT_COMPLETE | 40 | True |
| JR-CheapSummary（静态简化表示诊断）；seed29 | ACTUAL_FIT_COMPLETE | 40 | True |

Greedy在当前冻结实现中用q+w(W)对同一native请求菜单排序，仍允许共同native调用；它不是完全不调用native的Greedy-only策略。共同paid prefix收益另列，不能归给学习。P1用q+结构P1代理排序。

## 固定调用回放：单workpoint先看，pooled单列

### DEVELOPMENT / native_1000ms

| 方法 | 1call增秒↑ | 2call增秒↑ | 4call增秒↑ | 1call观察池后悔值↓ | 物理来源数 |
|---|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 586.088090 | 769.877603 | 837.380106 | 277.855652 | 2 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 734.939437 | 814.116376 | 850.481476 | 129.004305 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 626.627468 | 792.201380 | 837.114340 | 237.316274 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 573.963583 | 779.055699 | 837.380106 | 289.980159 | 2 |
| Independent-replacement+CHILS（静态独立替代对照） | 0.000000 | 7.028680 | 223.205820 | 863.943742 | 2 |
| Greedy-rank+CHILS（静态贪心排序对照） | 582.861435 | 760.656945 | 837.380106 | 281.082307 | 2 |
### DEVELOPMENT / native_10ms

| 方法 | 1call增秒↑ | 2call增秒↑ | 4call增秒↑ | 1call观察池后悔值↓ | 物理来源数 |
|---|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 586.045445 | 769.834957 | 837.337460 | 275.197675 | 2 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 734.896791 | 814.073730 | 849.360229 | 126.346329 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 626.584822 | 792.158734 | 837.071694 | 234.658298 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 573.920937 | 779.013053 | 837.337460 | 287.322183 | 2 |
| Independent-replacement+CHILS（静态独立替代对照） | 0.000000 | 7.028680 | 223.205820 | 861.243120 | 2 |
| Greedy-rank+CHILS（静态贪心排序对照） | 582.818790 | 760.614299 | 837.337460 | 278.424330 | 2 |
### DEVELOPMENT / native_200ms

| 方法 | 1call增秒↑ | 2call增秒↑ | 4call增秒↑ | 1call观察池后悔值↓ | 物理来源数 |
|---|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 637.503362 | 789.073819 | 841.885060 | 225.936823 | 2 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 769.386241 | 823.572275 | 849.614114 | 94.053944 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 665.919200 | 811.780973 | 841.619294 | 197.520985 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 623.899783 | 799.730987 | 841.885060 | 239.540402 | 2 |
| Independent-replacement+CHILS（静态独立替代对照） | 0.000000 | 10.268653 | 236.277798 | 863.440185 | 2 |
| Greedy-rank+CHILS（静态贪心排序对照） | 632.403801 | 781.726066 | 841.885060 | 231.036384 | 2 |
### DEVELOPMENT / native_50ms

| 方法 | 1call增秒↑ | 2call增秒↑ | 4call增秒↑ | 1call观察池后悔值↓ | 物理来源数 |
|---|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 586.088090 | 769.877603 | 837.380106 | 275.197675 | 2 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 734.939437 | 814.116376 | 849.402874 | 126.346329 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 626.627468 | 792.201380 | 837.114340 | 234.658298 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 573.963583 | 779.055699 | 837.380106 | 287.322183 | 2 |
| Independent-replacement+CHILS（静态独立替代对照） | 0.000000 | 7.028680 | 223.205820 | 861.285766 | 2 |
| Greedy-rank+CHILS（静态贪心排序对照） | 582.861435 | 760.656945 | 837.380106 | 278.424330 | 2 |
### DEVELOPMENT / pooled_workpoints

| 方法 | 1call增秒↑ | 2call增秒↑ | 4call增秒↑ | 1call观察池后悔值↓ | 物理来源数 |
|---|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 586.088090 | 586.088090 | 637.503362 | 277.855652 | 2 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 734.939437 | 735.994248 | 769.386241 | 129.004305 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 626.584822 | 626.627468 | 665.919200 | 237.358920 | 2 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 573.963583 | 573.963583 | 623.899783 | 289.980159 | 2 |
| Independent-replacement+CHILS（静态独立替代对照） | 0.000000 | 0.000000 | 0.000000 | 863.943742 | 2 |
| Greedy-rank+CHILS（静态贪心排序对照） | 582.861435 | 582.861435 | 632.403801 | 281.082307 | 2 |

该菜单可以选择不同native时长，固定调用数不等于匹配计算成本；此表只描述冷快照观察标签上的选择，不支持实际完整D优势。

## 实际完整D与caller返回

### CALIBRATION / calibration_60s / D=60.000000s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 2126.165126 | 766.085813 | 100.00% | 3.667 | 2.346553 | 2 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 2318.024793 | 957.945481 | 100.00% | 4.000 | 2.473849 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 2147.447764 | 787.368451 | 100.00% | 4.000 | 2.428474 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 2147.447764 | 787.368451 | 100.00% | 4.000 | 2.381501 | 2 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 1360.079313 | 0.000000 | 100.00% | 2.333 | 1.484727 | 2 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 2147.447764 | 787.368451 | 100.00% | 4.000 | 2.173590 | 2 | 0 | 0 |
### VALIDATION / quarter / D=0.974731s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 279.155926 | 133.941888 | 8.33% | ≥0.750 | 0.990419 | 2 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 82.800245 | 9.616465 | 8.33% | ≥1.000 | 0.986821 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 449.627017 | 112.229910 | 16.67% | ≥2.167 | 0.985976 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 928.358497 | 296.957720 | 33.33% | ≥1.000 | 0.981185 | 2 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 1311.471960 | 1.733936 | 75.00% | ≥1.833 | 0.838050 | 2 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 1056.747985 | 430.899608 | 33.33% | ≥1.833 | 0.979024 | 2 | 0 | 0 |
### VALIDATION / half / D=1.949462s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 380.516095 | 31.572182 | 25.00% | ≥1.583 | 1.943705 | 2 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 166.439518 | 25.575379 | 8.33% | ≥1.250 | 1.962365 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 1990.171075 | 575.606961 | 83.33% | ≥3.833 | 1.410779 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 907.960610 | 98.025884 | 58.33% | ≥3.083 | 1.682458 | 2 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 1263.446632 | 0.000000 | 75.00% | ≥1.833 | 1.487526 | 2 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 233.566682 | 9.616465 | 16.67% | ≥2.250 | 1.959552 | 2 | 0 | 0 |
### VALIDATION / wide / D=3.898924s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 2219.320073 | 581.255362 | 100.00% | 4.000 | 2.422663 | 2 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 2204.985943 | 566.921232 | 100.00% | 4.000 | 2.487553 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 2225.438468 | 587.373756 | 100.00% | 4.000 | 2.384057 | 2 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 2219.320073 | 581.255362 | 100.00% | 4.000 | 2.312335 | 2 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 1638.064711 | 0.000000 | 100.00% | 2.667 | 1.679945 | 2 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 2219.320073 | 581.255362 | 100.00% | 4.000 | 2.124856 | 2 | 0 | 0 |
### TEST / quarter / D=0.974731s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 573.315168 | 139.723816 | 20.83% | ≥0.667 | 0.985163 | 4 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 199.470562 | 30.494000 | 8.33% | ≥0.875 | 0.987177 | 4 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 616.527236 | 199.624806 | 20.83% | ≥1.667 | 0.985543 | 4 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 745.898065 | 201.946135 | 25.00% | ≥1.417 | 0.984009 | 4 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 909.570105 | 0.000000 | 50.00% | ≥1.417 | 0.882721 | 4 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 511.768543 | 137.744195 | 25.00% | ≥2.167 | 0.970764 | 4 | 0 | 0 |
### TEST / half / D=1.949462s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 895.582995 | 353.735280 | 29.17% | ≥2.167 | 1.930794 | 4 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 686.488416 | 189.986033 | 29.17% | ≥2.167 | 1.881161 | 4 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 2550.928897 | 817.421155 | 95.83% | ≥3.917 | 1.322915 | 4 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 1034.273628 | 295.773750 | 45.83% | ≥2.417 | 1.866448 | 4 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 902.276501 | 0.000000 | 54.17% | ≥1.542 | 1.512264 | 4 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 614.651801 | 178.270575 | 25.00% | ≥2.292 | 1.951285 | 4 | 0 | 0 |
### TEST / wide / D=3.898924s

| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JointRecovery（静态 max4 内部诊断）；seed17 | 2509.600171 | 703.055325 | 100.00% | 4.000 | 2.362871 | 4 | 0 | 0 |
| JointRecovery（静态 max4 内部诊断）；seed29 | 2614.343814 | 807.798968 | 100.00% | 4.000 | 2.417900 | 4 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed17 | 2575.064072 | 768.519226 | 100.00% | 4.000 | 2.351665 | 4 | 0 | 0 |
| JR-CheapSummary（静态简化表示诊断）；seed29 | 2509.600171 | 703.055325 | 100.00% | 4.000 | 2.372109 | 4 | 0 | 0 |
| Independent-replacement+CHILS（静态独立替代对照） | 1806.544846 | 0.000000 | 100.00% | 2.667 | 1.591830 | 4 | 0 | 0 |
| Greedy-rank+CHILS（静态贪心排序对照） | 2509.600171 | 703.055325 | 100.00% | 4.000 | 2.111409 | 4 | 0 | 0 |

strict增秒采用driver冻结口径：已实际执行且收据明确失败/未交付的运行有显式strict=0；原始gain、成本缺测仍保留未知。未启动运行或缺运行receipt是WAITING，不填0。有效按时返回率要求有效完整收据且caller实际返回不超过D，执行失败不计成功。

完整caller曲线中的strict共同prefix之外净收益，要求实际按时交付；内核内部已有收益或迟到收益不能冒充caller收益。部署驻留图/模型与原始全图初态是协议声明的setup，整体CLI wall另存，不能重复添加或冒充D内成本。未收到finished时，已观察native完成调用仅是总调用数的下界，表中以≥明确标记。

## 覆盖、等待项与可追溯数据

回放图显示阶段为DEVELOPMENT，完整D图为TEST；分别按TEST→VALIDATION→DEVELOPMENT顺序选择已完整的实际证据，不依优势挑选。未完成组仍保留CSV与WAITING标签；部分来源覆盖不是完整阶段确认。

- DEVELOPMENT / ACTUAL_COLD_CLONE_REPLAY_COMPLETE / E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005\execution\p1_followup\out\fixed_calls
- CALIBRATION / ACTUAL_MATRIX_COMPLETE / E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005\execution\p1_followup\out\calibration
- VALIDATION / ACTUAL_MATRIX_COMPLETE / E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005\execution\p2_confirmation\out\validation
- TEST / ACTUAL_MATRIX_COMPLETE / E:\01-Joycecyq\2026-AAMAS\data\两篇论文数据定制化构建\JointRecovery_STK_20261005\execution\p2_confirmation\out\test

可追溯文件：fit_history.csv、fit_receipts.csv、fixed_call_all_rows.csv、fixed_call_source_equal.csv、actual_deadline_all_receipts.csv、actual_deadline_source_equal.csv、seed_ranges.csv。所有输入和图输出哈希见P1_REPORT_MANIFEST.json。报告没有自动宣称模型优势；实际差异必须结合全部来源、强廉价对照和完整成本解释。
