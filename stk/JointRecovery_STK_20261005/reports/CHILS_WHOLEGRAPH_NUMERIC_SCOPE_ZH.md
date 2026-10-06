# 原版 CHILS-p1 整图数值域：待真实运行的门控

当前状态：**WHOLEGRAPH_NUMERIC_GUARD_PASS（48/48图，4次真实原生检查）**。云端独立job `chils_wholegraph_gate` 已实际运行并完整下载至 `execution/chils_wholegraph_gate/out`；实际 completion、逐图 gates、全部native输入/输出及收据均保留。该通过记录仅允许原版CHILS整图执行，不证明算法收益或时间优势。

范围固定为 r004–r011 的 48 张完整 STK 图：r004/r005 为开发来源，r006/r007 为验证来源，r008–r011 为测试来源，每个来源含 R12/R8 × 地面间隔 170/340/680 秒。所有图使用原始完整窗口时长，原版源码、二进制均保持不变。范围仅覆盖 `CHILS-p1-c1`、seed 17；不扩展合作种群配置或 BR64，也不沿用旧 cap256 响应子图证书。

## 数值依据

原始 duration float64 按既有 P0 约定转为微秒计奖整数 `round(duration / 1e-6)`，不能以两个独立舍入端点之差替代。原始时长仍用于外部最终计奖。这是求解接口序列化，并不宣称物理精度达到微秒。

整图域冻结为 n≤20,919、m≤232,636、S≤9,383,000,000,000 个微秒计奖单位。旧 S≤2^40 子图范围不能直接覆盖这些图。新门控逐图计算以下必要条件：

| 检查 | 冻结边界与含义 |
|---|---|
| 累计权重与扰动 | 正权重，成本/邻接权重≤S；双权重与随机噪声 `2S+2^29` 小于有符号 64 位上界 |
| 日志容量 | `8n(S+129)` 小于 2^63；最大域约 1.571×10^18，仍小于 9.223×10^18 |
| 索引 | n、2n、2m+1 均在有符号 32 位内；边指针与计奖使用 long long |
| 十进制读入 | 检查解析器加数字字符再减 `'0'` 的中间值：`max(n,m,max_weight)+48` 不溢出 |
| 整数转 double | S<2^53，整数值转换可精确表示 |
| 执行 ABI | 真实 Linux int32、longlong64、size_t64，单线程单种群 |

证明绑定官方 commit `515952724cd3dcc6c4365a340ecf0f1da782119a`、完整 src/include 下 9 个 `.c/.h` 文件集合及原始二进制 SHA。CRLF→LF 归一化只用于核对源码身份；门控同时保存实际文件字节 SHA，供执行器独立重算。`local_search_explore` 的 `c<il` 检查发生在递增前；每轮日志重置；正整数贪心改善次数受 S 约束，扰动至多 129 次。上述范围只证明整数运算与索引有界，不保证分配成功或满足实际完整调用截止时间。

## 一次必要的真实检查

云端 `--mode run` 首先核对真实 ABI、官方完整源码/二进制和 48 图数值条件，随后只运行以下 4 次原版调用：

1. 三个小图：路径、团、混合兼容结构，均有 TOTAL>2^40；穷举得到最优成员集，并通过真实 `-i` 参数传给原版二进制。验证输出在原图可行、成员整数求和等于原生报告，并保留最优热启动。
2. 一张完整开发图 `JR-DUAL-r004-R12-g0170`，100 ms 原生搜索设置；用与 P0 同样的三种贪心候选生成共同 incumbent，传入全部节点/全部边及真实热启动，检查完整原图可行性、报告整数目标与实际成员吻合、未丢失热启动整数目标，另按原时长计奖。100 ms 不含完整调用开销，也不是在线性能实验。

只有以上实际检查全部通过，才允许每图输出 `WHOLEGRAPH_NUMERIC_GATE_PASS`，且所有 `checks` 为 true。准备模式只输出 PENDING；缺文件、源码不一致、原生失败、超时、输出不可行或计奖不吻合均不能通过。保留协议、源码证明、完整输入/热启动/输出、stdout/stderr、逐次收据与 SHA，不重跑旧 183 项检查。

## 调用与交付

脚本：`code/chils_wholegraph_numeric_gate.py`。只由总代理启动云端命令；下列路径使用云端实际位置替换。

```text
python code/chils_wholegraph_numeric_gate.py --root DATASET_ROOT --runtime-root PAPER_RUNTIME --graphs DATASET_ROOT/graphs --chils ORIGINAL_CHILS_BINARY --chils-source ORIGINAL_CHILS_SOURCE --out FRESH_GATE_OUTPUT --mode run
```

若只准备 manifest，使用 `--mode prepare`，仍需另用全新输出目录运行实测。输出核心文件为 `source_proof.json`、`protocol.json`、`guard.json`、`completion.json` 和 `gates/<graph_id>.json`；原生记录位于 `native_checks/`。几何/执行代理须再次核对 NPZ、实际源码集合、二进制、数值域以及每个布尔检查，才可进行完整图 baseline 比较。

这份文件是实验接口范围说明，不是论文结果。不把门控通过当作已发表方法的性能比较；后者仍须公平使用相同原始 incumbent、完整调用 D 与原时长计奖。
