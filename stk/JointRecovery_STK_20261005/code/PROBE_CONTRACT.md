# P0 真实恢复探针契约

本程序只运行新 STK 图的 P0，不训练模型，不读取旧结果，不复用旧 calibration，不改变第一篇冻结代码。`--runtime-root` 指向只读算法闭包；输出协议记录每个依赖和本代码的 SHA-256。

## 图与收益

- NPZ `weights` 必须是原始 Decimal `end - start` 转 float64 的完整时长；保留原始 Decimal 端点。float64 端点的直接相减仅作 ≤1 ns 的 binary 舍入一致性检查，不改变更准确的原奖励。计奖使用原始秒和 `math.fsum`。
- `agents` 是卫星 owner 标签；全天同 owner 的接触仍可共选，实际冲突只由原始 edges 表示。
- `factor_indptr/factor_vertices` 是整日资源归属，不能当静态冲突团。P0 上界使用 actual local edges 推导的真冲突团划分。
- cap256、原始 incumbent 优先后 weight/ID 的恢复顺序，与冻结 `NeighborhoodScopeCache` 一致；每个候选必须与整个固定 base B 兼容。

## 真执行状态

初始完整调度实际执行全图三贪心 `w/(1+d)^e`，e∈{0,0.5,1}，取其最好结果。每个 action 执行同样三贪心并与实际可恢复 incumbent 比较，形成 W/L。全局 prefix 最好值形成当前 g*；其收益不算学习收益。

预声明 action seeds = `seed + trajectory * 1009`。每轨迹保存 prefix 后及一次预定 action 的真实 200 ms history 后两个状态。每个 alternative 克隆相同 original、B/R、实际 warm 和 g*；后一个 alternative 不继承前一个结果。历史已花的 action/cap/workpoint 留在覆盖表但不再调用，标注 unavailable。重复探针只是同状态有限搜索响应的复测，不是额外独立来源。

目标每图 12 个不同 controller states。去重键包含 original/best 调度、action scopes、warm 成员与 spent requests，排除 seed 标签和时间。相同完整调度可能有不同实际 warm/history 的分配状态；另报告 distinct original/best schedule 数，不把 controller state 数解释为 12 个物理独立样本。

## Native 数值与时间

CHILS `-p 1 -c 1 -r 17`。读取实际 src/include，检查 reward/cost/adjacent_weight 为 `long long`，并绑定实际二进制 hash。每个 recovery 域检查 n≤32760、正微秒权重和总 tick≤2^40。保留原 float 奖励；只为原生整数执行取最近微秒 tick，记录每顶点及加和舍入误差。原始端点/收益不修改、不制造相等，不能声称微秒物理精度或原 float 精确最优。

200/1000 ms 是 CHILS 自身 `-t` 搜索设置。本阶段不用旧完整 deadline 扣除 preparation；另实测 export/preparation、launch、native process、parse、完整 B∪T 可行性校验与原奖励重计、完整返回时间。`--repeats 2` 才能判断同方向的可重复预算差异。

程序只能在 native 最终输出时取得实际成员；日志中更早的内部声称不算可验证好解。warm 已在请求开始时持有，时间 0 可返回。超越 snapshot g* 的 native 好解最早可得时刻是完整 original graph 校验完成之后。父进程提供严格完整 deadline 的 P1/P2 要另测试。

## 输出与判断

每图目录保存 protocol、完整初态、trajectory scopes/coverage/history、snapshot request JSON、标量 CSV、summary。q+w(T)、(w(T)−L)/max(1,L)、[q+w(T)−g*]₊、完整成员和全部成本皆保存。无机会、失败、无输出、已花请求不删除。G1 的 30% 和 G3 的约 20% 是开发投入参考，不是样本筛选规则；G2 图信息优势与学习优势均不能由 P0 标签单独证明。

示例（只先跑一个图的 smoke，使用单独目录）：

```text
python p0_recovery_probes.py --runtime-root NEW/runtime --chils CHILS --chils-source CHILS_SOURCE --graph NEW/graphs/JR-DUAL-r000-R12-g0170.npz --out NEW/p0_smoke --states 1 --budgets-ms 200 1000 --repeats 1 --seed 17
```

正式 P0 将 out 换成 p0_probes，states=12、repeats=2。可以按图分配不同单核心并最多并发两个图，诊断中记录 inherited CPU affinity；正式完整 deadline 对比应串行测量。所有数据产物由负责人同步回本计划所在的 data 目录。
