# P1 最小拟合与分配契约

本阶段代码不代表已经执行或取得优势。只有负责人签收 P0 的实际净机会、写出 `allow_p1_execution: true` 的 gate 后，才可启动 `fit`。模块导入不读取实验结果、不训练、不启动求解器。

## 来源与规模

拟合只用 r000–r003；r004–r005 是 TRAIN 族内部开发 holdout，不能称最终 validation。每来源必须完整包含固定 R8/R12 × gap170/340/680 六图，所有实际 controller states 留存。r006–r007 留给 P2 validation 冻结；r008–r011 是不按机会/胜负筛选的 test。

只拟合 `ResidualCapacity` 与 `ResidualCheapSummary`，各 seeds17/29。沿用冻结的 h32、两层 graph encoder、h32 summary head、Adam0.001、40 epochs、每步4个完整图、辅助权重0.05（仅图模型）、排序权重1、temperature0.2。不运行全量消融、扩深度、找 best validation epoch 或搜超参数。保留 final epoch40 checkpoint。

损失按同状态 request 均值，再按 state/graph/source 分层均衡。P0 重复执行按同 action/workpoint 聚合实际平均返回 reward 和平均 mask（软 occupancy 标签），避免同输入的 wall-clock 扰动被强制标为冲突排序；重复数不能增加独立样本数。失败时已验证的 wrapper warm fallback 是真实执行返回，但失败标记仍保存，不能伪称 native 输出或隐藏执行风险。

## 单位、输入与标签

scale = r000–r003 各 R12-g0170 完整接触时长中位数的中位数，至少1秒。scale 仅由拟合来源的原始 contact duration 定义，之后冻结。q/L/U、权重、signed q+w(T) 目标和当前 g* 都除以同一 scale；预测乘同一 scale 回原始秒。不将原奖励改成 tick 或图总权重，不把 q 与 recovery 混用单位。

模型沿用实际 warm 的 L–U residual floor：`recovery=L+(U−L)*fraction`，q 为已知 offset。只编码当前原图、固定 B/R、真正已执行 warm 与 native workpoint。没有 P0 总 deadline 监督，因此 remaining-time head 特征在 fit/回放/实际运行统一固定0；真实剩余时间只负责外部准入/停止。本版不能主张剩余时间条件组件优势。

`features/*.observable.pt` 与 `.targets.pt` 分开保存，feature_index 保留真实 warm、ordered R/base、原图地址与 SHA。没有未来 T、失败结果、净收益或排序 oracle 进入模型输入。输出记录完整 P0/图绑定，能够从原会员集重计奖励。

每 epoch 保存实际 loss 各项、每 optimizer step 的梯度范数分布/非零比例，以及参数变化。平坦 regret 或单个零梯度不能被写成梯度崩塌结论。

## 固定调用数选择

`allocate` 比较 Capacity17/29、CheapSummary17/29、P1、Greedy。所有方法访问相同 unique available action/workpoint 集合，最多1/2/4次调用，cold clone 保持相同原状态和 warm。已花 history 请求不重新分配。

Greedy 按已执行 q+L 排序；P1 按 q+真实局部 clique-overlap P1 scalar 排序。即便代理认为额外收益为0，固定调用数试验仍对所有控制给同样 K 请求，不能靠让廉价控制不调用而制造额外 native 搜索优势。报告 snapshot 之外净增秒数、same-state conditional regret、来源级均值。

这是利用已有 cold-clone 实际标签进行的固定调用数选择回放；不宣称实际 online、多轮自适应或时间成本胜利。两次真实重复返回先平均，不挑最好一次。

输出同时包含每个 native workpoint 固定相同预算的独立表（`request_menu=native_200ms` 等），以及 pooled-workpoint 请求选择；不能把 pooled 的不同native计算投入冒称纯排序优势。所有方法相同分数时统一按action ID、native预算降序打破平局，防止仅因为廉价控制默认先选最短时间而被削弱。

## 实际完整 D 接口

`actual-policy` 在 Linux CPU 云端使用 resident graph/model 与现有已验证 original S。全图三贪心初态是初始调度 setup，单独报告；不是每次 decision 里的免费搜索。每个实际 decision 在 D 内重新建立 action/menu/scopes、执行三共同贪心 warm、获得 prefix g*、准备特征与推理，然后执行冷 clone 请求。不会把从 P0 文件读来的 warm 当免费准备。

native、export、解析、完整 B∪T 校验、worker 到 caller 的成员传输、caller 独立全图校验与返回均在 D 内。caller 始终持有原 S/此前已验证最好成员；未完成或迟到的 worker 不允许消耗该方案。每个新执行 job 使用独立进程组，停止仅影响本 job 的 native 子进程。

所有策略共享 fit 来源测得的 conditional request p95，用实际 remaining 做准入；这是估计，不是延时保证。完整 D 在开发端实测4调用策略成本后由负责人统一冻结（最大策略 p95 加≥20%起始余量），再做0.5/0.25 D。不能在 validation/test 看输赢以后加时。当前代码只接受显式 D，绝不推断某个有利 deadline。

记录 worker内部ready、caller收到/完整验证/实际返回及missed_return，strict on-time gain 对迟到归零。此软件边界不能证明物理硬实时；cleanup/reaping 是返回后的资源清理，不能冒充已可交付收益。

## 显式命令

Gate 示例（由负责人根据实际 P0 写，不能由拟合代码自行放行）：

```json
{"allow_p1_execution":true,"reason":"Actual P0 net recovery observed; complete evidence reviewed.","p0_analysis_sha256":"..."}
```

```text
python p1_fit_and_allocate.py fit --runtime-root NEW/runtime --graphs-dir NEW/graphs --p0-root NEW/P1_labels --p0-gate NEW/protocol/p1_gate.json --out NEW/p1_fit --device cuda
python p1_fit_and_allocate.py allocate --runtime-root NEW/runtime --graphs-dir NEW/graphs --p0-root NEW/P1_labels --fit-root NEW/p1_fit --sources 4 5 --out NEW/p1_fixed_calls --device cpu
python p1_fit_and_allocate.py actual-policy --runtime-root NEW/runtime --graph GRAPH.npz --fit-root NEW/p1_fit --chils CHILS --chils-source CHILS_SOURCE --policy ResidualCapacity --fit-seed 17 --deadline-seconds FROZEN_D --max-calls 4 --out NEW/actual_one_decision --device cpu
```

需要 runtime 内冻结的 `src/joint_recovery` model/packing 模块；无需旧 fit/label/calibration/结果。P0生成的新来源 r002–r005 数据及其完整实际标签需先完成，不能从旧训练集填补。
