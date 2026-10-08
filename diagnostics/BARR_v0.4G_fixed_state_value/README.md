# BARR v0.4G 固定状态计算价值诊断

本目录保存独立诊断工具。它从冻结 E-full 搜索中捕获已知严格正联合机会，再比较“立即执行该 pair”和“继续搜索”。这不是新的最终算法，也不使用 GNN 决策；复制源码中的学习模块不参与本协议。原有正式版本及标签不由本包修改。

每张图以 seed101 搜索360秒，在60/120/180秒之后依次捕获首个满足条件的不同 epoch，最多三个状态：两个单点收益都严格为负，联合收益为正。状态包含全部4个种群成员、队列和私有局部搜索状态、所有 RNG、归档、控制器、缓存及临时融合状态。没有可用状态的槽位保留缺失。

每个状态以 future seed 0/1901/1907/1913/1931 配对分叉。两臂各有新的360秒未来预算；共同解码与准备排除，动作、归档、共同待执行的融合反馈以及后续搜索计时。recover 只执行已经验证的 pair；两臂随后共用 pulse-only 局部搜索/融合策略，不再调用联合侦察、Kernel 或联合求解器。这是条件于已发现机会的动作价值实验，不能推出无条件机会频率或完整算法优越性。

在本目录使用 C++17、CMake≥3.16 构建：

~~~sh
cmake -S source/BARR_v0.4G -B source/BARR_v0.4G/build -DCMAKE_BUILD_TYPE=Release
cmake --build source/BARR_v0.4G/build --parallel 1
(cd source/BARR_v0.4G/build && ctest --output-on-failure)
~~~

原生入口接受用户提供的 BARR1 图：

~~~sh
source/BARR_v0.4G/build/barr_state_capture --input GRAPH.barr --output capture.json --state-dir NEW_SNAPSHOTS --seconds 360 --seed 101
source/BARR_v0.4G/build/barr_state_replay --input GRAPH.barr --snapshot STATE.bin --action recover --seconds 360 --future-seed 0 --output recover.json
source/BARR_v0.4G/build/barr_state_replay --input GRAPH.barr --snapshot STATE.bin --action continue --seconds 360 --future-seed 0 --output continue.json
~~~

所有输出路径应为全新且各臂不同。完整协议和字段见 fixed_state_design.json、source/BARR_v0.4G/docs/FIXED_STATE_SCHEMA.json。scripts/ 保存冻结的批处理、审计和分析工具；它们需要原协议登记、输入与运行环境，未打包的真实输入由使用者提供。examples/demo/ 是原源码自带的合成例子。

VALIDATION.json 记录真实服务器构建、6项 CTest、23项原源码 Python 测试、31项外部协议测试及实际 C++→Python fixture 联调通过。固定时钟 fixture 仅验证状态和接口，不能充当360秒真实性能结果。本包不含服务器凭据、SSH 部署工具、真实输入、二进制、云端原始结果或质量结论。首轮准备中的单测变量重名失败已原样保留，r1 仅改局部测试变量名称后重新验证。
