# BARR v0.4 — 冻结研究版

这里管理 2026-10-07 完成六分钟服务器实验并独立审计的 **BARR v0.4 最终源码**。最终实现为冻结的 **E full**：B 的 pair-fusion 逻辑，加上经过验证的组件消融与诊断开关。该目录与仓库已有论文版本、STK 数据、其他算法版本相互独立。

## 版本与文件

| 文件 | 用途 |
| --- | --- |
| `source/BARR_v0.4-final/` | 原始冻结的 50 个源码、测试、合成示例文件，逐字节保留 |
| `FINAL_VERSION.json` | 原始发布元数据：源码 SHA256、实测 Linux 二进制 SHA256、冻结参数、审计与实验范围 |
| `run_v04_final.py` | 原始冻结入口，严格绑定当时实测 Linux 二进制的 SHA256 |
| `verify_source.py` | 本次版本管理新增的源码完整性检查，不启动搜索 |

标签：`barr-v0.4`。本目录不包含真实 CP-SCALE-AU-L002 数据、实验归档、云控制脚本、SSH 凭据或编译产物。四个 `examples/demo/contacts_*.npz` 均为 1000 节点的合成示例，不能用来替代 37421 节点的真实实验输入。

冻结源码内的 `0.4E`、部分旧脚本的 `0.3` 字符串保留原样以维持字节一致性。**旧 `source/BARR_v0.4-final/scripts/run.py` 不是本版本的推荐入口**，其默认短预算与模式不能代表本次实验协议。

## 检查源码

以下命令均从本目录执行：

```bash
python verify_source.py
```

此命令核对原发布元数据本身、50 个源码文件及原入口的 SHA256。它不依赖 NumPy，不编译、不运行求解器。

## 从源码构建

推荐 Linux，要求 CMake >= 3.16、C++17 编译器和系统 Threads。核心无需 PyTorch 或 CHILS；Python 数据工具需要 NumPy。

```bash
cmake -S source/BARR_v0.4-final -B source/BARR_v0.4-final/build -DCMAKE_BUILD_TYPE=Release
cmake --build source/BARR_v0.4-final/build --parallel
ctest --test-dir source/BARR_v0.4-final/build --output-on-failure
python -m pip install -r source/BARR_v0.4-final/requirements.txt
python -m unittest discover -s source/BARR_v0.4-final/tests -p 'test_*.py' -v
```

Python 测试中的可选学习检查需要 PyTorch，未安装时会按原测试规则跳过；v0.4 full 不要求安装它。构建输出保持在源码树的 `build/` 下，是因为原 Python 测试按此路径查找原生程序。

原服务器已经实际通过全部 5 项 CTest、23 项 Python 测试及独立诊断审计；本次 Git 上传不重新执行质量实验。`BARR_TEST_FIXED_CLOCK` 只在私有 `pulse_ablation_properties` 测试目标中启用；生产求解器使用真实 `std::steady_clock`。

## 六分钟运行示例

先将一个合成示例转换为原生 BARR1 输入。权重转为微秒整数；保存的初解存在时沿用它，否则使用 `common_three_greedy`。此转换不是重新生成真实数据集的冻结输入。

```bash
mkdir -p inputs
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'source/BARR_v0.4-final/python')
from barr_io import load_npz, write_native
graph = load_npz(Path('source/BARR_v0.4-final/examples/demo/contacts_00.npz'))
write_native(Path('inputs/demo.barr'), graph, graph.ticks(1e-6))
PY
```

自行编译的二进制可能与实测二进制 SHA256 不同，应使用下面的完整显式参数运行，记录新编译产物的 SHA256。以下配置与冻结原入口的 `OPTIONS` 一致：单线程、4 解种群、`pair/full`，原生搜索 360 秒，诊断快照关闭。`mkdir` 要求新的输出目录，防止覆盖已有结果。

```bash
set -e
mkdir -p runs
mkdir runs/demo-360-seed17
sha256sum source/BARR_v0.4-final/build/barr_solver > runs/demo-360-seed17/binary.sha256
sha256sum inputs/demo.barr > runs/demo-360-seed17/input.sha256
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
./source/BARR_v0.4-final/build/barr_solver \
  --input inputs/demo.barr \
  --output runs/demo-360-seed17/native_result.json \
  --events runs/demo-360-seed17/events.jsonl \
  --checkpoint runs/demo-360-seed17/checkpoint.json \
  --seconds 360 --seed 17 \
  --threads 1 --population 4 \
  --mode pair --rank heuristic \
  --recovery-backend hybrid --kernel-nodes 128 \
  --kernel-seconds 0.03 --challengers 12 \
  --proposals 6 --execute-top 2 \
  --factor-width 10 --factor-boundary 10 \
  --factor-entries 262144 --local-iterations 64 \
  --local-seconds 0.025 --gate-fraction 0.05 \
  --gate-warmup 30 --gate-cooldown 0.5 \
  --event-stale 8 --pair-slice 0.5 \
  --pair-max-seeds 0 --pair-fusion-every 4 \
  --pair-policy fusion-refine --pair-component full
```

360 秒包含求解器内部种群构造与初始化；外部 Python 输入准备、BARR 图文件读取及事后验证另计。它不是旧 `barr_run_v1` 全流程 360 秒协议。比较时需控制可用 CPU 核、单线程和完整预算，并保留失败或资源异常记录。求解结果还应在原始完整图上重新检查目标值和所有冲突边；真实任务另外检查资源时间线。

如果持有原实验归档中的实测 Linux 二进制，原 `run_v04_final.py` 可按原协议使用（本仓库不分发该二进制）：

```bash
python run_v04_final.py --binary /path/to/tested/barr_solver \
  --input /path/to/verified/input.barr --run-dir runs/frozen-360-seed17 \
  --seconds 360 --seed 17 --dry-run
```

去掉 `--dry-run` 才执行搜索，`--cpu` 可指定一个允许使用的 Linux CPU。原入口会拒绝非原实验二进制或已有输出目录。自行编译后的运行不应声称复现了原二进制或逐事件相同轨迹。

## 算法与结论边界

代码实现廉价配对机会筛选、正见证先保留、再按预算结构化细化，以及稀疏融合脉冲。完整的 ≤2 插入证书仅针对固定当前解下的相应范围；受限 motif 类证书和 partial/超时结果不能当作全配对无收益，更不能推出 3+ 插入或全局最优。

最终独立对照采用 **CHILS（4 解种群基础对照）**、单线程、原生 360 秒，在 8 配置 × 3 个新种子的 24 对中为 17 胜、7 负，平均配对相对差 +0.122365549%。这是按用户选择控制种群规模的对照；官方默认 16 解并非 16 线程。本发布核心不包含 CHILS 实现；`tools/chils_p1_fixture.cpp` 只是需要外部上游源的可选旧接口，不是上述完整对照的执行器。

同二进制 E 消融中，full 对 pulse-only 为 9 胜、15 负，平均配对 −0.006382914%。因此联合组件对最终质量的净贡献尚未得到支持，不能把“检测到正联合提交”直接当作最终净质量收益。8 配置来自同一物理母源；不等于 8 个独立数据集。结果摘要与审计范围保存在原发布元数据，真实数据和唯一完整中文报告不在本次代码上传范围。

## 后续维护

本标签记录实测冻结基线。后续开发在新分支或新版本目录进行；不要改写该标签。修改源码后应重新生成新版本的哈希与验证记录，保留本版负结果与证书范围。仓库根目录的旧 `MANIFEST.json` 仍属于原论文发布快照，本目录由自己的 `FINAL_VERSION.json` 管理。

核心源码许可证为 [MIT](source/BARR_v0.4-final/LICENSE)。`requirements-learning.txt` 和学习模块为可选历史组件，不是 v0.4 full 的必需依赖；涉及外部 CHILS 时须另行遵守其上游许可证与固定版本。
