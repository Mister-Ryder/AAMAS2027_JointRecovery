# 已发表基线与新 STK 测试的对齐盘点

**盘点结论：最终方法性能报告应加入独立的整图已发表优化方法比较，不能只依靠 Capacity/CheapSummary/P1/Greedy 内部优先级对照。** 第一篇仓库已有真实发布基线接口、官方来源和旧执行结果。最小高价值组合是：CHILS-p1 整图 warm 改进、HiGHS-MILP 整图有限优化、显式改版 WeightedBR-64compat，以及真实付费 AnytimeGreedy。前两类整图性能和同恢复器优先级诊断回答不同问题，两者都应保留。

当前没有执行这些方法在新 STK 图上的测试。本文件是只读盘点和最小对照建议；未修改 P1/P2 协议、预算、数据、旧代码，未启动云任务。旧通用公开图、旧应用图和新 STK 图的结果严格分开。

## 1. 正式出处、现成接口及适用限制

正式发表信息已从下列出版社/会议及官方实现页面核实，引用中不使用预印本。

| 方法 | 正式发表出处 | 仓库实际可复用接口 | 新 STK 加权图的条件限制 |
|---|---|---|---|
| CHILS-p1 | Großmann, Langedal, Schulz, **SEA 2025**, LIPIcs 338, 22:1–22:18：[论文](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.22)，[官方源码](https://github.com/KarlsruheMIS/CHILS) | `experiments/v4_published_weighted.py: solve_published / command`；`experiments/v3_published_baselines.py: solve_chils / read_chils_solution` | 官方 `-i` 支持真实 S warm start；p1/c1 单线程。源权重为64位，但已有完整数值收据仅覆盖 S_ticks≤2^40，**不可直接复用来宣称新整图数值域已通过**。需新整图范围派生收据。 |
| WeightedBR | Lamm et al., **ALENEX 2019**, 144–158：[论文](https://epubs.siam.org/doi/10.1137/1.9781611975499.12)，[KaMIS 官方源码](https://github.com/KarlsruheMIS/KaMIS) | 原版 `solve_published(method="WeightedBR")`；兼容版 `experiments/v4_MT_baselines.py: solve_modified` + `v4_weighted_br_compat.py` | 原版32位范围不支持本次微秒整图。已有 **WeightedBR-64compat 是显式修改实现**：扩宽 weight/gain/flow 等存储，stable sort 替换 weight-index counting sort；不能写成未修改的原版。现收据同样 S_ticks≤2^40、n≤32760，整图需派生更大域收据，或先作为 cap256 恢复域对照。官方 CLI 无已验证外部 warm start；S 在外部 acceptance floor 中保留。 |
| m²wis / m²wis+s（代码名 M2WIS） | Großmann et al., **JGAA 2024**, 28(1), 439–473：[论文](https://jgaa.info/index.php/jgaa/article/view/2997) | `v4_baseline_setup.METHODS` 和 `solve_published(method="M2WIS"/"M2WIS+s")`；KaMIS `mmwis` | 原版32位总权重/中间域不支持本次微秒输入；已有64BITMODE并不解决其全部 weighted 中间量。无可直接复用的64位完整兼容实现或 warm start 接口；不能把 N/A 计成失败零增益来制造胜出。 |
| Struction-fast / strong | Gellner et al., **ALENEX 2021**, 128–142：[论文](https://epubs.siam.org/doi/10.1137/1.9781611976472.10) | `solve_published(method="Struction-fast"/"Struction-strong")`，官方 KaMIS 嵌套构建 | 原版32位 weighted 范围不支持新整图；现有原版单位权重 guards 也不能转用于完整时长权重。不是本轮最快可直接执行候选。 |
| HiGHS-MILP | 软件算法正式期刊依据：Huangfu & Hall, **Mathematical Programming Computation 2018**, 10, 119–142：[论文](https://link.springer.com/article/10.1007/s12532-017-0130-5)；[官方 HiGHS](https://highs.dev/)、[SciPy MILP 接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) | `experiments/v4_published_weighted.py: solve_highs_reference / solve_published(method="HiGHS-MILP")`；局部接口 `v4_backends.highs_backend` | 可建原边 x_u+x_v≤1、0/1变量、微秒目标；输入整数总和远低于2^53。但解、gap/dual仍是数值结果，不能冒充独立精确认证。SciPy MILP没有 x0：S由外部 acceptance floor保留，不能声称 solver获得 warm start。期刊论文讲 LP simplex，**不是专门提出本 MWIS/MIP 基线的论文**，应同时准确引用实际软件接口。 |
| DIFUSCO | Sun & Yang, **NeurIPS 2023**：[正式会议页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html) | `v3_published_baselines.OfficialDIFUSCO.solve`；已有 SAT 模型、50步×1/4 samples | 现接口明确拒绝非单位权重，released MIS模型不能直接承担本 STK 时长加权任务。把所有 contact 权重改为1会改变目标，不是同任务直接对照。 |
| COExpander | Ma et al., **ICML 2025**, PMLR 267, 42130–42164：[正式会议页](https://proceedings.mlr.press/v267/ma25r.html) | `experiments/v3_coexpander_baseline.py: OfficialCOExpander.solve / unit_input`；SAT-CM recipe | `unit_input` 明确只接受全1权重。本轮没有发布模型对应的 weighted 改版或重新训练结果；同样不能偷偷改目标后写作加权 STK 的先进方法比较。 |

同一算法的 population/thread/workpoint 配置不能冒称多个独立已发表方法。CHILS-p16-c16 是16线程配方；当前 learned-controller 是单CPU native恢复，不能混入同资源主表。旧 CHILS-p1 数值域审查还明确没有覆盖 p16 合作路径，不能仅改 CLI 后宣称已可执行。

第一篇 `tools/` 中本地有 `CHILS-v3`、`KaMIS`、`DIFUSCO-v3`、`COExpander-v3` 源目录；实际执行/build收据也已留存。**本次未 SSH 检查旧云二进制是否仍在**，因此“代码可复用”不等于确认旧云 executable 路径仍可直接调用。新 P0 已使用的 CHILS 二进制 SHA 为 `19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0`，与仓库接口的冻结 p1 binary 一致；完整原始构建身份仍由实际 runner 绑定。

## 2. 新图的真实数值域：为什么不能直接跑全部原版基线

仅读取已交付72张图的 JSON sidecar汇总，不加载图重建、不重新验证可行性：

- 最大 n=20,919，最大 m=232,636，最大单窗口时长718.955946664秒。
- 全顶点时长总和范围6,247,241.5884685535–9,382,668.260357596秒。native微秒 tick总和约6.247×10^12–9.383×10^12。
- 原版32位总权重上界为约2.147×10^9/4.295×10^9 ticks；已有 CHILS-p1/WeightedBR-64compat保守冻结域为2^40≈1.100×10^12 ticks。新整图超出两类现有域。
- 单个 cap256域保守总和≤256×最大窗口时长×10^6≈1.841×10^11 ticks，落在现有64位兼容/CHILS p1保守总和域内，仍需按实际域记录 n/m/S 与收据；不代表原版32位 KaMIS 支持它。

CHILS 的64位算术本身不意味着整图无法运行。现有 p1源推导中的队列/日志分配包络是 `8*n*(S_ticks+129)`，以及 `2*S_ticks+2^29` 等。用上述所有图的保守 n/S 组合计算，日志字节包络约1.571×10^18，小于2^63−1；这提示可用**原算法不变、仅新整图域的窄派生审查**支持新数据，而不需要重做旧公共实验。但本文件只是算术范围线索，不是一份新已签发的源/二进制通过收据。不得直接把现有2^40阈值删掉或改成 int64上限来跳过该契约。

WeightedBR-64compat若扩大整图范围，同样必须承接它真实源修改和独立guard的边界；除总权重外还有 flow、gain、日志/内存分配及索引包络。不能因编译成功就声称整个新域通过。

STK原始 reward仍是CSV端点差对应的float64完整时长。所有native基线统一使用**与P0相同的微秒 reward取整**，返回后按原始完整时长重评分并保留原S；记录单点和总取整误差。不能使用 `end_ticks-start_ticks` 冒充 duration tick，也不能将原始9位端点或资源边改成微秒后重新建图。微秒数值表示不提供物理精度保证。

## 3. 最小高价值对照名单与执行边界

### A. 整图性能表：检验拟投稿方法是否值得替代成熟优化

| 建议纳入 | 最小配方 | 回答的问题 | 前提 |
|---|---|---|---|
| **CHILS-p1 whole-graph warm** | 官方 p1/c1、相同原 S、相同原边和 reward microticks，直接全图改进 | 学习局部协调是否优于成熟 MWIS 搜索，而不是只优于内部 P1控制 | 先补上述整图源范围派生收据；native allowance须留出完整导出/启动/parse/validation/return成本 |
| **HiGHS-MILP whole-graph** | 同图0/1模型、1线程、固定 time limit、外部保留原S | 与通用有限优化器比较实际交付质量/返回率 | 记录 SciPy/HiGHS版本；无warmstart、数值gap不当精确证明；无新解/超时/失败照实 |
| **WeightedBR-64compat whole-graph，条件就绪后纳入** | 显式modified ALENEX2019兼容版、1线程、同原S acceptance floor | 覆盖与ILS不同的branch-and-reduce家族 | 需扩大当前已审数值域的独立收据；实现标签明确modified，不能写original WeightedBR |
| **AnytimeGreedyPortfolio** | 同原S、真实付费菜单/原边、每个可行改进及时验证保留 | 防止方法只胜出于较弱或批处理超时的廉价控制 | 仓库自实现经典组合对照，不是独立已发表完整算法；与当前“Greedy+CHILS”优先级控制另列 |

若 WeightedBR整图域暂未就绪，先执行前两种独立优化方法＋真实greedy和拟投稿模型；明确覆盖边界，不能声称已有多个**原版已发表 MWIS 算法**完成 STK整图比较。即便期刊软件与modified方法具有正式依据，也不能把它们包装成三种未修改的已发表 MWIS算法。

### B. 同恢复域表：检验响应结构与请求选择机制

保留当前 Capacity两个seed、CheapSummary两个seed、P1、Greedy +同CHILS-p1内核、四工作点、完整D。它隔离优先级影响，有价值，但不能取代 A 表。

可加一个高价值而较小的 **WeightedBR-64compat cap256恢复器** 或 **HiGHS cap256恢复器** 对照：固定同动作域和现有warm，采用事先固定的廉价请求顺序/同菜单，所有准备、冷启动、求解、重评分均计费。这是论文方法的协议内适配，必须写“published solver adapted to the same recovery scope”，不是原论文整图配方；不要从已看到的测试结果选一个有利顺序、域或工作点。scope域成本与整图成本分开，不把一个恢复标签当一项完整调度性能。

作为后续独立算法候选，DynWVC1/2有正式 **IJCAI2018**依据：[会议页](https://www.ijcai.org/proceedings/2018/196)。MWVC补集恰为同图MWIS，可比性原则成立。但当前第一篇没有已绑定的 DynWVC实现/适配/执行收据，官方 weighted算术域与初始化接口尚未盘点；因此它不是本轮“现成可执行”承诺，也未列入本轮最小就绪名单。需要增加原版独立MWIS/MWVC家族时可再落实，不能凭正式出处造一行结果。

## 4. 相同 original S 和完整 caller D 的公平合同

1. 从新数据固定构造相同 original S：`code/p0_recovery_probes.py: global_incumbent` 实际执行全图三个指数0/0.5/1的greedy并择最佳。新 `p1_actual_policy` 的 initial-mode也以此为起点。各方法使用同一S成员和原时长价值，不另换弱初态给我方制造相对改善。
2. 输入图/模型/二进制登记等 resident初始化边界遵循已经冻结的 P2，而每决策导出、warm文件、矩阵/子域、启动、搜索、parse、原图全成员验证、原时长重评分和最终调用者返回都计入 D。一次性setup成本单列；不能把我方resident成本和外部solver cold成本混称同样的边界。
3. 所有方法共用已经冻结的 D：wide=3.8989239679276935、half=1.9494619839638467、quarter=0.9747309919819234秒。不要据验证/测试性能另调 D。新增已发表对照可用**独立追加协议和记录**，不覆盖当前P2cells、拟合或选择结果。
4. native搜索 t不能直接取完整D后再加导出/启动等成本，否则必然迟到的表不能证明我方时间优势。固定的开发成本余量或实际剩余时间用于确定 allowance；真实外部caller测得最终返回才判断按时，不使用native内部“最好解时间”或日志标注冒充已物化返回。
5. 返回后重新检查全部原图边并按完整原时长评分：若新解差于S/失败/无有效解则保留S，记录其真实结果类别；若完整调用者迟到则strict可交付额外收益按协议记0，晚到质量单列。预先不支持的数值域 N/A 是不可比，不能变成一次失败的零收益并制造优势。
6. 主要表至少给原完整目标秒 ↑、strict_on_time增秒 ↑、相对原S增量% ↑、共同paid-prefix外净增秒 ↑（我方分项）、完整caller耗时秒 ↓、按时率 ↑、实际有效返回/声明执行数 ↑、失败/迟到/不可用。分母和成本边界清楚；物理源→配置等权，两seed范围不冒充CI。
7. 只在开发源决定对照实现、成本余量和参数；P2 r006/r007及r008–r011的所有六配置保留。不以“某类图基线过强/方法没优势”为由切场景或删失败来源。

## 5. 可复用 API 的具体接入注意事项

- [v4_published_weighted.py](E:/01-Joycecyq/2026-AAMAS/第一篇/experiments/v4_published_weighted.py) 提供 `solve_published(graph, method, seconds, seed, setup=..., receipt_directory=..., initial_mode="published", decision_seconds=...)`，返回 `(accepted_mask, raw_mask, details)`。`seconds`是soft搜索额，不是完整 D；外层必须再计原时长重评分和最终caller返回。不能把该函数已有整数tick acceptance直接视为原时长 acceptance：还需用 `raw_mask` 对原float64时长重新择优保留S。
- 整数图容器是 [PublicGraph](E:/01-Joycecyq/2026-AAMAS/第一篇/src/joint_recovery/v4_public_data.py)，字段 `weights:int64, edge_u/v:int32, initial_mask:int8, cliques, metadata`，`validate_mask/objective`可复用。新STK资源成员CSR不是静态clique，不能照资源全天成员填 `cliques`；可置空并使用真实原边。原float64时长另保留，不丢弃。
- 原官方CHILS CLI由 `command("CHILS-p1",...)` 生成：`-g graph -p 1 -c 1 -r seed -t native_allowance -o solution -i original_S.ids`。`read_chils_solution`接一基顶点ID；`read_vertex_id_solution`亦有严格唯一范围检查。
- [v3_published_baselines.solve_chils](E:/01-Joycecyq/2026-AAMAS/第一篇/experiments/v3_published_baselines.py) 默认 `weight_scale=1000`；[v4_backends.chils_backend](E:/01-Joycecyq/2026-AAMAS/第一篇/experiments/v4_backends.py) 对非整数 reward也会选1000。**不能原样用于本轮微秒合同**。需显式同P0的1e6刻度/整数tick导出，保留原时长rescore和取整界。当前新P0的 `NativeCHILS` 原生只接局部 `scope`，不能仅把整个图当替换数组就继承其2^40域通过状态。
- [v4_MT_baselines.solve_modified](E:/01-Joycecyq/2026-AAMAS/第一篇/experiments/v4_MT_baselines.py) 复用 `WeightedBR-64compat` 的真实modified源和release build收据；其源数值范围由 [v4_weighted_br_compat.py](E:/01-Joycecyq/2026-AAMAS/第一篇/experiments/v4_weighted_br_compat.py) `range_reason`固定。MT6 runner有旧公开图/seed/目录合同，**不要直接改旧runner来输入新STK**；复用API做独立新增适配即可。
- HiGHS整图API是 `solve_highs_reference(PublicGraph, seconds, seed)`，局部 `highs_backend(graph,scope,workpoint,warm,remaining_seconds)`；后者把已知warm作为外部fallback，绝不宣称传入solver。新整图定时必须把MILP矩阵构建也计入。
- 真实greedy接口 [run_anytime_greedy](E:/01-Joycecyq/2026-AAMAS/第一篇/src/joint_recovery/v4_anytime_classical.py) 参数 `graph, selected, proposal_factory, deadline_seconds, caps, exponents, resource_cliques, scope_cache_factory, known_warm_factory`；新案同cap256使用固定 `caps=(256,)`及兼容原动作的scope cache，不把资源CSR当全天clique。

## 6. 以前确实做过的比较及差距：不转写成新 STK 结果

### 旧公开单位权重图：2040条已执行，尚非我方对比

固定40图（DIMACS16、SATLIB-UF18、CBS6）×三共同初态；1878有效返回，162进程失败保留。原始发布表 [UNIT_PUBLIC_ORIGINAL_INDEPENDENT_RESULTS.md](E:/01-Joycecyq/2026-AAMAS/第一篇/docs/research_v4/UNIT_PUBLIC_ORIGINAL_INDEPENDENT_RESULTS.md) 和 [published_original_table.csv](E:/01-Joycecyq/2026-AAMAS/第一篇/results/v4_unit_public_original/independent_audit_01/published_original_table.csv) 包含全部17配置。

| 旧公开配方 | 等三族相对共同初态增益% ↑ | 成本秒 ↓ | 有效返回/120 ↑ | 可用于新STK主表？ |
|---|---:|---:|---:|---|
| CHILS-p1，2s soft搜索 | 24.468 | 2.0438 cold | 120/120 | 只有新实测后可用 |
| m²wis，2s | 13.548 | 5.5979 cold | 93/120 | 本微秒加权域不支持，旧值不可迁移 |
| Struction-fast，2s | 13.607 | 2.0483 cold | 120/120 | 同上 |
| WeightedBR原版，2s | 9.437 | 5.2060 cold | 93/120 | 同上 |
| DIFUSCO SAT50×1 | 8.724 | 1.4209 resident | 120/120 | 单位目标，不可直接迁移 |
| COExpander SAT-CM | 6.347 | 0.0730 resident | 120/120 | 单位目标，不可直接迁移 |

这些是相对各公开图共同初态的收益，不是最优率，也不是我方超越这些方法的百分比。cold CPU与GPU resident费用边界不同，不能由这张表推出统一D速度优势。发布checkpoint是否见过该公开集合也不能当已确认独立性。

### 旧 MT6 原始加权公开开发图：324声明工作点

原报告 [MT_BASELINES_INDEPENDENT_RESULTS.md](E:/01-Joycecyq/2026-AAMAS/第一篇/docs/research_v4/MT_BASELINES_INDEPENDENT_RESULTS.md) 保存CHILS/BR64compat/HiGHS和原版数值 N/A：CHILS 2s accepted诊断平均增益1.4349%；BR64compat为1.3816%；HiGHS为1.4004%。CHILS全部54实际返回都晚于当时整段soft搜索时间所设的cold期限；这暴露了旧比较合同给native整段时间后再加开销的问题，**不能沿用该合同来制造时间优势**。BR64compat在四个大图的raw解低于发布S，HiGHS四大图均无新mask，原版WeightedBR/m²wis/Struction各54行数值N/A。不是六种全支持的新STK基线。

### 旧第一篇方法开发：廉价控制曾明确更强

旧已拒绝首轮1512单位中，D=0.556时 FactorizedCapacity strict gain/原incumbent=0.029691935，而 AnytimeGreedyPortfolio=2.911448991（[旧拒绝报告](E:/01-Joycecyq/2026-AAMAS/第一篇/docs/research_v4/ONLINE_VALIDATION_REJECTED_CANDIDATE_RESULTS.md)）。后续旧残差开发表同D的 ResidualCapacity=0.575424228、AnytimeGreedyPortfolio=2.342658130，ResidualCapacity返回中共同前缀以上净收益为0（[旧残差表](E:/01-Joycecyq/2026-AAMAS/第一篇/results/v4_residual_online/independent_analysis_01/residual_online_table.csv)）。这些是旧开发数据/时限合同的无量纲比率，不是百分比优越性、更不是新STK结论；不能把论文只放弱 P1控制当成解决了这个差距。

新STK P1当前只有内部开发签收：r004/r005匹配200ms回放，Capacity两seed平均比Greedy在K1/K2/K4分别多71.041/24.597/3.865秒；包含seed不稳定和不利来源，仍不是对CHILS整图、已发表方法或P2完整D的最终优势（[新P1签收](E:/01-Joycecyq/2026-AAMAS/data/两篇论文数据定制化构建/JointRecovery_STK_20261005/reports/P1_DEVELOPMENT_SIGNOFF_ZH.md)）。

## 7. AAMAS 领域出处与数值基线边界

Picard 的 [AAMAS2022卫星协调论文](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1056.pdf) 正式研究带exclusive orbit portions的观测调度，并比较拍卖/DCOP/MILP。它适合说明真实卫星多agent协调背景，但其请求满足、可移动观测开始时间和exclusive-owner合同不同于本次不可切分完整contact窗口的MWIS。当前仓库没有这个方法到新图的忠实实现；不能只改owner名称就在表中加“CBBA/AAMAS2022”结果。领域出处和可执行同任务基线都应真实，不能互相替代。

**交接建议：先补原S/fullcallerD整图 CHILS-p1 和 HiGHS 的真实性能，再把明确modified且数值域就绪的 WeightedBR-64compat补入。保留最强真实greedy。现有P2队列继续，新增发布对照用追加协议执行；最终报告再凭实际收齐结果形成有引用的对比表。** 本次只提供代码和事实边界，没有任何新STK发布方法性能数值或胜出承诺。
