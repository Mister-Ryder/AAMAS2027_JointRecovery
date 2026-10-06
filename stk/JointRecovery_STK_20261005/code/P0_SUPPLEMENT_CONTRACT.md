# 50/10 ms 补充探针

只能在负责人根据真实 P0 机会写出 `allow_short_probe_execution: true` gate 后启动。输入必须是两次重复、完整200/1000 ms 的正式 P0/out，不能是 smoke。

补充不生成初态、不生成动作、不运行共同贪心、不运行历史搜索。逐项从冻结 snapshot/scopes 重建 ordered R 与整个 B，核对 original/best 完整成员、q、eligible-before-cap、actual warm、scope SHA 和 controller-state identity。绑定原图、binary、原 P0 adapter/算法闭包和每个原 summary/protocol/snapshot 文件 SHA；差异明确失败。

每张图先完成所有冻结 states 的50 ms pass，再做10 ms pass；每个请求两次repeat，native seed17/单线程、同actual warm和cold clone。原200/1000 rows逐字义保留在新的JSON中，然后 append 新rows；原文件不修改，完成后再核对源SHA。任何state仍按原来源组划分，数量和独立物理来源不增加。

新fresh out具有同P0 summary/snapshot结构，现有 P1load_examples 可消费其四workpoint数据。不把原P0目录和supplement目录同时提供给P1（它会拒绝重复graph_id）。只补充输入指定的原正式输出；不能按我方算法胜负删图或删状态。

```json
{"allow_short_probe_execution":true,"reason":"Actual P0 opportunity observed; supplement on exact frozen states authorized.","p0_analysis_sha256":"..."}
```

```text
python p0_supplement.py --runtime-root NEW_CAPSULE/runtime --graphs-dir NEW/graphs --source-out OLD_FORMAL_JOB/out --out NEW/short_budget_supplement --chils CHILS --chils-source CHILS_SOURCE --p0-gate NEW/protocol/short_probe_gate.json
```

代码已通过一个必要的真实core语义小guard：禁用initial_known_warm后仍可准确重建冻结scope/warm；ordered R修改会失败；没有native调用。它不替代之后的真实补充执行。
