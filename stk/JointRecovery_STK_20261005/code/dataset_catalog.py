"""Update the JointRecovery dataset index from frozen, small metadata files.

Read-only with respect to geometry, graphs and execution outputs. This tool
does not start STK, load NPZ arrays, scan native scene contents, recompute large
file hashes, evaluate a model or reinterpret incomplete data as completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


CSV_DICTIONARY = [
    ("source_group", "string", "来源组；同母库所有站网、gap、状态和恢复请求必须同组划分"),
    ("geometry_id / replicate_id", "string", "物理族标识 / 重新传播的物理实现标识"),
    ("epoch_utc", "ISO UTC", "规划起点及轨道历元；relative seconds的零点"),
    ("contact_id / pass_id", "string", "完整接触唯一标识 / 卫星—站点流的原始过境序号"),
    ("satellite_id", "string", "卫星实体；同时通信容量为1；也是owner"),
    ("site_id / antenna_id", "string", "站点 / 单副天线资源标识"),
    ("start_rel_seconds / end_rel_seconds", "decimal seconds", "权威端点；9位小数；不裁切、不整数秒化；区间[s,e)"),
    ("duration_seconds", "decimal seconds", "严格等于冻结CSV的end−start；唯一收益，无业务权重"),
    ("start_utc / end_utc", "STK UTCG", "辅助人工显示；STK默认毫秒显示，不能代替权威相对端点"),
    ("boundary_crossing", "string", "contacts.csv恒为none；边界表left/right/both；原始表另有outside_before/after"),
    ("access_settings_hash", "SHA-256", "对应access_settings_actual.json中的真实Access设置"),
    ("padding_boundary_clipped", "boolean", "仅raw/boundary表：传播padding边界上可能被截断；不得作为完整规划接触"),
]
GRAPH_DICTIONARY = [
    ("start_decimal_seconds / end_decimal_seconds / duration_decimal_seconds", "string[N]", "冻结CSV原小数；收益为端点差"),
    ("start_ns / end_ns", "int64[N], ns", "9位小数的整数表示；原始冲突边在此精度建立"),
    ("start_ticks / end_ticks", "int64[N], us", "求解器微秒序列化；不是物理精度声明"),
    ("weights", "float64[N], seconds", "完整接触时长；无重新赋权"),
    ("owner / agents / satellite_id_mapping", "integer[N] / string[168]", "owner=卫星的索引映射；不构成全天团"),
    ("edges / edge_u / edge_v", "integer[E,2] / integer[E]", "去重无向冲突边，u<v"),
    ("edge_types", "uint8[E]", "位1=同天线冲突，位2=同卫星冲突；3=同时满足两类"),
    ("mother_contact_index / contact_id", "integer[N] / string[N]", "R8/R12顶点回溯到同母contacts.csv；保留原窗口"),
    ("vertex_factors / factor_indptr / factor_vertices", "CSR-style arrays", "时间资源归属：每个顶点关联一副地面天线和一个卫星通道；成员并非静态团"),
    ("factor_ids / factor_kind / factor_capacity / factor_gap_seconds", "arrays", "天线/卫星资源类型，容量恒1，各自gap"),
    ("source_group / split / graph_id / contacts_sha256 / parameters_sha256", "scalar strings", "来源分组、派生图和冻结输入证据"),
    ("duration_greedy_witness_vertices", "integer[K]", "建图验收的可行调度见证；不是学习算法结果或最优证书"),
]


def read_json(path, warnings):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append({"path": str(path), "reason": str(exc)})
        return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def dictionary_json(rows):
    return [{"field": field, "type_or_unit": kind, "meaning_zh": meaning} for field, kind, meaning in rows]


def frozen_outputs(manifest):
    outputs = manifest.get("outputs", [])
    if isinstance(outputs, dict):
        return {name: {"sha256": sha} for name, sha in outputs.items()}
    return {item["path"]: {key: value for key, value in item.items() if key != "path"} for item in outputs}


def build_catalog(root):
    warnings = []
    parameters_path = root / "protocol" / "JointRecovery_parameters.json"
    parameters = read_json(parameters_path, warnings)
    if parameters is None:
        raise ValueError("The frozen parameter JSON could not be read")
    graphs = []
    for path in sorted((root / "graphs").glob("*.json")):
        data = read_json(path, warnings)
        if data is None or "graph_id" not in data:
            continue
        arrays_path = path.with_suffix(".npz")
        graphs.append({"graph_id": data["graph_id"], "source_group": data["source_group"],
                       "replicate_id": data["replicate_id"], "split": data["split"].upper(),
                       "station_view": data["station_view"], "station_count": data["station_count"],
                       "ground_gap_seconds": data["ground_gap_seconds"], "satellite_gap_seconds": data["satellite_gap_seconds"],
                       "vertices": data["vertices"], "edges": data["edges"], "mean_degree": data["mean_degree"],
                       "cross_agent_edge_fraction": data["cross_agent_edge_fraction"],
                       "duration_range_seconds": [data["min_contact_duration_seconds"], data["max_contact_duration_seconds"]],
                       "contacts_sha256": data["contacts_sha256"], "parameters_sha256": data["parameters_sha256"],
                       "npz_sha256": data["npz_sha256"], "npz_present": arrays_path.is_file(),
                       "metadata_path": path.relative_to(root).as_posix(), "npz_path": arrays_path.relative_to(root).as_posix()})
    sources = []
    for specification in parameters["scene_definitions"]:
        scene_id = specification["scene_id"]
        scene_dir = root / "raw_geometry" / scene_id
        manifest_path = scene_dir / "manifest.json"
        manifest = read_json(manifest_path, warnings) if manifest_path.is_file() else None
        success = bool(manifest and manifest.get("status") == "success")
        recorded_status = manifest.get("status") if manifest else "not_generated"
        linked_graphs = [g for g in graphs if g["source_group"] == specification["source_group"]]
        receipts = frozen_outputs(manifest) if success else {}
        source = {"scene_id": scene_id, "source_group": specification["source_group"], "replicate_id": specification["replicate_id"],
                  "geometry_id": specification["geometry_id"], "split": specification["split"].upper(),
                  "epoch_utc": specification["epoch_utc"], "planning_stop_utc": specification["horizon_stop_utc"],
                  "recorded_status": recorded_status, "completed": success, "counts": manifest.get("counts") if success else None,
                  "planned_satellites": len(specification["satellites"]), "planned_sites": len(parameters["stations"]),
                  "completed_graph_count": sum(g["npz_present"] for g in linked_graphs) if success else 0,
                  "planned_graph_count": len(parameters["station_views"]) * len(parameters["graph_configurations"]),
                  "manifest_path": manifest_path.relative_to(root).as_posix() if manifest else None,
                  "paths": {"contacts": f"raw_geometry/{scene_id}/contacts.csv", "boundary_contacts": f"raw_geometry/{scene_id}/boundary_contacts.csv",
                            "padded_raw_access": f"raw_geometry/{scene_id}/access_raw.csv", "original_access_reports": f"raw_geometry/{scene_id}/raw_access_reports.jsonl",
                            "editable_scene": f"raw_geometry/{scene_id}/scene/{scene_id}.sc", "scene_dependencies_directory": f"raw_geometry/{scene_id}/scene"},
                  "frozen_receipts": {name: receipts.get(name) for name in ("contacts.csv", "boundary_contacts.csv", "access_raw.csv", "raw_access_reports.jsonl", f"scene/{scene_id}.sc")},
                  "parameters_sha256": manifest.get("parameters_sha256") if manifest else None,
                  "geometry_generator_sha256": manifest.get("generator_sha256") if manifest else None,
                  "save_recovery_generator_sha256": manifest.get("save_recovery_generator_sha256") if manifest else None,
                  "spot_check_count": manifest.get("spot_check_pair_count") if success else None,
                  "spot_check_counts_match": manifest.get("spot_check_all_counts_match") if success else None,
                  "spot_check_within_5ms": manifest.get("spot_check_all_within_5ms") if success else None}
        sources.append(source)
    complete_groups = {s["source_group"] for s in sources if s["completed"]}
    for graph in graphs:
        graph["completed"] = graph["source_group"] in complete_groups and graph["npz_present"]
    summary = {"planned_mother_geometries": len(sources), "completed_mother_geometries": sum(s["completed"] for s in sources),
               "planned_graphs": sum(s["planned_graph_count"] for s in sources), "completed_graphs": sum(g["completed"] for g in graphs),
               "full_contained_contacts": sum(s["counts"]["full_contained"] for s in sources if s["completed"]),
               "boundary_crossing_contacts": sum(s["counts"]["crossing"] for s in sources if s["completed"]),
               "counts_are_unique_mother_windows_not_graph_duplicates": True,
               "split_counts": {split: {"planned": sum(s["split"] == split for s in sources),
                                          "completed": sum(s["split"] == split and s["completed"] for s in sources)}
                                for split in ("TRAIN", "VALIDATION", "TEST")}}
    return {"catalog_format_version": "1.0", "updated_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(root), "index_method": "small manifests and graph JSON only; metadata hashes reused; no STK/NPZ load/large-file rehash",
            "summary": summary, "sources": sources, "graphs": graphs, "station_views": parameters["station_views"],
            "graph_configurations": parameters["graph_configurations"], "settings": parameters["common_settings"],
            "data_dictionary": {"contact_csv": dictionary_json(CSV_DICTIONARY), "graph_npz": dictionary_json(GRAPH_DICTIONARY)},
            "warnings": warnings}


def md_link(path, label):
    return f"[{label}](../{path})"


def table_dictionary(rows):
    lines = ["| 字段 | 类型 / 单位 | 含义 |", "|---|---|---|"]
    lines.extend(f"| `{field}` | {kind} | {meaning} |" for field, kind, meaning in rows)
    return lines


def render_markdown(catalog):
    summary = catalog["summary"]
    local_update = datetime.fromisoformat(catalog["updated_utc"]).astimezone(timezone(timedelta(hours=8)))
    lines = ["# JointRecovery 新建STK数据集目录与数据字典", "",
             f"更新：{local_update:%Y-%m-%d %H:%M:%S}（北京时间）。根目录：`{catalog['dataset_root']}`。", "",
             f"**实际完成 {summary['completed_mother_geometries']} / {summary['planned_mother_geometries']} 个计划母机会库，"
             f"{summary['completed_graphs']} / {summary['planned_graphs']} 张计划派生图。** "
             f"成功母库共有 {summary['full_contained_contacts']:,} 条完整规划接触与 {summary['boundary_crossing_contacts']:,} 条跨界接触。", "",
             "计划数量与已完成数量分别统计。不同站网/gap重复使用同母窗口，不能把派生图节点相加当成新增物理接触或独立样本。"
             "进行中、失败及尚未生成的母库不进入完成计数。本目录不推断模型已训练或学习方法有优势。", "",
             "## 来源组与阶段", "", "| 来源组 | 划分 | 规划起点UTC | 几何状态 | 完整窗口 | 跨界窗口 | 完成派生图 |", "|---|---|---|---|---:|---:|---:|"]
    status_names = {"success": "已完成并冻结", "not_generated": "尚未生成", "running": "进行中，未冻结",
                    "failed": "失败，未冻结", "save_recovery_running": "保存恢复中，未冻结", "save_recovery_failed": "保存恢复失败，未冻结"}
    for source in catalog["sources"]:
        counts = source["counts"] or {}
        lines.append(f"| {source['source_group']} | {source['split']} | {source['epoch_utc']} | {status_names.get(source['recorded_status'], source['recorded_status'])} | "
                     f"{counts.get('full_contained', '—')} | {counts.get('crossing', '—')} | {source['completed_graph_count']}/{source['planned_graph_count']} |")
    lines += ["", "TRAIN r000–r005（含P0已曝光开发来源）；VALIDATION r006–r007；TEST r008–r011。"
              "同一来源的R8/R12、全部gap、初态及恢复域必须始终绑定同一split；日期不同不自动证明统计独立。", "",
              "## 已完成派生图", "", "| 图 | 来源组 | 站网 | τg / τs（秒） | 顶点 | 冲突边 | 平均度 | 跨owner边占比 |", "|---|---|---|---|---:|---:|---:|---:|"]
    completed_graphs = [g for g in catalog["graphs"] if g["completed"]]
    if completed_graphs:
        for graph in completed_graphs:
            lines.append(f"| {md_link(graph['metadata_path'], graph['graph_id'])} | {graph['source_group']} | {graph['station_view']} | "
                         f"{graph['ground_gap_seconds']} / {graph['satellite_gap_seconds']} | {graph['vertices']:,} | {graph['edges']:,} | "
                         f"{graph['mean_degree']:.3f} | {graph['cross_agent_edge_fraction']:.2%} |")
    else:
        lines.append("| 尚无完成图 | — | — | — | — | — | — | — |")
    lines += ["", "各图的数组位于同名`.npz`，统计与SHA位于同名`.json`。R8固定为"
              "GS01、GS03、GS04、GS06、GS07、GS09、GS10、GS12；只删除不在子网内的整条接触，保留窗口和收益完全不变。", "",
              "## 原始机会库与可编辑场景", ""]
    for source in catalog["sources"]:
        if not source["completed"]:
            continue
        lines += [f"### {source['scene_id']}", "", f"来源：`{source['source_group']}`，划分：{source['split']}。", "",
                  f"- {md_link(source['paths']['contacts'], '完整规划接触 contacts.csv')}：72小时内完整落入的窗口。",
                  f"- {md_link(source['paths']['boundary_contacts'], '跨规划边界 boundary_contacts.csv')}：单列保留，不裁短，不进入图。",
                  f"- {md_link(source['paths']['padded_raw_access'], '含padding的 access_raw.csv')}与{md_link(source['paths']['original_access_reports'], '原始Access报告 JSONL')}：原始端点与padding边界标记。",
                  f"- {md_link(source['paths']['editable_scene'], '可编辑STK场景')}及同目录全部卫星/站点依赖：不能只复制`.sc`。",
                  f"- {md_link(source['manifest_path'], '执行manifest')}：实际轨道、约束、30/15秒抽样、保存技术桥接与逐文件证据。", "",
                  "| 冻结项目 | SHA-256（复用已生成receipt，不重新扫描大文件） |", "|---|---|"]
        for name, receipt in source["frozen_receipts"].items():
            lines.append(f"| `{name}` | `{receipt['sha256'] if receipt else '尚无冻结receipt'}` |")
        lines += [f"| 参数JSON | `{source['parameters_sha256']}` |",
                  f"| 原几何生成代码 | `{source['geometry_generator_sha256']}` |", "",
                  f"固定抽样对：{source['spot_check_count']}；15秒与30秒区间数一致：{source['spot_check_counts_match']}；端点最大差≤5ms：{source['spot_check_within_5ms']}。", ""]
    lines += ["## 固定问题契约", "",
              "1. 顶点是一次完整几何可见区间`[s,e)`；收益`w=e−s`，目标为静态星地链路总时长。没有业务任务、窗口切片、部分移动、统一时长或人为重赋权。",
              "2. 每站1副天线、每星1个同时通信通道；`owner=卫星`。资源因子是时间归属，不是全天只能选一次的静态团。",
              "3. 起点排序u/v：同天线且`sv < eu + τg`，或同卫星且`sv < eu + τs`，则冲突；取并集。包含和同起点也冲突；恰好满足gap等号时兼容。τs=150秒，τg=170/340/680秒。",
              "4. 168星双壳层、WGS84理想化站网、最低仰角15°、J2传播、初始J2000；规划72小时，两端各额外传播1小时，轨道epoch仍为规划起点。完整端点不裁切；跨界和padding外窗口另存。",
              "5. Access精确事件开启、最大步长30秒、收敛0.001秒、光行时关闭；12固定对用15秒复查。CSV9位小数、求解器微秒tick均是数值序列化，不是物理精度声明。",
              "6. 原始UTC是辅助毫秒显示，相对秒端点是建边及计奖依据。STK11中文路径保存限制通过独占ASCII临时路径和全依赖复制解决，实际数据仍在本目录。",
              "7. 同母库的站网、gap、局部状态和恢复请求共用来源组，不能跨训练/验证/测试。零机会、失败、无增益和迟到请求须保留于后续独立执行记录。", "",
              "## 接触CSV数据字典", ""]
    lines += table_dictionary(CSV_DICTIONARY)
    lines += ["", "## 图NPZ数据字典", ""]
    lines += table_dictionary(GRAPH_DICTIONARY)
    lines += ["", "## 更新方式", "", "在每个几何/建图阶段完成后运行：", "", "```text", "python code/dataset_catalog.py --dataset-root <本数据集绝对目录>", "```", "",
              "工具只读取小型manifest和图JSON，复用已冻结SHA；不会启动STK、加载NPZ、重算场景、重新校验全部窗口或重复大规模实验。"
              "程序仅更新`reports/DATASET_CATALOG_ZH.md`和`reports/dataset_catalog.json`。"]
    if catalog["warnings"]:
        lines += ["", "## 索引读取警告", ""]
        lines.extend(f"- `{warning['path']}`：{warning['reason']}" for warning in catalog["warnings"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve(strict=True)
    catalog = build_catalog(root)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "dataset_catalog.json", catalog)
    markdown = reports / "DATASET_CATALOG_ZH.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text(render_markdown(catalog), encoding="utf-8")
    temporary.replace(markdown)
    print(json.dumps({"status": "catalog_updated", "summary": catalog["summary"], "warnings": len(catalog["warnings"]),
                      "markdown": str(markdown), "json": str(reports / "dataset_catalog.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
