"""Inherited satellite conflict graphs and explicitly local transfer windows.

The conflict predicate is an unmodified vendored SNSD v5.1 predicate. Window
graphs are induced bounded subproblems, not full-schedule completions: contacts
and incumbent commitments outside a selected node set are not represented.
Their overlapping variants must remain grouped by source and time boundary.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .core import Graph
from .vendor.snsd_core.data import load_arcs
from .vendor.snsd_core.graph import build_conflict_graph, partition_equal_time


SOURCE_HASHES = {
    "C3": "ec95f50c11d800f051e218aa1e414df873ddd12e1f71ce911da3ba28adff647e",
    "W3": "73fb7cfe7d5f6e522cf1e437f86b5b5b1f1a1652a0a68e6bc434f41ba30b485e",
}
HORIZON = 259200
AGENTS = 6


def _source(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stem in SOURCE_HASHES and digest != SOURCE_HASHES[path.stem]:
        raise ValueError(f"Satellite source hash mismatch: {path}")
    dataset = load_arcs(path)
    native = build_conflict_graph(dataset.arcs)
    parts = partition_equal_time(dataset.arcs, native, k=AGENTS,
                                 horizon_start=0, horizon_end=HORIZON)
    return dataset, native, parts, digest


def _convert(dataset, native, parts, name, source_ids=None):
    if source_ids is None:
        source_ids = tuple(range(native.num_nodes))
    else:
        source_ids = tuple(sorted(int(node) for node in source_ids))
    remap = {node: local for local, node in enumerate(source_ids)}
    adjacency = tuple(
        frozenset(remap[int(peer)] for peer in native.neighbors(node)
                  if int(peer) in remap)
        for node in source_ids
    )
    return Graph(
        weights=np.asarray([dataset.arcs[node].weight for node in source_ids], dtype=np.float64),
        agents=np.asarray([int(parts.partition_id[node]) for node in source_ids], dtype=np.int64),
        adjacency=adjacency,
        name=name,
    )


def load_full_satellite_graph(path) -> Graph:
    """Load one complete CSV with all inherited constraints and six owners.

    This adapter supports later full-schedule experiments. It performs no
    optimization and applies no approximation to the original conflict graph.
    """
    dataset, native, parts, _ = _source(path)
    return _convert(dataset, native, parts, f"{Path(path).stem}_full")


def _window_ids(dataset, native, parts, boundary_index, offset, max_nodes):
    boundary_time = boundary_index * (HORIZON // AGENTS)
    center = boundary_time + offset
    starts = np.asarray([arc.link_st for arc in dataset.arcs], dtype=np.int64)
    left_agent, right_agent = boundary_index - 1, boundary_index
    crosses = []
    for first, second in parts.cut_edges:
        first, second = int(first), int(second)
        if {int(parts.partition_id[first]), int(parts.partition_id[second])} != {left_agent, right_agent}:
            continue
        if max(abs(int(starts[first]) - boundary_time),
               abs(int(starts[second]) - boundary_time)) > 1800:
            continue
        crosses.append((first, second))
    if not crosses:
        raise ValueError(f"No inherited cut edge at boundary {boundary_index}")
    crosses.sort(key=lambda edge: (
        abs(float(starts[edge[0]] + starts[edge[1]]) / 2.0 - center),
        edge[0], edge[1]))

    # Preserve real cross-agent dependencies before truncating the local pool.
    chosen = set()
    seed_limit = max(2, max_nodes // 4)
    for first, second in crosses:
        addition = {first, second} - chosen
        if len(chosen) + len(addition) <= seed_limit:
            chosen.update(addition)
        if len(chosen) >= seed_limit:
            break

    core = np.flatnonzero(np.abs(starts - center) <= 900)
    core = sorted(map(int, core), key=lambda node: (abs(int(starts[node]) - center), node))
    core_limit = max(len(chosen), (2 * max_nodes) // 3)
    for node in core:
        if len(chosen) >= core_limit:
            break
        chosen.add(node)

    # Original resource-conflict neighbors provide a one-hop endpoint halo.
    # The remaining outside graph is intentionally omitted, not assumed free
    # when reporting full-schedule performance.
    neighbors = set()
    for node in sorted(chosen):
        neighbors.update(int(peer) for peer in native.neighbors(node))
    for node in sorted(neighbors - chosen,
                       key=lambda node: (abs(int(starts[node]) - center), node)):
        if len(chosen) >= max_nodes:
            break
        chosen.add(node)
    if len(chosen) < max_nodes:
        for node in core:
            if len(chosen) >= max_nodes:
                break
            chosen.add(node)
    return tuple(sorted(chosen)), boundary_time, center


def load_satellite_graphs_with_metadata(data_dir, per_dataset=20, max_nodes=160):
    """Return local graphs plus auditable source row IDs and grouping keys."""
    if per_dataset < 1:
        raise ValueError("per_dataset must be positive")
    if max_nodes < 4:
        raise ValueError("max_nodes must be at least four")
    graphs, metadata = [], []
    variants = int(math.ceil(per_dataset / (AGENTS - 1)))
    offsets = np.linspace(-900, 900, variants) if variants > 1 else np.asarray([0.0])
    for source_name in ("C3", "W3"):
        path = Path(data_dir) / f"{source_name}.csv"
        dataset, native, parts, digest = _source(path)
        for index in range(per_dataset):
            boundary_index = index % (AGENTS - 1) + 1
            variant = index // (AGENTS - 1)
            offset = int(round(float(offsets[variant])))
            source_ids, boundary_time, center = _window_ids(
                dataset, native, parts, boundary_index, offset, max_nodes)
            name = f"{source_name}_window_{index:02d}"
            graph = _convert(dataset, native, parts, name, source_ids)
            graph_edges = sum(len(peers) for peers in graph.adjacency) // 2
            cross_edges = sum(
                int(graph.agents[first] != graph.agents[second])
                for first, peers in enumerate(graph.adjacency)
                for second in peers if first < second)
            if cross_edges == 0:
                raise ValueError(f"No cross-agent edge survived window sampling: {name}")
            graphs.append(graph)
            metadata.append({
                "name": name,
                "source_file": str(path),
                "source_sha256": digest,
                "source_graph_hash": native.graph_hash,
                "source_row_ids": list(source_ids),
                "group_key": f"{source_name}_boundary_{boundary_index}",
                "boundary_index": boundary_index,
                "boundary_time": boundary_time,
                "window_center": center,
                "window_offset": offset,
                "core_half_width": 900,
                "cut_endpoint_half_width": 1800,
                "nodes": len(source_ids),
                "edges": graph_edges,
                "cross_agent_edges": cross_edges,
                "owner_ids": sorted(set(map(int, graph.agents))),
                "external_commitments_ignored": True,
                "scope": "induced local transfer subproblem; overlapping variants are not independent instances",
            })
    return graphs, metadata


def load_satellite_graphs(data_dir, per_dataset=20, max_nodes=160) -> list[Graph]:
    """Load deterministic C3/W3 boundary windows, preserving inherited owners.

    Default output has 20 overlapping local windows per source file. Four
    offsets of each of five original 12-hour boundaries retain genuine cut
    edges. Split and aggregate by source/boundary, not by window variation.
    """
    return load_satellite_graphs_with_metadata(data_dir, per_dataset, max_nodes)[0]
