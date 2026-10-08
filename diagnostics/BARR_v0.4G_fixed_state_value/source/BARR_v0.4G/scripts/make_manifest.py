#!/usr/bin/env python3
"""Create a manifest from NPZ graph metadata; never invent physical source splits."""
from pathlib import Path
import argparse,csv,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from barr_io import load_npz
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pattern',default='*.npz',help='recursive glob; restrict to graph directories')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); rows=[]; sources={}
    for path in sorted(a.root.rglob(a.pattern)):
        g=load_npz(path); split=str(g.metadata['split']).upper(); source=g.metadata['source_group']
        if not source or split not in {'TRAIN','VAL','DEV','TEST'}:
            raise ValueError(f'{path}: missing explicit source_group/split; do not random-split sibling physical scenarios')
        if source in sources and sources[source]!=split:
            raise ValueError(f'{source}: appears in multiple data splits')
        sources[source]=split
        rows.append(dict(graph=str(path.resolve()),graph_id=g.metadata['graph_id'],source_group=source,split=split))
    if not rows:raise ValueError('no graphs matched')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.out)
