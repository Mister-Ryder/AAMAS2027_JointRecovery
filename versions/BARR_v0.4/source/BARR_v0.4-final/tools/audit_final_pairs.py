"""Read-only complete <=2 outsider audit on an archived final strong mask.

Run only after the performance process exits; output belongs to mechanism audit.
No local/native search or candidate parameter tuning occurs here. Python 3.8+.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', required=True, type=Path)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--mask-json', required=True, type=Path, help='Completed native result or checkpoint, selected is an ID list.')
    parser.add_argument('--output-dir', required=True, type=Path, help='Fresh mechanism-only directory, never a performance cell directory.')
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--core', type=int, help='Approved extra physical CPU, e.g. 12..16; no affinity by default.')
    args = parser.parse_args()
    if args.seconds < 0 or args.core is not None and args.core < 0:
        raise ValueError('Negative seconds/core')
    data = json.loads(args.mask_json.read_text(encoding='utf-8'))
    selected = data['selected']
    if not isinstance(selected,list) or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in selected) or len(set(selected))!=len(selected):
        raise ValueError('selected must be a unique nonnegative vertex ID list')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = args.output_dir / 'incumbent.txt'
    frozen.write_text('BARRPAIR1 {}\n{}\n'.format(len(selected),' '.join(map(str,selected))), encoding='ascii')
    output = args.output_dir / 'pair_audit.json'
    command = [str(args.oracle.resolve()),str(args.graph.resolve()),str(frozen.resolve()),str(output.resolve()),str(args.seconds)]
    if args.core is not None:
        command = ['taskset','-c',str(args.core)] + command
    started = time.monotonic()
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    wall = time.monotonic()-started
    (args.output_dir/'stdout.txt').write_text(run.stdout,encoding='utf-8')
    (args.output_dir/'stderr.txt').write_text(run.stderr,encoding='utf-8')
    receipt = {'purpose':'Post-hoc frozen final-mask mechanism evidence only, excluded from every performance mean and trajectory.',
               'created_utc':datetime.now(timezone.utc).isoformat(),'command':command,'process_wall_seconds':wall,'exit_code':run.returncode,
               'graph_path':str(args.graph.resolve()),'graph_sha256':sha(args.graph),'mask_json_path':str(args.mask_json.resolve()),
               'mask_json_sha256':sha(args.mask_json),'frozen_ids_sha256':sha(frozen),'oracle_path':str(args.oracle.resolve()),
               'oracle_sha256':sha(args.oracle),'wrapper_sha256':sha(Path(__file__)),'selected_count':len(selected),
               'performance_search_called':False,'source_mask_written':False}
    if run.returncode == 0:
        result = json.loads(output.read_text(encoding='utf-8'))
        if result['incumbent_selected_count'] != len(selected):
            raise ValueError('Oracle selected count mismatch')
        value = data.get('tick_value',data.get('tick_objective'))
        if value is not None and int(value) != result['incumbent_ticks']:
            raise ValueError('Native frozen objective does not match graph/mask')
        receipt['pair_audit_sha256'] = sha(output)
        receipt['complete'] = result['complete']
    (args.output_dir/'provenance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if run.returncode != 0:
        raise RuntimeError('Read-only pair oracle failed; see preserved stderr and provenance')
    print(json.dumps({'complete':result['complete'],'lower_gain_ticks':result['lower_gain_ticks'],
                      'upper_gain_ticks':result['upper_gain_ticks'],'positive_singletons':result['positive_singletons_lower_count'],
                      'positive_pairs':result['positive_pairs_lower_count'],'audit_seconds':result['audit_seconds']}))


if __name__ == '__main__':
    main()
