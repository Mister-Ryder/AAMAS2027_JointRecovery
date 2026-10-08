"""Read-only compact status; never polls, launches or stops a solver."""
from collections import Counter
from pathlib import Path
import argparse
import json
from g_protocol import guard_root, now, read, live_native_processes


def status(root):
    output = {'utc': now(), 'native_processes': live_native_processes(root), 'phases': {}}
    for phase in ['capture', 'fork']:
        rows = [read(path) for path in sorted((root / 'results' / phase).glob('*/result.json'))]
        item = {'completed_results': len(rows), 'statuses': dict(Counter(row.get('status') for row in rows)),
                'errors': sum(row.get('status') != 'COMPLETE' for row in rows),
                'resource_flagged': sum(row.get('resource_qualified') is False for row in rows)}
        registration = root / ('registration_' + phase + '.json')
        if registration.exists():
            item['expected_positions'] = read(registration)['positions']
        completion = root / ('completion_' + phase + '.json')
        if completion.exists():
            item['completion'] = read(completion)
        output['phases'][phase] = item
    for name in ['chain_completion.json', 'chain_failure.json', 'finalization_v04g_completion.json']:
        if (root / name).exists():
            output[name] = read(root / name)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(status(guard_root(args.root))), flush=True)
