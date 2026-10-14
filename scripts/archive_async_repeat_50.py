"""Archive a fresh native-async repeat and its consumed replay trace."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.records import read_json, sha256
from scripts.archive_fresh_50_trials import archive, plot
from scripts.archive_traced_50 import validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    assert protocol['registered_before_launch'] and protocol['runtime_order_override'] is None
    assert len(protocol['run_directories']) == protocol['additional_complete_trials']
    records = []
    for path in protocol['run_directories']:
        run = ROOT / path
        assert read_json(run / 'native_async_runtime.json')['runtime_order_override'] is None
        trace_audit = validate_trace(run)
        record = archive(run)
        dest = ROOT / record['published_records']
        for name in ['native_async_runtime.json', 'initialization_proof.json',
                     'replay_trace_summary.json', 'consumed_replay_batches.jsonl.gz']:
            shutil.copyfile(run / name, dest / name)
        record['trace_audit'] = trace_audit
        records.append(record)
    report = {'protocol': args.protocol.as_posix(), 'protocol_sha256': sha256(args.protocol),
              'trials': records, 'all_registered_trials_reported': True,
              'fresh_from_scratch': True, 'runtime_order_override': None,
              'paper_replacement_accepted': False,
              'historical_state_identified': False,
              'limitation': 'Native asynchronous replay order is captured for each repeat; the original run has no corresponding consumed-index trace, so exact historical scheduling cannot be tested.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.plot_output:
        plot(records, args.plot_output)
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({p.resolve().relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / 'experiments').rglob('*'))
        if p.is_file() and p != manifest and '__pycache__' not in p.parts}, indent=2) + '\n')
    print(json.dumps({'trials': [{'run': x['run'], 'mean': x['comparison']['new_mean'],
        'status': x['comparison']['status'], 'integrity': x['audit']['status']} for x in records]}))


if __name__ == '__main__':
    main()
