"""Publish both registered complete continuous trials and their consumed traces."""
import argparse
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    protocol_path = ROOT / 'docs/continuous_trace_pair_protocol_20261014.json'
    protocol = read_json(protocol_path)
    assert protocol['registered_before_launch'] and protocol['additional_complete_trials'] == 2
    trials = []
    for relative in protocol['run_directories']:
        run = ROOT / relative
        trace = validate_trace(run)
        record = archive(run)
        target = ROOT / record['published_records']
        for name in ['initialization_proof.json', 'replay_trace_summary.json', 'consumed_replay_batches.jsonl.gz']:
            shutil.copyfile(run / name, target / name)
        record['trace_audit'] = trace
        trials.append(record)
    report = {'protocol': protocol_path.relative_to(ROOT).as_posix(), 'protocol_sha256': sha256(protocol_path),
        'trials': trials, 'all_registered_trials_reported': True, 'fresh_from_scratch': True,
        'all_original_learning_parameters_and_seed': True,
        'performance_candidates': [record['run'] for record in trials if record['comparison']['status'].startswith('REPRODUCED')],
        'paper_replacement_accepted': False,
        'limitation': 'Every registered trial retained. A matched candidate still requires visual trend review and fresh computational trace replay.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.plot_output:
        plot(trials, args.plot_output)
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / 'experiments').rglob('*'))
        if p.is_file() and p != manifest and '__pycache__' not in p.parts}, indent=2) + '\n')
    print(json.dumps({'rates': [r['comparison']['new_mean'] for r in trials],
        'candidates': report['performance_candidates'], 'all_trials_reported': True}))


if __name__ == '__main__':
    main()
