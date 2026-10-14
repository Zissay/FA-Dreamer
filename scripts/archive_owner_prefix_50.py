"""Retain the preregistered estimated-prefix full trial, including failure."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    protocol_path = ROOT / 'docs/refined_owner_prefix_full_protocol_20261014.json'
    protocol = read_json(protocol_path)
    assert protocol['registered_before_launch'] and protocol['additional_complete_trials'] == 1
    run = ROOT / protocol['run_directories'][0]
    profile_path = ROOT / protocol['profile']
    assert sha256(profile_path) == protocol['profile_sha256']
    profile = read_json(profile_path)
    runtime = read_json(run / 'owner_prefix_runtime.json')
    assert runtime['creator_owner_prefix'] == profile['creator_owner_prefix']
    assert runtime['model_seed'] == 3501 and runtime['uniform_sampler_seed'] == 0
    assert runtime['workers'] == 2 and runtime['sources'] == 16 and runtime['prefetch'] == 4
    assert runtime['learning_parameters_unchanged'] and not runtime['driver_changed']
    assert not runtime['historical_state_identified'] and runtime['external_weights_or_observations'] == []
    assert runtime['jax_default_matmul_precision'] is None
    launch = read_json(run / 'reproduction_launch.json')
    assert launch['initial_creator_owner_profile_sha256'] == sha256(profile_path)
    parent = read_json(ROOT / 'docs/refined_fixed_parent_20261014.json')
    with gzip.open(run / 'consumed_replay_batches.jsonl.gz', 'rt') as stream:
        initial = [json.loads(next(stream))['sequence_start_frames'] for _ in range(5)]
    assert initial == parent['sequence_start_frames']
    trace = validate_trace(run)
    record = archive(run)
    target = ROOT / record['published_records']
    for name in ['owner_prefix_runtime.json', 'initialization_proof.json', 'replay_trace_summary.json',
                 'consumed_replay_batches.jsonl.gz']:
        shutil.copyfile(run / name, target / name)
    shutil.copyfile(profile_path, target / profile_path.name)
    record['trace_audit'] = trace
    record['declared_loader_runtime'] = runtime
    report = {'protocol': protocol_path.relative_to(ROOT).as_posix(),
              'protocol_sha256': sha256(protocol_path), 'trials': [record],
              'all_registered_trials_reported': True, 'fresh_from_scratch': True,
              'first_five_batches_match_declared_diagnostic': True,
              'previous_diagnostic_trigger_met': False, 'historical_state_identified': False,
              'paper_replacement_accepted': False,
              'limitation': 'Target-informed estimated startup order; not unique historical execution recovery or typical-performance proof. Candidate needs visual review and independent fresh computational verification.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.plot_output:
        plot([record], args.plot_output)
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / 'experiments').rglob('*'))
        if p.is_file() and p != manifest and '__pycache__' not in p.parts}, indent=2) + '\n')
    print(json.dumps({'rate': record['comparison']['new_mean'],
                      'status': record['comparison']['status'], 'integrity': record['audit']['status']}))


if __name__ == '__main__':
    main()
