"""Audit and publish a completed fresh observer run, including setup failures."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_fresh_50_trials import archive, plot
from reproduction_tools.records import read_json, sha256


def validate_trace(run):
    summary = read_json(run / 'replay_trace_summary.json')
    proof = read_json(run / 'initialization_proof.json')
    assert summary['success'] and summary['mode'] == 'observe_original_async_batches'
    assert summary['checkpoint_restore'] is False
    assert summary['historical_observations_loaded'] is False
    assert proof['fresh_random_initialization'] and proof['callback_frames_at_initialization'] == 0
    assert proof['checkpoint_restore'] is False and proof['seed'] == 3501
    path = run / 'consumed_replay_batches.jsonl.gz'
    digest = hashlib.sha256()
    count = 0
    previous_frame = 0
    with gzip.open(path, 'rb') as stream:
        for count, line in enumerate(stream, 1):
            digest.update(line)
            choice = json.loads(line)
            frame = choice['callback_frame']
            starts = choice['sequence_start_frames']
            assert choice['update'] == count
            assert previous_frame <= frame <= summary['callback_frames']
            assert len(starts) == 16
            assert all(isinstance(start, int) and 1 <= start and start + 63 * 20 <= frame for start in starts)
            previous_frame = frame
    assert count == summary['recorded_optimizer_updates'] == summary['attempted_updates'] == summary['recorded_batch_choices']
    assert digest.hexdigest() == summary['trace_raw_sha256']
    assert sha256(path) == summary['trace_gzip_sha256']
    assert summary['callback_frames'] == 51000
    return {'status': 'PASS', 'updates': count, 'callback_frames': summary['callback_frames'],
            'all_batches_have16_available64_frame_sequences': True,
            'contiguous_update_indices': True, 'trace_hashes_match': True,
            'initialization': proof, 'summary': summary,
            'limitation': 'Indices identify consumed replay starts; no historical sampling-order trace exists for comparison.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    protocol_path = ROOT / 'docs/traced_fresh_protocol_20261014.json'
    protocol = read_json(protocol_path)
    run = args.run_dir.resolve()
    assert protocol['registered_before_launch']
    assert run == (ROOT / protocol['executed_setup_retry_run_dir']).resolve()
    trace_audit = validate_trace(run)
    record = archive(run)
    destination = ROOT / record['published_records']
    for name in ['initialization_proof.json', 'replay_trace_summary.json', 'consumed_replay_batches.jsonl.gz']:
        shutil.copyfile(run / name, destination / name)
    setup_records = []
    for failure in protocol['setup_failures_recorded']:
        source = ROOT / 'runs/50kmh' / failure['run']
        target = ROOT / 'experiments/50kmh/diagnostics/tracing_setup_20261014' / failure['run']
        target.mkdir(parents=True, exist_ok=True)
        for name in ['reproduction_launch.json', 'reproduction_exit.json', 'pretraining_source.tar.gz',
                     'reproduction_launcher.py', 'initialization_proof.json', 'replay_trace_summary.json',
                     'consumed_replay_batches.jsonl.gz', 'experiment.json', 'config.yaml',
                     'pip-freeze-trainer.txt', 'pip-freeze-rate_server.txt']:
            if (source / name).exists():
                shutil.copyfile(source / name, target / name)
        for name in ['console.out.log', 'console.err.log']:
            if (source / name).exists():
                shutil.copyfile(source / name, target / name.replace('.log', '.txt'))
        if (source / 'training_queries.csv').exists():
            with (source / 'training_queries.csv').open('rb') as stream, gzip.open(target / 'training_queries.csv.gz', 'wb') as output:
                shutil.copyfileobj(stream, output)
        (target / 'setup_failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
        setup_records.append({**failure, 'published_records': target.relative_to(ROOT).as_posix()})
    report = {'protocol': protocol_path.relative_to(ROOT).as_posix(), 'protocol_sha256': sha256(protocol_path),
              'trials': [record], 'all_registered_trials_reported': True, 'fresh_from_scratch': True,
              'setup_failures': setup_records, 'trace_audit': trace_audit,
              'paper_replacement_accepted': False, 'ppo_common_seed_comparison_reproduced': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.plot_output:
        plot([record], args.plot_output)
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / 'experiments').rglob('*'))
        if p.is_file() and p != manifest and '__pycache__' not in p.parts}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'mean': record['comparison']['new_mean'],
                      'performance_status': record['comparison']['status'], 'trace_updates': trace_audit['updates'],
                      'authenticity_status': record['audit']['status']}))


if __name__ == '__main__':
    main()
