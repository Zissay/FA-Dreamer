"""Publish the declared fresh FP32 numerical-runtime control without relabeling it."""
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
    protocol_path = ROOT / 'docs/fp32_fresh_protocol_20261014.json'
    protocol = read_json(protocol_path)
    assert protocol['registered_before_launch'] and protocol['additional_complete_trials'] == 1
    run = ROOT / protocol['run_directories'][0]
    runtime = read_json(run / 'numeric_runtime.json')
    assert runtime['jax_default_matmul_precision'] == 'float32'
    assert not runtime['driver_modified'] and runtime['original_learning_parameters_unchanged']
    launch = read_json(run / 'reproduction_launch.json')
    assert launch['numeric_runtime_override'] == {'JAX_DEFAULT_MATMUL_PRECISION': 'float32'}
    trace = validate_trace(run)
    record = archive(run)
    target = ROOT / record['published_records']
    for name in ['numeric_runtime.json', 'initialization_proof.json', 'replay_trace_summary.json', 'consumed_replay_batches.jsonl.gz']:
        shutil.copyfile(run / name, target / name)
    record['trace_audit'] = trace
    record['declared_numeric_runtime'] = runtime
    report = {'protocol': protocol_path.relative_to(ROOT).as_posix(), 'protocol_sha256': sha256(protocol_path),
        'trials': [record], 'all_registered_trials_reported': True, 'fresh_from_scratch': True,
        'numeric_runtime_control': True, 'historical_matmul_precision_known': False,
        'performance_and_trend_candidate': record['comparison']['status'].startswith('REPRODUCED'),
        'paper_replacement_accepted': False,
        'limitation': 'ExplicitFP32 matrix precision must be disclosed. Historical numerical runtime is unknown; a candidate requires visual trend and fresh computational trace-replay validation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.plot_output:
        plot([record], args.plot_output)
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / 'experiments').rglob('*'))
        if p.is_file() and p != manifest and '__pycache__' not in p.parts}, indent=2) + '\n')
    print(json.dumps({'rate': record['comparison']['new_mean'], 'status': record['comparison']['status'],
        'numeric_runtime': runtime['jax_default_matmul_precision'], 'integrity': record['audit']['status']}))


if __name__ == '__main__':
    main()
