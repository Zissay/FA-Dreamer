"""Fresh computational verification of a completed matching candidate's indices.

Reads only candidate metadata and consumed replay indices. Builds a new random
model and new observations from empty replay. Never reads checkpoint weights or
candidate observations. Numerical runtime selection is copied and disclosed.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--gpu-rate-socket')
    parser.add_argument('--dry-run', action='store_true')
    args, remaining = parser.parse_known_args()
    from reproduction_tools.records import read_json, sha256
    source = args.source_run.resolve()
    trace = source / 'consumed_replay_batches.jsonl.gz'
    assert read_json(source / 'training_complete.json')['real_actions'] == 50000
    proof = read_json(source / 'initialization_proof.json')
    assert proof['fresh_random_initialization'] and not proof['checkpoint_restore']
    assert proof['callback_frames_at_initialization'] == 0 and proof['seed'] == 3501
    summary = read_json(source / 'replay_trace_summary.json')
    assert summary['success'] and summary['recorded_optimizer_updates'] == 12175
    assert sha256(trace) == summary['trace_gzip_sha256']
    precision = (read_json(source / 'numeric_runtime.json')['jax_default_matmul_precision']
                 if (source / 'numeric_runtime.json').exists() else None)
    assert precision in (None, 'float32')
    if precision:
        os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = precision
    else:
        os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
    if not args.gpu_rate_socket:
        from reproduction_tools.comparison import compare_historical_50
        comparison = compare_historical_50(source)
        if not args.dry_run:
            assert comparison['status'].startswith('REPRODUCED'), 'Only an audited matching candidate can launch this validation'
        from reproduction_tools import training
        original_plan = training.plan
        def plan(speed, algorithm, run_dir, python=None, gpu_python=None):
            execution = original_plan(speed, algorithm, run_dir, python, gpu_python)
            execution['command'][2] = str(Path(__file__).resolve())
            execution['command'] += ['--source-run', str(source)]
            execution['capture_extra_sources'] = [str(Path(__file__).resolve()), str(ROOT / 'scripts/trace_fresh_50.py')]
            execution['replay_trace_mode'] = 'fresh_computational_playback'
            execution['sampling_trace_input_sha256'] = sha256(trace)
            execution['numeric_runtime_override'] = {'JAX_DEFAULT_MATMUL_PRECISION': precision}
            execution['external_weights_or_observations'] = []
            return execution
        training.plan = plan
        training.launch(50, 'dreamer', args.run_dir, dry_run=args.dry_run)
        return
    sys.modules['tensorflow'] = None
    import jax
    if precision:
        jax.config.update('jax_default_matmul_precision', precision)
    assert jax.config.jax_default_matmul_precision == precision
    (args.run_dir / 'computational_validation.json').write_text(json.dumps({
        'candidate': source.name, 'trace_input_sha256': sha256(trace),
        'random_initialization_required': True, 'empty_initial_replay_required': True,
        'external_weights_or_observations': [], 'sampling_indices_only': True,
        'jax_default_matmul_precision': precision,
        'limitation': 'Deterministic index playback from a declared candidate; not recovery of the historical asynchronous execution schedule.'}, indent=2) + '\n')
    script = ROOT / 'scripts/trace_fresh_50.py'
    sys.argv = [str(script), '--run-dir', str(args.run_dir), '--gpu-rate-socket', args.gpu_rate_socket,
                '--replay-trace', str(trace), *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
