"""Original-parameter fresh training with explicit FP32 dot precision.

Network compute dtype was alreadyfloat32. This declares the additional numeric
runtime choice instead of relying on the GPU's default matrix precision. It
does not change learning parameters, seeds, samples, losses or the driver.
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
    parser.add_argument('--gpu-rate-socket')
    parser.add_argument('--dry-run', action='store_true')
    args, remaining = parser.parse_known_args()
    os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'float32'
    if not args.gpu_rate_socket:
        from reproduction_tools import training
        original_plan = training.plan
        def plan(speed, algorithm, run_dir, python=None, gpu_python=None):
            execution = original_plan(speed, algorithm, run_dir, python, gpu_python)
            execution['command'][2] = str(Path(__file__).resolve())
            execution['capture_extra_sources'] = [str(Path(__file__).resolve()), str(ROOT / 'scripts/trace_fresh_50.py')]
            execution['numeric_runtime_override'] = {'JAX_DEFAULT_MATMUL_PRECISION': 'float32'}
            execution['original_learning_parameters_unchanged'] = True
            execution['historical_matmul_precision_known'] = False
            return execution
        training.plan = plan
        training.launch(50, 'dreamer', args.run_dir, dry_run=args.dry_run)
        return
    sys.modules['tensorflow'] = None
    import jax
    jax.config.update('jax_default_matmul_precision', 'float32')
    assert jax.config.jax_default_matmul_precision == 'float32'
    (args.run_dir / 'numeric_runtime.json').write_text(json.dumps({
        'jax_default_matmul_precision': jax.config.jax_default_matmul_precision,
        'network_compute_dtype': 'float32, unchanged', 'driver_modified': False,
        'original_learning_parameters_unchanged': True,
        'limitation': 'Historical dot precision was not recorded; this is an explicit arithmetic-runtime control.'}, indent=2) + '\n')
    script = ROOT / 'scripts/trace_fresh_50.py'
    sys.argv = [str(script), '--run-dir', str(args.run_dir), '--gpu-rate-socket', args.gpu_rate_socket, *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
