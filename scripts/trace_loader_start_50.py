"""Declared original-parameter control of the two replay-loader start order.

Only creator launch order is reversed. Sources, lane assignment, queue depth,
uniform sampler seed, model seed, learning math and environment are unchanged.
This runtime scheduling control is explicit, not identified historical state.
"""
import argparse
import importlib
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
    # Keep this scheduling control separate from the preceding FP32 control.
    os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
    if not args.gpu_rate_socket:
        from reproduction_tools import training
        original_plan = training.plan
        def plan(speed, algorithm, run_dir, python=None, gpu_python=None):
            execution = original_plan(speed, algorithm, run_dir, python, gpu_python)
            execution['command'][2] = str(Path(__file__).resolve())
            execution['capture_extra_sources'] = [str(Path(__file__).resolve()), str(ROOT / 'scripts/trace_fresh_50.py')]
            execution['runtime_loader_creator_start_order'] = [1, 0]
            execution['historical_creator_execution_order_known'] = False
            return execution
        training.plan = plan
        training.launch(50, 'dreamer', args.run_dir, dry_run=args.dry_run)
        return
    sys.modules['tensorflow'] = None
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0, str(preserved_vendor()))
    from dreamerv3 import train as native
    import jax
    assert jax.config.jax_default_matmul_precision is None
    e = native.embodied
    module = importlib.import_module(e.Prefetch.__module__)
    original_init = e.Prefetch.__init__
    def init(prefetcher, sources, workers=0, prefetch=4):
        if not workers:
            return original_init(prefetcher, sources, workers, prefetch)
        assert workers == 2 and prefetch == 4 and len(sources) == 16
        prefetcher.workers = workers
        prefetcher._running = True; prefetcher._threads = []; prefetcher._queues = []
        assignments = [([], []) for _ in range(workers)]
        for index, source in enumerate(sources):
            queue = module.queuelib.Queue(prefetch)
            prefetcher._queues.append(queue)
            assignments[index % workers][0].append(source)
            assignments[index % workers][1].append(queue)
        for index in (1, 0):
            creator = module.threading.Thread(target=prefetcher._creator, args=assignments[index], daemon=True)
            creator.start(); prefetcher._threads.append(creator)
        prefetcher._batches = module.queuelib.Queue(prefetch)
        batcher = module.threading.Thread(target=prefetcher._batcher,
            args=(prefetcher._queues, prefetcher._batches), daemon=True)
        batcher.start(); prefetcher._threads.append(batcher)
        prefetcher._once = False
    e.Prefetch.__init__ = init
    (args.run_dir / 'loader_runtime.json').write_text(json.dumps({
        'creator_start_order': [1, 0], 'sources': 16, 'workers': 2, 'prefetch': 4,
        'source_lane_assignment': 'Unchanged round robin: worker0 even lanes, worker1 odd lanes',
        'sampler': 'Unchanged uniform sampler seed0', 'driver_changed': False,
        'learning_parameters_unchanged': True, 'historical_order_identified': False,
        'jax_default_matmul_precision': jax.config.jax_default_matmul_precision,
        'limitation': 'Explicit loader startup control; subsequent scheduling remains asynchronous.'}, indent=2) + '\n')
    script = ROOT / 'scripts/trace_fresh_50.py'
    sys.argv = [str(script), '--run-dir', str(args.run_dir), '--gpu-rate-socket', args.gpu_rate_socket, *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
