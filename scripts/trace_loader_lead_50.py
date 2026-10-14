"""Fresh original-parameter control of a32-call initial creator lead.

Two original source queues retain their lane assignments and depth. A condition
serializes the creators' sampling calls in fixed blocks, an explicit scheduling
control. No historical execution order, data or model state is supplied.
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
    parser.add_argument('--first-creator', type=int, choices=[0], default=0)
    parser.add_argument('--sample-block', type=int, choices=[16], default=16)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--creator-lead', type=int, choices=[32], default=32)
    args, remaining = parser.parse_known_args()
    os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
    if not args.gpu_rate_socket:
        from reproduction_tools import training
        original_plan = training.plan
        def plan(speed, algorithm, run_dir, python=None, gpu_python=None):
            execution = original_plan(speed, algorithm, run_dir, python, gpu_python)
            execution['command'][2] = str(Path(__file__).resolve())
            execution['command'] += ['--first-creator', str(args.first_creator), '--sample-block', str(args.sample_block), '--creator-lead', str(args.creator_lead)]
            execution['capture_extra_sources'] = [str(Path(__file__).resolve()), str(ROOT / 'scripts/trace_fresh_50.py')]
            execution['replay_creator_schedule'] = {'first': args.first_creator, 'lead': args.creator_lead, 'block': args.sample_block}
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
        condition = module.threading.Condition()
        turn = {'owner': args.first_creator, 'remaining': args.creator_lead}
        def creator(owner, assigned_sources, outputs):
            try:
                iterators = [source() for source in assigned_sources]
                while prefetcher._running:
                    for iterator, queue in zip(iterators, outputs):
                        with condition:
                            condition.wait_for(lambda: turn['owner'] == owner or not prefetcher._running)
                            if not prefetcher._running:
                                return
                            sample = next(iterator)
                            turn['remaining'] -= 1
                            if turn['remaining'] == 0:
                                turn.update(owner=1-owner, remaining=args.sample_block)
                                condition.notify_all()
                        # Never hold the sampling condition while a full queue waits.
                        queue.put(sample)
            except Exception as error:
                with condition:
                    prefetcher._running = False
                    condition.notify_all()
                error.stacktrace = ''.join(module.traceback.format_exception(*sys.exc_info()))
                outputs[0].put(error)
                raise
        for index in (0, 1):
            thread = module.threading.Thread(target=creator, args=(index, *assignments[index]), daemon=True)
            thread.start(); prefetcher._threads.append(thread)
        prefetcher._batches = module.queuelib.Queue(prefetch)
        batcher = module.threading.Thread(target=prefetcher._batcher,
            args=(prefetcher._queues, prefetcher._batches), daemon=True)
        batcher.start(); prefetcher._threads.append(batcher)
        prefetcher._once = False
    e.Prefetch.__init__ = init
    (args.run_dir / 'loader_runtime.json').write_text(json.dumps({
        'creator_schedule': {'first': args.first_creator, 'lead': args.creator_lead, 'block': args.sample_block},
        'workers': 2, 'sources': 16, 'prefetch': 4,
        'sampler_seed': 0, 'model_seed': 3501,
        'learning_parameters_unchanged': True, 'driver_changed': False,
        'jax_default_matmul_precision': jax.config.jax_default_matmul_precision,
        'historical_order_identified': False,
        'limitation': 'Explicit serialized creator sampling schedule; not the identified historical schedule.'}, indent=2) + '\n')
    script = ROOT / 'scripts/trace_fresh_50.py'
    sys.argv = [str(script), '--run-dir', str(args.run_dir), '--gpu-rate-socket', args.gpu_rate_socket, *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
