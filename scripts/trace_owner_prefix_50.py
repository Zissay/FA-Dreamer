"""Fresh original-parameter training with an explicit estimated creator prefix.

Only initial sampling call ownership is prescribed. Afterwards both creators
sample asynchronously. No prior weights, observations or policy actions enter
training. Estimated order is disclosed, not claimed historical unique state.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--owner-prefix',type=Path,required=True)
    parser.add_argument('--gpu-rate-socket')
    parser.add_argument('--dry-run',action='store_true')
    args, remaining = parser.parse_known_args()
    profile_path = args.owner_prefix.resolve()
    profile = json.loads(profile_path.read_text())
    owners = profile['creator_owner_prefix']
    assert 80 <= len(owners) <= 113 and all(type(x) is int and x in (0,1) for x in owners)
    counts = [0,0]
    for index,owner in enumerate(owners):
        counts[owner] += 1
        assert abs(counts[0]-counts[1]) <= 33
        if min(counts) >= 40:
            assert index == len(owners)-1, 'Prefix ends at first complete five batches'
    assert min(counts) == 40
    assert profile['model_seed']==3501 and profile['uniform_sampler_seed']==0
    os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION',None)
    if not args.gpu_rate_socket:
        from reproduction_tools import training
        from reproduction_tools.records import sha256
        original_plan = training.plan
        def plan(speed,algorithm,run_dir,python=None,gpu_python=None):
            execution = original_plan(speed,algorithm,run_dir,python,gpu_python)
            execution['command'][2]=str(Path(__file__).resolve())
            execution['command'] += ['--owner-prefix',str(profile_path)]
            execution['capture_extra_sources']=[str(Path(__file__).resolve()),str(ROOT/'scripts/trace_fresh_50.py'),str(profile_path)]
            execution['initial_creator_owner_profile_sha256']=sha256(profile_path)
            execution['estimated_initial_order_not_unique_historical_state']=True
            execution['external_weights_or_observations']=[]
            return execution
        training.plan=plan
        training.launch(50,'dreamer',args.run_dir,dry_run=args.dry_run)
        return
    sys.modules['tensorflow']=None
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0,str(preserved_vendor()))
    from dreamerv3 import train as native
    import jax
    assert jax.config.jax_default_matmul_precision is None
    e=native.embodied
    module=importlib.import_module(e.Prefetch.__module__)
    original_init=e.Prefetch.__init__
    def init(prefetcher,sources,workers=0,prefetch=4):
        if not workers:
            return original_init(prefetcher,sources,workers,prefetch)
        assert workers==2 and prefetch==4 and len(sources)==16
        prefetcher.workers=workers
        prefetcher._running=True; prefetcher._threads=[]; prefetcher._queues=[]
        assignments=[([],[]) for _ in range(workers)]
        for index,source in enumerate(sources):
            queue=module.queuelib.Queue(prefetch)
            prefetcher._queues.append(queue)
            assignments[index%workers][0].append(source)
            assignments[index%workers][1].append(queue)
        condition=module.threading.Condition()
        cursor={'index':0}
        def creator(owner,assigned_sources,outputs):
            try:
                iterators=[source() for source in assigned_sources]
                while prefetcher._running:
                    for iterator,queue in zip(iterators,outputs):
                        sampled=False
                        with condition:
                            condition.wait_for(lambda: cursor['index']>=len(owners) or owners[cursor['index']]==owner or not prefetcher._running)
                            if not prefetcher._running:
                                return
                            if cursor['index']<len(owners):
                                sample=next(iterator); sampled=True
                                cursor['index'] += 1; condition.notify_all()
                        if not sampled:
                            sample=next(iterator)
                        queue.put(sample)
            except Exception as error:
                with condition:
                    prefetcher._running=False; condition.notify_all()
                error.stacktrace=''.join(module.traceback.format_exception(*sys.exc_info()))
                prefetcher._batches.put(error)
                raise
        prefetcher._batches=module.queuelib.Queue(prefetch)
        for index in (0,1):
            thread=module.threading.Thread(target=creator,args=(index,*assignments[index]),daemon=True)
            thread.start(); prefetcher._threads.append(thread)
        batcher=module.threading.Thread(target=prefetcher._batcher,args=(prefetcher._queues,prefetcher._batches),daemon=True)
        batcher.start(); prefetcher._threads.append(batcher)
        prefetcher._once=False
    e.Prefetch.__init__=init
    (args.run_dir/'owner_prefix_runtime.json').write_text(json.dumps({
        'creator_owner_prefix':owners,'subsequent_creators':'Original asynchronous sampling calls',
        'workers':2,'sources':16,'prefetch':4,'uniform_sampler_seed':0,'model_seed':3501,
        'jax_default_matmul_precision':None,'driver_changed':False,
        'learning_parameters_unchanged':True,'historical_state_identified':False,
        'estimated_order_input':profile_path.name,
        'external_weights_or_observations':[],
        'limitation':'Estimated finite startup order inferred from earliest actor fingerprint; not uniquely recovered historical state or typical performance proof.'},indent=2)+'\n')
    script=ROOT/'scripts/trace_fresh_50.py'
    sys.argv=[str(script),'--run-dir',str(args.run_dir),'--gpu-rate-socket',args.gpu_rate_socket,*remaining]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':
    main()
