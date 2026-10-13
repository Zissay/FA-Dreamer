"""Fresh original50km/h training with consumed-replay indices recorded.

Recording observes original two-loader batches without changing their contents,
sampling calls, learning parameters or model RNG. Timing overhead is explicit.
Optional playback uses recorded indices to select freshly generated observations;
it loads no model weights or old observations and is a separate diagnostic mode.
"""
import argparse,gzip,hashlib,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpu-rate-socket')
    parser.add_argument('--replay-trace',type=Path)
    args,core_args=parser.parse_known_args()
    if not args.gpu_rate_socket:
        from reproduction_tools import training
        original_plan=training.plan
        def plan(speed,algorithm,run_dir,python=None,gpu_python=None):
            result=original_plan(speed,algorithm,run_dir,python,gpu_python)
            result['command'][2]=str(Path(__file__).resolve())
            if args.replay_trace:
                result['command']+=['--replay-trace',str(args.replay_trace.resolve())]
            result['capture_extra_sources']=[str(Path(__file__).resolve())]
            result['replay_trace_mode']='playback' if args.replay_trace else 'observe_original_async_batches'
            return result
        training.plan=plan
        training.launch(50,'dreamer',args.run_dir)
        return
    sys.modules['tensorflow']=None
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0,str(preserved_vendor()))
    source=ROOT/'experiments/50kmh/source/dreamer'
    sys.path.insert(0,str(source))
    import numpy as np
    from dreamerv3 import train as native,jaxagent
    e=native.embodied
    # Native make_replay imports the top-level embodied namespace. Patching
    # dreamerv3.embodied creates a distinct module instance and misses the live class.
    from embodied.replay.generic import Generic
    assert issubclass(e.replay.Uniform,Generic)
    controller={'frames':0,'updates':0,'completed_updates':0,'recorded_choices':0,
                'id_to_frame':{},'start_to_key':{},'replay':None}
    trace_path=args.run_dir/'consumed_replay_batches.jsonl.gz'
    output=gzip.open(trace_path,'wb')
    trace_hash=hashlib.sha256()
    input_file=gzip.open(args.replay_trace,'rt') if args.replay_trace else None
    original_add=Generic.add
    original_init=jaxagent.JAXAgent.__init__
    original_train=jaxagent.JAXAgent.train
    def add(replay,step,worker=0,load=False):
        assert not load,'Fresh traced training cannot load pre-existing replay'
        assert controller['replay'] is None or controller['replay'] is replay
        controller['replay']=replay
        controller['frames']+=1
        original_add(replay,step,worker,load)
        stream=replay.streams[worker]
        controller['id_to_frame'][stream[-1]['id'].tobytes()]=controller['frames']
        if len(stream)==replay.length and replay.counters[worker]==0:
            first=controller['id_to_frame'][stream[0]['id'].tobytes()]
            controller['start_to_key'][first]=replay.sampler.keys[-1]
    def init(model,*values,**kwargs):
        original_init(model,*values,**kwargs)
        digest=hashlib.sha256();arrays=model.save()
        for value in arrays:
            a=np.asarray(value);digest.update(str(a.shape).encode());digest.update(str(a.dtype).encode());digest.update(a.tobytes())
        info={'fresh_random_initialization':True,'checkpoint_restore':False,'seed':3501,
              'variable_arrays':len(arrays),'initial_variable_digest':digest.hexdigest(),
              'callback_frames_at_initialization':controller['frames']}
        assert controller['frames']==0
        (args.run_dir/'initialization_proof.json').write_text(json.dumps(info,indent=2)+'\n')
    def train(model,data,state=None):
        controller['updates']+=1
        if input_file:
            line=input_file.readline()
            if not line:raise ValueError('Recorded sampling trace ended early')
            choice=json.loads(line)
            assert choice['update']==controller['updates'] and choice['callback_frame']==controller['frames']
            replay=controller['replay'];samples=[]
            for start in choice['sequence_start_frames']:
                seq=replay.table[controller['start_to_key'][start]]
                sample={k:e.convert([s[k] for s in seq]) for k in seq[0]}
                sample['is_first'][0]=True;samples.append(sample)
            data={k:np.stack([s[k] for s in samples]) for k in samples[0]}
        starts=[controller['id_to_frame'][value.tobytes()] for value in np.asarray(data['id'])[:,0,:]]
        assert len(starts)==16 and all(start+63*20<=controller['frames'] for start in starts)
        choice={'update':controller['updates'],'callback_frame':controller['frames'],
                'sequence_start_frames':starts}
        encoded=(json.dumps(choice,separators=(',',':'))+'\n').encode()
        output.write(encoded);trace_hash.update(encoded)
        controller['recorded_choices']+=1
        result=original_train(model,data,state)
        controller['completed_updates']+=1
        return result
    Generic.add=add
    jaxagent.JAXAgent.__init__=init
    jaxagent.JAXAgent.train=train
    core=source/'dreamer_simple/train.py'
    sys.argv=[str(core),'--run-dir',str(args.run_dir),'--gpu-rate-socket',args.gpu_rate_socket,*core_args]
    succeeded=False
    try:
        runpy.run_path(str(core),run_name='__main__')
        if input_file:assert not input_file.readline(),'Recorded sampling trace has unused updates'
        succeeded=True
    finally:
        output.close()
        if input_file:input_file.close()
        info={'mode':'playback' if args.replay_trace else 'observe_original_async_batches',
              'training_seed':3501,'data_loaders':2,'callback_frames':controller['frames'],
              'recorded_optimizer_updates':controller['completed_updates'],
              'attempted_updates':controller['updates'],'recorded_batch_choices':controller['recorded_choices'],
              'success':succeeded,
              'checkpoint_restore':False,'historical_observations_loaded':False,
              'trace_raw_sha256':trace_hash.hexdigest(),
              'trace_gzip_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'timing_limitation':'Observer writes can affect asynchronous timing; record mode does not alter selected samples or random draws',
              'playback_limitation':'Playback repeats recorded stochastic indices using own fresh observations; separate from original async recipe'}
        (args.run_dir/'replay_trace_summary.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':main()
