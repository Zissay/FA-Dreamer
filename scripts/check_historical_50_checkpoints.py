"""Read-only sampled-policy validation of historical50km/h checkpoints.

Separate diagnostic actions; never training data or a replacement training curve.
Uses recovered original policy code, unchanged weights and the same50-step task.
"""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
sys.modules['tensorflow']=None
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduction_tools.training import preserved_vendor
sys.path.insert(0,str(preserved_vendor()))
sys.path.insert(0,str(ROOT/'experiments/50kmh/source/dreamer'))
import numpy as np
from dreamer_simple.train import Agent,model_config,e,native,FromGym,MatchedDreamerEnv,matched_configs,install_truncated_normal_actor
from dreamerv3 import jaxutils

def digest(model):
    h=hashlib.sha256()
    for value in model.save():
        a=np.asarray(value);h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run',type=Path,required=True)
    parser.add_argument('--fresh-run',type=Path)
    parser.add_argument('--milestones',type=int,nargs='+',default=[30000,35000,40000,50000])
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--evaluation-seed',type=int,default=3501)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert args.episodes>0 and args.episodes%20==0
    folder=ROOT/'experiments/50kmh'
    hp=json.loads((folder/'configs/dreamer.json').read_text(encoding='utf-8'))
    cfg=model_config(hp,3501,'cpu')
    ch,task=matched_configs(json.loads((folder/'configs/task.json').read_text(encoding='utf-8')),3501)
    install_truncated_normal_actor(stable_sampling=True)
    envs=[native.wrap_env(FromGym(MatchedDreamerEnv(ch,task,potential_model=True)),cfg) for _ in range(20)]
    model=Agent(envs[0].obs_space,envs[0].act_space,e.Counter(),cfg)
    paths=[('historical',step,args.historical_run/f'milestone_checkpoints/checkpoint_{step:06d}.pkl') for step in args.milestones]
    if args.fresh_run:
        paths.append(('fresh_source_only',50000,args.fresh_run/'checkpoint.pkl'))
    report={'evaluation_only':True,'training_updates':0,'sampling_mode':'original model.policy(mode=train), sampled latent and action',
        'episodes_per_checkpoint':args.episodes,'episode_steps':50,'channel_seed':7,
        'evaluation_rng_seed':args.evaluation_seed,'results':[]}
    pack=lambda obs:{key:np.stack([row[key] for row in obs]) for key in obs[0]}
    for label,step,path in paths:
        with path.open('rb') as f:checkpoint=pickle.load(f)
        model.load(checkpoint['agent'])
        before=digest(model)
        model.rng=jaxutils.RNG(args.evaluation_seed)
        terminal=[];positions=[]
        for batch in range(args.episodes//20):
            obs=pack([env.step({'action':np.zeros(4,np.float32),'reset':True}) for env in envs])
            state=None
            for _ in range(50):
                outs,state=model.policy(obs,state,mode='train')
                obs=pack([env.step({'action':act,'reset':False}) for env,act in zip(envs,outs['action'])])
            assert np.asarray(obs['is_terminal']).all()
            terminal.extend(np.asarray(obs['rate_target'],float).reshape(-1).tolist())
            positions.extend((np.asarray(obs['vector'][:,160:164],float)*0.25).tolist())
        assert digest(model)==before,'Read-only policy evaluation modified checkpoint variables'
        result={'label':label,'training_milestone':step,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'mean':float(np.mean(terminal)),'std':float(np.std(terminal)),
            'min':min(terminal),'max':max(terminal),'terminal_rates':terminal,'positions_m':positions,
            'checkpoint_variables_unchanged':True,'diagnostic_actions':args.episodes*50}
        report['results'].append(result)
        print(json.dumps({key:result[key] for key in ['label','training_milestone','mean','std']}),flush=True)
    for env in envs:env.close()
    report['total_separate_diagnostic_actions']=sum(r['diagnostic_actions'] for r in report['results'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
