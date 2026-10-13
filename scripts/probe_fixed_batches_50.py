"""First five updates with fixed prefill batches; no checkpoint or full-result claim."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.modules['tensorflow']=None
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduction_tools.training import preserved_vendor
sys.path.insert(0,str(preserved_vendor()))
sys.path.insert(0,str(ROOT/'experiments/50kmh/source/dreamer'))
import numpy as np
from dreamer_simple.train import Agent,model_config,e,native,FromGym,MatchedDreamerEnv,matched_configs,install_truncated_normal_actor

def digest(values):
    h=hashlib.sha256()
    for x in values:
        a=np.asarray(x);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix-file',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reverse-batch-slots',action='store_true',help='Controlled lane-order change only; same samples per batch')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    stored=np.load(args.prefix_file)
    prefix={k:stored[k][:,:117] for k in stored.files}
    assert int(prefix['is_first'][:,:115].sum())==60
    assert int(prefix['is_terminal'][:,:115].sum())==40
    # Frames0--114 contain only seeded uniform-prefill transitions. The first
    # actor was generated at frame115 before the first optimizer update.
    replay=e.replay.Uniform(64,200000,directory=None,seed=0)
    for t in range(115):
        for worker in range(20):replay.add({k:v[worker,t] for k,v in prefix.items()},worker)
    source=replay.dataset()
    batches=[]
    for _ in range(5):
        rows=[next(source) for _ in range(16)]
        batches.append({k:np.stack([r[k] for r in rows]) for k in rows[0]})
    batch_hashes=[digest([b[k] for k in sorted(b) if k!='id']) for b in batches]
    if args.reverse_batch_slots:
        batches=[{k:v[::-1].copy() for k,v in batch.items()} for batch in batches]
    consumed_hashes=[digest([b[k] for k in sorted(b) if k!='id']) for b in batches]
    hp=json.loads((ROOT/'experiments/50kmh/configs/dreamer.json').read_text())
    cfg=model_config(hp,3501,'cuda')
    ch,task=matched_configs(json.loads((ROOT/'experiments/50kmh/configs/task.json').read_text()),3501)
    install_truncated_normal_actor(stable_sampling=True)
    env=native.wrap_env(FromGym(MatchedDreamerEnv(ch,task,potential_model=True)),cfg)
    model=Agent(env.obs_space,env.act_space,e.Counter(),cfg)
    before=digest(model.save())
    obs_keys=list(env.obs_space)
    obs=lambda t:{k:prefix[k][:,t] for k in obs_keys}
    first,state=model.policy(obs(115),None,mode='train')
    initial_action_error=float(np.max(np.abs(first['action']-prefix['action'][:,115])))
    assert initial_action_error<1e-6,initial_action_error
    train_state=None;metrics=[]
    for batch in batches:
        _,train_state,mets=model.train(batch,train_state)
        metrics.append({k:float(v) for k,v in mets.items() if k in (
            'model_opt_loss','model_opt_grad_norm','actor_opt_loss','actor_opt_grad_norm',
            'extr_critic_critic_opt_loss','extr_critic_critic_opt_grad_norm')})
    second,_=model.policy(obs(116),state,mode='train')
    values=model.save()
    np.savez(args.output_dir/'variables_after5.npz',**{f'v{i:03d}':v for i,v in enumerate(values)})
    np.save(args.output_dir/'action_after5.npy',second['action'])
    report={'diagnostic_only':True,'fresh_random_initialization':True,'checkpoint_restore':False,
        'training_seed':3501,'sampler_seed':0,'optimizer_updates':5,'new_environment_actions':0,
        'prefill_frames':115,'prefill_real_actions':2240,'minibatch_policy':'Fixed synchronous control for diagnosis; not the original asynchronous full-training recipe',
        'initial_model_digest':before,'final_model_digest':digest(values),'batch_hashes':batch_hashes,
        'consumed_batch_hashes':consumed_hashes,'reverse_batch_slots':args.reverse_batch_slots,
        'initial_actor_action_max_error_vs_historical':initial_action_error,'metrics':metrics,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'experiments/50kmh/source/dreamer/dreamer_simple/train.py',
            ROOT/'experiments/50kmh/source/dreamer/dreamer_simple/agent.py',Path(__file__)]}}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    env.close()
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
