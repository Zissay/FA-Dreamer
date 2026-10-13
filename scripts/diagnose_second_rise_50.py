"""Reconstructed continuation from the historical30k plateau; not a fresh run.

Restores model/optimizer states, original settings and reconstructible JAX RNG
call count. Uses only the pre30k replay prefix. Historical loader queues and
pre-update policy carry were not saved: approximate continuation is explicit.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
sys.modules['tensorflow']=None
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--prefix-file',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu-rate-socket')
    args=p.parse_args()
    checkpoint_path=args.parent_run/'milestone_checkpoints/checkpoint_030000.pkl'
    assert sha(checkpoint_path)=='cf5df546a46e043c09b1c8279f1618f38b67d7ef50d165d172fff8ea66cee29e'
    assert sha(args.prefix_file)=='797b6b5da5ae69739da54b49d19b26b9539f373fe5fa5f5fb198f57a5093079a'
    if not args.gpu_rate_socket:
        # Reuse complete pretraining source/config capture and service cleanup.
        from reproduction_tools import training
        normal=training.plan
        def diagnostic_plan(speed,algorithm,run_dir,python=None,gpu_python=None):
            execution=normal(speed,algorithm,run_dir,python,gpu_python)
            execution['command']=[execution['command'][0],'-u',str(Path(__file__).resolve()),
                '--parent-run',str(args.parent_run.resolve()),'--prefix-file',str(args.prefix_file.resolve()),
                '--run-dir',execution['run_dir'],'--gpu-rate-socket',execution['socket']]
            execution['training_interactions']=20000
            execution['diagnostic_continuation']=True
            execution['capture_extra_sources']=[str(Path(__file__).resolve())]
            return execution
        training.plan=diagnostic_plan
        training.launch(50,'dreamer',args.run_dir)
        return
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0,str(preserved_vendor()))
    sys.path.insert(0,str(ROOT/'experiments/50kmh/source/dreamer'))
    import numpy as np
    from dreamer_simple.train import Agent,model_config,e,native,FromGym,MatchedDreamerEnv,matched_configs,install_truncated_normal_actor,TrainingLog
    from dreamerv3 import jaxutils
    from dataclasses import asdict
    case=ROOT/'experiments/50kmh'
    hp=json.loads((case/'configs/dreamer.json').read_text(encoding='utf-8'))
    original=json.loads((args.parent_run/'experiment.json').read_text(encoding='utf-8'))
    assert hp=={**original['hyperparameters'],'data_loaders':2}
    assert original['seed']==3501
    path=args.parent_run/'milestone_checkpoints/checkpoint_030000.pkl'
    with path.open('rb') as f:checkpoint=pickle.load(f)
    assert checkpoint['step']==30620
    with np.load(args.prefix_file) as stored:prefix={k:stored[k] for k in stored.files}
    assert prefix['vector'].shape==(20,1530,545) and int(prefix['is_terminal'].sum())==600
    # Original log IDs were allocated by threads. Recover actual driver slots
    # from the first seeded20-command batch before replay insertion.
    with (args.parent_run/'training_queries.csv').open(newline='') as f:
        first=[r for r in csv.DictReader(f) if int(r['steps'])<=20]
    expected=np.random.RandomState(1003504).uniform(-1.,1.,(20,4)).astype(np.float32)
    worker_to_log=np.full(20,-1,int)
    for row in first:
        command=np.array([float(row['a'+str(i)]) for i in range(4)],np.float32)
        errors=np.max(np.abs(expected-command),axis=1)
        worker=int(errors.argmin());assert errors[worker]<1e-7
        worker_to_log[worker]=int(row['env_index'])
    assert len(set(worker_to_log.tolist()))==20 and min(worker_to_log)>=0
    prefix={k:v[worker_to_log] for k,v in prefix.items()}
    args.run_dir.mkdir(parents=True,exist_ok=True)
    cfg=model_config(hp,3501,'cuda');cfg.save(e.Path(args.run_dir/'config.yaml'))
    ch,task=matched_configs(json.loads((case/'configs/task.json').read_text()),3501,20,args.gpu_rate_socket)
    assert asdict(ch)==original['channel']
    install_truncated_normal_actor(stable_sampling=True)
    log=TrainingLog(args.run_dir,20,20000);log.source='actor_policy'
    ctor=lambda:native.wrap_env(FromGym(MatchedDreamerEnv(ch,task,log,True)),cfg)
    env=e.BatchEnv([e.Parallel(ctor,'thread') for _ in range(20)],parallel=True)
    counter=e.Counter();counter.load(checkpoint['step'])
    model=Agent(env.obs_space,env.act_space,counter,cfg);model.load(checkpoint['agent'])
    steps={k:int(np.asarray(v)) for k,v in model.varibs.items() if k.endswith('/step/value') and np.asarray(v).size==1}
    actor_steps=next(v for k,v in steps.items() if k.endswith('/actor_opt/step/value'))
    assert actor_steps==7080
    replay=native.make_replay(cfg,e.Path(args.run_dir/'replay'))
    for j in range(1530):
        for i in range(20):replay.add({k:v[i,j] for k,v in prefix.items()},i)
    # Last historical checkpoint follows a center reset. Reconstruct that reset
    # and a policy carry with the last policy RNG key and current saved weights.
    # Original carry used weights before five updates; exact carry is unknown.
    reset={k:np.zeros((20,)+s.shape,s.dtype) for k,s in env.act_space.items()}
    reset['reset']=np.ones(20,bool)
    obs=env.step(reset);assert np.asarray(obs['is_first']).all()
    policy_calls=(checkpoint['step']-2300)//20
    last_policy_call_index=1+(actor_steps-5)+(policy_calls-1)
    model.rng=jaxutils.RNG(3501)
    for _ in range(last_policy_call_index):model.rng.next()
    acts,policy_state=model.policy(obs,None,mode='train')
    for _ in range(5):model.rng.next()
    acts['reset']=np.zeros(20,bool)
    for i in range(20):replay.add({k:v[i] for k,v in {**obs,**acts}.items()},i)
    dataset=iter(model.dataset(replay.dataset))
    ratio=e.when.Ratio(hp['train_ratio']/(hp['batch_size']*hp['batch_length']))
    ratio._prev=2301+(actor_steps-1)/ratio._ratio
    assert ratio._prev==30617
    driver=e.Driver(env);driver._state=policy_state;driver._acts=acts
    driver.on_step(lambda t,w:counter.increment());driver.on_step(replay.add)
    stats=e.Metrics();state=None
    logger=e.Logger(counter,[e.logger.JSONLOutput(e.Path(args.run_dir),'metrics.jsonl')])
    def update(tran,worker):
        nonlocal state
        for _ in range(ratio(counter)):
            _,state,mets=model.train(next(dataset),state)
            assert all(np.isfinite(v).all() for k,v in mets.items() if k.endswith(('_opt_loss','_opt_grad_norm')))
            stats.add(mets,prefix='train')
        model.sync()
    driver.on_step(update)
    save(args.run_dir/'experiment.json',{'diagnostic_continuation':True,'fresh_from_scratch':False,
        'exact_historical_resume':False,'parent_checkpoint':str(path),'parent_checkpoint_sha256':sha(path),
        'historical_prefix_actions':30000,'additional_real_action_budget':20000,'training_seed':3501,
        'hyperparameters':hp,'channel':asdict(ch),'environment':asdict(task),
        'prefix_file_sha256':sha(args.prefix_file),'future_original_observations_used':False,
        'optimizer_states_restored':steps,'driver_worker_to_historical_log_id':worker_to_log.tolist(),
        'reconstructed_policy_rng_call_count':policy_calls,'reconstructed_total_rng_calls':1+actor_steps+policy_calls,
        'replay_sampler_seed':0,'replay_sampler_state':'restarted; historical sampled queues/order were not saved',
        'policy_carry':'recomputed at center with recovered last-policy RNG key and saved post-update weights',
        'training_carry':'reset; replay marks the first element of every sampled sequence as is_first'})
    try:
        next_log=int(counter)+1020
        while int(counter)<51000:
            driver(lambda *x:model.policy(*x,mode='train'),steps=min(100,51000-int(counter)))
            if int(counter)>=next_log:
                logger.add(stats.result());logger.write();next_log+=1020
                print('CONTINUATION additional_real_actions='+str(log.count),flush=True)
        assert log.count==20000
        with (args.run_dir/'checkpoint.pkl').open('wb') as f:
            pickle.dump({'agent':model.save(),'step':counter.save(),'replay':replay.save()},f)
        logger.add(stats.result());logger.write()
        save(args.run_dir/'training_complete.json',{'additional_real_actions':log.count,
            'lineage_actions':50000,'fresh_from_scratch':False,'status':'complete'})
    finally:log.close();env.close()
    with (args.run_dir/'training_queries.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    terminal=[r for r in rows if int(r['episode_terminal'])==1]
    assert len(terminal)==400
    save(args.run_dir/'result.json',{'last_5k_mean':float(np.mean([float(r['rate']) for r in terminal if int(r['steps'])>15000])),
        'terminal_records':400,'additional_actions':20000,'scope':'Reconstructed historical continuation, not fresh50k reproduction'})

if __name__=='__main__':main()
