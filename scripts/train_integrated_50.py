"""Fresh30k followed by own-state20k using the restored-training procedure.

No historical weights, replay or results are training inputs. The30k boundary
restarts replay sampling and reconstructs policy carry, so this is a declared
two-stage experiment, not the original uninterrupted training protocol.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def rows(path):
    with Path(path).open(newline='') as stream:
        return sorted(csv.DictReader(stream), key=lambda row: int(row['steps']))


def worker_map(records, commands):
    import numpy as np
    mapping = np.full(20, -1, int)
    assert len(records) == 20
    for record in records:
        action = np.array([float(record[f'a{i}']) for i in range(4)], np.float32)
        errors = np.max(np.abs(commands - action), axis=1)
        worker = int(errors.argmin())
        assert errors[worker] < 1e-7 and mapping[worker] == -1
        mapping[worker] = int(record['env_index'])
    assert sorted(mapping.tolist()) == list(range(20))
    return mapping.tolist()


def prepare_runtime():
    sys.modules['tensorflow'] = None
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0, str(preserved_vendor()))
    sys.path.insert(0, str(ROOT / 'experiments/50kmh/source/dreamer'))


def model_digest(model):
    import numpy as np
    hashed = hashlib.sha256()
    for value in model.save():
        array = np.asarray(value)
        hashed.update(str(array.shape).encode()); hashed.update(str(array.dtype).encode())
        hashed.update(array.tobytes())
    return hashed.hexdigest()


def fresh30(run, socket):
    prepare_runtime()
    import numpy as np
    from dreamerv3 import train as native, jaxagent
    from embodied.replay.generic import Generic
    from fa_afdm_dreamerv3.dreamer_matched_env import TrainingLog
    e = native.embodied
    assert issubclass(e.replay.Uniform, Generic)
    collected = {i: [] for i in range(20)}
    observed = {'frames': 0, 'model': None, 'log': None, 'replay': None}
    original_add, original_init = Generic.add, jaxagent.JAXAgent.__init__
    original_call, original_log = e.Driver.__call__, TrainingLog.__init__

    class StageComplete(BaseException):
        pass

    def log_init(log, *values, **kwargs):
        original_log(log, *values, **kwargs)
        observed['log'] = log

    def init(model, *values, **kwargs):
        original_init(model, *values, **kwargs)
        assert observed['frames'] == 0
        observed['model'] = model
        save_json(run / 'initialization_proof.json', {
            'seed': 3501, 'fresh_random_initialization': True, 'external_checkpoint_restore': False,
            'callback_frames': 0, 'initial_variable_digest': model_digest(model)})

    def add(replay, step, worker=0, load=False):
        assert not load
        observed['frames'] += 1
        observed['replay'] = replay
        # Exactly the first30 episodes; omit the reset starting episode31.
        if len(collected[worker]) < 1530:
            collected[worker].append({k: np.asarray(v).copy() for k, v in step.items()
                                      if k != 'id' and not k.startswith('log_')})
        original_add(replay, step, worker, load)

    def call(driver, *values, **kwargs):
        result = original_call(driver, *values, **kwargs)
        if observed['log'].count >= 30000:
            assert observed['log'].count == 30000 and observed['frames'] == 30620
            assert all(len(collected[i]) == 1530 for i in range(20))
            prefix = {key: np.stack([[step[key] for step in collected[i]] for i in range(20)])
                      for key in collected[0][0]}
            assert prefix['vector'].shape == (20, 1530, 545)
            assert int(prefix['is_terminal'].sum()) == 600
            np.savez_compressed(run / 'prefix_replay.npz', **prefix)
            model = observed['model']
            optimizer_steps = {key: int(np.asarray(value)) for key, value in model.varibs.items()
                               if key.endswith('/step/value') and np.asarray(value).size == 1}
            assert len(optimizer_steps) == 3 and set(optimizer_steps.values()) == {7080}
            checkpoint = {'step': observed['frames'], 'agent': model.save(), 'replay': observed['replay'].save()}
            with (run / 'checkpoint_030000.pkl').open('wb') as stream:
                pickle.dump(checkpoint, stream)
            save_json(run / 'stage_complete.json', {
                'real_actions': 30000, 'callback_frames': 30620, 'optimizer_steps': optimizer_steps,
                'checkpoint_sha256': digest(run / 'checkpoint_030000.pkl'),
                'prefix_sha256': digest(run / 'prefix_replay.npz'), 'prefix_worker_order': 'native_driver',
                'model_digest': model_digest(model), 'external_training_inputs': [],
                'observation_limit': 'Observer copies can affect asynchronous timing; samples/random calls unchanged'})
            raise StageComplete()
        return result

    TrainingLog.__init__ = log_init
    Generic.add = add
    jaxagent.JAXAgent.__init__ = init
    e.Driver.__call__ = call
    core = ROOT / 'experiments/50kmh/source/dreamer/dreamer_simple/train.py'
    sys.argv = [str(core), '--run-dir', str(run), '--queries', '50000', '--seed', '3501',
                '--num-envs', '20', '--platform', 'cuda', '--gpu-rate-socket', socket,
                '--config', str(ROOT / 'experiments/50kmh/configs/dreamer.json'), '--snapshot-every', '5000']
    try:
        runpy.run_path(str(core), run_name='__main__')
    except StageComplete:
        records = rows(run / 'training_queries.csv')
        assert [int(record['steps']) for record in records] == list(range(1, 30001))
        commands = np.random.RandomState(1003504).uniform(-1., 1., (20, 4)).astype(np.float32)
        proof = json.loads((run / 'stage_complete.json').read_text())
        proof['worker_to_log_id'] = worker_map(records[:20], commands)
        save_json(run / 'stage_complete.json', proof)
        print('FRESH30_COMPLETE actions=30000', flush=True)
    else:
        raise RuntimeError('Original trainer reached50k without stopping at the declared boundary')


def resume20(run, parent, socket):
    prepare_runtime()
    import numpy as np
    from dataclasses import asdict
    from dreamer_simple.train import (Agent, model_config, e, native, FromGym, MatchedDreamerEnv,
        matched_configs, install_truncated_normal_actor, TrainingLog)
    from dreamerv3 import jaxutils
    parent_proof = json.loads((parent / 'stage_complete.json').read_text())
    assert parent_proof['external_training_inputs'] == []
    assert digest(parent / 'checkpoint_030000.pkl') == parent_proof['checkpoint_sha256']
    assert digest(parent / 'prefix_replay.npz') == parent_proof['prefix_sha256']
    assert parent.resolve() == run.parent.resolve() / 'stage1_fresh30k'
    hp = json.loads((ROOT / 'experiments/50kmh/configs/dreamer.json').read_text())
    original = json.loads((parent / 'experiment.json').read_text())
    assert original['from_checkpoint'] is None and original['seed'] == 3501
    assert original['hyperparameters'] == hp
    with (parent / 'checkpoint_030000.pkl').open('rb') as stream:
        checkpoint = pickle.load(stream)
    assert checkpoint['step'] == 30620
    with np.load(parent / 'prefix_replay.npz') as stored:
        prefix = {key: stored[key] for key in stored.files}
    assert prefix['vector'].shape == (20, 1530, 545) and int(prefix['is_terminal'].sum()) == 600
    run.mkdir(parents=True, exist_ok=False)
    cfg = model_config(hp, 3501, 'cuda'); cfg.save(e.Path(run / 'config.yaml'))
    channel, task = matched_configs(json.loads((ROOT / 'experiments/50kmh/configs/task.json').read_text()), 3501, 20, socket)
    assert asdict(channel) == original['channel']
    install_truncated_normal_actor(stable_sampling=True)
    log = TrainingLog(run, 20, 20000); log.source = 'actor_policy'
    ctor = lambda: native.wrap_env(FromGym(MatchedDreamerEnv(channel, task, log, True)), cfg)
    env = e.BatchEnv([e.Parallel(ctor, 'thread') for _ in range(20)], parallel=True)
    counter = e.Counter(); counter.load(checkpoint['step'])
    model = Agent(env.obs_space, env.act_space, counter, cfg); model.load(checkpoint['agent'])
    assert model_digest(model) == parent_proof['model_digest']
    optimizer_steps = {key: int(np.asarray(value)) for key, value in model.varibs.items()
                       if key.endswith('/step/value') and np.asarray(value).size == 1}
    assert optimizer_steps == parent_proof['optimizer_steps']
    replay = native.make_replay(cfg, e.Path(run / 'replay'))
    for frame in range(1530):
        for worker in range(20):
            replay.add({key: value[worker, frame] for key, value in prefix.items()}, worker)
    reset = {key: np.zeros((20,) + space.shape, space.dtype) for key, space in env.act_space.items()}
    reset['reset'] = np.ones(20, bool)
    obs = env.step(reset); assert np.asarray(obs['is_first']).all()
    actor_steps = next(value for key, value in optimizer_steps.items() if key.endswith('/actor_opt/step/value'))
    policy_calls = (checkpoint['step'] - 2300) // 20
    last_call = 1 + (actor_steps - 5) + (policy_calls - 1)
    model.rng = jaxutils.RNG(3501)
    for _ in range(last_call):
        model.rng.next()
    actions, policy_state = model.policy(obs, None, mode='train')
    for _ in range(5):
        model.rng.next()
    actions['reset'] = np.zeros(20, bool)
    first_commands = np.asarray(actions['action']).copy()
    for worker in range(20):
        replay.add({key: value[worker] for key, value in {**obs, **actions}.items()}, worker)
    dataset = iter(model.dataset(replay.dataset))
    ratio = e.when.Ratio(hp['train_ratio'] / (hp['batch_size'] * hp['batch_length']))
    ratio._prev = 2301 + (actor_steps - 1) / ratio._ratio
    assert ratio._prev == 30617
    driver = e.Driver(env); driver._state = policy_state; driver._acts = actions
    driver.on_step(lambda transition, worker: counter.increment()); driver.on_step(replay.add)
    stats = e.Metrics(); state = None
    logger = e.Logger(counter, [e.logger.JSONLOutput(e.Path(run), 'metrics.jsonl')])

    def update(transition, worker):
        nonlocal state
        for _ in range(ratio(counter)):
            _, state, metrics = model.train(next(dataset), state)
            assert all(np.isfinite(value).all() for key, value in metrics.items()
                       if key.endswith(('_opt_loss', '_opt_grad_norm')))
            stats.add(metrics, prefix='train')
        model.sync()

    driver.on_step(update)
    save_json(run / 'experiment.json', {
        'seed': 3501, 'hyperparameters': hp, 'channel': asdict(channel), 'environment': asdict(task),
        'parent_run': str(parent), 'parent_checkpoint_sha256': parent_proof['checkpoint_sha256'],
        'prefix_sha256': parent_proof['prefix_sha256'], 'external_training_inputs': [],
        'optimizer_states_restored': optimizer_steps, 'restored_model_digest': model_digest(model),
        'reconstructed_total_rng_calls': 1 + actor_steps + policy_calls,
        'sampling_boundary': 'Sampler seed0 restarted, two new asynchronous loaders; training carry reset',
        'policy_boundary': 'Center observation and saved postupdate model used to reconstruct carry',
        'initial_actions_by_worker': first_commands.tolist()})
    try:
        next_log = int(counter) + 1020
        while int(counter) < 51000:
            driver(lambda *values: model.policy(*values, mode='train'), steps=min(100, 51000 - int(counter)))
            if int(counter) >= next_log:
                logger.add(stats.result()); logger.write(); next_log += 1020
                print(f'OWN_STATE_RESUME additional_actions={log.count}', flush=True)
        assert log.count == 20000
        with (run / 'checkpoint.pkl').open('wb') as stream:
            pickle.dump({'agent': model.save(), 'step': counter.save(), 'replay': replay.save()}, stream)
        logger.add(stats.result()); logger.write()
        records = rows(run / 'training_queries.csv')
        save_json(run / 'stage_complete.json', {'real_actions': 20000, 'callback_frames': int(counter),
            'worker_to_log_id': worker_map(records[:20], first_commands), 'external_training_inputs': [],
            'checkpoint_sha256': digest(run / 'checkpoint.pkl')})
    finally:
        log.close(); env.close()


def consolidate(root):
    from reproduction_tools.records import curve
    first, second = root / 'stage1_fresh30k', root / 'stage2_own20k'
    first_meta = json.loads((first / 'experiment.json').read_text())
    fields = list(rows(first / 'training_queries.csv')[0])
    combined = []
    for name, stage, offset in [('fresh30k', first, 0), ('own_checkpoint20k', second, 30000)]:
        mapping = json.loads((stage / 'stage_complete.json').read_text())['worker_to_log_id']
        inverse = {log_id: worker for worker, log_id in enumerate(mapping)}
        for record in rows(stage / 'training_queries.csv'):
            old_id = int(record['env_index']); worker = inverse[old_id]
            step = int(record['steps']) - old_id + worker + offset
            combined.append({**record, 'steps': step, 'env_index': worker, 'stage': name,
                             'stage_steps': record['steps'], 'stage_env_index': old_id})
    combined.sort(key=lambda record: int(record['steps']))
    assert [int(record['steps']) for record in combined] == list(range(1, 50001))
    with (root / 'training_queries.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields + ['stage', 'stage_steps', 'stage_env_index'])
        writer.writeheader(); writer.writerows(combined)
    first_meta.update(real_action_budget=50000, internal_frames=51000, from_checkpoint=None,
        integrated_two_stage=True, external_training_inputs=[],
        own_checkpoint_restored_at_real_action=30000,
        protocol_difference='Own-state sampler/carry reconstruction at30k; not the original uninterrupted protocol')
    save_json(root / 'experiment.json', first_meta)
    shutil.copyfile(first / 'config.yaml', root / 'config.yaml')
    shutil.copyfile(first / 'initialization_proof.json', root / 'initialization_proof.json')
    with (root / 'metrics.jsonl').open('w') as output:
        for name, stage in [('fresh30k', first), ('own_checkpoint20k', second)]:
            for line in (stage / 'metrics.jsonl').read_text().splitlines():
                value = json.loads(line); value['stage'] = name
                output.write(json.dumps(value) + '\n')
    score = curve(root / 'training_queries.csv')['last_5k_mean']
    save_json(root / 'training_complete.json', {'real_actions': 50000, 'status': 'complete',
        'initialization': 'Fresh random model, empty replay', 'stage_action_counts': [30000, 20000],
        'external_training_inputs': [], 'own_checkpoint_restore': True})
    save_json(root / 'result.json', {'last_5k_mean': score, 'scope': 'Fresh two-stage integrated trial',
        'external_historical_weights_or_observations': False,
        'uninterrupted_historical_protocol_reproduced': False})
    print(f'INTEGRATED_COMPLETE actions=50000 last5k_mean={score}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-rate-socket')
    parser.add_argument('--phase', choices=['orchestrate', 'fresh30', 'resume20'], default='orchestrate')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not args.gpu_rate_socket:
        from reproduction_tools import training
        original_plan = training.plan
        def integrated_plan(speed, algorithm, run_dir, python=None, gpu_python=None):
            execution = original_plan(speed, algorithm, run_dir, python, gpu_python)
            execution['command'] = [execution['command'][0], '-u', str(Path(__file__).resolve()),
                '--run-dir', execution['run_dir'], '--gpu-rate-socket', execution['socket']]
            execution['capture_extra_sources'] = [str(Path(__file__).resolve())]
            execution['integrated_two_stage'] = True
            execution['stage_action_counts'] = [30000, 20000]
            execution['external_training_inputs'] = []
            execution['boundary_protocol'] = 'Own fresh30k checkpoint and actual own replay; reset sampler/training carry and reconstruct policy carry'
            return execution
        training.plan = integrated_plan
        training.launch(50, 'dreamer', run, dry_run=args.dry_run)
    elif args.phase == 'fresh30':
        fresh30(run, args.gpu_rate_socket)
    elif args.phase == 'resume20':
        resume20(run, run.parent / 'stage1_fresh30k', args.gpu_rate_socket)
    else:
        for phase, directory in [('fresh30', 'stage1_fresh30k'), ('resume20', 'stage2_own20k')]:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', phase,
                       '--run-dir', str(run / directory), '--gpu-rate-socket', args.gpu_rate_socket]
            print(f'INTEGRATED_START phase={phase}', flush=True)
            with (run / f'{phase}.out.log').open('w') as output, (run / f'{phase}.err.log').open('w') as error:
                subprocess.run(command, stdout=output, stderr=error, check=True)
            print(f'INTEGRATED_DONE phase={phase}', flush=True)
        consolidate(run)


if __name__ == '__main__':
    main()
