"""Read-only original/fresh checkpoint comparison on fixed historical routes.

The held-out high-rate route is retrospective diagnostic data. It never enters
replay, optimizer updates, a training launch, or the reported training curve.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run', type=Path, required=True)
    parser.add_argument('--fresh-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.modules['tensorflow'] = None
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0, str(preserved_vendor()))
    sys.path.insert(0, str(ROOT / 'experiments/50kmh/source/dreamer'))
    import numpy as np
    import jax
    import jax.numpy as jnp
    from dreamerv3 import train as native, ninjax as nj, jaxutils
    from dreamerv3.embodied.envs.from_gym import FromGym
    from dreamer_simple.agent import Agent
    from dreamer_simple.train import model_config
    from fa_afdm_dreamerv3.dreamer_matched_env import MatchedDreamerEnv, matched_configs, phase_encode
    from fa_afdm_dreamerv3.dreamer_truncated_actor import install_truncated_normal_actor
    hp = json.loads((ROOT / 'experiments/50kmh/configs/dreamer.json').read_text())
    cfg = model_config(hp, 3501, 'cuda')
    channel, task = matched_configs(json.loads((ROOT / 'experiments/50kmh/configs/task.json').read_text()), 3501)
    env = MatchedDreamerEnv(channel, task, potential_model=True); env.reset()
    core = env.core
    initial_vector, initial_rate = phase_encode(core._observation(), channel), core.rate
    with (args.historical_run / 'training_queries.csv').open(newline='') as stream:
        records = list(csv.DictReader(stream))
    vectors, actions, rates = [], [], []
    for start in (28000, 48000):
        for worker in range(20):
            track = sorted([r for r in records if start < int(r['steps']) <= start + 1000
                            and int(r['env_index']) == worker], key=lambda r: int(r['episode_step']))
            assert len(track) == 50
            v, r, a = [initial_vector], [initial_rate], []
            for row in track:
                core.u = np.array([float(row[k]) for k in ['u_x', 'u_y']], np.float32)
                core.q = np.array([float(row[k]) for k in ['q_x', 'q_y']], np.float32)
                core.t = int(row['episode_step'])
                v.append(phase_encode(core._observation(), channel)); r.append(float(row['rate']))
                a.append([float(row[f'a{i}']) for i in range(4)])
            vectors.append(v); rates.append(r); actions.append(a + [[0] * 4])
    vector, action = jnp.asarray(np.stack(vectors, 1)), jnp.asarray(np.stack(actions, 1))
    truth = np.stack(rates, 1)
    install_truncated_normal_actor(stable_sampling=True)
    wrapped = native.wrap_env(FromGym(MatchedDreamerEnv(channel, task, potential_model=True)), cfg)
    model = Agent(wrapped.obs_space, wrapped.act_space, native.embodied.Counter(), cfg)

    def digest():
        h = hashlib.sha256()
        for value in model.save():
            a = np.asarray(value); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()

    def diagnostic(vector, action):
        wm = model.agent.wm
        initial = wm.rssm.initial(vector.shape[1])
        previous = jnp.concatenate([jnp.zeros_like(action[:1]), action[:-1]], 0)
        first = jnp.broadcast_to(jnp.arange(51)[:, None] == 0, (51, vector.shape[1]))
        def roll(prior, inputs):
            observation, control, reset = inputs
            posterior, _ = wm.rssm.obs_step(prior, control, wm.encoder({'vector': observation}), reset)
            return posterior
        posteriors = jaxutils.scan(roll, (vector, previous, first), initial, False)
        trajectory = {**posteriors, 'vector': vector, 'action': action,
                      'cont': (vector[..., 164] < 1).astype(jnp.float32)}
        predicted_rate = wm.heads['reward'](trajectory).mean()
        critic = model.agent.task_behavior.ac.critics['extr']
        short = {key: value[:11] for key, value in trajectory.items()}
        _, returns, _ = critic.score(short)
        return predicted_rate, returns[0], critic.net(trajectory).mean()[10]

    fn = jax.jit(nj.pure(diagnostic))
    paths = [('original30k', args.historical_run / 'milestone_checkpoints/checkpoint_030000.pkl'),
             ('fresh30k', args.fresh_checkpoint)]
    report = {'evaluation_only': True, 'optimizer_updates': 0, 'new_training_actions': 0,
              'held_out_future_routes_used_only_for_diagnosis': True, 'results': {},
              'limitation': 'Retrospective route estimates do not establish a causal explanation or fresh reproduction.'}
    for name, path in paths:
        with path.open('rb') as stream:
            model.load(pickle.load(stream)['agent'])
        before = digest()
        evaluated = [jax.device_get(fn(model.varibs, jax.random.PRNGKey(seed), vector, action)[0])
                     for seed in (91, 92, 93, 94)]
        pred, returned, value = [np.mean([out[index] for out in evaluated], 0) for index in range(3)]
        result = {}
        for label, group in [('plateau_route', slice(0, 20)), ('high_rate_route', slice(20, 40))]:
            result[label] = {'predicted_h10_lambda_return': float(returned[group].mean()),
                            'predicted_value_at10': float(value[group].mean()),
                            'predicted_terminal_rate': float(pred[-1, group].mean()),
                            'actual_terminal_rate': float(truth[-1, group].mean()),
                            'actual_full_return': float((1.2 * truth[-1, group] - truth[0, group]).mean())}
        assert digest() == before
        result['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['checkpoint_variables_unchanged'] = True
        report['results'][name] = result
        print(json.dumps({name: result}), flush=True)
    wrapped.close(); env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
