"""Diagnostic fingerprint search for the original first5 minibatch order.

All optimizer inputs come from this task's own fresh uniform-prefill replay.
Historical early actions are comparison fingerprints only. Temporary diagnostic
weights are never supplied to a benchmark run. No final-rate fitting is done.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return sorted(csv.DictReader(stream), key=lambda row: int(row['steps']))


def worker_order(records):
    import numpy as np
    commands = np.random.RandomState(1003504).uniform(-1., 1., (20, 4)).astype(np.float32)
    mapping = [-1] * 20
    for row in records[:20]:
        command = np.array([float(row[f'a{i}']) for i in range(4)], np.float32)
        difference = abs(commands - command).max(axis=1)
        index = int(difference.argmin())
        assert difference[index] < 1e-7 and mapping[index] == -1
        mapping[index] = int(row['env_index'])
    assert sorted(mapping) == list(range(20))
    return mapping


def action_block(records, mapping, end):
    import numpy as np
    block = {int(row['env_index']): [float(row[f'a{i}']) for i in range(4)]
             for row in records if end - 20 < int(row['steps']) <= end}
    return np.asarray([block[index] for index in mapping], np.float32)


def candidate_batches(initial_worker, lead, burst, available=1040, discard=0):
    import numpy as np
    draws = np.random.default_rng(0).integers(0, available, size=400+discard)[discard:] + 1
    owners = [initial_worker] * lead
    owner = 1 - initial_worker
    while len(owners) < len(draws):
        owners.extend([owner] * burst); owner = 1 - owner
    counts = [0, 0]; starts = np.zeros((5, 16), int)
    for owner, start in zip(owners, draws):
        number = counts[owner]
        if number < 40:
            starts[number // 8, 2 * (number % 8) + owner] = start
        counts[owner] += 1
        # A creator cannot get arbitrarily far ahead of the other's queues.
        if abs(counts[0] - counts[1]) > 33:
            return None
        if min(counts) >= 40:
            assert (starts > 0).all()
            return starts.tolist()
    raise RuntimeError('Incomplete diagnostic interleaving')


def random_candidate_batches(generator, available=1040, discard=0):
    """Finite variable-burst owner schedule, bounded by original queue capacity."""
    import numpy as np
    draws = np.random.default_rng(0).integers(0, available, size=160+discard)[discard:] + 1
    counts = [0, 0]; starts = np.zeros((5, 16), int); owners = []
    owner = int(generator.integers(0, 2)); remaining = int(generator.integers(1, 34))
    for start in draws:
        if counts[owner] - counts[1-owner] >= 33:
            owner = 1-owner; remaining = int(generator.integers(1, 17))
        number = counts[owner]
        if number < 40:
            starts[number // 8, 2 * (number % 8) + owner] = start
        counts[owner] += 1; owners.append(owner); remaining -= 1
        assert abs(counts[0]-counts[1]) <= 33
        if min(counts) >= 40:
            assert (starts > 0).all()
            return starts.tolist(), owners
        if remaining == 0:
            owner = 1-owner; remaining = int(generator.integers(1, 17))
    raise RuntimeError('Incomplete bounded diagnostic interleaving')


def batches_from_owners(owners, available=1040, discard=0):
    import numpy as np
    draws = np.random.default_rng(0).integers(0, available, size=len(owners)+discard)[discard:] + 1
    counts = [0, 0]; starts = np.zeros((5, 16), int)
    for index, (owner, start) in enumerate(zip(owners, draws)):
        number = counts[owner]
        if number < 40:
            starts[number // 8, 2*(number % 8)+owner] = start
        counts[owner] += 1
        if abs(counts[0]-counts[1]) > 33:
            return None
        if min(counts) >= 40:
            return starts.tolist(), list(owners[:index+1])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--own-prefix', type=Path, required=True)
    parser.add_argument('--historical-run', type=Path, required=True)
    parser.add_argument('--control-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls-only', action='store_true')
    parser.add_argument('--allow-numeric-differences', action='store_true', help='Declared arithmetic-runtime comparison, not an order-identification result')
    parser.add_argument('--random-interleavings', type=int, default=0)
    parser.add_argument('--random-only', action='store_true')
    parser.add_argument('--diagnostic-schedule-seed', type=int, default=20261014)
    parser.add_argument('--initial-sequence-count', type=int, choices=[1040, 1041], default=1040)
    parser.add_argument('--neighbor-interleavings', type=int, default=0)
    parser.add_argument('--neighbor-owner-prefixes', type=Path)
    parser.add_argument('--refine-owner-prefix', type=Path)
    parser.add_argument('--fixed-owner-prefix', type=Path)
    parser.add_argument('--sampler-discard-draws',type=int,choices=[0,16,32,64,128,144,256,512],default=0)
    args = parser.parse_args()
    assert 0 <= args.random_interleavings <= 512
    assert not args.random_only or args.random_interleavings
    assert 0 <= args.neighbor_interleavings <= 512
    assert not (args.neighbor_interleavings and args.random_interleavings)
    assert not args.neighbor_interleavings or args.neighbor_owner_prefixes or args.refine_owner_prefix
    sys.modules['tensorflow'] = None
    from reproduction_tools.training import preserved_vendor
    sys.path.insert(0, str(preserved_vendor()))
    sys.path.insert(0, str(ROOT / 'experiments/50kmh/source/dreamer'))
    import numpy as np
    import jax
    from dreamer_simple.train import Agent, model_config, native, e, FromGym, MatchedDreamerEnv, matched_configs, install_truncated_normal_actor
    from dreamerv3 import jaxutils
    hp = json.loads((ROOT / 'experiments/50kmh/configs/dreamer.json').read_text())
    cfg = model_config(hp, 3501, 'cuda')
    channel, task = matched_configs(json.loads((ROOT / 'experiments/50kmh/configs/task.json').read_text()), 3501)
    install_truncated_normal_actor(stable_sampling=True)
    env = native.wrap_env(FromGym(MatchedDreamerEnv(channel, task, potential_model=True)), cfg)
    model = Agent(env.obs_space, env.act_space, e.Counter(), cfg)
    initial = model.save()
    with np.load(args.own_prefix) as stored:
        prefix = {key: stored[key] for key in stored.files}
    assert prefix['vector'].shape == (20, 1530, 545)
    obs = lambda frame: {key: value[:, frame] for key, value in prefix.items() if key not in ('action', 'reset', 'id')}
    old_records = read_rows(args.historical_run / 'training_queries.csv')
    old_map = worker_order(old_records)
    target = action_block(old_records, old_map, 2300)
    initial_target = action_block(old_records, old_map, 2280)
    candidates = []
    for run in args.control_runs:
        with gzip.open(run / 'consumed_replay_batches.jsonl.gz', 'rt') as stream:
            batches = [json.loads(next(stream))['sequence_start_frames'] for _ in range(5)]
        assert max(max(batch) for batch in batches) <= 1040
        actual = read_rows(run / 'training_queries.csv')
        candidates.append((f'actual_trace:{run.name}', batches, action_block(actual, worker_order(actual), 2300)))
    seen = set()
    for owner in (0, 1):
        if args.controls_only or args.random_only or args.neighbor_interleavings:
            break
        for lead in (0, 1, 2, 4, 8, 16, 24, 32, 33):
            for burst in (1, 2, 4, 8, 16, 32):
                batches = candidate_batches(owner, lead, burst, args.initial_sequence_count,args.sampler_discard_draws)
                if batches is None:
                    continue
                signature = tuple(sum(batches, []))
                if signature not in seen:
                    candidates.append((f'owner{owner}_lead{lead}_burst{burst}', batches, None)); seen.add(signature)

    candidate_owners = {}
    candidate_bases = {}
    if args.fixed_owner_prefix:
        selected=json.loads(args.fixed_owner_prefix.read_text())
        fixed=batches_from_owners(selected['creator_owner_prefix'],args.initial_sequence_count,args.sampler_discard_draws)
        assert fixed is not None
        if not args.sampler_discard_draws:
            assert fixed[0]==selected['sequence_start_frames']
        candidates.append(('fixed_selected_prefix',fixed[0],None))
        candidate_owners['fixed_selected_prefix']=fixed[1]
    if args.random_interleavings:
        assert not args.controls_only
        baseline = candidate_batches(0, 32, 16, args.initial_sequence_count,args.sampler_discard_draws)
        candidates.append(('prior_best_owner0_lead32_burst16', baseline, None))
        generator = np.random.default_rng(args.diagnostic_schedule_seed)
        generated = 0
        while generated < args.random_interleavings:
            batches, owners = random_candidate_batches(generator, args.initial_sequence_count,args.sampler_discard_draws)
            signature = tuple(sum(batches, []))
            if signature in seen:
                continue
            seen.add(signature)
            label = f'variable_interleaving_{generated:04d}'
            candidates.append((label, batches, None)); candidate_owners[label] = owners
            generated += 1

    if args.neighbor_interleavings:
        assert not args.controls_only
        if args.refine_owner_prefix:
            selected = json.loads(args.refine_owner_prefix.read_text())
            bases = [{'run': selected['candidate'], 'feasible_creator_owner_prefix': selected['creator_owner_prefix']}]
            baseline = batches_from_owners(selected['creator_owner_prefix'],args.initial_sequence_count,args.sampler_discard_draws)
            assert baseline is not None and baseline[0] == selected['sequence_start_frames']
            candidates.append(('selected_refinement_base',baseline[0],None))
            candidate_owners['selected_refinement_base']=baseline[1]
        else:
            recorded = json.loads(args.neighbor_owner_prefixes.read_text())
            bases = [x for x in recorded['trials'] if 'continuous_trace' in x['run']]
            assert len(bases) == 2
            candidates.append(('prior_best_owner0_lead32_burst16', candidate_batches(0,32,16,args.initial_sequence_count,args.sampler_discard_draws), None))
        generator = np.random.default_rng(args.diagnostic_schedule_seed)
        generated = 0; attempts = 0
        seen.update(tuple(sum(x[1], [])) for x in candidates)
        while generated < args.neighbor_interleavings:
            attempts += 1
            assert attempts < 100000, 'Finite local candidate generation exhausted'
            base = bases[int(generator.integers(len(bases)))]
            owners = list(base['feasible_creator_owner_prefix'])
            for _ in range(int(generator.integers(1,17))):
                positions = [i for i in range(len(owners)-1) if owners[i] != owners[i+1]]
                index = positions[int(generator.integers(len(positions)))]
                owners[index], owners[index+1] = owners[index+1], owners[index]
            candidate = batches_from_owners(owners,args.initial_sequence_count,args.sampler_discard_draws)
            if candidate is None:
                continue
            batches, owners = candidate
            signature = tuple(sum(batches, []))
            if signature in seen:
                continue
            seen.add(signature)
            label = f'neighbor_interleaving_{generated:04d}'
            candidates.append((label,batches,None)); candidate_owners[label]=owners
            candidate_bases[label]=base['run']; generated += 1

    def evaluate(starts):
        model.load(initial); model.rng = jaxutils.RNG(3501); model.rng.next()
        first_actions, policy_state = model.policy(obs(115), None, mode='train')
        normalized = (np.clip(first_actions['action'], -1, 1) + 1) / 2 * 2 - 1
        if not args.allow_numeric_differences:
            np.testing.assert_allclose(normalized, initial_target, rtol=0, atol=1e-7)
        train_state = None
        for indices in starts:
            samples = []
            for start in indices:
                worker, frame = (start - 1) % 20, (start - 1) // 20
                sample = {key: value[worker, frame:frame + 64].copy() for key, value in prefix.items()}
                sample['is_first'][0] = True
                sample['id'] = np.zeros((64, 16), np.uint8)
                samples.append(sample)
            data = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
            _, train_state, _ = model.train(data, train_state)
        action, _ = model.policy(obs(116), policy_state, mode='train')
        return (np.clip(action['action'], -1, 1) + 1) / 2 * 2 - 1

    results = []
    for label, starts, known_actions in candidates:
        action = evaluate(starts)
        result = {'candidate': label, 'historical_action_rmse': float(np.sqrt(np.mean((action - target) ** 2))),
                  'historical_action_max_error': float(abs(action - target).max()), 'sequence_start_frames': starts}
        if label in candidate_owners:
            result['creator_owner_prefix'] = candidate_owners[label]
        if label in candidate_bases:
            result['current_trace_neighbor_base'] = candidate_bases[label]
        if known_actions is not None:
            result['known_trace_action_max_error'] = float(abs(action - known_actions).max())
            if not args.allow_numeric_differences:
                assert result['known_trace_action_max_error'] < 1e-5, result
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != 'sequence_start_frames'}), flush=True)
    results.sort(key=lambda result: result['historical_action_rmse'])
    report = {'diagnostic_only': True, 'training_seed': 3501, 'updates_per_candidate': 5,
        'matmul_precision': jax.config.jax_default_matmul_precision,
        'numeric_runtime_comparison': args.allow_numeric_differences,
        'candidate_count': len(results), 'new_environment_actions': 0,
        'random_interleavings': args.random_interleavings,
        'neighbor_interleavings': args.neighbor_interleavings,
        'candidate_initial_sequence_count': args.initial_sequence_count,
        'candidate_sampler_discard_draws': args.sampler_discard_draws,
        'diagnostic_schedule_seed': args.diagnostic_schedule_seed if args.random_interleavings or args.neighbor_interleavings else None,
        'diagnostic_seed_is_not_model_or_uniform_sampler_seed': True,
        'optimizer_inputs': 'Own fresh uniform-prefill replay; no historical observations or checkpoint weights',
        'historical_actions_used_as_comparison_fingerprint': True,
        'temporary_weights_used_for_full_training': False,
        'historical_first5_order_identified': results[0]['historical_action_max_error'] < 1e-5,
        'fingerprint_consistency_is_not_unique_historical_state_identification': True,
        'results': results, 'limitation': 'Finite block-interleaving family; a near match is not proof of unique original order or full50k reproduction.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    env.close()


if __name__ == '__main__':
    main()
