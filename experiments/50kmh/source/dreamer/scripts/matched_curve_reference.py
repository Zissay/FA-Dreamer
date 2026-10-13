"""Plot-only comparison validation. Never imported by the learning agent."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'output/pdf/exhaustive_matched_center_grid21.json'
PPO = ROOT / 'runs/user/t1_ppo_fixed_center_full_seed3004_500000q'


def validate_ppo(meta, ppo_dir=PPO):
    ppo = json.loads((ppo_dir / 'config.json').read_text())
    channel = meta['channel']
    # The common environment constructs unspecified channel fields from defaults.
    from fa_afdm_dreamerv3.channel import FAAFDMConfig
    from dataclasses import asdict
    fields = asdict(FAAFDMConfig())
    candidate = {k: ppo.get(k, v) for k, v in fields.items()}
    candidate['seed'] = ppo['channel_seed']
    assert candidate == channel, 'PPO and Dreamer channel mismatch'
    env = meta['environment']
    assert ppo['fixed_center_start'] and not env['random_reset']
    assert ppo['train_start_pool'] is None and env['reset_position_pool'] is None
    assert ppo['episode_steps'] == env['max_steps'] == 50
    for key in ('action_step', 'terminal_rate_weight', 'reward_mode',
                'dense_rate_coefficient', 'target_bonus', 'direct_position_action',
                'continuing_task', 'finite_horizon_terminal', 'boundary_penalty'):
        assert ppo[key] == env[key], f'Environment mismatch: {key}'


def reference_for(meta, path=REFERENCE):
    if not path.exists():
        return None
    ref = json.loads(path.read_text())
    env = meta['environment']
    radius = env['max_steps'] * env['action_step']
    assert ref['status'] == 'complete'
    assert ref['method'] == 'exact_exhaustive_cartesian_grid'
    assert ref['channel'] == meta['channel'], 'Exhaustive reference channel mismatch'
    assert not env['random_reset'] and env['reset_position_pool'] is None
    assert ref['start_position_m'] == [0.] * 4
    assert ref['episode_steps'] == env['max_steps']
    assert ref['action_step_m'] == env['action_step']
    assert ref['reachable_bounds_m'] == [-radius, radius]
    assert ref['query_count'] == ref['grid_points_per_coordinate'] ** 4
    assert ref['max_cpu_gpu_error'] < 1e-5 and not ref['training_data_used']
    return ref


def draw_reference(ax, ref):
    if ref is not None:
        ax.axhline(ref['best_rate_bps_per_hz'], color='#333333', linestyle='--',
                   linewidth=1.7, label='Exhaustive search (grid)')
