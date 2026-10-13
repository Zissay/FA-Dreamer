"""Audit real training logs and plot observed terminal-rate distributions."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig
from matched_curve_reference import validate_ppo, reference_for, draw_reference


def records(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def grouped(episodes, maximum):
    groups = defaultdict(list)
    for row in episodes:
        step = int(row['steps'])
        if step <= maximum:
            groups[step].append(float(row['terminal_rate']))
    x = np.asarray(sorted(groups))
    y = np.asarray([np.mean(groups[s]) for s in x])
    low = np.asarray([min(groups[s]) for s in x])
    high = np.asarray([max(groups[s]) for s in x])
    assert np.all(np.diff(x) > 0)
    return x, y, low, high


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--ppo-dir', type=Path, default=None,
                   help='Optional PPO run for a comparison curve')
    a = p.parse_args()
    meta = json.loads((a.run_dir/'experiment.json').read_text())
    amount = int(meta.get('num_envs', 20))
    if a.ppo_dir is not None:
        validate_ppo(meta, a.ppo_dir)
    reference = reference_for(meta)
    q = sorted(records(a.run_dir/'training_queries.csv'), key=lambda r: int(r['steps']))
    episodes = sorted(records(a.run_dir/'episode_terminal_rates.csv'), key=lambda r: (int(r['steps']), int(r['env_index'])))
    n = len(q)
    assert n == meta['real_action_budget'] and (a.run_dir/'training_complete.json').exists()
    assert [int(r['steps']) for r in q] == list(range(1, n+1))
    coords = np.asarray([[float(r[k]) for k in ('u_x','u_y','q_x','q_y')] for r in q])
    assert np.isfinite(coords).all() and np.max(np.abs(coords)) <= .25000001
    sources = [r['source'] for r in q]
    assert set(sources) <= {'uniform_prefill','actor_policy'}
    first_actor = sources.index('actor_policy')
    assert all(s == 'actor_policy' for s in sources[first_actor:])
    by_env = defaultdict(list)
    for r in q:
        by_env[int(r['env_index'])].append(r)
    assert len(by_env) == amount
    expected_episodes = {}
    for env, rows in by_env.items():
        pos = np.zeros(4)
        total = 0.
        for i, r in enumerate(rows):
            t = i % 50 + 1
            assert int(r['episode_step']) == t
            action = np.asarray([float(r[f'a{k}']) for k in range(4)])
            after = np.asarray([float(r[k]) for k in ('u_x','u_y','q_x','q_y')])
            assert np.max(np.abs(action)) <= 1.000001
            np.testing.assert_allclose(after, np.clip(pos + .001*action, -.25, .25), atol=3e-8)
            total += float(r['reward'])
            assert int(r['episode_terminal']) == int(t == 50)
            if t == 50:
                steps = (i+1) * len(by_env)
                expected_episodes[(steps, env)] = (float(r['rate']), total)
                pos, total = np.zeros(4), 0.
            else:
                pos = after
    assert len(expected_episodes) == len(episodes) == n//50
    for r in episodes:
        expected = expected_episodes[(int(r['steps']), int(r['env_index']))]
        np.testing.assert_allclose([float(r['terminal_rate']), float(r['episode_return'])], expected, atol=1e-9)
        # At most 50 mm displacement is possible here, so boundary costs are zero.
        np.testing.assert_allclose(float(r['episode_return']), 1.2*float(r['terminal_rate'])-float(r['initial_rate']), atol=1e-8)
    channel = FAAFDMChannel(FAAFDMConfig(**meta['channel']))
    rate_errors = []
    for i in np.linspace(0,n-1,32,dtype=int):
        rate_errors.append(abs(channel.rate(coords[i,:2],coords[i,2:])-float(q[i]['rate'])))
    assert max(rate_errors) < 1e-5, rate_errors
    d = grouped(episodes, n)
    ppo = grouped(records(a.ppo_dir/'episode_terminal_rates.csv'), n) if a.ppo_dir else None
    window = min(100, len(episodes)//4)
    rates = np.asarray([float(r['terminal_rate']) for r in episodes])
    start, end = float(rates[:window].mean()), float(rates[-window:].mean())
    block_window=max(1,window//amount)
    peak_window=float(np.max(np.convolve(d[1],np.ones(block_window)/block_window,mode='valid')))
    regression=peak_window-end
    rolling = np.convolve(d[1], np.ones(block_window)/block_window, mode='valid')
    rolling_steps = d[0][block_window-1:]
    late_rolling = rolling[rolling_steps >= n/2]
    late_drawdown = float(np.max(np.maximum.accumulate(late_rolling)-late_rolling))
    # Positive first-to-last difference alone does not establish convergence.
    if regression > 1.:
        trend='improved_then_regressed'
    elif end-start >= 1.:
        trend='observed_improvement_not_convergence_proof'
    else:
        trend='no_clear_improvement'
    summary = dict(real_actions=n, complete_episodes=len(episodes), source_counts=dict(Counter(sources)),
                   num_envs=amount, curve_points=len(d[0]), terminal_record_interval=50*amount,
                   coordinate_max_abs_m=float(np.max(np.abs(coords))),
                   rate_window_episodes=window, first_terminal_mean=start, last_terminal_mean=end,
                   change=end-start, last_terminal_std=float(rates[-window:].std()),
                   last_window_fraction_at_least_15=float(np.mean(rates[-window:] >= 15)),
                   last_batch_terminal_mean=float(d[1][-1]), max_actual_terminal=float(rates.max()),
                   best_window_terminal_mean=peak_window,late_regression_from_best_window=regression,
                   max_second_half_5k_window_drawdown=late_drawdown,
                   min_second_half_5k_window_mean=float(late_rolling.min()),
                   audit='PASS', cpu_gpu_rate_check_count=32, max_cpu_gpu_rate_error=max(rate_errors),
                   frozen_evaluation_included=False,
                   shade='none',
                   single_seed=True, trend=trend)
    summary['exhaustive_reference'] = reference
    summary['ppo_source_run'] = str(a.ppo_dir) if a.ppo_dir else None
    summary['curve_metric'] = 'actual episode terminal rate' if amount == 1 else '20-episode terminal mean'
    if ppo is not None:
        summary['ppo_same_budget_last_window_terminal_mean'] = float(
            np.mean(ppo[1][-max(1, window // 20):]))
        summary['ppo_curve_metric'] = 'episode terminal rate grouped at PPO timestamps'
        summary['ppo_curve_truncated_at_real_actions'] = n
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/'audit.json').write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':14,'pdf.fonttype':42})
    for comparison in ((False, True) if ppo is not None else (False,)):
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        series = [(d, '#2462ce', 'Proposed Alg. 1')]
        if comparison:
            series.append((ppo, '#d93635', 'PPO'))
        for (x,y,low,high), color, label in series:
            ax.plot(x, y, color=color, linewidth=2, label=label)
        if comparison:
            draw_reference(ax, reference)
        ax.set(xlim=(0,n), xlabel='Steps', ylabel='AFDM rate (bps/Hz)',
               title='Training curves' if comparison else 'Dreamer training curve')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '0' if v == 0 else f'{v/1000:g}K'))
        ax.grid(alpha=.2)
        ax.tick_params(direction='in', top=True, right=True)
        if comparison:
            if amount == 1:
                ax.legend(loc='best',framealpha=.95)
            else:
                ax.legend(loc='lower center',bbox_to_anchor=(.58,.05),ncol=1,framealpha=.9)
        fig.tight_layout()
        name = 'dreamer_ppo_training_comparison' if comparison else 'dreamer_training_curve'
        fig.savefig(a.output_dir/f'{name}.pdf', bbox_inches='tight')
        fig.savefig(a.output_dir/f'{name}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
