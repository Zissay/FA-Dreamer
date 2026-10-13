"""Verify new actions in a historical30k checkpoint continuation; no fresh claim."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.baselines import channel_for
from reproduction_tools.records import curve, read_json, sha256


def audit(run):
    run = Path(run)
    meta = read_json(run / 'experiment.json')
    launch = read_json(run / 'reproduction_launch.json')
    complete = read_json(run / 'training_complete.json')
    assert read_json(run / 'reproduction_exit.json')['returncode'] == 0
    assert complete['additional_real_actions'] == 20000 and complete['lineage_actions'] == 50000
    assert launch['pretraining_capture'] and launch['training_interactions'] == 20000
    assert meta['training_seed'] == 3501 and meta['historical_prefix_actions'] == 30000
    assert not meta['future_original_observations_used'] and not meta['fresh_from_scratch']
    original = read_json(ROOT / 'experiments/50kmh/historical_target/dreamer/experiment.json')
    assert meta['hyperparameters'] == {**original['hyperparameters'], 'data_loaders': 2}
    assert meta['channel'] == original['channel']
    assert meta['parent_checkpoint_sha256'] == 'cf5df546a46e043c09b1c8279f1618f38b67d7ef50d165d172fff8ea66cee29e'
    assert meta['prefix_file_sha256'] == '797b6b5da5ae69739da54b49d19b26b9539f373fe5fa5f5fb198f57a5093079a'
    assert len(meta['optimizer_states_restored']) == 3
    assert set(meta['optimizer_states_restored'].values()) == {7080}
    assert '--xla_gpu_deterministic_ops' not in launch['environment']['XLA_FLAGS']
    assert sha256(run / 'reproduction_launcher.py') == launch['launcher_sha256']
    archive = run / 'pretraining_source.tar.gz'
    assert sha256(archive) == launch['source_archive_sha256']
    with tarfile.open(archive) as source:
        for rel, digest in launch['source_sha256'].items():
            assert hashlib.sha256(source.extractfile(rel).read()).hexdigest() == digest
    recovery = read_json(ROOT / 'experiments/50kmh/historical_target/source/recovery_manifest.json')
    for rel, digest in recovery['recovered_sha256'].items():
        assert launch['source_sha256']['experiments/50kmh/source/dreamer/' + rel] == digest
    log = run / 'training_queries.csv'
    if log.exists():
        stream = log.open('rt', newline='')
    else:
        log = log.with_suffix('.csv.gz')
        stream = gzip.open(log, 'rt', newline='')
    with stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r['steps']))
    assert [int(r['steps']) for r in rows] == list(range(1, 20001))
    physics = channel_for(50)
    initial = physics.rate(np.zeros(2), np.zeros(2))
    last = np.zeros((20, 4)); previous = np.full(20, initial); count = np.zeros(20, int)
    max_pos = 0.; max_reward = 0.; errors = []
    for row in rows:
        worker = int(row['env_index'])
        action = np.array([float(row[f'a{i}']) for i in range(4)])
        position = np.array([float(row[k]) for k in ['u_x', 'u_y', 'q_x', 'q_y']])
        assert np.isfinite([*action, *position, float(row['rate']), float(row['reward'])]).all()
        assert np.max(np.abs(action)) <= 1.000001
        t = count[worker] % 50 + 1; terminal = t == 50
        if t == 1:
            last[worker] = 0; previous[worker] = initial
        assert int(row['episode_step']) == t and bool(int(row['episode_terminal'])) == terminal
        assert row['source'] == 'actor_policy'
        max_pos = max(max_pos, float(np.max(np.abs(position - last[worker] - .001 * action))))
        rate = float(row['rate']); reward = rate - previous[worker] + (.2 * rate if terminal else 0)
        max_reward = max(max_reward, abs(float(row['reward']) - reward))
        if terminal:
            errors.append(abs(physics.rate(position[:2], position[2:]) - rate))
        last[worker] = position; previous[worker] = rate; count[worker] += 1
    assert (count == 1000).all() and len(errors) == 400
    assert max_pos < 1e-7 and max_reward < 1e-10 and max(errors) < 1e-10
    terminals = [r for r in rows if int(r['episode_terminal'])]
    means = [float(np.mean([float(r['rate']) for r in terminals if end - 1000 < int(r['steps']) <= end]))
             for end in range(1000, 20001, 1000)]
    old = curve(ROOT / 'experiments/50kmh/historical_target/dreamer/terminal_interactions.csv')
    raw_hash = hashlib.sha256(gzip.open(log, 'rb').read()).hexdigest() if log.suffix == '.gz' else sha256(log)
    return {'status': 'PASS', 'scope': 'Original30k checkpoint plus20k new actions; not fresh50k',
            'new_real_actions': 20000, 'terminal_records': 400, 'training_seed': 3501,
            'independent_cpu_terminal_checks': 400, 'max_physical_rate_error': max(errors),
            'max_position_transition_error': max_pos, 'max_reward_error': max_reward,
            'final_5k_mean': float(np.mean(means[-5:])), 'historical_final_mean': old['last_5k_mean'],
            'difference': float(np.mean(means[-5:]) - old['last_5k_mean']), 'batch_means_new_20k': means,
            'first_new_terminal_ge17_at_lineage_step': min(
                (int(r['steps']) + 30000 for r in terminals if float(r['rate']) >= 17), default=None),
            'source_archive_sha256': sha256(archive), 'raw_new_log_sha256': raw_hash,
            'original_learning_parameters_preserved': True, 'only_pre30k_historical_observations_used': True,
            'model_actor_critic_optimizer_states_restored': True,
            'fresh_from_scratch': False, 'exact_historical_resume': False}


def plot(report, output):
    from reproduction_tools.plotting import canvas, font, COLORS
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    old = curve(ROOT / 'experiments/50kmh/historical_target/dreamer/terminal_interactions.csv')
    lineage = np.concatenate([old['batch_mean'][:30], report['batch_means_new_20k']])
    shown = np.array([lineage[max(0, i-4):i+1].mean() for i in range(50)])
    fig, ax = canvas('Environment Interaction Steps')
    ax.plot(old['steps'], old['displayed'], color=COLORS[0], lw=2, label='Original online training')
    ax.plot(old['steps'][30:], shown[30:], color=COLORS[2], lw=2, label='30k checkpoint continuation')
    ax.axvline(30000, color='0.5', ls=':', lw=1.4)
    ax.text(.03, .96, 'Diagnostic continuation; not a fresh run', transform=ax.transAxes,
            fontproperties=font(14), va='top')
    ax.set_xlim(0, 50000); ax.set_ylim(4, 20)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: '0' if x == 0 else f'{x/1000:.0f}k'))
    ax.legend(loc='lower right', prop=font(17), framealpha=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font(13))
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Use a separate diagnostic PNG path')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot-output', type=Path, help='Separate diagnostic PNG, not a paper panel')
    args = parser.parse_args()
    report = audit(args.run_dir)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    if args.plot_output:
        plot(report, args.plot_output)
    print(text)
