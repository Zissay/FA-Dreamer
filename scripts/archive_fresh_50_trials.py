"""Archive every declared fresh trial, including failures, without replacing paper data."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.authenticity import audit_run
from reproduction_tools.comparison import compare_historical_50
from reproduction_tools.records import curve, read_json, sha256


def archive(run):
    run = Path(run).resolve()
    if read_json(run / 'training_complete.json')['real_actions'] != 50000:
        raise ValueError('A complete fresh50k trial is required')
    audit = audit_run(50, 'dreamer', run)
    comparison = compare_historical_50(run)
    dest = ROOT / 'experiments/50kmh/reproduction_attempts' / run.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['experiment.json', 'config.yaml', 'metrics.jsonl', 'reproduction_launch.json',
                 'reproduction_launcher.py', 'reproduction_exit.json', 'pretraining_source.tar.gz',
                 'pip-freeze-trainer.txt', 'pip-freeze-rate_server.txt', 'training_complete.json']:
        shutil.copyfile(run / name, dest / name)
    for name in ['console.out.log', 'console.err.log']:
        shutil.copyfile(run / name, dest / name.replace('.log', '.txt'))
    shutil.copytree(run / 'reproduction_configs', dest / 'reproduction_configs', dirs_exist_ok=True)
    with (run / 'training_queries.csv').open('rb') as source, gzip.open(dest / 'training_queries.csv.gz', 'wb') as output:
        shutil.copyfileobj(source, output)
    with (run / 'training_queries.csv').open(newline='') as source:
        reader = csv.DictReader(source)
        with (dest / 'terminal_interactions.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, reader.fieldnames); writer.writeheader()
            for row in reader:
                if int(row['episode_terminal']):
                    writer.writerow(row)
    checkpoints = [{'filename': str(p.relative_to(run)), 'sha256': sha256(p), 'bytes': p.stat().st_size,
                    'weights_published': False}
                   for p in sorted((run / 'milestone_checkpoints').glob('*.pkl'))]
    for name, data in [('authenticity_audit.json', audit), ('historical_comparison.json', comparison),
                       ('local_checkpoints.json', checkpoints)]:
        (dest / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return {'run': run.name, 'published_records': dest.relative_to(ROOT).as_posix(),
            'audit': audit, 'comparison': comparison}


def plot(records, output):
    from reproduction_tools.plotting import canvas, font, COLORS
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Use a separate diagnostic PNG path')
    original = curve(ROOT / 'experiments/50kmh/historical_target/dreamer/terminal_interactions.csv')
    fig, ax = canvas('Environment Interaction Steps')
    ax.plot(original['steps'], original['displayed'], lw=2, color=COLORS[0], label='Original seed3501')
    for index, record in enumerate(records):
        values = curve(ROOT / record['published_records'] / 'terminal_interactions.csv')
        ax.plot(values['steps'], values['displayed'], lw=2, color=COLORS[index + 1],
                label=f'Fresh repeat{index + 1}, seed3501')
    ax.set_xlim(0, 50000); ax.set_ylim(4, 20)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: '0' if x == 0 else f'{x / 1000:.0f}k'))
    ax.legend(loc='lower right', prop=font(15), framealpha=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font(13))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--protocol', type=Path)
    source.add_argument('--recorded-report', type=Path, help='Replot published archives without models/GPU')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    if args.recorded_report:
        if not args.plot_output:
            parser.error('--recorded-report requires --plot-output')
        recorded = read_json(args.recorded_report)
        assert recorded['all_registered_trials_reported']
        plot(recorded['trials'], args.plot_output)
        sys.exit(0)
    if not args.output:
        parser.error('--protocol requires --output')
    protocol = read_json(args.protocol)
    assert protocol['registered_before_launch']
    paths = protocol['run_directories']
    assert len(paths) == protocol['additional_complete_trials']
    # Require all registered trials: no post-hoc best-run selection.
    records = [archive(ROOT / path) for path in paths]
    scores = [r['comparison']['new_mean'] for r in records]
    report = {'protocol': args.protocol.as_posix(), 'protocol_sha256': sha256(args.protocol), 'trials': records,
              'all_registered_trials_reported': True, 'fresh_from_scratch': True,
              'all_training_seeds': [r['audit']['seed'] for r in records],
              'mean_across_repeats': float(np.mean(scores)), 'std_across_repeats': float(np.std(scores)),
              'near_target_trials': sum(r['comparison']['status'].startswith('REPRODUCED') for r in records),
              'paper_replacement_accepted': False, 'ppo_common_seed_comparison_reproduced': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / 'experiments').rglob('*'))
        if p.is_file() and p != manifest and '__pycache__' not in p.parts}, indent=2) + '\n', encoding='utf-8')
    if args.plot_output:
        plot(records, args.plot_output)
    print(json.dumps({'scores': scores, 'near_target_trials': report['near_target_trials'],
                      'all_registered_trials_reported': True}, indent=2))
