"""One entry point for the original30k + reconstructed20k training workflow.

The first stage is an existing historical run. This does not turn checkpoint
continuation into a new, independent fresh50k experiment.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.records import curve, read_json, sha256
from scripts.audit_second_rise_50 import audit

CHECKPOINT_SHA = 'cf5df546a46e043c09b1c8279f1618f38b67d7ef50d165d172fff8ea66cee29e'
PREFIX_SHA = '797b6b5da5ae69739da54b49d19b26b9539f373fe5fa5f5fb198f57a5093079a'


def read_rows(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix('.csv.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='', encoding='utf-8') as stream:
        return sorted(csv.DictReader(stream), key=lambda row: int(row['steps']))


def prefix_rows(parent):
    original = read_json(parent / 'experiment.json')
    expected = read_json(ROOT / 'experiments/50kmh/historical_target/dreamer/experiment.json')
    if original != expected:
        raise ValueError('Original stage metadata differs from the preserved historical experiment')
    # Authenticate the source file. Post30k bytes are hashed only; they are never
    # inserted into the restored learner or the exported first-stage prefix.
    expected_hash = read_json(ROOT / 'docs/historical_50kmh_physical_check.json')['dreamer']['raw_source_sha256']
    if sha256(parent / 'training_queries.csv') != expected_hash:
        raise ValueError('Original source log differs from the physically audited historical file')
    rows = [row for row in read_rows(parent / 'training_queries.csv') if int(row['steps']) <= 30000]
    if [int(row['steps']) for row in rows] != list(range(1, 30001)):
        raise ValueError('Original stage must contain30000 distinct real interactions')
    historical = [row for row in read_rows(ROOT / 'experiments/50kmh/historical_target/dreamer/terminal_interactions.csv')
                  if int(row['steps']) <= 30000]
    terminals = [row for row in rows if int(row['episode_terminal'])]
    if len(terminals) != 600 or terminals != historical:
        raise ValueError('Original stage terminals do not match the preserved historical record')
    return rows


def plan(parent, prefix, run):
    parent, prefix, run = (Path(path).resolve() for path in (parent, prefix, run))
    checkpoint = parent / 'milestone_checkpoints/checkpoint_030000.pkl'
    if sha256(checkpoint) != CHECKPOINT_SHA or sha256(prefix) != PREFIX_SHA:
        raise ValueError('Original30k checkpoint or replay-prefix hash mismatch')
    meta = read_json(parent / 'experiment.json')
    hp = read_json(ROOT / 'experiments/50kmh/configs/dreamer.json')
    if meta['seed'] != 3501 or meta['from_checkpoint'] is not None:
        raise ValueError('Original stage must originate from fresh seed3501 training')
    if hp != {**meta['hyperparameters'], 'data_loaders': 2}:
        raise ValueError('Recovered learning parameters differ from the original stage')
    rows = prefix_rows(parent)
    command = [sys.executable, '-u', str(ROOT / 'scripts/diagnose_second_rise_50.py'),
               '--parent-run', str(parent), '--prefix-file', str(prefix), '--run-dir', str(run)]
    return {'scope': 'Original30k + reconstructed20k; checkpoint-dependent recovery',
            'fresh_from_scratch': False, 'exact_historical_resume': False,
            'training_seed': 3501, 'speed_kmh': 50, 'hyperparameters': hp, 'channel': meta['channel'],
            'parent_run': str(parent), 'checkpoint_sha256': CHECKPOINT_SHA,
            'prefix_file': str(prefix), 'prefix_file_sha256': PREFIX_SHA,
            'existing_stage_real_actions': len(rows), 'new_stage_action_budget': 20000,
            'lineage_action_budget': 50000, 'resume_command': command,
            'run_dir': str(run), 'inputs_include_post30k_original_observations': False,
            'restored': ['world_model', 'actor', 'critic', 'three_optimizer_states'],
            'reconstructed': ['pre30k_replay', 'policy_rng_call_count', 'schedule_counter'],
            'unavailable_historical_state': ['asynchronous_loader_queues', 'sampler_order', 'preupdate_policy_carry'],
            'limitation': 'Historical first stage is reused. Replay sampler and training carry restart; policy carry is reconstructed. No guarantee of the same result on another execution.'}


def plot_stages(path, output):
    from reproduction_tools.plotting import canvas, font, COLORS
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    values = curve(path)
    fig, ax = canvas('Environment Interaction Steps')
    x, y = values['steps'], values['displayed']
    ax.plot(x[:30], y[:30], color=COLORS[0], lw=2, label='Original first30k')
    ax.plot(x[29:], y[29:], color=COLORS[2], lw=2, label='Restored +20k')
    ax.axvline(30000, color='0.5', ls=':', lw=1.4)
    ax.text(.03, .96, 'Checkpoint recovery; two recorded stages', transform=ax.transAxes,
            fontproperties=font(14), va='top')
    ax.set_xlim(0, 50000); ax.set_ylim(4, 20)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: '0' if value == 0 else f'{value / 1000:.0f}k'))
    ax.legend(loc='lower right', prop=font(16), framealpha=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font(13))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def assemble(parent, continuation, output, execution_plan=None, make_plot=False):
    parent, continuation, output = (Path(path).resolve() for path in (parent, continuation, output))
    if output.exists():
        raise FileExistsError(f'Fresh consolidated output directory required: {output}')
    # Verify all new actions/terminal physics and source snapshots before exporting.
    report = audit(continuation)
    old_rows = prefix_rows(parent)
    new_rows = read_rows(continuation / 'training_queries.csv')
    meta = read_json(continuation / 'experiment.json')
    if sha256(parent / 'milestone_checkpoints/checkpoint_030000.pkl') != meta['parent_checkpoint_sha256']:
        raise ValueError('Continuation checkpoint does not belong to this parent')
    fields = list(old_rows[0])
    if any(set(row) != set(fields) for row in new_rows):
        raise ValueError('Stage logs must retain matching action/rate fields')
    output.mkdir(parents=True)
    prefix_text = io.StringIO(newline='')
    writer = csv.DictWriter(prefix_text, fields); writer.writeheader(); writer.writerows(old_rows)
    encoded_prefix = prefix_text.getvalue().encode('utf-8')
    with gzip.open(output / 'original_first30k.csv.gz', 'wb') as stream:
        stream.write(encoded_prefix)
    combined = []
    for stage, offset, rows in [('original_first30k', 0, old_rows), ('restored_next20k', 30000, new_rows)]:
        for row in rows:
            combined.append({**row, 'steps': str(int(row['steps']) + offset),
                             'stage': stage, 'stage_steps': row['steps'],
                             'stage_env_index': row['env_index']})
    assert [int(row['steps']) for row in combined] == list(range(1, 50001))
    terminals = [row for row in combined if int(row['episode_terminal'])]
    with gzip.open(output / 'lineage_interactions.csv.gz', 'wt', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(combined[0])); writer.writeheader(); writer.writerows(combined)
    terminal_path = output / 'lineage_terminal_interactions.csv'
    with terminal_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(combined[0])); writer.writeheader(); writer.writerows(terminals)
    values = curve(terminal_path)
    with (output / 'lineage_curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['steps', 'terminal_batch_mean', 'causal_trailing5_mean', 'stage'])
        for step, raw, shown in zip(values['steps'], values['batch_mean'], values['displayed']):
            writer.writerow([int(step), float(raw), float(shown), 'original_first30k' if step <= 30000 else 'restored_next20k'])
    if abs(values['last_5k_mean'] - report['final_5k_mean']) > 1e-12:
        raise ValueError('Consolidation changed the final5k metric')
    manifest = {'scope': 'Two-stage lineage export; original30k plus reconstructed20k',
                'fresh_from_scratch': False, 'exact_historical_resume': False,
                'original_stage_actions': 30000, 'new_stage_actions': 20000, 'total_lineage_actions': 50000,
                'training_seed': 3501, 'hyperparameters': meta['hyperparameters'], 'channel': meta['channel'],
                'parent_run': str(parent), 'continuation_run': str(continuation),
                'parent_checkpoint_sha256': meta['parent_checkpoint_sha256'], 'prefix_file_sha256': meta['prefix_file_sha256'],
                'first30k_export_raw_sha256': hashlib.sha256(encoded_prefix).hexdigest(),
                'original600_terminals_match_historical_record': True,
                'workflow_source_sha256': sha256(Path(__file__)),
                'new20k_audit': report, 'final_5k_mean': values['last_5k_mean'],
                'stage_boundary_real_actions': 30000,
                'environment_indices': 'Stage-local IDs retained; not assumed to identify the same worker across stages',
                'causal_display': '1k terminal batches with trailing five-batch average across the labeled stage boundary',
                'paper_replacement_accepted': False,
                'limitation': 'Uses the historical first30k. Loader order and policy carry were reconstructed; does not establish a fresh50k reproduction.'}
    if execution_plan:
        manifest['execution_plan'] = execution_plan
    if make_plot:
        plot_stages(terminal_path, output / 'lineage_curve.png')
    manifest['output_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'lineage_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('plan', 'resume'):
        child = sub.add_parser(name, help='Validate historical inputs' if name == 'plan' else 'Resume20k then audit and consolidate')
        child.add_argument('--parent-run', type=Path, required=True)
        child.add_argument('--prefix-file', type=Path, required=True)
        child.add_argument('--run-dir', type=Path, required=True)
        if name == 'resume':
            child.add_argument('--plot', action='store_true')
    child = sub.add_parser('assemble', help='Audit and consolidate already completed stages without training')
    child.add_argument('--parent-run', type=Path, required=True)
    child.add_argument('--continuation-run', type=Path, required=True)
    child.add_argument('--output-dir', type=Path, required=True)
    child.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.action == 'assemble':
        result = assemble(args.parent_run, args.continuation_run, args.output_dir, make_plot=args.plot)
    else:
        execution = plan(args.parent_run, args.prefix_file, args.run_dir)
        if args.action == 'plan':
            result = execution
        else:
            if args.run_dir.exists():
                raise FileExistsError('Resume requires a fresh output directory')
            subprocess.run(execution['resume_command'], check=True)
            result = assemble(args.parent_run, args.run_dir, args.run_dir / 'integrated', execution, args.plot)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
