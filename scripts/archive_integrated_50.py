"""Verify and publish a complete fresh30k + own-state20k trial."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.records import read_json, sha256
from scripts.archive_fresh_50_trials import archive, plot


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return sorted(csv.DictReader(stream), key=lambda record: int(record['steps']))


def variable_digest(values):
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_lineage(run):
    run = Path(run).resolve()
    launch = read_json(run / 'reproduction_launch.json')
    first, second = run / 'stage1_fresh30k', run / 'stage2_own20k'
    a, b = read_json(first / 'stage_complete.json'), read_json(second / 'stage_complete.json')
    first_meta, second_meta = read_json(first / 'experiment.json'), read_json(second / 'experiment.json')
    proof = read_json(first / 'initialization_proof.json')
    assert launch['integrated_two_stage'] and launch['external_training_inputs'] == []
    assert proof['fresh_random_initialization'] and not proof['external_checkpoint_restore']
    assert proof['callback_frames'] == 0 and proof['seed'] == 3501
    assert a['real_actions'] == 30000 and b['real_actions'] == 20000
    assert a['callback_frames'] == 30620 and b['callback_frames'] == 51000
    assert a['external_training_inputs'] == b['external_training_inputs'] == second_meta['external_training_inputs'] == []
    assert first_meta['from_checkpoint'] is None
    assert first_meta['hyperparameters'] == second_meta['hyperparameters']
    assert first_meta['channel'] == second_meta['channel']
    assert first_meta['seed'] == second_meta['seed'] == 3501
    assert second_meta['parent_checkpoint_sha256'] == a['checkpoint_sha256'] == sha256(first / 'checkpoint_030000.pkl')
    assert second_meta['prefix_sha256'] == a['prefix_sha256'] == sha256(first / 'prefix_replay.npz')
    assert second_meta['optimizer_states_restored'] == a['optimizer_steps']
    assert len(a['optimizer_steps']) == 3 and set(a['optimizer_steps'].values()) == {7080}
    assert second_meta['restored_model_digest'] == a['model_digest']
    assert Path(second_meta['parent_run']).name == first.name
    # Only this local run's trusted checkpoint is read, after its recorded hash matches.
    with (first / 'checkpoint_030000.pkl').open('rb') as stream:
        checkpoint = pickle.load(stream)
    assert checkpoint['step'] == 30620 and variable_digest(checkpoint['agent']) == a['model_digest']
    with (second / 'checkpoint.pkl').open('rb') as stream:
        final = pickle.load(stream)
    assert final['step'] == 51000
    optimizer_indices = [index for index, value in enumerate(checkpoint['agent'])
                         if np.asarray(value).shape == () and np.asarray(value).dtype.kind in 'iu'
                         and int(value) == 7080]
    # Flattened state also includes Adam/moment/target-update counters. The
    # three named outer optimizer counters were verified separately above.
    assert len(optimizer_indices) >= 3
    assert all(int(final['agent'][index]) == 12175 for index in optimizer_indices)
    own_records = read_rows(first / 'training_queries.csv')
    new_records = read_rows(second / 'training_queries.csv')
    combined = read_rows(run / 'training_queries.csv')
    expected_rows = []
    for name, stage_rows, offset, mapping in [('fresh30k', own_records, 0, a['worker_to_log_id']),
            ('own_checkpoint20k', new_records, 30000, b['worker_to_log_id'])]:
        assert sorted(mapping) == list(range(20))
        inverse = {log_id: worker for worker, log_id in enumerate(mapping)}
        for record in stage_rows:
            local_id = int(record['env_index']); worker = inverse[local_id]
            expected_rows.append({**record, 'steps': str(int(record['steps']) - local_id + worker + offset),
                'env_index': str(worker), 'stage': name, 'stage_steps': record['steps'],
                'stage_env_index': str(local_id)})
    expected_rows.sort(key=lambda record: int(record['steps']))
    assert combined == expected_rows
    assert len(own_records) == 30000 and len(new_records) == 20000
    assert [int(record['steps']) for record in combined] == list(range(1, 50001))
    assert len([record for record in own_records if int(record['episode_terminal'])]) == 600
    assert len([record for record in new_records if int(record['episode_terminal'])]) == 400
    # Prefix is actual own driver data; authenticate each acted transition against
    # own action/rate logs, including next-action alignment within each episode.
    prefix_checks = 0
    with np.load(first / 'prefix_replay.npz') as prefix:
        assert prefix['vector'].shape == (20, 1530, 545)
        assert int(prefix['is_terminal'].sum()) == 600
        for worker, log_id in enumerate(a['worker_to_log_id']):
            track = [record for record in own_records if int(record['env_index']) == log_id]
            acted = np.flatnonzero(~prefix['is_first'][worker].reshape(-1))
            assert len(acted) == len(track) == 1500
            for local, (frame, record) in enumerate(zip(acted, track)):
                assert bool(prefix['is_terminal'][worker, frame]) == bool(int(record['episode_terminal']))
                assert abs(float(np.asarray(prefix['rate_target'][worker, frame]).item()) - float(record['rate'])) < 2e-6
                assert abs(float(np.asarray(prefix['reward'][worker, frame]).item()) - float(record['reward'])) < 2e-6
                next_action = np.zeros(4, np.float32) if int(record['episode_terminal']) else np.array(
                    [float(track[local + 1][f'a{i}']) for i in range(4)], np.float32)
                # Replay contains policy commands before NormalizeAction;
                # CSV contains executed commands after its float32 arithmetic.
                command = np.clip(prefix['action'][worker, frame], -1, 1)
                executed = (command + 1) / 2 * 2 - 1
                np.testing.assert_allclose(executed, next_action, rtol=0, atol=1e-7)
                prefix_checks += 1
    return {'status': 'PASS', 'fresh_random_initialization': proof, 'external_training_inputs': [],
            'first_stage_actions': 30000, 'second_stage_actions': 20000,
            'all50000_stage_action_fields_preserved': True, 'own_checkpoint_hash_verified': True,
            'restored_model_variables_match_own30k': True, 'three_optimizer_steps_at_boundary': 7080,
            'three_optimizer_steps_at_completion': 12175, 'own_prefix_action_alignment_checks': prefix_checks,
            'tracked_integer_update_counter_arrays': len(optimizer_indices),
            'stage1_proof': a, 'stage2_proof': b, 'stage2_metadata': second_meta,
            'original_uninterrupted_protocol': False,
            'limitation': 'Own-state restart changes replay sampling/training carry and reconstructs policy carry at30k.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    protocol_path = ROOT / 'docs/integrated_fresh_protocol_20261014.json'
    protocol = read_json(protocol_path)
    run = args.run_dir.resolve()
    assert protocol['registered_before_launch'] and run == (ROOT / protocol['run_dir']).resolve()
    lineage = check_lineage(run)
    record = archive(run)
    # Correct the general integrity audit's blanket no-restore field: no external
    # checkpoint was used, but this run restored its own30k state internally.
    record['audit'].update(checkpoint_restore_used=True, external_checkpoint_restore_used=False,
        own_checkpoint_restore_used=True, fresh_random_initialization_of_complete_lineage=True)
    original_comparison = record['comparison']
    matched = original_comparison['status'].startswith('REPRODUCED')
    original_comparison['parameter_seed_performance_gate_status'] = original_comparison['status']
    original_comparison['status'] = 'INTEGRATED_FRESH_PERFORMANCE_AND_TREND_MATCHED' if matched else 'NOT_REPRODUCED'
    original_comparison['original_uninterrupted_protocol_reproduced'] = False
    destination = ROOT / record['published_records']
    for label in ['stage1_fresh30k', 'stage2_own20k']:
        stage, target = run / label, destination / label
        target.mkdir(exist_ok=True)
        for name in ['stage_complete.json', 'experiment.json', 'config.yaml', 'initialization_proof.json', 'metrics.jsonl']:
            if (stage / name).exists():
                shutil.copyfile(stage / name, target / name)
        with (stage / 'training_queries.csv').open('rb') as stream, gzip.open(target / 'training_queries.csv.gz', 'wb') as output:
            shutil.copyfileobj(stream, output)
        model_files = list(stage.glob('*.pkl')) + list((stage / 'milestone_checkpoints').glob('*.pkl'))
        save = [{'file': str(path.relative_to(stage)), 'sha256': sha256(path), 'bytes': path.stat().st_size,
                 'weights_published': False} for path in model_files]
        (target / 'local_models.json').write_text(json.dumps(save, indent=2) + '\n')
    for phase in ['fresh30', 'resume20']:
        for stream in ['out', 'err']:
            shutil.copyfile(run / f'{phase}.{stream}.log', destination / f'{phase}.{stream}.txt')
    shutil.copyfile(run / 'initialization_proof.json', destination / 'initialization_proof.json')
    shutil.copyfile(run / 'result.json', destination / 'result.json')
    for name, value in [('authenticity_audit.json', record['audit']), ('historical_comparison.json', original_comparison),
                        ('own_state_lineage_audit.json', lineage)]:
        (destination / name).write_text(json.dumps(value, indent=2) + '\n')
    report = {'protocol': protocol_path.relative_to(ROOT).as_posix(), 'protocol_sha256': sha256(protocol_path),
              'trials': [record], 'all_registered_trials_reported': True, 'fresh_from_scratch_lineage': True,
              'own_state_lineage_audit': lineage, 'performance_and_trend_matched': matched,
              'original_uninterrupted_protocol_reproduced': False, 'paper_replacement_accepted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.plot_output:
        plot([record], args.plot_output)
    manifest = ROOT / 'experiments/MANIFEST.sha256.json'
    manifest.write_text(json.dumps({path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted((ROOT / 'experiments').rglob('*'))
        if path.is_file() and path != manifest and '__pycache__' not in path.parts}, indent=2) + '\n')
    print(json.dumps({'mean': record['comparison']['new_mean'], 'status': record['comparison']['status'],
        'integrity': record['audit']['status'], 'lineage': lineage['status']}))


if __name__ == '__main__':
    main()
