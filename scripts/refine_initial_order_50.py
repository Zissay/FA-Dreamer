"""Bounded early fingerprint refinement; all temporary weights remain diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol_path = ROOT / 'docs/refine_order_micro_protocol_20261014.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['registered_before_execution'] and protocol['maximum_rounds'] == 4
    parent_path = ROOT / 'docs/neighbor_order_micro_result_20261014.json'
    parent = json.loads(parent_path.read_text())['results'][0]
    assert 'creator_owner_prefix' in parent
    environment = os.environ.copy()
    environment.pop('JAX_DEFAULT_MATMUL_PRECISION',None)
    summaries = []
    for number in range(1,5):
        base_path = ROOT / f'docs/refine_order_round{number}_parent_20261014.json'
        output_path = ROOT / f'docs/refine_order_round{number}_result_20261014.json'
        if base_path.exists() or output_path.exists():
            raise FileExistsError('Refinement artifacts must be fresh')
        base_path.write_text(json.dumps(parent,indent=2)+'\n')
        source = ROOT / 'scripts/infer_initial_batch_order_50.py'
        command = [sys.executable,'-u',str(source),
            '--own-prefix', str(ROOT/'runs/50kmh/dreamer_seed3501_integrated_fresh_20261014/stage1_fresh30k/prefix_replay.npz'),
            '--historical-run','/mnt/e/fa_afdm_dreamerv3/runs/user/t1_dreamer_h10_scheduled_env20_seed3501_50k',
            '--control-runs',str(ROOT/'runs/50kmh/dreamer_seed3501_continuous_trace1_20261014'),
            str(ROOT/'runs/50kmh/dreamer_seed3501_continuous_trace2_20261014'),
            '--neighbor-interleavings','512','--refine-owner-prefix',str(base_path),
            '--diagnostic-schedule-seed',str(20261014+number),'--output',str(output_path)]
        launch = {'round':number,'captured_before_execution_utc':datetime.now(timezone.utc).isoformat(),
            'command':command,'source_sha256':digest(source),'protocol_sha256':digest(protocol_path),
            'parent_path':base_path.relative_to(ROOT).as_posix(),'parent_sha256':digest(base_path),
            'previous_report':parent_path.relative_to(ROOT).as_posix(),'previous_report_sha256':digest(parent_path),
            'new_environment_actions':0,'temporary_weights_used_for_full_training':False}
        (ROOT/f'docs/refine_order_round{number}_launch_20261014.json').write_text(json.dumps(launch,indent=2)+'\n')
        print(f'START_DIAGNOSTIC_ROUND {number}',flush=True)
        with (ROOT/f'runs/refine_order_round{number}_20261014.out.log').open('w') as out, (ROOT/f'runs/refine_order_round{number}_20261014.err.log').open('w') as err:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err,check=True)
        report = json.loads(output_path.read_text())
        assert report['candidate_count'] == 515 and report['neighbor_interleavings'] == 512
        baseline = next(row for row in report['results'] if row['candidate']=='selected_refinement_base')
        assert abs(baseline['historical_action_rmse']-parent['historical_action_rmse']) < 1e-7
        assert abs(baseline['historical_action_max_error']-parent['historical_action_max_error']) < 1e-7
        best = report['results'][0]
        trigger = best['historical_action_rmse']<=0.02 and best['historical_action_max_error']<=0.1
        summaries.append({'round':number,'all515_candidates_reported':True,
            'best_rmse':best['historical_action_rmse'],'best_max_error':best['historical_action_max_error'],
            'fingerprint_trigger_met':trigger,'historical_state_identified':False,
            'report':output_path.relative_to(ROOT).as_posix()})
        (ROOT/'docs/refine_order_progress_20261014.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(json.dumps(summaries[-1]),flush=True)
        if trigger:
            break
        parent, parent_path = best, output_path


if __name__ == '__main__':
    main()
