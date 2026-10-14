"""Complete both declared creator scheduling controls, retaining all outcomes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / 'docs/loader_cycle_pair_protocol_20261014.json').read_text())
    assert protocol['registered_before_launch'] and len(protocol['trials']) == 2
    previous = json.loads((ROOT / protocol['launch_after_completed_failure']).read_text())
    assert not previous['performance_and_trend_candidate']
    for trial in protocol['trials']:
        if (ROOT / trial['run']).exists():
            raise FileExistsError(trial['run'])
    outcomes = []
    for trial in protocol['trials']:
        command = [sys.executable, '-u', str(ROOT / 'scripts/trace_loader_cycle_50.py'),
            '--run-dir', str(ROOT / trial['run']), '--first-creator', str(trial['first']),
            '--sample-block', str(trial['block'])]
        print('START_FRESH_SCHEDULE ' + trial['run'], flush=True)
        result = subprocess.run(command, cwd=ROOT)
        outcomes.append({'run': trial['run'], 'returncode': result.returncode})
        print(json.dumps(outcomes[-1]), flush=True)
    (ROOT / 'runs/50kmh/loader_cycle_pair_exit_20261014.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    if any(trial['returncode'] for trial in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
