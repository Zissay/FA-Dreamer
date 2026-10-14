"""Complete every preregistered original-parameter fresh traced trial."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol_path = ROOT / 'docs/continuous_trace_pair_protocol_20261014.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['registered_before_launch'] and protocol['additional_complete_trials'] == 2
    for relative in protocol['run_directories']:
        run = ROOT / relative
        if run.exists():
            raise FileExistsError(f'All registered trials require fresh directories: {run}')
    outcomes = []
    for relative in protocol['run_directories']:
        command = [sys.executable, '-u', str(ROOT / 'scripts/trace_fresh_50.py'), '--run-dir', str(ROOT / relative)]
        print(f'START_FRESH_TRACE {relative}', flush=True)
        result = subprocess.run(command, cwd=ROOT)
        outcomes.append({'run': relative, 'returncode': result.returncode})
        print(json.dumps(outcomes[-1]), flush=True)
    (ROOT / 'runs/50kmh/continuous_trace_pair_exit_20261014.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    if any(result['returncode'] for result in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
