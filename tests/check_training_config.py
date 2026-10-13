"""Check the real preserved trainer's parsed/resolved config without training.

Run once per speed/algorithm in its own interpreter, as documented in README.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.records import case_dir, read_json
from reproduction_tools.training import plan, preserved_vendor

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=int, choices=(50, 500), required=True)
parser.add_argument("--algorithm", choices=("dreamer", "ppo"), required=True)
args = parser.parse_args()
folder = case_dir(args.speed)
source = folder / "source" / args.algorithm
if args.algorithm == "dreamer":
    sys.path.insert(0, str(preserved_vendor()))
sys.path.insert(0, str(source))
entry = source / ("dreamer_simple/train.py" if args.algorithm == "dreamer" else "train_grpo_cuda_miso_ppo.py")
spec = importlib.util.spec_from_file_location("preserved_training_config", entry)
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)
if args.algorithm == "ppo":
    execution = plan(args.speed, "ppo", ROOT / "runs/config_check")
    sys.argv = execution["command"][2:]
    actual = vars(trainer.parse_args())
    expected = read_json(folder / "configs/ppo.json")
    for key, value in expected.items():
        if key not in ("run_dir", "gpu_rate_socket"):
            assert actual[key] == value, (key, actual[key], value)
else:
    from ruamel.yaml import YAML
    hp = read_json(folder / "configs/dreamer.json")
    protocol = read_json(folder / "configs/protocol.json")
    actual = trainer.model_config(hp, protocol["dreamer_seed"], "cpu")
    expected = YAML(typ="safe").load((folder / "recorded/dreamer/resolved_config.yaml").read_text())
    expected = trainer.e.Config(expected).flat
    actual = actual.flat
    implicit = expected.keys() - actual.keys()
    if implicit:
        assert args.speed == 50 and implicit == {"afdm.imagination_start_mode"}
        assert expected["afdm.imagination_start_mode"] == "uniform"
        assert expected["afdm.imagination_start_count"] == 0
    # The older 50 km/h candidate uses all replay starts; this named mode was added later.
    assert actual.keys() - expected.keys() == set()
    assert expected.keys() - actual.keys() == implicit
    for key, value in expected.items():
        if key != "jax.platform" and key not in implicit:
            assert actual[key] == value, (key, actual[key], value)
print(json.dumps({"speed_kmh": args.speed, "algorithm": args.algorithm, "trainer_config_check": "PASS",
                  "implicit_historical_fields": sorted(implicit) if args.algorithm == "dreamer" else []}))
