"""Small checks for temporal leakage and experiment/source confusion."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.records import curve, case_dir, load_results, read_json
from reproduction_tools.baselines import channel_for
from reproduction_tools.training import plan
from reproduction_tools.verify import verify_all


class ReproductionChecks(unittest.TestCase):
    def test_published_records_and_sources(self):
        summary = verify_all()
        self.assertEqual(set(summary), {"50", "500"})

    def test_future_rates_cannot_change_earlier_display(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "terminal.csv"
            def write(late):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["steps", "rate", "episode_terminal", "u_x", "u_y", "q_x", "q_y"])
                    for batch in range(50):
                        for env in range(20):
                            writer.writerow([batch * 1000 + 981 + env,
                                             1000 if late and batch >= 25 else batch,
                                             1, 0, 0, 0, 0])
            write(False)
            original = curve(path)
            write(True)
            altered = curve(path)
            np.testing.assert_array_equal(original["displayed"][:25], altered["displayed"][:25])
            self.assertNotEqual(original["displayed"][-1], altered["displayed"][-1])

    def test_speed_selects_its_own_channel(self):
        low, high = channel_for(50), channel_for(500)
        self.assertAlmostEqual(high.cfg.fixed_doppler_hz / low.cfg.fixed_doppler_hz, 10)

    def test_all_paper_training_budgets_are_50k(self):
        for speed in (50, 500):
            for algorithm in ("dreamer", "ppo"):
                execution = plan(speed, algorithm, ROOT / "runs/test")
                command = execution["command"]
                flag = "--queries" if algorithm == "dreamer" else "--timesteps"
                self.assertEqual(command[command.index(flag) + 1], "50000")
                self.assertEqual(execution["training_interactions"], 50000)
                if algorithm == "ppo":
                    self.assertEqual(command[command.index("--ent-coef-anneal-steps") + 1], "20000")

    def test_new_log_loader_checks_channel_before_plotting(self):
        for speed in (50, 500):
            folder = case_dir(speed)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dreamer, ppo = root / "dreamer", root / "ppo"
                dreamer.mkdir()
                ppo.mkdir()
                shutil.copyfile(folder / "recorded/dreamer/experiment.json", dreamer / "experiment.json")
                shutil.copyfile(folder / "recorded/dreamer/terminal_interactions.csv", dreamer / "training_queries.csv")
                shutil.copyfile(folder / "recorded/ppo/config.json", ppo / "config.json")
                shutil.copyfile(folder / "recorded/ppo/terminal_interactions.csv", ppo / "training_interaction_rates.csv")
                result = load_results(speed, dreamer, ppo)
                self.assertAlmostEqual(result["dreamer"]["last_5k_mean"], read_json(folder / "expected_results.json")["dreamer"])
                config = read_json(ppo / "config.json")
                config["fixed_doppler_hz"] *= 2
                (ppo / "config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "channel"):
                    load_results(speed, dreamer, ppo)


if __name__ == "__main__":
    unittest.main()
