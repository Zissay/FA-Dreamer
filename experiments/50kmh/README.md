# 50 km/h experiment

Paper panel (a), and the 50 km/h group of panel (c).

* Dreamer: seed 3501, 50,000 genuine interactions, H=10, 20 environments.
* PPO: seed 3004, 50,000 genuine interactions, with the recorded learning
  parameters and absolute 20k entropy schedule.
* Channel seed 7, Doppler 1297.193703548369 Hz, 1 mm action increments,
  50-action center-start episodes. Complete configs are under `configs/`.
* Source: 2026-09-11 portable historical package. Original launch source
  and dependency provenance are incomplete. Two PPO support modules were
  recovered from the retained workspace; see `source/ppo_source_manifest.json`.

## Entry points (run from the repository root)

```bash
python scripts/reproduce.py plot --speed 50
python scripts/reproduce.py train --speed 50 --algorithm dreamer --run-dir runs/50kmh/dreamer_seed3501 --dry-run
python scripts/reproduce.py train --speed 50 --algorithm ppo --run-dir runs/50kmh/ppo_seed3004 --dry-run
```

Remove `--dry-run` only after installing the separate training environments
described in the root README. Both learners use the paper's 50k budget.

Expected final-window means (bps/Hz): Dreamer 18.837550331174892;
PPO 15.441476879350997. See `expected_results.json` for all independent references.
Archived terminal-only CSVs retain 1,000 real terminal actions per learner,
with original interaction indices and positions. `recorded/provenance.json`
records the original complete interaction-file hashes and the prefix extraction rule.

Historical parameters are retained; bitwise reproduction of the original
training trajectory is not established. Exact reconstruction of its recorded
curve and final-window statistics is available without retraining.
