# 500 km/h experiment

Paper panel (b), and the 500 km/h group of panel (c).

* Dreamer BX: seed 5215, 50,000 genuine interactions, H=50, 20 environments.
* PPO: seed 5215, 50,000 genuine interactions, same channel and task.
* Channel seed 7, Doppler 12971.93703548369 Hz, 1 mm action increments,
  50-action center-start episodes. Complete configs are under `configs/`.
* Both algorithm sources come from their own pretraining snapshots.
  Selected-source hashes and archive provenance are retained under `source/`
  and `source_snapshot_provenance.json`.

The Dreamer configuration retains 10k stratified-goal prefill, H=50,
observed state-rate learning, real-data replay/imitation parameters,
known kinematic feature rollout, and physical-feature feedback gradients.
These are part of the measured implementation; they are not replaced with
an unmodified default DreamerV3 configuration.

## Entry points (run from the repository root)

```bash
python scripts/reproduce.py plot --speed 500
python scripts/reproduce.py train --speed 500 --algorithm dreamer --run-dir runs/500kmh/dreamer_seed5215 --dry-run
python scripts/reproduce.py train --speed 500 --algorithm ppo --run-dir runs/500kmh/ppo_seed5215 --dry-run
```

Install the separate Dreamer/GPU and PPO/CPU environments in the root README.
Use `--gpu-python` to select the GPU rate interpreter when launching PPO.
Output directories must be fresh; no old checkpoint or replay is loaded.

Expected final-window means (bps/Hz): Dreamer 18.301657504417648;
PPO 15.277229841515. See `expected_results.json` for all independent references.
Grid/random references are not imported into either learner or replay.
New training can vary across hardware/backends; compare its real logs and
statistics rather than overwriting them with the archived paper results.
