# FA-Dreamer

Code to reproduce the training curves and final-rate comparison of the proposed
Dreamer-based fluid-antenna position optimization for AFDM (receiver speeds 50 and
500 km/h), against PPO and exhaustive-search references.

## Layout

```text
dreamer/      Dreamer agent, world model and AFDM environment (shared by both speeds)
ppo/          PPO baseline and AFDM environment
configs/      50kmh/ and 500kmh/: dreamer.json, ppo.json, task.json (channel and task)
baselines/    Exhaustive-search and random-position reference rates for each speed
vendor/       dreamerv3==1.3.0 source used for all runs (extracted automatically)
scripts/      train.py (one run) and plot_figures.py (paper figures)
```

## Setup

Linux or WSL2, Python 3.12, an NVIDIA GPU with CUDA 12.

```bash
python3.12 -m venv .venv-dreamer && .venv-dreamer/bin/pip install -r requirements/dreamer.txt
python3.12 -m venv .venv-ppo && .venv-ppo/bin/pip install -r requirements/ppo.txt
```

The PPO network runs on CPU; both methods evaluate the AFDM rate on the GPU through a
rate server started from the Dreamer environment.

## Train

Each run uses 50,000 real environment interactions, 20 parallel environments and
training seed 5215, and needs a new run directory.

```bash
.venv-dreamer/bin/python scripts/train.py --speed 50  --algorithm dreamer --run-dir runs/dreamer_50
.venv-dreamer/bin/python scripts/train.py --speed 500 --algorithm dreamer --run-dir runs/dreamer_500
.venv-ppo/bin/python scripts/train.py --speed 50  --algorithm ppo --run-dir runs/ppo_50  --gpu-python .venv-dreamer/bin/python
.venv-ppo/bin/python scripts/train.py --speed 500 --algorithm ppo --run-dir runs/ppo_500 --gpu-python .venv-dreamer/bin/python
```

A Dreamer run takes about 70 minutes on one RTX 4060; a PPO run takes about 10 minutes.
If JAX cannot find `ptxas`, set `XLA_CUDA_DATA_DIR` to a CUDA toolkit directory.

## Plot

```bash
.venv-dreamer/bin/python scripts/plot_figures.py --dreamer-50 runs/dreamer_50 --ppo-50 runs/ppo_50 \
    --dreamer-500 runs/dreamer_500 --ppo-500 runs/ppo_500 --output figures
```

This writes `training_curve_50.pdf`, `training_curve_500.pdf` and
`rate_bar_50kmh_vs_500kmh.pdf`. Curves show the mean of the 20 online terminal rates of
each 1K-interaction batch, smoothed by a causal trailing mean over up to five batches.
Final rates are the mean of the last 100 online terminal rates (last 5K interactions).

## Settings and expected results

The two speeds share all code. They differ in the Doppler frequency (`task.json`,
`ppo.json`) and in two Dreamer settings (`dreamer.json`):

| Dreamer setting | 50 km/h | 500 km/h |
|---|---|---|
| `train_ratio` | 192 | 128 |
| late actor learning-rate decay (`schedule.late_lr_decay_start/end`, optimizer updates) | 5000-5600 | 7700-8300 |

Final rates obtained with these settings (bps/Hz):

| Speed | Proposed | PPO | DAFT-domain exhaustive search | Time-domain exhaustive search | Random position |
|---|---|---|---|---|---|
| 50 km/h | 18.242 | 15.284 | 19.416 | 13.258 | 5.920 |
| 500 km/h | 18.302 | 15.277 | 19.418 | 13.252 | 5.921 |

Dreamer runs use `--xla_gpu_deterministic_ops`; on the same GPU and software versions a
run repeats its trajectory. Other hardware or library versions can change the trajectory,
and with other seeds the agent may converge to a different rate.

## License

`vendor/dreamerv3-installed.tar.gz` contains DreamerV3 (see `vendor/LICENSE.dreamerv3`).
