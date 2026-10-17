# FA-Dreamer

Code to reproduce the training-curve results of the proposed Dreamer-based joint
transmit/receive fluid-antenna position optimization for AFDM, compared with PPO and
exhaustive-search references.

## Layout

```text
dreamer/      Dreamer agent, world model and AFDM environment
ppo/          PPO baseline and AFDM environment
configs/      dreamer.json, ppo.json, task.json (channel and task parameters)
baselines/    Exhaustive-search and random-position reference rates
vendor/       dreamerv3==1.3.0 source used for all runs (extracted automatically)
scripts/      train.py (one run) and plot_figures.py (training-curve figure)
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
.venv-dreamer/bin/python scripts/train.py --algorithm dreamer --run-dir runs/dreamer
.venv-ppo/bin/python scripts/train.py --algorithm ppo --run-dir runs/ppo --gpu-python .venv-dreamer/bin/python
```

A Dreamer run takes about 70 minutes on one RTX 4060; a PPO run takes a few minutes.
If JAX cannot find `ptxas`, set `XLA_CUDA_DATA_DIR` to a CUDA toolkit directory.

## Plot

```bash
.venv-dreamer/bin/python scripts/plot_figures.py --dreamer runs/dreamer --ppo runs/ppo --output figures
```

This writes `figures/training_curve.pdf`. Curves show the mean of the 20 online terminal
rates of each 1K-interaction batch, smoothed by a causal trailing mean over up to five
batches. Final rates are the mean of the last 100 online terminal rates (last 5K
interactions).

## Expected results

Final rates (bps/Hz):

| Proposed | PPO | DAFT-domain exhaustive search | Time-domain exhaustive search | Random position |
|---|---|---|---|---|
| 18.302 | 15.277 | 19.418 | 13.252 | 5.921 |

Dreamer runs use `--xla_gpu_deterministic_ops`; on the same GPU and software versions a
run repeats its trajectory. Other hardware or library versions can change the trajectory,
and with other seeds the agent may converge to a different rate.

## License

`vendor/dreamerv3-installed.tar.gz` contains DreamerV3 (see `vendor/LICENSE.dreamerv3`).
