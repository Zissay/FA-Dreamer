# Dependency records

* `dreamer500-launch.txt`: exact pip-freeze bytes in the Dreamer BX pretrain snapshot.
* `ppo500-launch.txt`: exact PPO interpreter pip-freeze captured before its seed-5215 run.
* `dreamer50-trainer-launch.txt` and `dreamer50-rate_server-launch.txt`: actual environments captured before the fresh seed-3501 reconstruction.
* `ppo50-trainer-launch.txt` and `ppo50-rate_server-launch.txt`: actual CPU learner and CUDA server environments captured before the fresh seed-3501 PPO run.

The Dreamer/GPU server and PPO CPU learner used separate Python 3.12 environments.
Their Torch and Stable-Baselines3 versions differ; do not combine both full freezes
into one environment. Curated install entry points are one directory above.

The historical 50 km/h launch dependency lock could not be established.
The new 50 km/h case has its own captured launch environments and complete new logs.

CUDA toolkit/driver compatibility is system-dependent. The launcher preserves
the recorded deterministic XLA flags and automatically searches the trainer's
installed `nvidia/cuda_nvcc*` package. Set `XLA_CUDA_DATA_DIR` to a compatible
toolkit root containing `bin/ptxas` and `nvvm/libdevice/libdevice.10.bc` when needed.
