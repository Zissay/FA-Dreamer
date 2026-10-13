# Dependency records

* `dreamer500-launch.txt`: exact pip-freeze bytes in the Dreamer BX pretrain snapshot.
* `ppo500-launch.txt`: exact PPO interpreter pip-freeze captured before its seed-5215 run.

The Dreamer/GPU server and PPO CPU learner used separate Python 3.12 environments.
Their Torch and Stable-Baselines3 versions differ; do not combine both full freezes
into one environment. Curated install entry points are one directory above.

No original launch dependency freeze exists for the 50 km/h experiments.
The 500 km/h environments are provided as compatible rerun environments,
not represented as evidence of the original 50 km/h package versions.

CUDA toolkit/driver compatibility is system-dependent. The launcher preserves
the recorded deterministic XLA flags and automatically searches the trainer's
installed `nvidia/cuda_nvcc*` package. Set `XLA_CUDA_DATA_DIR` to a compatible
toolkit root containing `bin/ptxas` and `nvvm/libdevice/libdevice.10.bc` when needed.
