"""Train Dreamer or PPO for one receiver speed (50,000 real interactions, seed 5215).

Run on Linux/WSL with an NVIDIA GPU, e.g.
  .venv-dreamer/bin/python scripts/train.py --speed 50 --algorithm dreamer --run-dir runs/dreamer_50
  .venv-ppo/bin/python scripts/train.py --speed 50 --algorithm ppo --run-dir runs/ppo_50 \
      --gpu-python .venv-dreamer/bin/python
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def cuda_toolkit(python):
    """Directory with ptxas/libdevice for XLA (XLA_CUDA_DATA_DIR overrides)."""
    value = os.environ.get("XLA_CUDA_DATA_DIR") or subprocess.check_output([python, "-c", (
        "from pathlib import Path; import site; "
        "r=[p for s in site.getsitepackages() for p in sorted(Path(s).glob('nvidia/cuda_nvcc*')) "
        "if (p/'bin/ptxas').is_file() and (p/'nvvm/libdevice/libdevice.10.bc').is_file()]; "
        "print(r[0] if r else '')")], text=True).strip()
    if not value or not (Path(value) / "bin/ptxas").is_file():
        raise RuntimeError("Set XLA_CUDA_DATA_DIR to a CUDA toolkit containing bin/ptxas")
    return value


def dreamer_runtime():
    """Extract the preserved dreamerv3==1.3.0 source used for the paper runs."""
    target = ROOT / "vendor/runtime"
    if not (target / "dreamerv3").is_dir():
        with tarfile.open(ROOT / "vendor/dreamerv3-installed.tar.gz") as archive:
            archive.extractall(target, filter="data")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=int, choices=(50, 500), required=True)
    p.add_argument("--algorithm", choices=("dreamer", "ppo"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=5215)
    p.add_argument("--gpu-python", default=sys.executable,
                   help="Python of the Dreamer/JAX environment, used for the GPU rate server")
    a = p.parse_args()
    run = a.run_dir.resolve()
    if run.exists():
        raise FileExistsError(f"Use a fresh run directory: {run}")
    configs = ROOT / "configs" / f"{a.speed}kmh"
    source = ROOT / a.algorithm
    socket = f"/tmp/fa_dreamer_{a.speed}_{a.algorithm}_{os.getpid()}.sock"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=".55")

    if a.algorithm == "dreamer":
        env["PYTHONPATH"] = os.pathsep.join([str(dreamer_runtime()), env.get("PYTHONPATH", "")])
        env["XLA_FLAGS"] = (f"--xla_gpu_cuda_data_dir={cuda_toolkit(sys.executable)} "
                            "--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops")
        command = [sys.executable, "-u", "dreamer_simple/train.py", "--run-dir", str(run),
                   "--config", str(configs / "dreamer.json"), "--task-config", str(configs / "task.json"),
                   "--queries", "50000", "--seed", str(a.seed), "--num-envs", "20",
                   "--platform", "cuda", "--gpu-rate-socket", socket, "--snapshot-every", "5000"]
    else:
        config = json.loads((configs / "ppo.json").read_text())
        config.update(run_dir=str(run), seed=a.seed, gpu_rate_backend="rpc", gpu_rate_socket=socket)
        command = [sys.executable, "-u", "train_grpo_cuda_miso_ppo.py"]
        for key, value in config.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                command += [flag] if value else []
            elif value is not None:
                command += [flag, str(value)]

    run.parent.mkdir(parents=True, exist_ok=True)
    log = run.parent / f"{run.name}.rate_server.log"
    with log.open("w") as server_log:
        server = subprocess.Popen([a.gpu_python, "scripts/gpu_rate_server.py", "--socket", socket],
                                  cwd=source, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not Path(socket).exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"GPU rate server did not start; see {log}")
                time.sleep(0.1)
            subprocess.run(command, cwd=source, env=env, check=True)
        finally:
            server.terminate()
            server.wait(timeout=10)
            Path(socket).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
