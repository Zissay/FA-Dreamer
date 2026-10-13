"""Check the preserved CUDA rate server against its own CPU channel implementation."""
import argparse
from dataclasses import asdict
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction_tools.records import case_dir, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=int, choices=(50, 500), required=True)
    parser.add_argument("--algorithm", choices=("dreamer", "ppo"), required=True)
    parser.add_argument("--gpu-python", default=sys.executable)
    args = parser.parse_args()
    if os.name == "nt":
        raise RuntimeError("Run this service check on Linux/WSL")
    folder = case_dir(args.speed)
    source = folder / "source" / args.algorithm
    sys.path.insert(0, str(source))
    from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig
    cfg = FAAFDMConfig(**read_json(folder / "configs/protocol.json")["channel"])
    channel = FAAFDMChannel(cfg)
    positions = np.random.default_rng(1775).uniform(-.05, .05, (20, 4))
    expected = np.array([channel.rate(p[:2], p[2:]) for p in positions])
    with tempfile.TemporaryDirectory(prefix="fa_gpu_check_") as directory:
        socket = Path(directory) / "rate.sock"
        with (Path(directory) / "server.log").open("w+") as log:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
            server = subprocess.Popen([args.gpu_python, str(source / "scripts/gpu_rate_server.py"),
                                       "--socket", str(socket)], cwd=source, env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while not socket.exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    time.sleep(.1)
                with Client(str(socket), family="AF_UNIX") as connection:
                    connection.send({"config": asdict(cfg), "u": positions[:, :2], "q": positions[:, 2:]})
                    if not connection.poll(30):
                        raise TimeoutError("CUDA rate request timed out")
                    actual = connection.recv()
                    if isinstance(actual, dict):
                        raise RuntimeError(actual)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    print(json.dumps({"speed_kmh": args.speed, "algorithm": args.algorithm,
                      "positions": 20, "max_cpu_gpu_difference": float(np.max(np.abs(actual - expected))),
                      "gpu_rate_service_check": "PASS"}))


if __name__ == "__main__":
    main()
