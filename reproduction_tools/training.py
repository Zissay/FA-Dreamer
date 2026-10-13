"""Launch the preserved trainer and capture source/config/dependency provenance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from .records import ROOT, case_dir, read_json, sha256


def arguments(config):
    args = []
    for key, value in config.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    return args


def preserved_vendor():
    archive = ROOT / "vendor/dreamerv3-installed.tar.gz"
    target = ROOT / "vendor/runtime"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "archive.sha256"
    digest = sha256(archive)
    if marker.exists() and marker.read_text().strip() == digest:
        return target
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            path = (target / member.name).resolve()
            if not path.is_relative_to(target.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe vendor archive entry: {member.name}")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.extractfile(member).read())
    marker.write_text(digest + "\n")
    return target


def capture_source_archive(run, manifest):
    """Save source bytes alongside the pretraining hash list before starting processes."""
    archive = run / "pretraining_source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for rel, digest in manifest.items():
            path = ROOT / rel
            if sha256(path) != digest:
                raise RuntimeError(f"Source changed during pretraining capture: {rel}")
            output.add(path, arcname=rel, recursive=False)
        output.add(ROOT / "vendor/dreamerv3-installed.tar.gz", arcname="vendor/dreamerv3-installed.tar.gz", recursive=False)
    return sha256(archive)


def plan(speed, algorithm, run_dir, python=None, gpu_python=None):
    folder = case_dir(speed)
    protocol = read_json(folder / "configs/protocol.json")
    run = Path(run_dir).resolve()
    source = folder / "source" / algorithm
    # PPO owns creation of its output directory; keep launcher provenance in the parent.
    training_run = run / "training" if algorithm == "ppo" else run
    # Keep virtual-environment executable symlinks: resolving them selects system Python.
    trainer_python = str(Path(python).absolute()) if python else sys.executable
    gpu_python = str(Path(gpu_python).absolute()) if gpu_python else trainer_python
    socket = f"/tmp/fa_dreamer_{speed}_{algorithm}_{os.getpid()}.sock"
    if algorithm == "dreamer":
        cli = ["--run-dir", str(run), "--config", str(folder / "configs/dreamer.json"),
               "--queries", "50000", "--seed", str(protocol["dreamer_seed"]),
               "--num-envs", "20", "--platform", "cuda", "--gpu-rate-socket", socket,
               "--snapshot-every", "5000"]
        if speed == 500:
            cli += ["--task-config", str(folder / "configs/task.json")]
        entry = source / "dreamer_simple/train.py"
    else:
        cfg = read_json(folder / "configs/ppo.json")
        cfg.update(run_dir=str(training_run), gpu_rate_backend="rpc", gpu_rate_socket=socket)
        if cfg["timesteps"] != protocol["training_interactions"] or cfg["timesteps"] != 50000:
            raise ValueError("The paper experiments use 50,000 real training interactions")
        if cfg["resume_run"] is not None:
            raise ValueError("The preserved experiments start from empty models")
        cli = arguments(cfg)
        entry = source / "train_grpo_cuda_miso_ppo.py"
    command = [trainer_python, "-u", str(entry), *cli]
    server = [gpu_python, str(source / "scripts/gpu_rate_server.py"), "--socket", socket]
    return {"speed_kmh": speed, "algorithm": algorithm, "run_dir": str(run),
            "source_root": str(source), "command": command, "rate_server": server,
            "socket": socket, "training_run_dir": str(training_run),
            "training_interactions": protocol["training_interactions"]}


def cuda_toolkit(interpreter, environment):
    """Use the original launcher's ptxas/libdevice discovery rule."""
    value = environment.get("XLA_CUDA_DATA_DIR")
    if not value:
        code = (
            "from pathlib import Path; import site; "
            "roots = [p for s in site.getsitepackages() "
            "for p in sorted(Path(s).glob('nvidia/cuda_nvcc*')) "
            "if (p/'bin/ptxas').is_file() and (p/'nvvm/libdevice/libdevice.10.bc').is_file()]; "
            "print(roots[0] if roots else '')"
        )
        value = subprocess.check_output([interpreter, "-c", code], text=True).strip()
    root = Path(value) if value else None
    if not root or not os.access(root / "bin/ptxas", os.X_OK) or not (root / "nvvm/libdevice/libdevice.10.bc").is_file():
        raise RuntimeError("Set XLA_CUDA_DATA_DIR to a CUDA toolkit with bin/ptxas and nvvm/libdevice/libdevice.10.bc")
    return str(root.resolve())


def launch(speed, algorithm, run_dir, python=None, gpu_python=None, dry_run=False):
    execution = plan(speed, algorithm, run_dir, python, gpu_python)
    if dry_run:
        print(json.dumps(execution, indent=2))
        return execution
    if os.name == "nt":
        raise RuntimeError("Run training on Linux/WSL: the preserved GPU rate service uses Unix sockets")
    run = Path(execution["run_dir"])
    if run.exists():
        raise FileExistsError(f"Fresh output directory required: {run}")
    folder = case_dir(speed)
    protocol = read_json(folder / "configs/protocol.json")
    env = os.environ.copy()
    runtime_profile = protocol.get("runtime", {})
    if speed == 50 and algorithm == "dreamer" and runtime_profile.get("pythonhashseed") is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = str(protocol[f"{algorithm}_seed"])
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=".55")
    if algorithm == "dreamer":
        env["XLA_CUDA_DATA_DIR"] = cuda_toolkit(execution["command"][0], env)
        runtime = preserved_vendor()
        env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
        flags = f"--xla_gpu_cuda_data_dir={env['XLA_CUDA_DATA_DIR']} --xla_gpu_enable_triton_gemm=false"
        if runtime_profile.get("xla_gpu_deterministic_ops", True):
            flags += " --xla_gpu_deterministic_ops"
        env["XLA_FLAGS"] = flags
    run.mkdir(parents=True)
    source = Path(execution["source_root"])
    paths = sorted([p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "output" not in p.parts]
                   + list((folder / "configs").glob("*"))
                   + [Path(__file__), ROOT / "reproduction_tools/records.py", ROOT / "scripts/reproduce.py"]
                   + [Path(p) for p in execution.get("capture_extra_sources", [])])
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    captured_configs = run / "reproduction_configs"
    captured_configs.mkdir()
    for path in (folder / "configs").glob("*.json"):
        shutil.copyfile(path, captured_configs / path.name)
    for kind, interpreter in [("trainer", execution["command"][0]), ("rate_server", execution["rate_server"][0])]:
        freeze = subprocess.check_output([interpreter, "-m", "pip", "freeze"], text=True)
        (run / f"pip-freeze-{kind}.txt").write_text(freeze, encoding="utf-8")
    provenance = {**execution, "source_sha256": manifest,
                  "source_archive_sha256": capture_source_archive(run, manifest),
                  "launcher_sha256": sha256(__file__), "vendor_sha256": sha256(ROOT / "vendor/dreamerv3-installed.tar.gz"),
                  "environment": {k: env.get(k) for k in (
                      "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS",
                      "XLA_CUDA_DATA_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                      "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")},
                  "python_versions": {k: subprocess.check_output([p, "--version"], text=True).strip()
                                      for k, p in [("trainer", execution["command"][0]), ("rate_server", execution["rate_server"][0])]},
                  "pretraining_capture": True}
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        provenance["gpu_inventory"] = subprocess.check_output(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as error:
        provenance["gpu_inventory"] = {"unavailable": type(error).__name__}
    (run / "reproduction_launch.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(__file__, run / "reproduction_launcher.py")
    socket = Path(execution["socket"])
    with (run / "rate_server.log").open("w") as server_log:
        server = subprocess.Popen(execution["rate_server"], cwd=source, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while not socket.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"Rate server failed; inspect {run / 'rate_server.log'}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU rate server did not create its socket")
                time.sleep(.1)
            with (run / "console.out.log").open("w") as out, (run / "console.err.log").open("w") as err:
                result = subprocess.run(execution["command"], cwd=source, env=env, stdout=out, stderr=err)
            (run / "reproduction_exit.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
            if result.returncode:
                raise RuntimeError(f"Trainer exited {result.returncode}; inspect {run / 'console.err.log'}")
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            socket.unlink(missing_ok=True)
    return execution
