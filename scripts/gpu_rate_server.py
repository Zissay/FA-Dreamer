"""CUDA AFDM-rate service for CPU-only PPO processes.

The process owns CUDA PyTorch and receives complete action batches over a Unix
socket.  It has no policy logic and cannot alter actions or rewards.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

# The server is launched as ``python scripts/gpu_rate_server.py``.  Insert the
# repository root so it can import the local FA-AFDM package without relying
# on the caller's PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    channels: dict[tuple[tuple[str, object], ...], FAAFDMChannel] = {}
    listener = Listener(args.socket, family="AF_UNIX")
    print(f"GPU rate server listening on {args.socket}", flush=True)
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    if request.get("op") == "shutdown":
                        return
                    config = request["config"]
                    key = tuple(sorted(config.items()))
                    channel = channels.get(key)
                    if channel is None:
                        channel = FAAFDMChannel(FAAFDMConfig(**config))
                        channels[key] = channel
                    rates = channel.rate_batch_gpu(
                        np.asarray(request["u"], dtype=np.float64),
                        np.asarray(request["q"], dtype=np.float64),
                        device=args.device,
                    )
                    connection.send(np.asarray(rates, dtype=np.float64))
            except EOFError:
                pass
            except BaseException:
                connection.send({"error": traceback.format_exc(limit=4)})
            finally:
                connection.close()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
