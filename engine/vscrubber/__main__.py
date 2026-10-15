# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import os
import socket
import sys


def main():
    ap = argparse.ArgumentParser(prog="vscrubber", description="Video Scrubber local engine")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--data-dir")
    args = ap.parse_args()
    if args.data_dir:
        os.environ["VS_DATA_DIR"] = args.data_dir
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Bind first so port 0 can pick a free port with no race; the app prints the ready line once serving.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", args.port))
    os.environ["VS_PORT"] = str(sock.getsockname()[1])

    import uvicorn

    from .server import app

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    server.run(sockets=[sock])


if __name__ == "__main__":
    sys.exit(main())
