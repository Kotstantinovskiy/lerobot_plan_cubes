#!/usr/bin/env python3
"""Compute action chunks on a GPU machine for inference/run_policy.py --server on the laptop with the arm.

The laptop keeps the robot, the cameras, the target box, the crop and the safety envelope, and sends a frame whenever
it needs a chunk. The joint envelope comes from the checkpoint's statistics, those of the data it trained on, so the
laptop needs neither a checkpoint nor a dataset, and this machine's dataset may have grown since training.

It listens on localhost only, one client at a time; reach it through an SSH tunnel from the laptop:
  ssh -N -L 18765:127.0.0.1:18765 <this machine>
Messages are numpy archives (remote_policy.py), so another user of a shared machine can ask for chunks but not run code.
"""

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time

import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.utils import get_safe_torch_device

from remote_policy import pack, unpack
from run_policy import (
    PROJECT,
    compute_chunk,
    dataset_observation,
    load_config,
    load_policy,
    saved_stats,
    target_box_style,
)
import top_crop

CHUNK_SIZE = 50  # of both smolvla_base and pi05_base; the laptop decides how much of a chunk it runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="Local pretrained_model folder")
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtc", action="store_true", help="Allow run_policy.py --rtc (Real-Time Chunking)")
    parser.add_argument("--rtc-horizon", type=int, default=20, help="As in run_policy.py")
    parser.add_argument("--rtc-weight", type=float, default=10.0, help="As in run_policy.py")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_meta = LeRobotDatasetMetadata(config["repo_id"], root=config["dataset_root"])
    rtc = None
    if args.rtc:
        rtc = RTCConfig(enabled=True, execution_horizon=args.rtc_horizon, max_guidance_weight=args.rtc_weight,
                        prefix_attention_schedule=RTCAttentionSchedule.EXP)
    policy, pre, post = load_policy(args.policy, args.device, CHUNK_SIZE, ds_meta, rtc)
    device = get_safe_torch_device(args.device)
    stats = saved_stats(args.policy) or ds_meta.stats
    info = {
        "type": policy.config.type,
        "policy": args.policy,
        "rtc": args.rtc,
        "target_box": target_box_style(args.policy),
        "top_crop": top_crop.load(args.policy),
        "features": ds_meta.features,
        "fps": ds_meta.fps,
        "low": stats["action"]["min"].tolist(),
        "high": stats["action"]["max"].tolist(),
    }

    def answer(request):
        if "recorded_episode" in request:
            episode, offset_s = int(request["recorded_episode"]), float(request["recorded_offset_s"])
            return pack(**dataset_observation(ds_meta, config["dataset_root"], episode, offset_s))
        task = str(request.pop("task"))
        guidance = {}
        if "inference_delay" in request:
            guidance["inference_delay"] = int(request.pop("inference_delay"))
        if "prev_chunk_left_over" in request:
            guidance["prev_chunk_left_over"] = torch.from_numpy(request.pop("prev_chunk_left_over")).to(device)
        start = time.perf_counter()
        chunk, raw = compute_chunk(policy, pre, post, request, device, task, ds_meta.robot_type, **guidance)
        print(f"Chunk in {time.perf_counter() - start:.2f} s{' (guided)' if guidance else ''}", flush=True)
        return pack(chunk=chunk, raw=raw.float().cpu().numpy())

    with Listener(("127.0.0.1", args.port)) as listener:
        while True:
            print(f"Ready on 127.0.0.1:{args.port}: {args.policy} on {args.device}", flush=True)
            with listener.accept() as connection:
                print("Client connected", flush=True)
                connection.send_bytes(json.dumps(info).encode())
                try:
                    while True:
                        connection.send_bytes(answer(unpack(connection.recv_bytes())))
                except (EOFError, ConnectionError):
                    print("Client left", flush=True)


if __name__ == "__main__":
    main()
