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
import functools
import json
import sys
import time
from collections.abc import Callable
from multiprocessing.connection import Connection, Listener
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.utils import get_safe_torch_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
import top_crop  # noqa: E402
from local_policy import (  # noqa: E402
    DEFAULT_RTC_HORIZON,
    DEFAULT_RTC_WEIGHT,
    compute_chunk,
    dataset_observation,
    load_policy,
    make_rtc_config,
    saved_stats,
    target_box_style,
)
from project_config import PROJECT, load_config  # noqa: E402
from remote_policy import decode_cameras, pack, unpack  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 18765
CHUNK_SIZE = 50  # of both smolvla_base and pi05_base; the laptop decides how much of a chunk it runs
WARM_UP_CHUNKS = 2
WARM_UP_TASK = "warm up"
WARM_UP_RTC_DELAY = 5  # steps
CAMERA_DTYPES = ("image", "video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="Local pretrained_model folder")
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtc", action="store_true", help="Allow run_policy.py --rtc (Real-Time Chunking)")
    parser.add_argument(
        "--rtc-horizon",
        type=int,
        default=DEFAULT_RTC_HORIZON,
        help="As in run_policy.py",
    )
    parser.add_argument(
        "--rtc-weight",
        type=float,
        default=DEFAULT_RTC_WEIGHT,
        help="As in run_policy.py",
    )
    parser.add_argument("--denoise-steps", type=int, help="Instead of the policy's 10: fewer are faster, less exact")
    return parser.parse_args()


def make_blank_frame(features: dict) -> dict:
    """All-zero arrays of the dataset's observation features: uint8 for cameras, float32 for the rest."""
    frame = {}
    for key, feature in features.items():
        if key.startswith("observation."):
            dtype = np.uint8 if feature["dtype"] in CAMERA_DTYPES else np.float32
            frame[key] = np.zeros(feature["shape"], dtype=dtype)
    return frame


def warm_up(compute: Callable, blank_frame: dict) -> torch.Tensor:
    """Blank chunks before the first client: the first ones on a fresh process take 2-3 times as long.

    `compute` is compute_chunk with the policy filled in. Returns the last chunk as the model gave it.
    """
    for _ in range(WARM_UP_CHUNKS):
        _, raw_chunk = compute(frame=blank_frame, task=WARM_UP_TASK)
    return raw_chunk


def warm_up_guidance(compute: Callable, blank_frame: dict, raw_chunk: torch.Tensor) -> None:
    """One blank chunk with RTC's guidance as clients with --rtc ask for it, continuing `raw_chunk`."""
    compute(
        frame=blank_frame,
        task=WARM_UP_TASK,
        inference_delay=WARM_UP_RTC_DELAY,
        prev_chunk_left_over=raw_chunk[WARM_UP_RTC_DELAY:],
    )


def make_client_info(args: argparse.Namespace, policy_type: str, ds_meta: LeRobotDatasetMetadata) -> dict:
    """What a client is told on connecting, as JSON-able values: the policy's settings and the dataset's facts."""
    stats = saved_stats(args.policy) or ds_meta.stats
    return {
        "type": policy_type,
        "policy": args.policy,
        "rtc": args.rtc,
        "target_box": target_box_style(args.policy),
        "top_crop": top_crop.load(args.policy),
        "features": ds_meta.features,
        "fps": ds_meta.fps,
        "low": stats["action"]["min"].tolist(),
        "high": stats["action"]["max"].tolist(),
    }


def answer_recorded_frame(request: dict, ds_meta: LeRobotDatasetMetadata, dataset_root: str) -> bytes:
    """The recorded observation a client's --dry-run asks for, as a message."""
    observation = dataset_observation(
        ds_meta,
        dataset_root,
        int(request["recorded_episode"]),
        float(request["recorded_offset_s"]),
    )
    return pack(**observation)


def pop_guidance(request: dict, device: torch.device) -> dict:
    """RTC's arguments taken out of a request, ready for compute_chunk; empty for a chunk without guidance."""
    guidance = {}
    if "inference_delay" in request:
        guidance["inference_delay"] = int(request.pop("inference_delay"))
    if "prev_chunk_left_over" in request:
        guidance["prev_chunk_left_over"] = torch.from_numpy(request.pop("prev_chunk_left_over")).to(device)
    return guidance


def answer_chunk(request: dict, compute: Callable, device: torch.device) -> bytes:
    """The action chunk for a request's frame, as a message: `chunk` in dataset units, `raw` as the model gave it."""
    task = str(request.pop("task"))
    guidance = pop_guidance(request, device)
    start = time.perf_counter()
    frame = decode_cameras(request)
    chunk, raw_chunk = compute(frame=frame, task=task, **guidance)
    guided_note = " (guided)" if guidance else ""
    print(f"Chunk in {time.perf_counter() - start:.2f} s{guided_note}", flush=True)
    return pack(chunk=chunk, raw=raw_chunk.float().cpu().numpy())


def serve_client(connection: Connection, info: dict, answer: Callable) -> None:
    """Send a new client `info`, then answer its requests until it leaves."""
    print("Client connected", flush=True)
    connection.send_bytes(json.dumps(info).encode())
    try:
        while True:
            connection.send_bytes(answer(unpack(connection.recv_bytes())))
    except (EOFError, ConnectionError):
        print("Client left", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ds_meta = LeRobotDatasetMetadata(config["repo_id"], root=config["dataset_root"])
    rtc = make_rtc_config(args.rtc_horizon, args.rtc_weight) if args.rtc else None
    loaded = load_policy(
        args.policy,
        args.device,
        CHUNK_SIZE,
        ds_meta,
        rtc,
    )
    if args.denoise_steps:
        loaded.policy.config.num_inference_steps = args.denoise_steps
    device = get_safe_torch_device(args.device)
    compute = functools.partial(
        compute_chunk,
        policy=loaded.policy,
        preprocess=loaded.preprocess,
        postprocess=loaded.postprocess,
        device=device,
        robot_type=ds_meta.robot_type,
    )
    blank_frame = make_blank_frame(ds_meta.features)
    raw_chunk = warm_up(compute, blank_frame)
    if args.rtc:
        warm_up_guidance(compute, blank_frame, raw_chunk)
    info = make_client_info(args, loaded.policy.config.type, ds_meta)

    def answer(request: dict) -> bytes:
        if "recorded_episode" in request:
            return answer_recorded_frame(request, ds_meta, config["dataset_root"])
        return answer_chunk(request, compute, device)

    with Listener((HOST, args.port)) as listener:
        while True:
            print(f"Ready on {HOST}:{args.port}: {args.policy} on {args.device}", flush=True)
            with listener.accept() as connection:
                serve_client(connection, info, answer)


if __name__ == "__main__":
    main()
