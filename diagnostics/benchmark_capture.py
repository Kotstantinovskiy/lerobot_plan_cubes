#!/usr/bin/env python3
"""Measure cameras, arm reads and PNG writing without sending motor commands.

This is a capture benchmark, not a full recording or a training demonstration.
Temporary images are removed; measured timings are saved to a JSON report.
"""

import argparse
import json
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

import draccus
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.image_writer import AsyncImageWriter
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, so_leader  # noqa: F401
from lerobot.utils.robot_utils import precise_sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from project_config import PROJECT, load_config  # noqa: E402

MS_PER_S = 1000
DELAYED_READ_MS = 200
IMAGE_WRITER_THREADS = 8
RESERVED_OVERHEAD_S = 0.003


class CaptureTimings(NamedTuple):
    frame_count: int
    elapsed_s: float
    work_times_s: list[float]  # each frame's reads, enqueueing and reserved overhead
    stages_ms: dict[str, list[float]]  # the time of each stage on each frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--read-timeout-ms", type=int, default=200)
    args = parser.parse_args()
    if min(args.fps, args.seconds, args.read_timeout_ms) <= 0:
        parser.error("All numeric arguments must be positive")
    if (args.width is None) != (args.height is None):
        parser.error("Specify both width and height or neither")
    if args.width is not None and min(args.width, args.height) <= 0:
        parser.error("Image size must be positive")
    return args


def disconnect_if_connected(camera) -> None:
    if camera.is_connected:
        camera.disconnect()


def connect_devices(stack: ExitStack, robot, leader) -> None:
    for bus in (robot.bus, leader.bus):
        bus.connect()
        stack.callback(bus.disconnect, disable_torque=False)
    for camera in robot.cameras.values():
        stack.callback(disconnect_if_connected, camera)
        camera.connect()


def read_arms(robot, leader, stages_ms: dict[str, list[float]], previous: float) -> float:
    """Read both arms, adding each read's time to `stages_ms`; `previous` and the result are perf_counter times."""
    for name, bus in (("follower_read", robot.bus), ("leader_read", leader.bus)):
        bus.sync_read("Present_Position")
        now = time.perf_counter()
        stages_ms[name].append((now - previous) * MS_PER_S)
        previous = now
    return previous


def read_cameras(
    cameras,
    read_timeout_ms: int,
    stages_ms: dict[str, list[float]],
    previous: float,
) -> tuple[dict, float]:
    """Read every camera, adding each read's time to `stages_ms`; returns the frames and the perf_counter time after."""
    frames = {}
    for name, camera in cameras.items():
        frames[name] = camera.async_read(timeout_ms=read_timeout_ms)
        now = time.perf_counter()
        read_ms = (now - previous) * MS_PER_S
        stages_ms[name].append(read_ms)
        if read_ms > DELAYED_READ_MS:
            print(f"Delayed fresh frame: {name}, {read_ms:.0f} ms", flush=True)
        previous = now
    return frames, previous


def capture_for(robot, leader, writer: AsyncImageWriter, image_dir: Path, args: argparse.Namespace) -> CaptureTimings:
    """Read the arms and cameras and queue the frames as PNGs at args.fps for args.seconds."""
    stages_ms = {name: [] for name in ["follower_read", "leader_read", *robot.cameras, "enqueue"]}
    work_times_s = []
    frame_count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        tick = time.perf_counter()
        previous = read_arms(
            robot,
            leader,
            stages_ms,
            tick,
        )
        frames, previous = read_cameras(
            robot.cameras,
            args.read_timeout_ms,
            stages_ms,
            previous,
        )
        for name, frame in frames.items():
            writer.save_image(frame, image_dir / name / f"{frame_count:06d}.png")
        stages_ms["enqueue"].append((time.perf_counter() - previous) * MS_PER_S)
        # Reserve 3 ms for motor command/record metadata overhead, without moving the arms.
        precise_sleep(RESERVED_OVERHEAD_S)
        work_times_s.append(time.perf_counter() - tick)
        precise_sleep(max(0, 1 / args.fps - (time.perf_counter() - tick)))
        frame_count += 1
    return CaptureTimings(
        frame_count,
        time.perf_counter() - start,
        work_times_s,
        stages_ms,
    )


def work_summary(timings: CaptureTimings, fps: int) -> dict[str, float]:
    """Frame count, achieved frame rate and how each frame's work compares with the budget of 1 / fps seconds."""
    return {
        "frames": timings.frame_count,
        "elapsed_s": timings.elapsed_s,
        "actual_fps": timings.frame_count / timings.elapsed_s,
        "work_p50_ms": float(np.percentile(timings.work_times_s, 50) * MS_PER_S),
        "work_p95_ms": float(np.percentile(timings.work_times_s, 95) * MS_PER_S),
        "over_budget_fraction": float(np.mean(np.array(timings.work_times_s) > 1 / fps)),
    }


def stage_summary(stages_ms: dict[str, list[float]], camera_names: list[str]) -> dict[str, dict]:
    """Mean and largest time of each stage, and how many reads of each camera were delayed."""
    return {
        "stages_mean_ms": {name: float(np.mean(times)) for name, times in stages_ms.items()},
        "stages_max_ms": {name: float(max(times)) for name, times in stages_ms.items()},
        "camera_reads_over_200ms": {
            name: int(np.sum(np.array(stages_ms[name]) > DELAYED_READ_MS)) for name in camera_names
        },
    }


def measure(stack: ExitStack, robot, leader, args: argparse.Namespace) -> dict:
    """Capture to a temporary folder and return the measurements, keeping the report's order of keys."""
    image_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="cube-capture-")))
    for name in robot.cameras:
        (image_dir / name).mkdir()
    writer = AsyncImageWriter(num_processes=0, num_threads=IMAGE_WRITER_THREADS)
    stack.callback(writer.stop)
    timings = capture_for(
        robot,
        leader,
        writer,
        image_dir,
        args,
    )
    pending_images = writer.queue.qsize()
    drain_start = time.perf_counter()
    writer.wait_until_done()
    return {
        **work_summary(timings, args.fps),
        "pending_images": pending_images,
        "drain_s": time.perf_counter() - drain_start,
        "images_written": len(list(image_dir.glob("*/*.png"))),
        **stage_summary(timings.stages_ms, list(robot.cameras)),
    }


def main():
    args = parse_args()
    config = load_config()
    if args.width is not None:
        for camera in config["robot"]["cameras"].values():
            camera.update(width=args.width, height=args.height)
    robot = make_robot_from_config(draccus.decode(RobotConfig, config["robot"]))
    leader = make_teleoperator_from_config(draccus.decode(TeleoperatorConfig, config["teleop"]))
    result = {
        "target_fps": args.fps,
        "camera_configs": config["robot"]["cameras"],
        "display_data": False,
        "motor_commands_sent": False,
        "read_timeout_ms": args.read_timeout_ms,
    }
    with ExitStack() as stack:
        connect_devices(stack, robot, leader)
        measurements = measure(
            stack,
            robot,
            leader,
            args,
        )
        result.update(measurements)
    size = f"{args.width}x{args.height}" if args.width else "configured-named"
    report = PROJECT / "camera_check" / f"benchmark-{size}-{args.fps}fps-{args.seconds}s-{args.read_timeout_ms}ms.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
