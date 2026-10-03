#!/usr/bin/env python3
"""Exercise the real recorder; arms are only read unless --teleop. Output is diagnostic, not training data.

With --episodes 2 or more the report also gives the pauses between the recorder's loops: the one before each
later episode is the time it took to save the episode before it.
"""

import argparse
import functools
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

from lerobot.scripts import lerobot_record as recorder

from live_video import RECORDER_PATCHES
from project_config import PROJECT, load_config, positive_int
from record import build_command

DIAGNOSTIC_REPO_ID = "local/diagnostic_camera_check"
DIAGNOSTIC_COLOR = "diagnostic"
TELEOP_TASK = "Diagnostic teleoperated capture."
STATIONARY_TASK = "Diagnostic stationary capture. No motor commands."
RESET_TIME_S = 5


class LoopTiming(NamedTuple):
    """One run of the recorder's record_loop, with time.perf_counter() times; `frames` is None for a reset."""

    start: float
    end: float
    frames: int | None


def episode_reports(loops: list[LoopTiming]) -> list[dict[str, float]]:
    """Wall time, frames and frame rate of every loop that recorded an episode."""
    reports = []
    for loop in loops:
        if loop.frames is None:
            continue
        wall_time_s = loop.end - loop.start
        reports.append({"wall_time_s": wall_time_s, "frames": loop.frames, "actual_fps": loop.frames / wall_time_s})
    return reports


def pauses_between(loops: list[LoopTiming]) -> list[float]:
    """Seconds from the end of each loop to the start of the next."""
    return [later.start - earlier.end for earlier, later in zip(loops, loops[1:], strict=False)]


def diagnostic_config(root: Path, seconds: int, task: str) -> dict[str, Any]:
    """The project's config, turned to recording a throwaway dataset in `root`."""
    config = load_config()
    config.update(
        dataset_root=str(root),
        repo_id=DIAGNOSTIC_REPO_ID,
        episode_time_s=seconds,
        reset_time_s=RESET_TIME_S,
        display_data=False,
    )
    config["tasks"] = {DIAGNOSTIC_COLOR: f"{task} Not training data."}
    return config


def disconnect_if_connected(camera) -> None:
    if camera.is_connected:
        camera.disconnect()


def connect_bus(stack: ExitStack, bus) -> None:
    bus.connect()
    stack.callback(bus.disconnect, disable_torque=False)


def connect_read_only_robot(stack: ExitStack, robot) -> None:
    connect_bus(stack, robot.bus)
    for camera in robot.cameras.values():
        stack.callback(disconnect_if_connected, camera)
        camera.connect()


def make_read_only_robot(stack: ExitStack, make_robot, robot_config):
    """Stands in for the recorder's make_robot_from_config: the robot reads its arm and cameras and writes nothing."""
    robot = make_robot(robot_config)
    robot.connect = functools.partial(connect_read_only_robot, stack, robot)
    robot.disconnect = lambda: None  # ExitStack closes camera streams and serial ports.
    robot.send_action = lambda action: action  # Never writes a goal position or enables torque.
    return robot


def make_read_only_leader(stack: ExitStack, make_leader, leader_config):
    """Stands in for the recorder's make_teleoperator_from_config: the leader only reads its arm."""
    leader = make_leader(leader_config)
    leader.connect = functools.partial(connect_bus, stack, leader.bus)
    leader.disconnect = lambda: None
    return leader


def patch_arms_read_only(stack: ExitStack) -> None:
    make_robot = functools.partial(make_read_only_robot, stack, recorder.make_robot_from_config)
    make_leader = functools.partial(make_read_only_leader, stack, recorder.make_teleoperator_from_config)
    stack.enter_context(patch.object(recorder, "make_robot_from_config", make_robot))
    stack.enter_context(patch.object(recorder, "make_teleoperator_from_config", make_leader))


def measured_record_loop(record_loop, loops: list[LoopTiming], /, *args, **kwargs):
    """Stands in for the recorder's record_loop: runs it and appends its timing to `loops`."""
    start = time.perf_counter()
    result = record_loop(*args, **kwargs)
    end = time.perf_counter()
    dataset = kwargs.get("dataset")  # none in the reset between episodes
    frames = None if dataset is None else dataset.episode_buffer["size"]
    loops.append(LoopTiming(start, end, frames))
    return result


def patch_recorder(stack: ExitStack, loops: list[LoopTiming], command: list[str]) -> None:
    """Make the recorder time its loops into `loops` and take the arguments of `command`, as record.py runs it."""
    measured_loop = functools.partial(measured_record_loop, recorder.record_loop, loops)
    stack.enter_context(patch.object(recorder, "record_loop", measured_loop))
    stack.enter_context(patch.multiple(recorder, **RECORDER_PATCHES))  # as record.py runs it
    # Keyboard input is deliberately disabled for this timed diagnostic.
    key_events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}
    stack.enter_context(patch.object(recorder, "init_keyboard_listener", return_value=(None, key_events)))
    recorder_arguments = command[2:]  # after the interpreter and the script
    stack.enter_context(patch.object(sys, "argv", ["verify_recording", *recorder_arguments]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=positive_int, default=60)
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=1,
        help=f"Episodes in a row, with a {RESET_TIME_S} s reset between them",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Move the follower with the leader arm; output still stays in camera_check",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = PROJECT / "camera_check" / f"recording-check-{time.time_ns()}"
    config = diagnostic_config(root, args.seconds, TELEOP_TASK if args.teleop else STATIONARY_TASK)
    command, _ = build_command(
        config,
        [DIAGNOSTIC_COLOR],
        args.episodes,
        resume=False,
    )
    loops = []

    with ExitStack() as stack:
        if not args.teleop:
            patch_arms_read_only(stack)
        patch_recorder(stack, loops, command)
        recorder.main()

    report = {
        "motor_commands_sent": args.teleop,
        "target_fps": config["fps"],
        "dataset": str(root),
        "episodes": episode_reports(loops),
        "pauses_s": pauses_between(loops),
    }
    report_text = json.dumps(report, indent=2)
    (root / "diagnostic-report.json").write_text(report_text + "\n")
    print(report_text, flush=True)


if __name__ == "__main__":
    main()
