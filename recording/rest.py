#!/usr/bin/env python3
"""Fold the follower into the start pose of our recordings from any position.

Joints move in phases so a gripper lying on the table rises before the base turns: shoulder and elbow,
then wrist and gripper, then the base. A joint lagging behind its command pauses the motion until it
catches up. Works with torque on or off at the start; torque is disabled on exit, and the folded pose
rests on its own. Ctrl+C stops the motion and disables torque immediately. With --dry-run, or if the calibration does
not match, the arm is only read: its torque stays as it was, so an arm held up by it does not drop.
"""

import argparse
import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import draccus
import numpy as np
from lerobot.motors.motors_bus import SerialMotorsBus
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401

from project_config import CONFIG_PATH, load_config

PHASES = [("shoulder_lift", "elbow_flex"), ("wrist_flex", "wrist_roll", "gripper"), ("shoulder_pan",)]
DEFAULT_FPS = 20
DEFAULT_STEP = 1.0
MAX_LAG = 15.0  # normalized units between command and position before the motion pauses
CATCH_UP_S = 2.0
SETTLE_S = 1  # left to the arm after the last phase, before the position it ended at is read
MAX_STEP = 3  # the largest --step accepted
TORQUE_WRITE_RETRIES = 3


def phase_step_count(start: dict[str, float], end: dict[str, float], phase: tuple[str, ...], step: float) -> int:
    """How many frames a phase takes when none of its motors moves more than `step` per frame; at least one."""
    largest_move = max(abs(end[motor] - start[motor]) for motor in phase)
    return max(1, int(np.ceil(largest_move / step)))


def command_between(start: dict[str, float], end: dict[str, float], fraction: float) -> dict[str, float]:
    """Interpolate every motor between two poses: `fraction` 0 gives `start`, 1 gives `end`."""
    return {motor: start[motor] + fraction * (end[motor] - start[motor]) for motor in start}


def largest_lag(command: dict[str, float], present: dict[str, float]) -> tuple[str, float]:
    """Return the joint furthest from its command and that distance, leaving the gripper out."""
    # The gripper may be blocked by a cube it holds; it never stops the motion.
    lags = {motor: abs(command[motor] - present[motor]) for motor in command if motor != "gripper"}
    joint = max(lags, key=lags.get)
    return joint, lags[joint]


def wait_until_caught_up(bus: SerialMotorsBus, command: dict[str, float], fps: int) -> None:
    """Return once every joint is within MAX_LAG of `command`; raise RuntimeError if that takes over CATCH_UP_S."""
    deadline = time.perf_counter() + CATCH_UP_S
    while True:
        present = bus.sync_read("Present_Position")
        joint, lag = largest_lag(command, present)
        if lag <= MAX_LAG:
            return
        if time.perf_counter() > deadline:
            raise RuntimeError(f"{joint} stays {lag:.1f} units from its command")
        time.sleep(1 / fps)


def fold_to_rest(
    bus: SerialMotorsBus,
    rest: dict[str, float],
    fps: int = DEFAULT_FPS,
    step: float = DEFAULT_STEP,
    log: Callable[[str], Any] = print,
) -> dict[str, float]:
    """Move a connected, torque-enabled follower bus to `rest` ({motor: position}) in phases.

    `step` is the largest change per frame per joint, in normalized units. Returns the position the arm ended at;
    raises RuntimeError when a joint does not catch up with its command.
    """
    start = {motor: float(position) for motor, position in bus.sync_read("Present_Position").items()}
    for phase in PHASES:
        end = {**start, **{motor: float(rest[motor]) for motor in phase}}
        step_count = phase_step_count(
            start,
            end,
            phase,
            step,
        )
        log(f"Moving {', '.join(phase)} over {step_count / fps:.1f} s.")
        for step_number in range(1, step_count + 1):
            tick = time.perf_counter()
            command = command_between(start, end, step_number / step_count)
            bus.sync_write("Goal_Position", command)
            wait_until_caught_up(bus, command, fps)
            time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
        start = end
    time.sleep(SETTLE_S)
    return bus.sync_read("Present_Position")


def format_pose(pose: dict[str, float]) -> str:
    return ", ".join(f"{motor} {position:.1f}" for motor, position in pose.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_STEP,
        help="Max change per frame per joint (normalized units)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read positions and print the plan; nothing moves and torque stays as it is",
    )
    args = parser.parse_args()
    if not 0 < args.step <= MAX_STEP:
        parser.error(f"Step must be in (0, {MAX_STEP}]")
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    rest = config["rest_pose"]
    fields = dict(config["robot"])
    fields.update(cameras={})
    robot = make_robot_from_config(draccus.decode(RobotConfig, fields))
    if set(robot.calibration) != set(robot.bus.motors) or set(rest) != set(robot.bus.motors):
        raise RuntimeError(f"Calibration or rest_pose does not cover every motor: {robot.calibration_fpath}")
    # Torque_Enable writes occasionally get no status packet (seen on ids 2, 4 and 6); rewriting is idempotent.
    robot.bus.enable_torque = functools.partial(robot.bus.enable_torque, num_retry=TORQUE_WRITE_RETRIES)
    robot.bus.disable_torque = functools.partial(robot.bus.disable_torque, num_retry=TORQUE_WRITE_RETRIES)
    has_taken_over = False
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; refusing automatic calibration")
        present = robot.bus.sync_read("Present_Position")
        print("Now:  " + format_pose(present), flush=True)
        print("Rest: " + format_pose({motor: rest[motor] for motor in present}), flush=True)
        if args.dry_run:
            return
        has_taken_over = True
        if not all(robot.bus.sync_read("Torque_Enable").values()):
            robot.bus.sync_write("Goal_Position", present)  # no jump to a stale goal when torque comes on
            robot.configure()  # also enables torque; skipped when it is on, as it briefly releases the joints
        final = fold_to_rest(
            robot.bus,
            rest,
            step=args.step,
            log=lambda message: print(message, flush=True),
        )
        print("Done: " + format_pose(final), flush=True)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=has_taken_over)
            torque_state = "torque disabled" if has_taken_over else "torque left as it was"
            print(f"Follower disconnected; {torque_state}.", flush=True)


if __name__ == "__main__":
    main()
