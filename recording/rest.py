#!/usr/bin/env python3
"""Fold the follower into the start pose of our recordings from any position.

Joints move in phases so a gripper lying on the table rises before the base turns: shoulder and elbow,
then wrist and gripper, then the base. A joint lagging behind its command pauses the motion until it
catches up. Works with torque on or off at the start; torque is disabled on exit, and the folded pose
rests on its own. Ctrl+C stops the motion and disables torque immediately.
"""

import argparse
import functools
from pathlib import Path
import time

import draccus
import numpy as np

from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from record import load_config

PROJECT = Path(__file__).resolve().parents[1]
PHASES = [("shoulder_lift", "elbow_flex"), ("wrist_flex", "wrist_roll", "gripper"), ("shoulder_pan",)]
MAX_LAG = 15.0  # normalized units between command and position before the motion pauses
CATCH_UP_S = 2.0


def fold_to_rest(bus, rest, fps=20, step=1.0, log=print):
    """Move a connected, torque-enabled follower bus to `rest` ({motor: position}) in phases."""
    target = {m: float(v) for m, v in bus.sync_read("Present_Position").items()}

    def wait_for(command):
        # The gripper may be blocked by a cube it holds; it never stops the motion.
        deadline = time.perf_counter() + CATCH_UP_S
        while True:
            present = bus.sync_read("Present_Position")
            lag = {m: abs(command[m] - present[m]) for m in command if m != "gripper"}
            joint = max(lag, key=lag.get)
            if lag[joint] <= MAX_LAG:
                return
            if time.perf_counter() > deadline:
                raise RuntimeError(f"{joint} stays {lag[joint]:.1f} units from its command")
            time.sleep(1 / fps)

    for phase in PHASES:
        start = dict(target)
        end = {**target, **{m: float(rest[m]) for m in phase}}
        steps = max(1, int(np.ceil(max(abs(end[m] - start[m]) for m in phase) / step)))
        log(f"Moving {', '.join(phase)} over {steps / fps:.1f} s.")
        for k in range(1, steps + 1):
            tick = time.perf_counter()
            command = {m: start[m] + k / steps * (end[m] - start[m]) for m in start}
            bus.sync_write("Goal_Position", command)
            wait_for(command)
            time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
        target = end
    time.sleep(1)
    return bus.sync_read("Present_Position")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--step", type=float, default=1.0, help="Max change per frame per joint (normalized units)")
    parser.add_argument("--dry-run", action="store_true", help="Read positions and print the plan; nothing moves")
    args = parser.parse_args()
    if not 0 < args.step <= 3:
        parser.error("Step must be in (0, 3]")
    config = load_config(args.config)
    rest = config["rest_pose"]
    fields = dict(config["robot"])
    fields.update(cameras={})
    robot = make_robot_from_config(draccus.decode(RobotConfig, fields))
    if set(robot.calibration) != set(robot.bus.motors) or set(rest) != set(robot.bus.motors):
        raise RuntimeError(f"Calibration or rest_pose does not cover every motor: {robot.calibration_fpath}")
    # Torque_Enable writes occasionally get no status packet (seen on ids 2, 4 and 6); rewriting is idempotent.
    robot.bus.enable_torque = functools.partial(robot.bus.enable_torque, num_retry=3)
    robot.bus.disable_torque = functools.partial(robot.bus.disable_torque, num_retry=3)
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; refusing automatic calibration")
        present = robot.bus.sync_read("Present_Position")
        print("Now:  " + ", ".join(f"{m} {v:.1f}" for m, v in present.items()), flush=True)
        print("Rest: " + ", ".join(f"{m} {rest[m]:.1f}" for m in present), flush=True)
        if args.dry_run:
            return
        if not all(robot.bus.sync_read("Torque_Enable").values()):
            robot.bus.sync_write("Goal_Position", present)  # no jump to a stale goal when torque comes on
            robot.configure()  # also enables torque; skipped when it is on, as it briefly releases the joints
        final = fold_to_rest(robot.bus, rest, step=args.step, log=lambda s: print(s, flush=True))
        print("Done: " + ", ".join(f"{m} {v:.1f}" for m, v in final.items()), flush=True)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)
            print("Follower disconnected; torque disabled.", flush=True)


if __name__ == "__main__":
    main()
