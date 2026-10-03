#!/usr/bin/env python3
"""Replay a local episode on the follower with a gradual move to its start."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import pandas as pd
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401

from follower_bus import connect_free_bus
from project_config import CONFIG_PATH, load_config

APPROACH_S = 4  # how long the move to the episode's start takes at full speed
MAX_RELATIVE_TARGET = 10.0
MAX_TRACKING_ERROR = 15  # normalized units between a joint and its next target before the replay stops
PROGRESS_PERIOD_S = 10


def slow_trajectory(actions: np.ndarray, speed: float) -> np.ndarray:
    """Keep the command rate fixed while interpolating a slower trajectory.

    `actions` has shape (frames, joints) and `speed` is in (0, 1]; the result has shape (commands, joints).
    Raises ValueError for any other speed.
    """
    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("Speed must be greater than 0 and no greater than 1")
    last_index = len(actions) - 1
    command_count = int(np.ceil(last_index / speed)) + 1
    positions = np.minimum(np.arange(command_count) * speed, last_index)
    left = positions.astype(int)
    right = np.minimum(left + 1, last_index)
    weights = (positions - left)[:, None]
    return actions[left] * (1 - weights) + actions[right] * weights


def read_episode_frames(root: Path, episode: int) -> pd.DataFrame:
    """Read an episode's rows from the dataset's parquet files, in frame order; ValueError if it has none."""
    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    frames = pd.concat([pd.read_parquet(path) for path in parquet_paths])
    frames = frames[frames.episode_index == episode].sort_values("frame_index")
    if frames.empty:
        raise ValueError("Episode not found")
    return frames


def recorded_actions(frames: pd.DataFrame, action_names: list[str]) -> np.ndarray:
    """Return the actions of an episode's rows, shape (frames, joints); ValueError if malformed or not finite."""
    actions = np.stack(frames.action).astype(float)
    has_expected_shape = actions.shape == (len(frames), len(action_names))
    if not has_expected_shape or not np.isfinite(actions).all():
        raise ValueError("Invalid recorded actions")
    return actions


def build_follower(config: dict[str, Any]):
    fields = dict(config["robot"])
    fields.update(cameras={}, max_relative_target=MAX_RELATIVE_TARGET)
    return make_robot_from_config(draccus.decode(RobotConfig, fields))


def check_follower(robot, action_names: list[str]) -> None:
    """Raise unless the robot has the recorded joints and a calibration file for every motor."""
    if set(action_names) != set(robot.action_features):
        raise ValueError("Recorded joints do not match robot")
    if set(robot.calibration) != set(robot.bus.motors):
        raise RuntimeError(f"Missing or incomplete follower calibration: {robot.calibration_fpath}")


def take_over_follower(robot, action_names: list[str]) -> np.ndarray:
    """Enable torque on a free follower where it stands; returns that position in the order of the actions."""
    current = robot.bus.sync_read("Present_Position")
    initial = np.array([current[name.removesuffix(".pos")] for name in action_names])
    # Set current targets while torque is off, avoiding a jump to a stale goal on enable.
    robot.bus.sync_write("Goal_Position", current)
    robot.configure()
    return initial


def send_target(robot, action_names: list[str], target: np.ndarray) -> None:
    """Send one command of joint positions; RuntimeError if a joint is too far from it."""
    observation = robot.get_observation()
    errors = {name: abs(float(target[j]) - observation[name]) for j, name in enumerate(action_names)}
    joint = max(errors, key=errors.get)
    error = errors[joint]
    if error > MAX_TRACKING_ERROR:
        raise RuntimeError(f"{joint} is too far from next target ({error:.1f} normalized units)")
    robot.send_action(dict(zip(action_names, map(float, target), strict=False)))


def sleep_out_frame(tick: float, fps: int) -> None:
    """Sleep for what is left of a frame that began at `tick` (time.perf_counter)."""
    time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))


def move_to_start(
    robot,
    action_names: list[str],
    initial: np.ndarray,
    first_action: np.ndarray,
    *,
    approach_s: float,
    fps: int,
) -> None:
    print(f"Moving gradually to episode start ({approach_s:g} seconds).", flush=True)
    for weight in np.linspace(0, 1, int(approach_s * fps) + 1):
        tick = time.perf_counter()
        send_target(robot, action_names, initial + weight * (first_action - initial))
        sleep_out_frame(tick, fps)


def replay_actions(robot, action_names: list[str], actions: np.ndarray, *, fps: int, episode: int) -> None:
    print(f"Replaying episode {episode}: {len(actions)} frames at {fps} fps.", flush=True)
    start = time.perf_counter()
    for index, action in enumerate(actions):
        tick = time.perf_counter()
        send_target(robot, action_names, action)
        sleep_out_frame(tick, fps)
        if (index + 1) % (PROGRESS_PERIOD_S * fps) == 0:
            print(f"Replayed {index + 1}/{len(actions)} frames.", flush=True)
    print(f"Replay completed in {time.perf_counter() - start:.2f} seconds.", flush=True)


def replay_on_follower(
    robot,
    action_names: list[str],
    actions: np.ndarray,
    *,
    approach_s: float,
    fps: int,
    episode: int,
) -> None:
    """Connect, move gradually to the first action and replay the rest.

    A follower that is not free is refused and left as it is; one taken over is always disconnected with torque off.
    """
    connect_free_bus(robot)
    try:
        initial = take_over_follower(robot, action_names)
        move_to_start(
            robot,
            action_names,
            initial,
            actions[0],
            approach_s=approach_s,
            fps=fps,
        )
        replay_actions(
            robot,
            action_names,
            actions,
            fps=fps,
            episode=episode,
        )
    finally:
        if robot.bus.is_connected:
            robot.disconnect()
            print("Follower disconnected; torque disabled.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, help="Dataset folder; defaults to dataset_root from the config")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed in (0, 1]; 0.5 is half speed",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and describe without opening the robot")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = args.dataset.expanduser().resolve() if args.dataset else Path(config["dataset_root"])
    info = json.loads((root / "meta/info.json").read_text())
    frames = read_episode_frames(root, args.episode)
    action_names = info["features"]["action"]["names"]
    actions = recorded_actions(frames, action_names)
    fps = info["fps"]
    actions = slow_trajectory(actions, args.speed)
    approach_s = APPROACH_S / args.speed
    print(
        f"Episode {args.episode}, speed {args.speed:g}x, {len(actions)} commands, "
        f"approximately {len(actions) / fps:.1f} seconds plus {approach_s:g} seconds approach.",
        flush=True,
    )
    robot = build_follower(config)
    check_follower(robot, action_names)
    print(f"Calibration loaded: {robot.calibration_fpath}", flush=True)
    if args.dry_run:
        print("Dry run: no ports opened; physical motor calibration has not been checked.", flush=True)
        return
    replay_on_follower(
        robot,
        action_names,
        actions,
        approach_s=approach_s,
        fps=fps,
        episode=args.episode,
    )


if __name__ == "__main__":
    main()
