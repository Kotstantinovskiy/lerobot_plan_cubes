#!/usr/bin/env python3
"""Exercise the real recorder; arms are only read unless --teleop. Output is diagnostic, not training data.

With --episodes 2 or more the report also gives the pauses between the recorder's loops: the one before each
later episode is the time it took to save the episode before it.
"""

from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from live_video import LiveVideoDataset
from record import PROJECT, build_command, load_config, positive_int
from lerobot.scripts import lerobot_record as recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=positive_int, default=60)
    parser.add_argument("--episodes", type=positive_int, default=1,
                        help="Episodes in a row, with a 5 s reset between them")
    parser.add_argument("--teleop", action="store_true",
                        help="Move the follower with the leader arm; output still stays in camera_check")
    args = parser.parse_args()
    config = load_config()
    root = PROJECT / "camera_check" / f"recording-check-{time.time_ns()}"
    config.update(dataset_root=str(root), repo_id="local/diagnostic_camera_check", episode_time_s=args.seconds,
                  reset_time_s=5, display_data=False)
    task = ("Diagnostic teleoperated capture." if args.teleop
            else "Diagnostic stationary capture. No motor commands.")
    config["tasks"] = {"diagnostic": f"{task} Not training data."}
    command, _ = build_command(config, "diagnostic", args.episodes, False)
    make_robot = recorder.make_robot_from_config
    make_leader = recorder.make_teleoperator_from_config
    record_loop = recorder.record_loop
    report = {"motor_commands_sent": args.teleop, "target_fps": config["fps"], "dataset": str(root), "episodes": []}
    loops = []  # (start, end, frames) of every episode and reset loop

    with ExitStack() as stack:
        def connect_bus(bus):
            bus.connect()
            stack.callback(bus.disconnect, disable_torque=False)

        def read_only_robot(cfg):
            robot = make_robot(cfg)

            def connect():
                connect_bus(robot.bus)
                for camera in robot.cameras.values():
                    stack.callback(lambda c=camera: c.disconnect() if c.is_connected else None)
                    camera.connect()

            robot.connect = connect
            robot.disconnect = lambda: None  # ExitStack closes camera streams and serial ports.
            robot.send_action = lambda action: action  # Never writes a goal position or enables torque.
            return robot

        def read_only_leader(cfg):
            leader = make_leader(cfg)
            leader.connect = lambda: connect_bus(leader.bus)
            leader.disconnect = lambda: None
            return leader

        def measured_loop(*args, **kwargs):
            start = time.perf_counter()
            result = record_loop(*args, **kwargs)
            dataset = kwargs.get("dataset")  # none in the reset between episodes
            loops.append((start, time.perf_counter(), None if dataset is None else dataset.episode_buffer["size"]))
            return result

        if not args.teleop:
            stack.enter_context(patch.object(recorder, "make_robot_from_config", read_only_robot))
            stack.enter_context(patch.object(recorder, "make_teleoperator_from_config", read_only_leader))
        stack.enter_context(patch.object(recorder, "record_loop", measured_loop))
        stack.enter_context(patch.object(recorder, "LeRobotDataset", LiveVideoDataset))  # as record.py runs it
        # Keyboard input is deliberately disabled for this timed diagnostic.
        stack.enter_context(patch.object(recorder, "init_keyboard_listener", return_value=(None, {
            "stop_recording": False, "exit_early": False, "rerecord_episode": False})))
        stack.enter_context(patch.object(sys, "argv", ["verify_recording", *command[2:]]))
        recorder.main()
    for start, end, frames in loops:
        if frames is not None:
            report["episodes"].append({"wall_time_s": end - start, "frames": frames,
                                       "actual_fps": frames / (end - start)})
    report["pauses_s"] = [later[0] - earlier[1] for earlier, later in zip(loops, loops[1:])]
    (root / "diagnostic-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
