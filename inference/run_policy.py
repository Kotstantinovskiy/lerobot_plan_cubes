#!/usr/bin/env python3
"""Let a SmolVLA or pi0.5 policy drive the follower inside a safety envelope; nothing is added to the dataset.

A trained checkpoint normalizes with the statistics saved in it, those of the data it trained on, whatever dataset lies
here now; a base policy from the Hub uses our dataset's. Every command is clipped to the joint range
seen in our demonstrations and to `--max-step` per frame. When the run ends, after Ctrl+C or an error,
the follower folds back to `rest_pose` from config.json (`--no-rest` skips it; a second Ctrl+C drops
torque at once), then torque is disabled. A side-by-side top/wrist video of the run is saved under camera_check/.

A policy trained with target boxes (target_box.json next to its weights) gets the same box: before the arm
moves, the cube of the commanded colour is found on the top camera and its box is drawn on every top frame.
A policy trained on the cropped top view (top_crop.json) gets the same crop; the video keeps the full frame with the
crop outlined in grey.

By default a chunk is computed when the previous one has run out (`--n-action-steps` actions), and the arm stands
still for the ~1.2 s that takes on a Mac. With `--replan K` the next chunk is computed in the background once K actions
of the current one have run, from the observation at that moment; when it is ready, the actions for the steps that
passed meanwhile are dropped and the arm switches to it without stopping. Replanning starts once the arm has moved
from its starting pose: the demonstrations keep 3.5 s of stillness before the first motion, so from a still arm the
policy plans ~2 s of waiting before it moves, and replanning from still frames would only ever run the wait.
`--start-threshold` sets how far a joint must go for that to count: pi0.5 stopped again after the end of its first chunk
had taken the arm just past the default 3 units, and stood still for the 30 s of a run, while 15 is clear of the wait.

A chunk computed in the background is a fresh sample that knows nothing of the one the arm is running: where it takes
over, the command jumps (on recorded episodes 2-3 times a normal step) and now and then turns to another decision.
With `--rtc` (Real-Time Chunking, for flow-matching policies such as SmolVLA) the background chunk is guided to agree
with what is left of the running one: its actions for the steps that pass while it is computed are held to the running
chunk's, and the following ones, up to `--rtc-horizon`, blend from those into its own. The run ends with the seams: the
largest joint change where a background chunk took over.

With `--server HOST:PORT` the chunks come from inference/policy_server.py on a GPU machine, and everything else stays
here. The joint envelope and the box and crop settings come from the server too, so this machine needs neither a
checkpoint nor a dataset, and `--policy` and `--device` have no effect.
"""

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.utils import get_safe_torch_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
import arm_control  # noqa: E402  (inference/ is the script's own directory)
import dry_run  # noqa: E402
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
from project_config import PROJECT, load_config, positive_int  # noqa: E402
from remote_policy import PlannedChunk, RemotePolicy  # noqa: E402

# What other scripts import from here; the code itself lives in local_policy.py.
__all__ = ["compute_chunk", "dataset_observation", "load_policy", "saved_stats", "target_box_style"]

VIDEO_FOLDER = PROJECT / "camera_check"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def add_chunk_arguments(parser: argparse.ArgumentParser) -> None:
    """The options for how chunks are computed and when the next one takes over."""
    parser.add_argument(
        "--n-action-steps",
        type=positive_int,
        default=50,
        help="Actions run from each chunk before the next one (without --replan)",
    )
    parser.add_argument(
        "--replan",
        type=positive_int,
        help="Compute the next chunk in the background after this many actions of the current one; "
        "must be below --n-action-steps",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=3.0,
        help="Joint change, in normalized units, after which the arm counts as moving and --replan starts",
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Guide each background chunk to continue the running one (Real-Time Chunking)",
    )
    parser.add_argument(
        "--rtc-horizon",
        type=positive_int,
        default=DEFAULT_RTC_HORIZON,
        help="Actions of the running chunk the next one is guided to agree with; keep it above the "
        "steps one inference takes (~14 on a Mac)",
    )
    parser.add_argument(
        "--rtc-weight",
        type=float,
        default=DEFAULT_RTC_WEIGHT,
        help="Upper bound of the RTC guidance; 10 suits SmolVLA's 10 denoising steps",
    )


def make_parser() -> argparse.ArgumentParser:
    """The command line of this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--policy", default="lerobot/smolvla_base", help="Hub id or local pretrained_model folder")
    parser.add_argument("--color", required=True, help="Key from tasks in config.json")
    parser.add_argument("--seconds", type=positive_int, default=20)
    # 2.0 stalled the elbow against gravity (the per-frame position error was too small to move it).
    parser.add_argument(
        "--max-step",
        type=float,
        default=5.0,
        help="Max change per frame per joint (normalized units)",
    )
    add_chunk_arguments(parser)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--server", help="HOST:PORT of policy_server.py, to compute chunks there instead of here")
    parser.add_argument("--dry-run", action="store_true", help="Run one inference on a recorded frame; no robot")
    parser.add_argument("--no-rest", action="store_true", help="Leave the arm where the run ends instead of folding it")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Once the policy is loaded, wait for Enter before the arm is powered (run_series.py)",
    )
    parser.add_argument("--summary", type=Path, help="Also write the run's summary to this JSON file")
    return parser


def facts_from_server(info: dict, address: str) -> arm_control.PolicyFacts:
    """The facts a policy server sends on connecting; the joint envelope is that of the policy's training data."""
    return arm_control.PolicyFacts(
        features=info["features"],
        fps=info["fps"],
        joint_names=info["features"]["action"]["names"],
        joint_low=np.asarray(info["low"], dtype=float),
        joint_high=np.asarray(info["high"], dtype=float),
        envelope_source="the policy's training data",
        policy_name=f"{info['policy']} at {address}",
    )


def facts_from_dataset(ds_meta: LeRobotDatasetMetadata, policy: str, device: str) -> arm_control.PolicyFacts:
    """The facts of the local dataset; the joint envelope is the range of its recorded actions."""
    return arm_control.PolicyFacts(
        features=ds_meta.features,
        fps=ds_meta.fps,
        joint_names=ds_meta.features["action"]["names"],
        joint_low=np.asarray(ds_meta.stats["action"]["min"], dtype=float),
        joint_high=np.asarray(ds_meta.stats["action"]["max"], dtype=float),
        envelope_source=f"{ds_meta.total_episodes} episodes",
        policy_name=f"{policy} on {device}",
    )


def describe_run(task: str, facts: arm_control.PolicyFacts) -> str:
    """The lines printed first of all: task, policy, joint envelope."""
    limits = zip(
        facts.joint_names,
        facts.joint_low,
        facts.joint_high,
        strict=False,
    )
    envelope = ", ".join(f"{name} [{low:.0f}, {high:.0f}]" for name, low, high in limits)
    return f"Task: {task}\nPolicy: {facts.policy_name}\nJoint envelope from {facts.envelope_source}: {envelope}"


def describe_control(args: argparse.Namespace) -> str:
    """The line printed when the policy takes over the arm."""
    if args.replan:
        replanning = f"next chunk in the background after {args.replan} actions"
    else:
        replanning = f"{args.n_action_steps} actions per chunk"
    if args.rtc:
        replanning += f", RTC over {args.rtc_horizon} actions, guidance up to {args.rtc_weight:g}"
    return (
        f"Policy controls the arm for {args.seconds} s (max {args.max_step} units/frame, {replanning}). Ctrl+C stops."
    )


def connect_to_server(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RemotePolicy:
    """The connection to --server. Exits through the parser if --rtc asks for what that server was not started with."""
    remote = RemotePolicy(args.server)
    if args.rtc and not remote.info["rtc"]:
        parser.error("--rtc needs policy_server.py started with --rtc")
    return remote


def use_server_policy(remote: RemotePolicy, task: str) -> tuple[Callable, arm_control.PolicyView]:
    """plan(frame, **guidance), which asks the server for a PlannedChunk for this task, and how the policy sees."""
    plan = functools.partial(remote.plan, task=task)
    return plan, arm_control.PolicyView(remote.info["target_box"], remote.info["top_crop"])


def load_local_policy(
    args: argparse.Namespace,
    ds_meta: LeRobotDatasetMetadata,
    task: str,
) -> tuple[Callable, arm_control.PolicyView]:
    """Load the policy: plan(frame, **guidance), which computes a PlannedChunk for this task here, and how it sees."""
    rtc = make_rtc_config(args.rtc_horizon, args.rtc_weight) if args.rtc else None
    policy, preprocess, postprocess = load_policy(
        args.policy,
        args.device,
        args.n_action_steps,
        ds_meta,
        rtc,
    )
    device = get_safe_torch_device(args.device)
    view = arm_control.PolicyView(target_box_style(args.policy), top_crop.load(args.policy))

    def plan(frame: dict, **guidance) -> PlannedChunk:
        start = time.perf_counter()
        actions, raw_actions = compute_chunk(
            policy,
            preprocess,
            postprocess,
            frame,
            device,
            task,
            ds_meta.robot_type,
            **guidance,
        )
        return PlannedChunk(actions, raw_actions, time.perf_counter() - start)

    return plan, view


def wait_for_enter() -> bool:
    """Ask the operator to set up the cubes and press Enter; False if they cancel instead."""
    try:
        input("Policy loaded. Set up the cubes, then press Enter to start (Ctrl+C cancels): ")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled before the arm moved.", flush=True)
        return False
    return True


def report_run(run: arm_control.ArmRun, state: arm_control.RunState) -> None:
    """Print how the run went, and write it to the --summary file if there is one."""
    seam_summary = arm_control.summarize_seams(state.seams)
    if seam_summary:
        print(
            f"Seams over {len(state.seams)} chunk changes: median {seam_summary['median']:.1f}, 90th percentile "
            f"{seam_summary['p90']:.1f}, max {seam_summary['max']:.1f} units",
            flush=True,
        )
    print(f"Summary: {asdict(state.stats)}\nVideo: {run.video.path}", flush=True)
    if not run.args.summary:
        return
    summary = {
        "color": run.args.color,
        "target_box": run.view.box,
        "stats": asdict(state.stats),
        "seams": seam_summary,
        "video": str(run.video.path),
        "args": vars(run.args),
    }
    run.args.summary.write_text(json.dumps(summary, indent=1, default=str))


def finish_run(run: arm_control.ArmRun, state: arm_control.RunState, config: dict) -> None:
    """However the run ended: fold the arm back unless --no-rest, turn torque off, close the video, report the run.

    An arm that was refused has its bus closed by now and is left as it is.
    """
    try:
        if run.robot.bus.is_connected:  # robot.is_connected is also False when only a camera has failed
            if not run.args.no_rest:
                arm_control.fold_if_holding(run.robot, config, run.facts.fps)
            arm_control.disconnect_arm(run.robot)
    finally:
        arm_control.close_video(run.video)
        if run.background is not None:
            run.background.shutdown(cancel_futures=True)
        report_run(run, state)


def run_on_arm(
    args: argparse.Namespace,
    config: dict,
    robot: Any,
    plan: Callable,
    facts: arm_control.PolicyFacts,
    view: arm_control.PolicyView,
) -> None:
    """Let the policy drive the arm for args.seconds and save a video of it under camera_check/.

    Ctrl+C ends the run normally; any other error is raised once finish_run has done its work.
    """
    timestamp = time.strftime(TIMESTAMP_FORMAT)
    video = arm_control.open_video(VIDEO_FOLDER / f"policy-run-{timestamp}-{args.color}.mp4", facts.fps)
    run = arm_control.ArmRun(
        robot=robot,
        plan=plan,
        background=ThreadPoolExecutor(max_workers=1) if args.replan else None,
        args=args,
        facts=facts,
        view=view,
        video=video,
    )
    state = arm_control.RunState()
    try:
        arm_control.connect_arm(robot)
        if view.box_style:  # the arm holds its pose; the box is found once, before it moves, as in the recordings
            view.box = arm_control.find_target(robot.get_observation()["top"], args.color)
        print(describe_control(args), flush=True)
        arm_control.drive_arm(run, state)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.", flush=True)
    finally:
        finish_run(run, state, config)


def check_chunk_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit through the parser on chunk options that would silently do nothing."""
    if args.rtc and not (args.replan or args.dry_run):
        parser.error("--rtc guides the chunks computed in the background; add --replan")
    # The loop looks for --replan actions run before it takes each one, so the last chunk action it can start from is
    # number --n-action-steps minus one.
    if args.replan and args.replan >= args.n_action_steps:
        parser.error(
            f"--replan {args.replan} would never start: a chunk runs out after --n-action-steps "
            f"({args.n_action_steps}) actions; use a smaller --replan"
        )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    check_chunk_arguments(parser, args)

    config = load_config(args.config)
    task = config["tasks"][args.color]
    remote = connect_to_server(parser, args) if args.server else None
    ds_meta = None
    if remote:
        facts = facts_from_server(remote.info, args.server)
    else:
        ds_meta = LeRobotDatasetMetadata(config["repo_id"], root=config["dataset_root"])
        facts = facts_from_dataset(ds_meta, args.policy, args.device)
    print(describe_run(task, facts), flush=True)

    # The robot comes first so configuration errors surface before the slow model load.
    robot = None if args.dry_run else arm_control.build_robot(config["robot"], args.max_step)
    plan, view = use_server_policy(remote, task) if remote else load_local_policy(args, ds_meta, task)
    if view.crop:
        print(f"Top camera cropped to {view.crop.get('box') or view.crop['quad']}", flush=True)

    if args.dry_run:
        recorded_observation = dry_run.make_recorded_source(remote, ds_meta, config["dataset_root"])
        dry_run.run_dry(
            args,
            plan,
            recorded_observation,
            facts,
            view,
        )
        return
    if args.wait and not wait_for_enter():
        return
    run_on_arm(
        args,
        config,
        robot,
        plan,
        facts,
        view,
    )


if __name__ == "__main__":
    main()
