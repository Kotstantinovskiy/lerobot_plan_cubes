"""The arm during a policy run: building and connecting it, the control loop, the run's video, folding it back.

The loop sends one clipped action per camera frame, taken from an action chunk. A chunk is computed in the loop when the
previous one has run out, or in the background with --replan; run_policy.py explains when and why, and imports this
module once it has put recording/ (rest.py) on the path.
"""

import argparse
import functools
import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import av
import cv2
import draccus
import numpy as np

# The camera and robot types are imported to register them: draccus.decode(RobotConfig, ...) finds them that way.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.utils import build_dataset_frame
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.constants import OBS_STR
from lerobot.utils.robot_utils import precise_sleep

import cube_boxes
import top_crop
from follower_bus import connect_free_bus
from remote_policy import PlannedChunk
from rest import fold_to_rest

BUS_RETRIES = 3
RECENT_DELAYS_KEPT = 5
VIDEO_CAMERAS = ("top", "wrist")
VIDEO_CAMERA_SIZE = (640, 360)  # width, height of each camera in the side-by-side video
VIDEO_SIZE = (1280, 360)
VIDEO_CODEC = "libx264"
VIDEO_OPTIONS = {"crf": "28"}
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_FRAME_FORMAT = "rgb24"


@dataclass(frozen=True)
class PolicyFacts:
    """What a run needs to know of the data behind its policy, from the policy server or the local dataset."""

    features: dict  # the dataset's features
    fps: int
    joint_names: list[str]
    joint_low: np.ndarray  # the joint envelope, in normalized units
    joint_high: np.ndarray
    envelope_source: str  # how the printed envelope describes its origin
    policy_name: str


@dataclass
class PolicyView:
    """How the policy was trained to see the top camera. `box` is its target, found once before the arm moves."""

    box_style: dict | None
    crop: dict | None
    box: list[int] | None = None


class RunVideo(NamedTuple):
    """The video being written: top and wrist cameras side by side."""

    path: Path
    container: Any
    stream: Any


@dataclass(frozen=True)
class ArmRun:
    """What stays the same during a run on the arm."""

    robot: Any
    plan: Callable  # (frame, **guidance) gives a PlannedChunk; guidance is RTC's inference_delay, prev_chunk_left_over
    background: ThreadPoolExecutor | None  # runs `plan` with --replan
    args: argparse.Namespace  # run_policy.py's command line
    facts: PolicyFacts
    view: PolicyView
    video: RunVideo


@dataclass
class RunStats:
    """The counters of a run, printed at its end and saved in its summary."""

    steps: int = 0
    chunks: int = 0
    max_infer_s: float = 0.0
    clipped_steps: int = 0
    late_steps: int = 0
    dropped_actions: int = 0


class PendingChunk(NamedTuple):
    """A chunk being computed in the background, and the step of the observation it was given."""

    job: Future
    observed_at_step: int


@dataclass
class RunState:
    """What changes during a run on the arm. Its stats and seams are reported however the run ends."""

    stats: RunStats = field(default_factory=RunStats)
    seams: list[float] = field(default_factory=list)  # largest joint change where a background chunk took over
    actions: list[np.ndarray] = field(default_factory=list)  # of the running chunk, left to run, in dataset units
    raw_actions: Any = None  # the same as the model gave them (normalized), which RTC holds the next chunk to
    steps_since_observation: int = 0  # the observation the running chunk was computed from
    last_action: np.ndarray | None = None  # taken from a chunk last step, before clipping; a seam is measured from it
    last_infer_seconds: float = 0.0  # of the chunk computed last
    pending: PendingChunk | None = None
    # The steps the last background chunks took to come.
    recent_delays: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_DELAYS_KEPT))
    first_pose: np.ndarray | None = None
    has_moved: bool = False


def build_robot(robot_fields: dict, max_step: float) -> Any:
    """The follower from config.json's "robot" section, not yet connected, with each command limited to `max_step`.

    Raises RuntimeError without a calibration of every motor.
    """
    fields = dict(robot_fields, max_relative_target=max_step)
    robot = make_robot_from_config(draccus.decode(RobotConfig, fields))
    if set(robot.calibration) != set(robot.bus.motors):
        raise RuntimeError(f"Missing or incomplete follower calibration: {robot.calibration_fpath}")
    # Torque_Enable occasionally gets no status packet right after the cameras start (seen on ids 4 and 6);
    # rewriting it is idempotent, so allow a few retries instead of aborting the connection. The first
    # Present_Position read after connecting fails the same way now and then.
    robot.bus.enable_torque = functools.partial(robot.bus.enable_torque, num_retry=BUS_RETRIES)
    robot.bus.sync_read = functools.partial(robot.bus.sync_read, num_retry=BUS_RETRIES)
    return robot


def connect_arm(robot: Any) -> None:
    """Connect bus and cameras and enable torque.

    Raises RuntimeError on bad calibration or with torque already on, leaving the arm as it is with the port closed.
    """
    connect_free_bus(robot)
    # With torque still off: replace any stale goal, so enabling torque cannot jump.
    robot.bus.sync_write("Goal_Position", robot.bus.sync_read("Present_Position"))
    robot.bus.disconnect(disable_torque=False)
    robot.connect(calibrate=False)


def fold_if_holding(robot: Any, config: dict, fps: int) -> None:
    """Fold a torque-enabled arm to config's rest_pose. Prints whatever stops that, even Ctrl+C, and returns.

    The whole config comes in because a missing rest_pose is such a stop: it must not keep the torque on.
    """
    try:
        if all(robot.bus.sync_read("Torque_Enable").values()):
            print("Folding back to rest; Ctrl+C drops torque at once.", flush=True)
            fold_to_rest(
                robot.bus,
                config["rest_pose"],
                fps,
                log=lambda message: print(message, flush=True),
            )
    except (Exception, KeyboardInterrupt) as error:
        print(f"Return to rest stopped: {error!r}", flush=True)


def disconnect_arm(robot: Any) -> None:
    """Disconnect bus and cameras with torque disabled; neither a failing camera nor Ctrl+C keeps the torque on."""
    try:
        robot.disconnect()
    except (Exception, KeyboardInterrupt):
        # A closed bus means robot.disconnect() got that far, and it disables torque before closing.
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)
    print("Follower disconnected; torque disabled.", flush=True)


def find_target(top_rgb: np.ndarray, color: str) -> list[int]:
    """The box of the largest cube of this colour on an HxWx3 uint8 RGB top camera frame; prints what was found.

    Raises RuntimeError on a frame of another size or without such a cube.
    """
    if top_rgb.shape[:2] != cube_boxes.TOP_FRAME_SHAPE:
        raise RuntimeError(
            f"Target boxes are tuned for a 1280x720 top camera, got {top_rgb.shape[1]}x{top_rgb.shape[0]}"
        )
    boxes = cube_boxes.detect(top_rgb)
    if boxes[color] is None:
        raise RuntimeError(f"No {color} cube found on the top camera; nothing to mark as the target")
    found = ", ".join(f"{name} {box}" for name, box in boxes.items() if box is not None)
    print(f"Target box: {color} {boxes[color]} (found: {found})", flush=True)
    return boxes[color]


def make_model_frame(observation: dict, view: PolicyView, features: dict) -> dict:
    """Model input from a robot observation; the box is drawn in place, so the video shows it too."""
    if view.box is not None:
        cube_boxes.draw_rgb(observation["top"], view.box, view.box_style)
    seen = observation
    if view.crop is not None:
        seen = {**observation, "top": top_crop.crop_hwc(observation["top"], view.crop)}
    # Copies: with --replan the frame is read by the background inference while the loop draws on the next one.
    copies = {key: np.array(value) if isinstance(value, np.ndarray) else value for key, value in seen.items()}
    return build_dataset_frame(features, copies, prefix=OBS_STR)


def seam_size(next_action: np.ndarray, last_action: np.ndarray) -> float:
    """The largest joint change between two actions, in normalized units: how far the command jumps at a takeover."""
    return np.abs(next_action - last_action).max()


def summarize_seams(seams: list[float]) -> dict | None:
    """Count, median, 90th percentile and maximum of the seams, for the printout and the summary; None without seams."""
    if not seams:
        return None
    return {
        "count": len(seams),
        "median": float(np.median(seams)),
        "p90": float(np.percentile(seams, 90)),
        "max": float(max(seams)),
    }


def track_motion(state: RunState, pose: np.ndarray, start_threshold: float) -> None:
    """Note whether any joint has gone further than `start_threshold` from the first pose (normalized units)."""
    if state.first_pose is None:
        state.first_pose = pose
    state.has_moved = state.has_moved or np.abs(pose - state.first_pose).max() > start_threshold


def start_chunk(state: RunState, planned: PlannedChunk) -> None:
    """Make these actions the running chunk."""
    state.actions = list(planned.actions)
    state.raw_actions = planned.raw_actions
    state.last_infer_seconds = planned.seconds
    state.stats.chunks += 1
    state.stats.max_infer_s = max(state.stats.max_infer_s, planned.seconds)


def take_background_chunk(state: RunState) -> None:
    """Switch to the chunk computed in the background, waiting for it if it is not ready; records its seam."""
    planned = state.pending.job.result()
    observed_at_step = state.pending.observed_at_step
    state.pending = None
    # The actions for the steps run since its observation are stale; the last one is kept in any case.
    stale_count = min(state.stats.steps - observed_at_step, len(planned.actions) - 1)
    state.steps_since_observation = stale_count
    state.recent_delays.append(stale_count)
    fresh_actions, fresh_raw_actions = planned.actions[stale_count:], planned.raw_actions[stale_count:]
    state.stats.dropped_actions += stale_count
    state.seams.append(seam_size(fresh_actions[0], state.last_action))
    start_chunk(state, PlannedChunk(fresh_actions, fresh_raw_actions, planned.seconds))


def compute_chunk_in_loop(run: ArmRun, state: RunState, frame: dict) -> None:
    """Compute the next chunk now, with the arm standing still meanwhile; its first --n-action-steps actions run."""
    planned = run.plan(frame)
    state.steps_since_observation = 0
    kept_count = run.args.n_action_steps
    start_chunk(state, PlannedChunk(planned.actions[:kept_count], planned.raw_actions[:kept_count], planned.seconds))


def update_running_chunk(run: ArmRun, state: RunState, frame: dict) -> None:
    """Switch to a background chunk that has come, or compute the next chunk now if the running one has run out."""
    is_chunk_waiting = state.pending is not None and (state.pending.job.done() or not state.actions)
    if is_chunk_waiting:  # waits only if the current chunk ran out
        take_background_chunk(state)
    elif not state.actions:  # every chunk without --replan, and with it until the arm has moved
        compute_chunk_in_loop(run, state, frame)


def is_replan_due(state: RunState, replan_after: int | None) -> bool:
    """Whether to start the next chunk in the background: with --replan, once the arm has moved, one job at a time."""
    if not replan_after or not state.has_moved or state.pending is not None:
        return False
    return state.steps_since_observation >= replan_after


def rtc_guidance(state: RunState, fps: int) -> dict:
    """RTC's arguments for a new chunk: its first actions are held to those the arm runs while it is computed."""
    return {
        "inference_delay": max(state.recent_delays, default=math.ceil(state.last_infer_seconds * fps)),
        "prev_chunk_left_over": state.raw_actions,
    }


def start_background_chunk(run: ArmRun, state: RunState, frame: dict) -> None:
    """Have the next chunk computed in the background from this step's frame."""
    guidance = rtc_guidance(state, run.facts.fps) if run.args.rtc else {}
    job = run.background.submit(run.plan, frame, **guidance)
    state.pending = PendingChunk(job, state.stats.steps)


def pop_next_action(state: RunState) -> np.ndarray:
    """Take the running chunk's next action, in dataset units, and remember it for the next seam."""
    state.last_action = state.actions.pop(0)
    state.raw_actions = state.raw_actions[1:]
    state.steps_since_observation += 1
    return state.last_action


def send_next_action(run: ArmRun, state: RunState) -> None:
    """Send the arm the running chunk's next action, clipped to the joint envelope."""
    action = pop_next_action(state)
    clipped = np.clip(action, run.facts.joint_low, run.facts.joint_high)
    run.robot.send_action(dict(zip(run.facts.joint_names, clipped.tolist(), strict=False)))
    state.stats.steps += 1
    state.stats.clipped_steps += bool((clipped != action).any())


def open_video(path: Path, fps: int) -> RunVideo:
    """Start the side-by-side video of a run at `path`."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream(VIDEO_CODEC, rate=fps, options=VIDEO_OPTIONS)
    stream.width, stream.height = VIDEO_SIZE
    stream.pix_fmt = VIDEO_PIXEL_FORMAT
    return RunVideo(path, container, stream)


def write_video_frame(video: RunVideo, observation: dict) -> None:
    """Add the observation's cameras, side by side, to the video."""
    resized = [cv2.resize(observation[key], VIDEO_CAMERA_SIZE, interpolation=cv2.INTER_AREA) for key in VIDEO_CAMERAS]
    frame = av.VideoFrame.from_ndarray(np.hstack(resized), format=VIDEO_FRAME_FORMAT)
    for packet in video.stream.encode(frame):
        video.container.mux(packet)


def close_video(video: RunVideo) -> None:
    """Write what the encoder still holds and close the file."""
    for packet in video.stream.encode():
        video.container.mux(packet)
    video.container.close()


def record_step(run: ArmRun, observation: dict) -> None:
    """Add this step to the video, with the crop's outline on the top camera if the policy has a crop."""
    if run.view.crop is not None:  # after the model's frame is made: only the video shows the outline
        top_crop.draw_outline(observation["top"], run.view.crop)
    write_video_frame(run.video, observation)


def print_progress(stats: RunStats, elapsed_seconds: float) -> None:
    """The line printed every second of a run."""
    print(
        f"{elapsed_seconds:5.1f}s step {stats.steps}, new chunks {stats.chunks}, clipped steps {stats.clipped_steps}",
        flush=True,
    )


def drive_arm(run: ArmRun, state: RunState) -> None:
    """Let the policy drive a connected arm for args.seconds, one action per frame at the dataset's fps.

    `state` is updated in place, so that a run cut short by Ctrl+C or an error can still report its stats and seams.
    """
    step_seconds = 1 / run.facts.fps
    start = time.perf_counter()
    while time.perf_counter() - start < run.args.seconds:
        tick = time.perf_counter()
        observation = run.robot.get_observation()
        pose = np.array([observation[name] for name in run.facts.joint_names])
        track_motion(state, pose, run.args.start_threshold)
        frame = make_model_frame(observation, run.view, run.facts.features)

        update_running_chunk(run, state, frame)
        if is_replan_due(state, run.args.replan):
            start_background_chunk(run, state, frame)
        send_next_action(run, state)

        record_step(run, observation)
        if state.stats.steps % run.facts.fps == 0:
            print_progress(state.stats, time.perf_counter() - start)
        elapsed = time.perf_counter() - tick
        state.stats.late_steps += elapsed > step_seconds
        precise_sleep(max(0.0, step_seconds - elapsed))
