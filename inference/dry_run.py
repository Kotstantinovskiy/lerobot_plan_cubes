"""run_policy.py --dry-run: the policy on recorded frames instead of the cameras, with no robot.

Imported by run_policy.py once it has put recording/ on the path, as arm_control.py needs it.
"""

import argparse
import functools
import math
from collections.abc import Callable

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

import arm_control
from local_policy import dataset_observation
from remote_policy import RemotePolicy

RECORDED_EPISODE = 1
FIRST_OFFSET_S = 3
RTC_CHECK_OFFSET_S = 6  # the arm is moving by then


def make_recorded_source(
    remote: RemotePolicy | None,
    ds_meta: LeRobotDatasetMetadata | None,
    dataset_root: str,
) -> Callable:
    """Recorded observations by offset_s, in seconds, from the server's dataset or the local one."""
    if remote:
        return functools.partial(remote.recorded_observation, episode=RECORDED_EPISODE)
    return functools.partial(
        dataset_observation,
        ds_meta,
        dataset_root,
        episode=RECORDED_EPISODE,
    )


def run_one_inference(
    plan: Callable,
    recorded_observation: Callable,
    facts: arm_control.PolicyFacts,
    view: arm_control.PolicyView,
    color: str,
) -> None:
    """One inference on a recorded frame; prints its time, its first action and what was clipped of that."""
    observation = recorded_observation(offset_s=FIRST_OFFSET_S)
    if view.box_style:
        view.box = arm_control.find_target(observation["top"], color)
    planned = plan(arm_control.make_model_frame(observation, view, facts.features))
    first_action = planned.actions[0]
    clipped_action = np.clip(first_action, facts.joint_low, facts.joint_high)
    clipped_joint_count = int((clipped_action != first_action).sum())
    state = np.round([observation[name] for name in facts.joint_names], 1).tolist()
    print(
        f"Dry run: one inference {planned.seconds:.2f}s, {clipped_joint_count} joints clipped.\n"
        f"  state  {state}\n  action {np.round(first_action, 1).tolist()}",
        flush=True,
    )


def check_rtc(
    plan: Callable,
    recorded_observation: Callable,
    facts: arm_control.PolicyFacts,
    view: arm_control.PolicyView,
) -> None:
    """Print the seam, with and without RTC, of a chunk taking over from another as with --replan.

    The later chunk is planned `delay` steps on, and run from there.
    """

    def frame_at(offset_s: float) -> dict:
        return arm_control.make_model_frame(recorded_observation(offset_s=offset_s), view, facts.features)

    running = plan(frame_at(RTC_CHECK_OFFSET_S))
    delay = min(math.ceil(running.seconds * facts.fps), len(running.actions) // 2)
    later_frame = frame_at(RTC_CHECK_OFFSET_S + delay / facts.fps)
    plain = plan(later_frame)
    guided = plan(later_frame, inference_delay=delay, prev_chunk_left_over=running.raw_actions[delay:])
    last_running_action = running.actions[2 * delay - 1]
    plain_seam = arm_control.seam_size(plain.actions[delay], last_running_action)
    guided_seam = arm_control.seam_size(guided.actions[delay], last_running_action)
    print(
        f"RTC check on frames {delay} steps apart: seam {plain_seam:.1f} without, {guided_seam:.1f} with (largest "
        f"joint change where the later chunk takes over); guided inference {guided.seconds:.2f}s",
        flush=True,
    )


def run_dry(
    args: argparse.Namespace,
    plan: Callable,
    recorded_observation: Callable,
    facts: arm_control.PolicyFacts,
    view: arm_control.PolicyView,
) -> None:
    """All of the dry run: one inference, and with --rtc the check of a takeover. Only prints."""
    run_one_inference(
        plan,
        recorded_observation,
        facts,
        view,
        args.color,
    )
    if args.rtc:
        check_rtc(
            plan,
            recorded_observation,
            facts,
            view,
        )
