"""A policy checkpoint on this machine: loading it, computing one action chunk, and what is saved next to its weights.

Shared by the arm (run_policy.py), the GPU server (policy_server.py) and offline checks (training/eval_offline.py, which
imports these names through run_policy.py), so a policy is loaded and called the same way everywhere.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_IMAGES
from safetensors.numpy import load_file

import pi05_inputs

# Our camera keys and the keys the base policy was pretrained with, as in training/train.py.
SMOLVLA_TOP_CAMERA = "observation.images.camera1"
SMOLVLA_CAMERA_RENAME = {
    "observation.images.top": SMOLVLA_TOP_CAMERA,  # SmolVLA: camera1 = top, camera2 = wrist
    "observation.images.wrist": "observation.images.camera2",
}
PI05_CAMERA_RENAME = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}
TARGET_BOX_FILE = "target_box.json"
SAVED_STATS_FILES = "policy_preprocessor_step_*_normalizer_processor.safetensors"
RECORDED_CAMERAS = ("top", "wrist")
RECORDED_VIDEO_BACKEND = "pyav"
DEFAULT_RTC_HORIZON = 20
DEFAULT_RTC_WEIGHT = 10.0


class LoadedPolicy(NamedTuple):
    """A policy in eval mode and the functions around it: robot frame to model input, model output to dataset units."""

    policy: Any
    preprocess: Callable
    postprocess: Callable


def target_box_style(policy_path: str | Path) -> dict | None:
    """Drawing style if the policy was trained with target boxes, else None."""
    path = Path(policy_path) / TARGET_BOX_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def saved_stats(policy_path: str | Path) -> dict[str, dict[str, np.ndarray]] | None:
    """The normalization statistics saved in a trained checkpoint ({feature: {stat: array}}), None for a Hub id.
    A dataset that has grown since training has other ones: 88 and 108 episodes differ by up to 4.8 joint units."""
    files = sorted(Path(policy_path).glob(SAVED_STATS_FILES))
    if not files:
        return None
    stats = {}
    for key, array in load_file(files[0]).items():
        feature, stat = key.rsplit(".", 1)
        stats.setdefault(feature, {})[stat] = array
    return stats


def make_rtc_config(execution_horizon: int, max_guidance_weight: float) -> RTCConfig:
    """Real-Time Chunking settings for load_policy: actions of the running chunk to agree with, largest guidance."""
    return RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )


def load_policy_config(path: str, device: str, n_action_steps: int, rtc: RTCConfig | None) -> PreTrainedConfig:
    """The checkpoint's config, set up to run on `device` with `n_action_steps` actions per chunk."""
    policy_config = PreTrainedConfig.from_pretrained(path)
    policy_config.pretrained_path = path
    policy_config.device = device
    policy_config.n_action_steps = n_action_steps
    if rtc is not None:
        policy_config.rtc_config = rtc
    if hasattr(policy_config, "load_vlm_weights"):
        policy_config.load_vlm_weights = False  # the policy checkpoint already holds every VLM weight
    return policy_config


def camera_rename_map(policy_config: PreTrainedConfig) -> dict[str, str]:
    """{our camera key: the key the checkpoint expects}; empty for a checkpoint that uses our keys."""
    if policy_config.type == pi05_inputs.POLICY_TYPE:
        return PI05_CAMERA_RENAME
    if SMOLVLA_TOP_CAMERA in policy_config.input_features:
        return SMOLVLA_CAMERA_RENAME
    return {}


def make_processors(
    policy_config: PreTrainedConfig,
    policy: Any,
    path: str,
    device: str,
    rename_map: dict[str, str],
    stats: dict,
) -> tuple[Callable, Callable]:
    """(preprocess, postprocess) for the checkpoint. For pi0.5 this downloads the tokenizer the first time."""
    features = {**policy.config.input_features, **policy.config.output_features}
    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": rename_map},
        "normalizer_processor": {"stats": stats, "features": features, "norm_map": policy.config.normalization_mapping},
    }
    if policy_config.type == pi05_inputs.POLICY_TYPE:  # the checkpoint names no tokenizer, training was given one
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer": pi05_inputs.load_tokenizer()}
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }
    return make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=path,
        dataset_stats=stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def resize_cameras_first(preprocess: Callable) -> Callable:
    """pi0.5's preprocessor behind a step that shrinks the camera frames as in training (inference/pi05_inputs.py)."""

    def resized_preprocess(observation):
        frames = {key: pi05_inputs.resize(value) for key, value in observation.items() if key.startswith(OBS_IMAGES)}
        return preprocess({**observation, **frames})

    return resized_preprocess


def load_policy(
    path: str,
    device: str,
    n_action_steps: int,
    ds_meta: LeRobotDatasetMetadata,
    rtc: RTCConfig | None = None,
) -> LoadedPolicy:
    """A checkpoint (Hub id or local folder) ready to compute chunks on `device`: (policy, preprocess, postprocess).

    It normalizes with the statistics saved in the checkpoint, or the dataset's for a base policy from the Hub.
    Raises RuntimeError if `rtc` is given for a policy without Real-Time Chunking.
    """
    # Matrix products as in training: on an H100 this cuts a pi0.5 chunk from 256 to 144 ms, changing it by at most
    # 0.03.
    torch.backends.cuda.matmul.allow_tf32 = True
    policy_config = load_policy_config(
        path,
        device,
        n_action_steps,
        rtc,
    )
    rename_map = camera_rename_map(policy_config)
    policy = make_policy(policy_config, ds_meta=ds_meta, rename_map=rename_map)
    if rtc is not None and getattr(policy, "rtc_processor", None) is None:
        raise RuntimeError(
            f"{policy_config.type} has no Real-Time Chunking; it is for flow-matching policies such as SmolVLA"
        )
    policy.eval()
    stats = saved_stats(path) or ds_meta.stats
    preprocess, postprocess = make_processors(
        policy_config,
        policy,
        path,
        device,
        rename_map,
        stats,
    )
    if policy_config.type == pi05_inputs.POLICY_TYPE:
        preprocess = resize_cameras_first(preprocess)
    return LoadedPolicy(policy, preprocess, postprocess)


def compute_chunk(
    policy: Any,
    preprocess: Callable,
    postprocess: Callable,
    frame: dict,
    device: torch.device,
    task: str,
    robot_type: str,
    **guidance,
) -> tuple[np.ndarray, torch.Tensor]:
    """The whole action chunk for a frame (chunk_size x joints): in dataset units and as the model gave it (normalized,
    what RTC compares). `guidance`: RTC's inference_delay and prev_chunk_left_over."""
    with torch.no_grad():  # not inference_mode: RTC's guidance takes a gradient, which that mode rules out
        prepared = prepare_observation_for_inference(
            dict(frame),
            device,
            task,
            robot_type,
        )
        batch = preprocess(prepared)
        raw_chunk = policy.predict_action_chunk(batch, **guidance)
        chunk = postprocess(raw_chunk.clone())
    return chunk[0].float().cpu().numpy().astype(float), raw_chunk[0]


def dataset_observation(ds_meta: LeRobotDatasetMetadata, root: str, episode: int, offset_s: float) -> dict:
    """A recorded frame in the same format as robot.get_observation(), for --dry-run."""
    dataset = LeRobotDataset(ds_meta.repo_id, root=root, video_backend=RECORDED_VIDEO_BACKEND)
    # round: an offset worked out as frames / fps can come back a hair under its frame
    index = int(ds_meta.episodes[episode]["dataset_from_index"]) + round(offset_s * ds_meta.fps)
    item = dataset[index]
    joint_names = ds_meta.features["observation.state"]["names"]
    observation = dict(zip(joint_names, item["observation.state"].tolist(), strict=False))
    for camera in RECORDED_CAMERAS:
        image = item[f"observation.images.{camera}"].permute(1, 2, 0).numpy()
        observation[camera] = (image * 255).round().astype(np.uint8)
    return observation
