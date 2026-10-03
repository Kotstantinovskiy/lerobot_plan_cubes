#!/usr/bin/env python3
"""lerobot_train with two data-loading fixes that leave the model inputs and their distribution unchanged.

1. Camera frames are shrunk inside the DataLoader workers, exactly as SmolVLA would shrink them. SmolVLA's first step
   on every camera image is resize_with_pad to 512x512 (bilinear, zero padding). Applying that same function in the
   workers, after augmentation, gives identical inputs: on a 512x512 image it is the identity. A sample then carries
   6 MB of images instead of 36 MB for our 1080p + 720p frames, and the pin-memory thread, which copied 2.3 GB per
   batch of 64 and held training at ~1.3 s per step, keeps up with the GPU. pi0.5 frames are shrunk to 224x224 as
   openpi shrinks them, with black bars (inference/pi05_inputs.py).
2. No epoch boundaries. lerobot_train re-iterates the DataLoader after every epoch, which restarts all workers and
   reopens every video; with 64 workers that stalled training ~30 s per epoch. A sampler that yields fresh random
   permutations back to back keeps the workers running; every epoch is still a new permutation of the dataset.
With TARGET_BOX=1 in the environment (train.py --target-box) the target cube's box from meta/cube_boxes.json is also
drawn on every top frame of its episode, right after decoding and before augmentation, so a shifted or rotated frame
moves the box with the cube. Each checkpoint then gets target_box.json with the drawing style, which tells
run_policy.py and eval_offline.py to draw the same box.
With TOP_CROP=1 (train.py --top-crop) the top frame is cropped to the left part of the table (inference/top_crop.py),
after the box and before augmentation, and each checkpoint gets top_crop.json so the arm and eval_offline.py crop alike.
The augmentation type "LightingGradient" (uneven light, see below) is added to LeRobot's ColorJitter/RandomAffine/....
The wrist camera gets only the photometric transforms (see COLOR_ONLY_CAMERAS); LeRobot applies one set to every camera.
With POLICY_TYPE=pi05 (train.py picks it from the base policy) the PaliGemma tokenizer comes from
inference/pi05_inputs.py, and pi05_base's third camera slot, which we have no camera for, is dropped (see make_policy).
Arguments are those of lerobot.scripts.lerobot_train; train.py starts this script instead of it.
"""

import dataclasses
import functools
import json
import math
import os
import shutil
import sys
from pathlib import Path

import accelerate.utils
import torch
from lerobot.datasets import transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.scripts import lerobot_train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cube_boxes  # noqa: E402
import pi05_inputs  # noqa: E402
import top_crop  # noqa: E402

SMOLVLA_IMAGE_SIZE = (512, 512)  # SmolVLAConfig.resize_imgs_with_padding of smolvla_base
PI05 = "pi05"  # LeRobot's name of the policy type
POLICY_TYPE = os.environ.get("POLICY_TYPE", "smolvla")
TARGET_BOX = os.environ.get("TARGET_BOX") == "1"
TOP_CROP = os.environ.get("TOP_CROP") == "1"
# The wrist camera is fixed to the gripper: rotating or shifting its frame moves the fingers relative to the cube, which
# never happens on the arm. As in openpi (pi0), it gets only the transforms that change colour and light.
COLOR_ONLY_CAMERAS = {"observation.images.wrist"}
GEOMETRIC_TRANSFORM_TYPES = {"RandomAffine"}
SAMPLER_EPOCHS = 100_000  # more than any run gets through, so the sampler never runs out
# run_policy.py and eval_offline.py look for these next to the weights of a checkpoint.
TARGET_BOX_FILE = "target_box.json"
TOP_CROP_FILE = "top_crop.json"
PRECISION_FILE = "train_precision.txt"  # what train.py --resume reads to continue with the same --precision
PRECISION = "bf16" if os.environ.get("ACCELERATE_MIXED_PRECISION") == "bf16" else "fp32"
original_getitem = LeRobotDataset.__getitem__
original_query_videos = LeRobotDataset._query_videos
original_save_checkpoint = lerobot_train.save_checkpoint
original_make_transform = transforms.make_transform_from_config
original_make_policy = lerobot_train.make_policy
original_make_processors = lerobot_train.make_pre_post_processors


class LightingGradient(torch.nn.Module):
    """Uneven light: the frame is multiplied by a gain that ramps linearly from one random value in `gain` to another
    along a random direction, like a lamp on one side (our recordings) against even daylight."""

    def __init__(self, gain=(0.6, 1.4)):
        super().__init__()
        self.gain = tuple(gain)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """A ...xHxW frame in [0, 1] as a new tensor of the same shape, multiplied by the ramp and clamped to [0, 1]."""
        low, high = torch.empty(2).uniform_(*self.gain).tolist()
        angle = torch.empty(()).uniform_(0, 2 * math.pi).item()
        height, width = image.shape[-2:]
        ys = torch.linspace(-0.5, 0.5, height).view(height, 1)
        xs = torch.linspace(-0.5, 0.5, width).view(1, width)
        ramp = (xs * math.cos(angle) + ys * math.sin(angle)).clamp(-0.5, 0.5) + 0.5
        gain = low + (high - low) * ramp
        return (image * gain.to(image)).clamp(0, 1)


def make_transform(cfg):
    """LeRobot's make_transform_from_config, which also knows "LightingGradient"."""
    return LightingGradient(**cfg.kwargs) if cfg.type == "LightingGradient" else original_make_transform(cfg)


def target_boxes(dataset) -> dict[int, list[int]]:
    """{episode number: box of its target cube}, read once from the dataset's meta/cube_boxes.json."""
    if not hasattr(dataset, "_target_boxes"):
        labels = json.loads((dataset.root / "meta" / "cube_boxes.json").read_text())["episodes"]
        dataset._target_boxes = {int(episode): label["boxes"][label["target"]] for episode, label in labels.items()}
    return dataset._target_boxes


def query_videos(self, query_timestamps, ep_idx):
    """LeRobotDataset._query_videos: {camera: decoded frames}, with the target box drawn, then the top frame cropped."""
    item = original_query_videos(self, query_timestamps, ep_idx)
    if TARGET_BOX:
        cube_boxes.draw_chw(item[cube_boxes.STYLE["camera"]], target_boxes(self)[ep_idx])
    if TOP_CROP:
        camera = top_crop.CROP["camera"]
        item[camera] = top_crop.crop_chw(item[camera]).contiguous()
    return item


# lerobot_train calls the next three by keyword, so their parameter names must stay those of LeRobot's functions.
def save_checkpoint(checkpoint_dir, *args, **kwargs):
    """LeRobot's save_checkpoint, then the box, crop and precision are added and older optimizer states deleted."""
    original_save_checkpoint(checkpoint_dir, *args, **kwargs)
    # Only the newest checkpoint keeps its optimizer state (25 GB for pi0.5): --resume continues from the last one.
    for state_dir in Path(checkpoint_dir).parent.glob("*/training_state"):
        if state_dir.parent != Path(checkpoint_dir):
            shutil.rmtree(state_dir)
    model_dir = Path(checkpoint_dir) / "pretrained_model"
    (model_dir / PRECISION_FILE).write_text(PRECISION)
    if TARGET_BOX:
        (model_dir / TARGET_BOX_FILE).write_text(json.dumps(cube_boxes.STYLE, indent=1))
    if TOP_CROP:
        (model_dir / TOP_CROP_FILE).write_text(json.dumps(top_crop.CROP, indent=1))


def make_policy(cfg, *args, ds_meta=None, rename_map=None, **kwargs):
    """LeRobot's make_policy, with the camera slots of a pi0.5 config that the dataset has no camera for removed."""
    # LeRobot would feed pi05_base's camera slot that the dataset lacks as a masked image. Nothing attends to its 256
    # tokens and they shift no positions (position ids count unmasked tokens only), so dropping the slot leaves the
    # outputs as they are, while SigLIP and every layer stop processing a quarter of the sequence.
    if cfg.type == PI05:
        cameras = {(rename_map or {}).get(key, key) for key in ds_meta.camera_keys}
        empty_slots = [key for key in cfg.image_features if key not in cameras]
        for key in empty_slots:
            del cfg.input_features[key]
    return original_make_policy(cfg, *args, ds_meta=ds_meta, rename_map=rename_map, **kwargs)


def make_pre_post_processors(policy_cfg, *args, **kwargs):
    """LeRobot's make_pre_post_processors, with the tokenizer of inference/pi05_inputs.py for pi0.5."""
    if policy_cfg.type == PI05:
        overrides = kwargs.setdefault("preprocessor_overrides", {})
        overrides["tokenizer_processor"] = {"tokenizer": pi05_inputs.load_tokenizer()}
    return original_make_processors(policy_cfg, *args, **kwargs)


def fit_image(image: torch.Tensor) -> torch.Tensor:
    """A CHW frame in [0, 1] shrunk to the policy's input size."""
    if POLICY_TYPE == PI05:
        return pi05_inputs.resize(image[None])[0]
    return resize_with_pad(image[None], *SMOLVLA_IMAGE_SIZE, pad_value=0)[0]


def camera_transforms(dataset, image_transforms, key):
    """The dataset's image transforms for one camera, without the geometric ones on COLOR_ONLY_CAMERAS."""
    if key not in COLOR_ONLY_CAMERAS:
        return image_transforms
    if not hasattr(dataset, "_color_transforms"):
        cfg = image_transforms._cfg
        color_tfs = {name: tf_cfg for name, tf_cfg in cfg.tfs.items() if tf_cfg.type not in GEOMETRIC_TRANSFORM_TYPES}
        dataset._color_transforms = transforms.ImageTransforms(dataclasses.replace(cfg, tfs=color_tfs))
    return dataset._color_transforms


def getitem(self, idx):
    """LeRobotDataset.__getitem__: one sample, with each camera frame augmented for that camera and shrunk."""
    # LeRobot would run self.image_transforms on every camera inside __getitem__; run them here per camera instead.
    image_transforms, self.image_transforms = self.image_transforms, None
    try:
        item = original_getitem(self, idx)
    finally:
        self.image_transforms = image_transforms
    for key in self.meta.camera_keys:
        if image_transforms is not None:
            item[key] = camera_transforms(self, image_transforms, key)(item[key])
        item[key] = fit_image(item[key])
    return item


class PermutationChainSampler(torch.utils.data.Sampler):
    """Random permutations of the dataset back to back, drawn from the (seeded) global torch RNG."""

    def __init__(self, size, epochs=SAMPLER_EPOCHS):
        self.size, self.epochs = size, epochs

    def __len__(self):
        return self.size * self.epochs

    def __iter__(self):
        for _ in range(self.epochs):
            yield from torch.randperm(self.size).tolist()


class DataLoader(torch.utils.data.DataLoader):
    """torch's DataLoader, except that shuffling goes through PermutationChainSampler, without epoch boundaries."""

    def __init__(self, dataset, *args, shuffle=False, sampler=None, **kwargs):
        if shuffle and sampler is None:
            shuffle, sampler = False, PermutationChainSampler(len(dataset))
        super().__init__(dataset, *args, shuffle=shuffle, sampler=sampler, **kwargs)


LeRobotDataset.__getitem__ = getitem
torch.utils.data.DataLoader = DataLoader  # lerobot_train looks it up as torch.utils.data.DataLoader at call time
transforms.make_transform_from_config = make_transform  # ImageTransforms looks it up in its module at build time
# These three are imported into lerobot_train's namespace and called from there.
lerobot_train.make_policy = make_policy
lerobot_train.make_pre_post_processors = make_pre_post_processors
lerobot_train.save_checkpoint = save_checkpoint
# On several GPUs the gradients live in DDP's communication buckets instead of beside a copy of them there (13.5 GB for
# pi0.5 in fp32). lerobot_train imports this class inside main(), so it gets the patched one.
accelerate.utils.DistributedDataParallelKwargs = functools.partial(
    accelerate.utils.DistributedDataParallelKwargs, gradient_as_bucket_view=True
)
if TARGET_BOX or TOP_CROP:
    LeRobotDataset._query_videos = query_videos

if __name__ == "__main__":
    lerobot_train.main()
