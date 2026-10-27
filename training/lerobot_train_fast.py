#!/usr/bin/env python3
"""lerobot_train with two data-loading fixes that leave the model inputs and their distribution unchanged.

1. Camera frames are shrunk inside the DataLoader workers, exactly as SmolVLA would shrink them. SmolVLA's first step
   on every camera image is resize_with_pad to 512x512 (bilinear, zero padding). Applying that same function in the
   workers, after augmentation, gives identical inputs: on a 512x512 image it is the identity. A sample then carries
   6 MB of images instead of 36 MB for our 1080p + 720p frames, and the pin-memory thread, which copied 2.3 GB per
   batch of 64 and held training at ~1.3 s per step, keeps up with the GPU.
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
Arguments are those of lerobot.scripts.lerobot_train; train.py starts this script instead of it.
"""

import dataclasses
import json
import math
import os
from pathlib import Path
import sys

import torch

from lerobot.datasets import transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.scripts import lerobot_train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cube_boxes  # noqa: E402
import top_crop  # noqa: E402

SIZE = (512, 512)  # SmolVLAConfig.resize_imgs_with_padding of smolvla_base
TARGET_BOX = os.environ.get("TARGET_BOX") == "1"
TOP_CROP = os.environ.get("TOP_CROP") == "1"
# The wrist camera is fixed to the gripper: rotating or shifting its frame moves the fingers relative to the cube, which
# never happens on the arm. As in openpi (pi0), it gets only the transforms that change colour and light.
COLOR_ONLY_CAMERAS = {"observation.images.wrist"}
GEOMETRIC = {"RandomAffine"}
original_getitem = LeRobotDataset.__getitem__
original_query_videos = LeRobotDataset._query_videos
original_save_checkpoint = lerobot_train.save_checkpoint
original_make_transform = transforms.make_transform_from_config


class LightingGradient(torch.nn.Module):
    """Uneven light: the frame is multiplied by a gain that ramps linearly from one random value in `gain` to another
    along a random direction, like a lamp on one side (our recordings) against even daylight."""

    def __init__(self, gain=(0.6, 1.4)):
        super().__init__()
        self.gain = tuple(gain)

    def forward(self, image):
        low, high = torch.empty(2).uniform_(*self.gain).tolist()
        angle = torch.empty(()).uniform_(0, 2 * math.pi).item()
        h, w = image.shape[-2:]
        ys = torch.linspace(-0.5, 0.5, h).view(h, 1)
        xs = torch.linspace(-0.5, 0.5, w).view(1, w)
        ramp = (xs * math.cos(angle) + ys * math.sin(angle)).clamp(-0.5, 0.5) + 0.5
        return (image * (low + (high - low) * ramp).to(image)).clamp(0, 1)


def make_transform(cfg):
    return LightingGradient(**cfg.kwargs) if cfg.type == "LightingGradient" else original_make_transform(cfg)


def query_videos(self, query_timestamps, ep_idx):
    item = original_query_videos(self, query_timestamps, ep_idx)
    if TARGET_BOX:
        if not hasattr(self, "_target_boxes"):
            labels = json.loads((self.root / "meta" / "cube_boxes.json").read_text())["episodes"]
            self._target_boxes = {int(ep): label["boxes"][label["target"]] for ep, label in labels.items()}
        cube_boxes.draw_chw(item[cube_boxes.STYLE["camera"]], self._target_boxes[ep_idx])
    if TOP_CROP:
        camera = top_crop.CROP["camera"]
        item[camera] = top_crop.crop_chw(item[camera]).contiguous()
    return item


def save_checkpoint(checkpoint_dir, *args, **kwargs):
    original_save_checkpoint(checkpoint_dir, *args, **kwargs)
    model_dir = Path(checkpoint_dir) / "pretrained_model"
    if TARGET_BOX:
        (model_dir / "target_box.json").write_text(json.dumps(cube_boxes.STYLE, indent=1))
    if TOP_CROP:
        (model_dir / "top_crop.json").write_text(json.dumps(top_crop.CROP, indent=1))


def camera_transforms(self, image_transforms, key):
    """The dataset's image transforms for one camera, without the geometric ones on COLOR_ONLY_CAMERAS."""
    if key not in COLOR_ONLY_CAMERAS:
        return image_transforms
    if not hasattr(self, "_color_transforms"):
        cfg = image_transforms._cfg
        self._color_transforms = transforms.ImageTransforms(dataclasses.replace(
            cfg, tfs={name: tf for name, tf in cfg.tfs.items() if tf.type not in GEOMETRIC}))
    return self._color_transforms


def getitem(self, idx):
    # LeRobot would run self.image_transforms on every camera inside __getitem__; run them here per camera instead.
    image_transforms, self.image_transforms = self.image_transforms, None
    try:
        item = original_getitem(self, idx)
    finally:
        self.image_transforms = image_transforms
    for key in self.meta.camera_keys:
        if image_transforms is not None:
            item[key] = camera_transforms(self, image_transforms, key)(item[key])
        item[key] = resize_with_pad(item[key][None], *SIZE, pad_value=0)[0]
    return item


class PermutationChainSampler(torch.utils.data.Sampler):
    """Random permutations of the dataset back to back, drawn from the (seeded) global torch RNG."""

    def __init__(self, size, epochs=100_000):
        self.size, self.epochs = size, epochs

    def __len__(self):
        return self.size * self.epochs

    def __iter__(self):
        for _ in range(self.epochs):
            yield from torch.randperm(self.size).tolist()


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, *args, shuffle=False, sampler=None, **kwargs):
        if shuffle and sampler is None:
            shuffle, sampler = False, PermutationChainSampler(len(dataset))
        super().__init__(dataset, *args, shuffle=shuffle, sampler=sampler, **kwargs)


LeRobotDataset.__getitem__ = getitem
torch.utils.data.DataLoader = DataLoader  # lerobot_train looks it up as torch.utils.data.DataLoader at call time
transforms.make_transform_from_config = make_transform  # ImageTransforms looks it up in its module at build time
if TARGET_BOX or TOP_CROP:
    LeRobotDataset._query_videos = query_videos
    lerobot_train.save_checkpoint = save_checkpoint  # imported into lerobot_train's namespace, called from there

if __name__ == "__main__":
    lerobot_train.main()
