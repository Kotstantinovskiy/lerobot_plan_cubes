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
Arguments are those of lerobot.scripts.lerobot_train; train.py starts this script instead of it.
"""

import json
import os
from pathlib import Path
import sys

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.scripts import lerobot_train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cube_boxes  # noqa: E402

SIZE = (512, 512)  # SmolVLAConfig.resize_imgs_with_padding of smolvla_base
TARGET_BOX = os.environ.get("TARGET_BOX") == "1"
original_getitem = LeRobotDataset.__getitem__
original_query_videos = LeRobotDataset._query_videos
original_save_checkpoint = lerobot_train.save_checkpoint


def query_videos(self, query_timestamps, ep_idx):
    item = original_query_videos(self, query_timestamps, ep_idx)
    if not hasattr(self, "_target_boxes"):
        labels = json.loads((self.root / "meta" / "cube_boxes.json").read_text())["episodes"]
        self._target_boxes = {int(ep): label["boxes"][label["target"]] for ep, label in labels.items()}
    cube_boxes.draw_chw(item[cube_boxes.STYLE["camera"]], self._target_boxes[ep_idx])
    return item


def save_checkpoint(checkpoint_dir, *args, **kwargs):
    original_save_checkpoint(checkpoint_dir, *args, **kwargs)
    (Path(checkpoint_dir) / "pretrained_model" / "target_box.json").write_text(json.dumps(cube_boxes.STYLE, indent=1))


def getitem(self, idx):
    item = original_getitem(self, idx)
    for key in self.meta.camera_keys:
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
if TARGET_BOX:
    LeRobotDataset._query_videos = query_videos
    lerobot_train.save_checkpoint = save_checkpoint  # imported into lerobot_train's namespace, called from there

if __name__ == "__main__":
    lerobot_train.main()
