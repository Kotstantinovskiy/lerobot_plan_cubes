#!/usr/bin/env python3
"""lerobot_train with camera frames shrunk inside the DataLoader workers, exactly as SmolVLA would shrink them.

SmolVLA's first step on every camera image is resize_with_pad to 512x512 (bilinear, zero padding). Applying that same
function in the workers, after augmentation, leaves the model inputs unchanged: on a 512x512 image it is the
identity. A sample then carries 6 MB of images instead of 36 MB for our 1080p + 720p frames, and the pin-memory
thread, which copied 2.3 GB per batch of 64 and held training at ~1.3 s per step, keeps up with the GPU.
Arguments are those of lerobot.scripts.lerobot_train; train.py starts this script instead of it.
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.scripts import lerobot_train

SIZE = (512, 512)  # SmolVLAConfig.resize_imgs_with_padding of smolvla_base
original_getitem = LeRobotDataset.__getitem__


def getitem(self, idx):
    item = original_getitem(self, idx)
    for key in self.meta.camera_keys:
        item[key] = resize_with_pad(item[key][None], *SIZE, pad_value=0)[0]
    return item


LeRobotDataset.__getitem__ = getitem

if __name__ == "__main__":
    lerobot_train.main()
