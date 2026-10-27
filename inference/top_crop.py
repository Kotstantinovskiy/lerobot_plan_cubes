"""Crop the top camera to the left part of the table: the white arm, the platform and the cubes.

In every recording the black leader arm and the operator's hand are visible in the right part of the top frame, and in
all 40 episodes they start moving 0.05-3 s before the follower. A policy can learn to wait for that motion, which never
comes when it drives the arm alone. Cutting them out removes the cue. Shared by training (training/lerobot_train_fast.py),
the arm (inference/run_policy.py) and offline checks (training/eval_offline.py), so a policy always sees the same view.
A checkpoint trained on the crop stores it next to its weights (top_crop.json).
Coordinates are pixels of the 1280x720 top camera frame: (x0, y0, x1, y1), inclusive-exclusive. The cubes of all 40
episodes lie within x 440-792, y 356-533; the operator's hand never comes left of x ~900.
"""

import json
from pathlib import Path

CROP = {"camera": "observation.images.top", "box": [360, 0, 880, 600]}


def load(policy_path):
    """The crop a policy was trained with, else None."""
    path = Path(policy_path) / "top_crop.json"
    return json.loads(path.read_text()) if path.is_file() else None


def crop_hwc(image, crop=CROP):
    """View of an HxWx3 frame (camera or video frame)."""
    x0, y0, x1, y1 = crop["box"]
    return image[y0:y1, x0:x1]


def crop_chw(image, crop=CROP):
    """View of a ...x3xHxW frame (LeRobot dataset format, with or without a batch dimension)."""
    x0, y0, x1, y1 = crop["box"]
    return image[..., y0:y1, x0:x1]
