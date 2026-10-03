"""Straighten the left part of the table out of the top camera frame: nothing but the table and what stands on it.

In the recordings the leader arm and the operator's body are visible around the table, and they start moving before
the follower does. A policy can learn to wait for that motion, which never comes when it drives the arm alone. Keeping
only the table removes them, though not the operator's shadow on it. Shared by training
(training/lerobot_train_fast.py), the arm (inference/run_policy.py) and offline checks (training/eval_offline.py), so a
policy always sees the same view. A checkpoint trained on the crop stores it next to its weights (top_crop.json); older
ones hold a plain "box".
"quad" is the area in pixels of the 1280x720 top camera frame: top-left, top-right, bottom-right, bottom-left corner.
It follows the table's far, left and near edges, which lean by about 3 degrees in the frame, and "size" is its width
and height once straightened. The cubes of all 155 episodes lie within x 414-840, y 260-578, so its right side passes
right of the table's middle (about 59% of its width), and left of the camera's clamp at x 860.
"""

import json
from pathlib import Path

import cv2
import numpy as np

CROP = {
    "camera": "observation.images.top",
    "quad": [[382, 131], [838, 99], [858, 572], [399, 597]],
    "size": [460, 470],
}
OUTLINE_BGR_OR_RGB = (128, 128, 128)  # grey reads the same in either channel order
OUTLINE_THICKNESS = 2

# OpenCV's thread pool can hang in forked data loader workers, and 64 workers with a pool each would oversubscribe.
cv2.setNumThreads(1)


def load(policy_path):
    """The crop a policy was trained with, else None."""
    path = Path(policy_path) / "top_crop.json"
    return json.loads(path.read_text()) if path.is_file() else None


def straighten(frame, crop=CROP):
    """The quad of an HxWxC frame as a new array of size["height"] x size["width"] x C, same data type."""
    width, height = crop["size"]
    view_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(np.float32(crop["quad"]), view_corners)
    return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)


def crop_hwc(image, crop=CROP):
    """The view of an HxWx3 frame (camera or video frame)."""
    if "box" in crop:
        x0, y0, x1, y1 = crop["box"]
        return image[y0:y1, x0:x1]
    return straighten(image, crop)


def crop_chw(image, crop=CROP):
    """The view of a ...x3xHxW tensor (LeRobot dataset format, with or without a batch dimension)."""
    if "box" in crop:
        x0, y0, x1, y1 = crop["box"]
        return image[..., y0:y1, x0:x1]
    frames = image.reshape(-1, *image.shape[-3:]).permute(0, 2, 3, 1).cpu().numpy()
    views = np.stack([straighten(frame, crop) for frame in frames]).transpose(0, 3, 1, 2)
    return image.new_tensor(views).reshape(*image.shape[:-3], *views.shape[1:])


def draw_outline(image, crop=CROP):
    """Draw the cropped area on an HxWx3 frame in place, for the videos of policy runs."""
    if "box" in crop:
        x0, y0, x1, y1 = crop["box"]
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), OUTLINE_BGR_OR_RGB, OUTLINE_THICKNESS)
        return
    cv2.polylines(image, [np.int32(crop["quad"])], True, OUTLINE_BGR_OR_RGB, OUTLINE_THICKNESS)
