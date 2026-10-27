"""Find the four coloured cubes on the top camera and mark the target one with a box.

Shared by dataset labelling (training/label_boxes.py), training (training/lerobot_train_fast.py) and the arm
(inference/run_policy.py), so the box a policy is trained with is drawn the same way when it runs.
Coordinates are pixels of the 1280x720 top camera frame: (x0, y0, x1, y1), inclusive-exclusive.
"""

import cv2
import numpy as np

COLORS = ("red", "blue", "green", "yellow")
# OpenCV HSV (H 0-180). The table is orange wood: its hue overlaps red and yellow but it is darker and less
# saturated where it is lit, so the cubes are separated by saturation and value. Tuned on 40 episodes' first frames.
HSV_RANGES = {
    "red": [((0, 170, 90), (8, 255, 255)), ((170, 170, 90), (180, 255, 255))],
    "yellow": [((22, 150, 110), (36, 255, 255))],
    "green": [((45, 120, 60), (90, 255, 255))],
    "blue": [((100, 150, 60), (130, 255, 255))],
}
# Table area in the top frame; the arm's base and gripper at rest (with red parts) are cut out.
TABLE = (300, 180, 1200, 700)
ARM = (500, 0, 720, 280)
CUBE_AREA = (300, 4000)  # pixels of the colour mask; a cube is ~35x35 px from the top camera

# How the target is marked. A checkpoint trained with boxes stores this next to its weights (target_box.json).
STYLE = {"camera": "observation.images.top", "rgb": [255, 255, 255], "thickness": 6, "margin": 8}


def detect(image_rgb):
    """{colour: box or None} for an HxWx3 uint8 RGB top camera frame."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    allowed = np.zeros(hsv.shape[:2], np.uint8)
    x0, y0, x1, y1 = TABLE
    allowed[y0:y1, x0:x1] = 1
    x0, y0, x1, y1 = ARM
    allowed[y0:y1, x0:x1] = 0
    kernel = np.ones((3, 3), np.uint8)
    boxes = {}
    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask & (allowed * 255), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        best = None
        for x, y, w, h, area in stats[1:count]:
            if not CUBE_AREA[0] <= area <= CUBE_AREA[1] or not 0.4 <= w / h <= 2.5 or area < 0.4 * w * h:
                continue
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
        boxes[color] = None if best is None else [int(best[0]), int(best[1]),
                                                  int(best[0] + best[2]), int(best[1] + best[3])]
    return boxes


def expand(box, shape, margin=STYLE["margin"]):
    h, w = shape
    x0, y0, x1, y1 = box
    return max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)


def draw_rgb(image_rgb, box, style=STYLE):
    """Draw the target box on an HxWx3 uint8 RGB frame in place."""
    x0, y0, x1, y1 = expand(box, image_rgb.shape[:2], style["margin"])
    t = style["thickness"]
    color = np.array(style["rgb"], np.uint8)
    image_rgb[y0:y0 + t, x0:x1] = color
    image_rgb[y1 - t:y1, x0:x1] = color
    image_rgb[y0:y1, x0:x0 + t] = color
    image_rgb[y0:y1, x1 - t:x1] = color
    return image_rgb


def draw_chw(image, box, style=STYLE):
    """Same box on a 3xHxW float frame in [0, 1] (LeRobot dataset format), in place."""
    x0, y0, x1, y1 = expand(box, image.shape[1:], style["margin"])
    t = style["thickness"]
    color = image.new_tensor(style["rgb"]).view(3, 1, 1) / 255
    image[:, y0:y0 + t, x0:x1] = color
    image[:, y1 - t:y1, x0:x1] = color
    image[:, y0:y1, x0:x0 + t] = color
    image[:, y0:y1, x1 - t:x1] = color
    return image
