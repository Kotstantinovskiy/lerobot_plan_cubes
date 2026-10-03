"""Find the four coloured cubes on the top camera and mark the target one with a box.

Shared by dataset labelling (training/label_boxes.py), training (training/lerobot_train_fast.py) and the arm
(inference/run_policy.py), so the box a policy is trained with is drawn the same way when it runs.
Coordinates are pixels of the 1280x720 top camera frame: (x0, y0, x1, y1), inclusive-exclusive.
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch  # only draw_chw meets a tensor; the pyramid scripts run without loading torch

COLORS = ("red", "blue", "green", "yellow")
TOP_FRAME_SHAPE = (720, 1280)  # height, width of the frame all of this is tuned for
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
MIN_CUBE_AREA, MAX_CUBE_AREA = 300, 4000  # pixels of the colour mask; a cube is ~35x35 px from the top camera
MIN_BOX_PROPORTION, MAX_BOX_PROPORTION = 0.4, 2.5  # width / height of a blob's box
MIN_BOX_FILL = 0.4  # the share of its box a blob covers
OPENING_KERNEL = np.ones((3, 3), np.uint8)
CLOSING_KERNEL = np.ones((7, 7), np.uint8)
# Pixels. On 94 frames of pyramid episodes one cube reaches 72 (with a lit patch of table), two touching start at 78.
TWO_CUBES_SIDE = 75
TOUCHING_CUBES = 2  # the groups a blob too long for one cube is split into
KMEANS_STOP = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
KMEANS_ATTEMPTS = 3

# How the target is marked. A checkpoint trained with boxes stores this next to its weights (target_box.json).
STYLE = {"camera": "observation.images.top", "rgb": [255, 255, 255], "thickness": 6, "margin": 8}


def color_masks(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """{colour: uint8 mask of its pixels on the table} for an HxWx3 uint8 RGB top camera frame."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    allowed = np.zeros(hsv.shape[:2], np.uint8)
    x0, y0, x1, y1 = TABLE
    allowed[y0:y1, x0:x1] = 1
    x0, y0, x1, y1 = ARM
    allowed[y0:y1, x0:x1] = 0
    masks = {}
    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cv2.morphologyEx(mask & (allowed * 255), cv2.MORPH_OPEN, OPENING_KERNEL)
        masks[color] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSING_KERNEL)
    return masks


def has_cube_shape(width: int, height: int, area: int) -> bool:
    """Whether a blob with this box (pixels) and this many mask pixels can be a cube."""
    has_cube_area = MIN_CUBE_AREA <= area <= MAX_CUBE_AREA
    has_cube_proportion = MIN_BOX_PROPORTION <= width / height <= MAX_BOX_PROPORTION
    fills_its_box = area >= MIN_BOX_FILL * width * height
    return has_cube_area and has_cube_proportion and fills_its_box


def find_blobs(mask: np.ndarray) -> list[list[int]]:
    """Boxes of the cube-sized blobs of a colour mask, largest first."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for x, y, width, height, area in stats[1:count]:
        if has_cube_shape(width, height, area):
            found.append((int(area), [int(x), int(y), int(x + width), int(y + height)]))
    found.sort(key=lambda pair: pair[0], reverse=True)
    return [box for _, box in found]


def detect_all(image_rgb: np.ndarray) -> dict[str, list[list[int]]]:
    """{colour: [box, ...], largest first} for an HxWx3 uint8 RGB top camera frame.

    Several cubes of a colour give several boxes. Two touching ones give a single box around both.
    """
    return {color: find_blobs(mask) for color, mask in color_masks(image_rgb).items()}


def split_blob(mask: np.ndarray, box: list[int]) -> list[list[int]]:
    """Two boxes from the box around two touching cubes: the mask pixels inside it clustered into two groups."""
    x0, y0, x1, y1 = box
    rows, columns = np.nonzero(mask[y0:y1, x0:x1])
    points = np.float32(np.column_stack((columns + x0, rows + y0)))
    _, groups, _ = cv2.kmeans(
        points,
        TOUCHING_CUBES,
        None,
        KMEANS_STOP,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )
    boxes = []
    for group in range(TOUCHING_CUBES):
        x, y, width, height = cv2.boundingRect(points[groups.ravel() == group])
        boxes.append([x, y, x + width, y + height])
    return boxes


def detect_cubes(image_rgb: np.ndarray) -> dict[str, list[list[int]]]:
    """{colour: [box, ...]} with one box per cube for an HxWx3 uint8 RGB top camera frame.

    Same as detect_all, except that a box too long for one cube is taken for two touching ones and split in two.
    """
    cubes = {}
    for color, mask in color_masks(image_rgb).items():
        cubes[color] = []
        for box in find_blobs(mask):
            is_two_cubes = max(box[2] - box[0], box[3] - box[1]) > TWO_CUBES_SIDE
            if is_two_cubes:
                cubes[color] += split_blob(mask, box)
            else:
                cubes[color].append(box)
    return cubes


def detect(image_rgb: np.ndarray) -> dict[str, list[int] | None]:
    """{colour: box of its largest cube, or None} for an HxWx3 uint8 RGB top camera frame."""
    return {color: found[0] if found else None for color, found in detect_all(image_rgb).items()}


def expand(box: list[int], shape: tuple[int, int], margin: int = STYLE["margin"]) -> tuple[int, int, int, int]:
    """The box grown by `margin` pixels on each side and kept inside a frame of `shape` (height, width)."""
    height, width = shape
    x0, y0, x1, y1 = box
    return max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin)


def draw_rgb(image_rgb: np.ndarray, box: list[int], style: dict = STYLE) -> np.ndarray:
    """Draw the target box on an HxWx3 uint8 RGB frame in place, and return the frame."""
    x0, y0, x1, y1 = expand(box, image_rgb.shape[:2], style["margin"])
    thickness = style["thickness"]
    color = np.array(style["rgb"], np.uint8)
    image_rgb[y0 : y0 + thickness, x0:x1] = color
    image_rgb[y1 - thickness : y1, x0:x1] = color
    image_rgb[y0:y1, x0 : x0 + thickness] = color
    image_rgb[y0:y1, x1 - thickness : x1] = color
    return image_rgb


def draw_chw(image: "torch.Tensor", box: list[int], style: dict = STYLE) -> "torch.Tensor":
    """Same box on a 3xHxW float frame in [0, 1] (LeRobot dataset format), in place, and return the frame."""
    x0, y0, x1, y1 = expand(box, image.shape[1:], style["margin"])
    thickness = style["thickness"]
    color = image.new_tensor(style["rgb"]).view(3, 1, 1) / 255
    image[:, y0 : y0 + thickness, x0:x1] = color
    image[:, y1 - thickness : y1, x0:x1] = color
    image[:, y0:y1, x0 : x0 + thickness] = color
    image[:, y0:y1, x1 - thickness : x1] = color
    return image
