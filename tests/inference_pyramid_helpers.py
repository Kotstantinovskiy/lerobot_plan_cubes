"""What the pyramid tests share: slot boxes, table states and synthetic top camera frames. Holds no tests itself."""

import json
import sys
from pathlib import Path

import numpy as np

INFERENCE = Path(__file__).resolve().parents[1] / "inference"
sys.path.insert(0, str(INFERENCE))

from pyramid_plan import (  # noqa: E402
    FINISH,
    NO_CHOICE,
    PICK_AND_PLACE,
    Cube,
    PlannedStep,
    TableState,
)

SLOT_BOXES = {
    "base_left": [560, 300, 620, 360],
    "base_right": [620, 300, 680, 360],
    "top": [600, 300, 640, 360],
}
COLORS_BY_SLOT = {"base_left": "red", "base_right": "blue", "top": "yellow"}
CUBE_SIDE = 40
SPLIT_TOLERANCE = 2  # pixels; where clustering puts the border between two touching cubes
CUBE_RGB = {"red": (230, 30, 30), "blue": (30, 30, 230), "green": (30, 200, 30), "yellow": (230, 220, 30)}
FREE_ALL = ["red", "blue", "green", "yellow"]
EMPTY_SLOTS = {"base_left": None, "base_right": None, "top": None}
PLACE_RED_LEFT = PlannedStep(PICK_AND_PLACE, "1", "base_left", "red first")
FINISHED = PlannedStep(FINISH, NO_CHOICE, NO_CHOICE, "done")


def box_centered_at(center_x, center_y):
    half = CUBE_SIDE // 2
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def make_state(free_colors, **slots):
    """A state whose free cubes have the ids 1, 2, ... in the order of `free_colors`."""
    free_cubes = [
        Cube(str(number), color, box_centered_at(400 + 60 * number, 500))
        for number, color in enumerate(free_colors, start=1)
    ]
    return TableState(free_cubes, {**EMPTY_SLOTS, **slots}, [])


def make_frame(cube_centers, more_cubes=()):
    """A grey 1280x720 RGB frame with square cubes at {colour: (x, y)}, then each (colour, (x, y)) of `more_cubes`."""
    frame = np.full((720, 1280, 3), 128, np.uint8)
    for color, center in [*cube_centers.items(), *more_cubes]:
        x0, y0, x1, y1 = box_centered_at(*center)
        frame[y0:y1, x0:x1] = CUBE_RGB[color]
    return frame


def completion(step_fields):
    return {"choices": [{"message": {"content": json.dumps(step_fields)}}]}


def step_lines(printed):
    return [line.strip() for line in printed.splitlines() if line.strip().startswith("step:")]
