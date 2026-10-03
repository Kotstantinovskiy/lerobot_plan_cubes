"""The choice of the target cube in training/label_boxes.py, without a dataset. Needs LeRobot, which the script uses."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
sys.path.insert(0, str(PROJECT / "inference"))

import cube_boxes  # noqa: E402
from label_boxes import find_moved_box, split_two_cubes  # noqa: E402

CUBE_ON_TABLE = [500, 450, 545, 495]
TWIN_ON_TABLE = [700, 400, 745, 445]
ON_CORK = [640, 320, 685, 365]
BASE_ON_CORK = [640, 320, 685, 365]
BASES_WITH_SAME_COLOR_TOP = [600, 310, 690, 370]
TWO_TOUCHING = [500, 450, 580, 490]
LEFT_CUBE_OF_TOUCHING = [500, 450, 540, 490]
RIGHT_CUBE_OF_TOUCHING = [540, 450, 580, 490]
CUBE_RGB = {"red": (230, 30, 30), "blue": (30, 30, 230), "green": (30, 200, 30), "yellow": (230, 220, 30)}


def make_frame(cubes):
    """A grey 1280x720 RGB frame with a filled box for each (colour, box)."""
    frame = np.full((720, 1280, 3), 128, np.uint8)
    for color, (x0, y0, x1, y1) in cubes:
        frame[y0:y1, x0:x1] = CUBE_RGB[color]
    return frame


class FindMovedBoxTests(unittest.TestCase):
    def test_single_cube_is_the_target(self):
        self.assertEqual(find_moved_box([CUBE_ON_TABLE], [ON_CORK]), (CUBE_ON_TABLE, True))

    def test_single_cube_that_stayed_is_still_the_target(self):
        self.assertEqual(find_moved_box([CUBE_ON_TABLE], [CUBE_ON_TABLE]), (CUBE_ON_TABLE, True))

    def test_of_two_cubes_the_one_that_moved_is_the_target(self):
        result = find_moved_box([TWIN_ON_TABLE, CUBE_ON_TABLE], [TWIN_ON_TABLE, ON_CORK])
        self.assertEqual(result, (CUBE_ON_TABLE, True))

    def test_the_first_listed_twin_that_stayed_is_not_the_target(self):
        result = find_moved_box([CUBE_ON_TABLE, TWIN_ON_TABLE], [CUBE_ON_TABLE, ON_CORK])
        self.assertEqual(result, (TWIN_ON_TABLE, True))

    def test_cube_placed_onto_its_twin_is_the_target_not_the_twin(self):
        target, is_certain = find_moved_box([BASE_ON_CORK, CUBE_ON_TABLE], [BASES_WITH_SAME_COLOR_TOP])
        self.assertEqual((target, is_certain), (CUBE_ON_TABLE, True))

    def test_two_cubes_gone_at_the_end_is_reported_as_uncertain(self):
        _, is_certain = find_moved_box([CUBE_ON_TABLE, TWIN_ON_TABLE], [ON_CORK])
        self.assertFalse(is_certain)

    def test_touching_cubes_give_the_one_that_left(self):
        target, is_certain = find_moved_box([TWO_TOUCHING], [RIGHT_CUBE_OF_TOUCHING, ON_CORK])
        self.assertEqual((target, is_certain), (LEFT_CUBE_OF_TOUCHING, True))

    def test_touching_cubes_without_one_left_nearby_are_uncertain(self):
        target, is_certain = find_moved_box([TWO_TOUCHING], [ON_CORK])
        self.assertEqual((target, is_certain), (TWO_TOUCHING, False))

    def test_split_mirrors_the_cube_that_stayed(self):
        self.assertEqual(split_two_cubes(TWO_TOUCHING, LEFT_CUBE_OF_TOUCHING), RIGHT_CUBE_OF_TOUCHING)


class DetectAllTests(unittest.TestCase):
    def test_two_cubes_of_a_colour_give_two_boxes_largest_first(self):
        small, large = [500, 450, 540, 490], [700, 400, 750, 450]
        found = cube_boxes.detect_all(make_frame([("red", small), ("red", large)]))
        self.assertEqual(found["red"], [large, small])

    def test_absent_colour_gives_an_empty_list(self):
        self.assertEqual(cube_boxes.detect_all(make_frame([("red", CUBE_ON_TABLE)]))["blue"], [])

    def test_touching_cubes_give_one_box_around_both(self):
        found = cube_boxes.detect_all(make_frame([("red", LEFT_CUBE_OF_TOUCHING), ("red", RIGHT_CUBE_OF_TOUCHING)]))
        self.assertEqual(found["red"], [TWO_TOUCHING])

    def test_detect_keeps_the_largest_cube_of_each_colour(self):
        small, large = [500, 450, 540, 490], [700, 400, 750, 450]
        found = cube_boxes.detect(make_frame([("red", small), ("red", large)]))
        self.assertEqual(found["red"], large)

    def test_detect_gives_none_for_an_absent_colour(self):
        self.assertIsNone(cube_boxes.detect(make_frame([("red", CUBE_ON_TABLE)]))["blue"])


if __name__ == "__main__":
    unittest.main()
