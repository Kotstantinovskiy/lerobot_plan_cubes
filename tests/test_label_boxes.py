"""Labelling in training/label_boxes.py: the target cube, the contact sheet. Needs LeRobot, as the script does."""

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
sys.path.insert(0, str(PROJECT / "inference"))

import cube_boxes  # noqa: E402
from label_boxes import (  # noqa: E402
    find_moved_box,
    label_episode,
    make_contact_sheet,
    make_tile,
    split_two_cubes,
)

CUBE_ON_TABLE = [500, 450, 545, 495]
TWIN_ON_TABLE = [700, 400, 745, 445]
ON_CORK = [640, 320, 685, 365]
BASE_ON_CORK = [640, 320, 685, 365]
BASES_WITH_SAME_COLOR_TOP = [600, 310, 690, 370]
TWO_TOUCHING = [500, 450, 580, 490]
LEFT_CUBE_OF_TOUCHING = [500, 450, 540, 490]
RIGHT_CUBE_OF_TOUCHING = [540, 450, 580, 490]
CUBE_RGB = {"red": (230, 30, 30), "blue": (30, 30, 230), "green": (30, 200, 30), "yellow": (230, 220, 30)}
FRAME_SHAPE = (720, 1280, 3)
GREY = 128
OTHER_CUBES = [("blue", [800, 350, 845, 395]), ("green", [900, 450, 945, 495]), ("yellow", [1000, 350, 1045, 395])]
RED_TASK = "pick the red cube"
COLOR_OF_TASK = {RED_TASK: "red"}
CAMERA = cube_boxes.STYLE["camera"]
LAST_FRAME = 59


def make_frame(cubes):
    """A grey 1280x720 RGB frame with a filled box for each (colour, box)."""
    frame = np.full(FRAME_SHAPE, GREY, np.uint8)
    for color, (x0, y0, x1, y1) in cubes:
        frame[y0:y1, x0:x1] = CUBE_RGB[color]
    return frame


class RedTaskEpisode:
    """A dataset of one episode with the red task; `scenes` is {first frame of a scene: its (colour, box) cubes}."""

    def __init__(self, scenes):
        self.scenes = scenes
        episode = {"dataset_from_index": 0, "length": LAST_FRAME + 1}
        self.meta = types.SimpleNamespace(episodes=[episode])

    def __getitem__(self, index):
        latest_start = max(start for start in self.scenes if start <= index)
        frame = torch.from_numpy(make_frame(self.scenes[latest_start])).permute(2, 0, 1).float() / 255
        return {CAMERA: frame, "task": RED_TASK}


def label_red_episode(scenes):
    return label_episode(RedTaskEpisode(scenes), 0, CAMERA, COLOR_OF_TASK, [0, 10, 20])


class FindMovedBoxTests(unittest.TestCase):
    def test_single_cube_is_the_target(self):
        self.assertEqual(find_moved_box([CUBE_ON_TABLE], [ON_CORK]), (CUBE_ON_TABLE, True))

    def test_largest_single_cube_is_not_taken_for_two(self):
        largest_cube = [500, 450, 572, 505]  # 72 pixels: the most one cube reached on 94 frames (cube_boxes.py)
        self.assertEqual(find_moved_box([largest_cube], [ON_CORK]), (largest_cube, True))

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


class LabelEpisodeTests(unittest.TestCase):
    def test_entry_holds_the_target_the_frame_and_all_boxes(self):
        entry, _, _ = label_red_episode({0: [("red", CUBE_ON_TABLE), *OTHER_CUBES]})
        expected_boxes = {"red": CUBE_ON_TABLE, **dict(OTHER_CUBES)}
        self.assertEqual(entry, {"target": "red", "frame": 0, "boxes": expected_boxes})

    def test_complete_detection_reports_nothing(self):
        _, problems, _ = label_red_episode({0: [("red", CUBE_ON_TABLE), *OTHER_CUBES]})
        self.assertEqual(problems, [])

    def test_hidden_target_is_taken_from_the_first_offset_that_shows_it(self):
        entry, _, _ = label_red_episode({0: OTHER_CUBES, 10: [("red", CUBE_ON_TABLE), *OTHER_CUBES]})
        self.assertEqual((entry["frame"], entry["boxes"]["red"]), (10, CUBE_ON_TABLE))

    def test_image_is_the_frame_the_boxes_come_from(self):
        _, _, image = label_red_episode({0: OTHER_CUBES, 10: [("red", CUBE_ON_TABLE), *OTHER_CUBES]})
        np.testing.assert_array_equal(image, make_frame([("red", CUBE_ON_TABLE), *OTHER_CUBES]))

    def test_target_never_seen_is_reported(self):
        entry, problems, _ = label_red_episode({0: OTHER_CUBES})
        self.assertEqual((entry["frame"], entry["boxes"]["red"]), (20, None))
        self.assertEqual(problems, ["эпизод 0: нет целевого red"])

    def test_missing_cube_of_another_colour_is_reported(self):
        _, problems, _ = label_red_episode({0: [("red", CUBE_ON_TABLE), *OTHER_CUBES[:2]]})
        self.assertEqual(problems, ["эпизод 0: не найдены yellow (цель red есть, кадр 0)"])

    def test_two_twins_gone_at_the_end_are_reported_as_uncertain(self):
        scenes = {
            0: [("red", CUBE_ON_TABLE), ("red", TWIN_ON_TABLE), *OTHER_CUBES],
            LAST_FRAME: [("red", ON_CORK), *OTHER_CUBES],
        }
        _, problems, _ = label_red_episode(scenes)
        self.assertEqual(problems, [f"эпизод 0: не ясно, какой red сдвинулся, взят {CUBE_ON_TABLE}: проверь глазами"])


class MakeTileTests(unittest.TestCase):
    table_x0, table_y0, table_x1, table_y1 = cube_boxes.TABLE

    def make_tile_of_red_cube(self):
        entry = {"target": "red", "frame": 0, "boxes": {"red": CUBE_ON_TABLE, "blue": None}}
        return make_tile(0, entry, make_frame([("red", CUBE_ON_TABLE)]))

    def test_tile_shows_the_table_area_at_half_size(self):
        table_height, table_width = self.table_y1 - self.table_y0, self.table_x1 - self.table_x0
        self.assertEqual(self.make_tile_of_red_cube().shape, (table_height // 2, table_width // 2, 3))

    def test_target_is_marked_with_the_white_box(self):
        band_y = CUBE_ON_TABLE[1] - cube_boxes.STYLE["margin"] + cube_boxes.STYLE["thickness"] // 2
        band_x = (CUBE_ON_TABLE[0] + CUBE_ON_TABLE[2]) // 2
        tile_pixel = self.make_tile_of_red_cube()[(band_y - self.table_y0) // 2, (band_x - self.table_x0) // 2]
        self.assertEqual(tile_pixel.tolist(), [255, 255, 255])

    def test_cube_keeps_its_colour_in_bgr(self):
        center_x, center_y = (CUBE_ON_TABLE[0] + CUBE_ON_TABLE[2]) // 2, (CUBE_ON_TABLE[1] + CUBE_ON_TABLE[3]) // 2
        tile_pixel = self.make_tile_of_red_cube()[(center_y - self.table_y0) // 2, (center_x - self.table_x0) // 2]
        self.assertEqual(tile_pixel.tolist(), list(reversed(CUBE_RGB["red"])))


class ContactSheetTests(unittest.TestCase):
    tile = np.full((2, 3, 3), 255, np.uint8)

    def test_full_rows_get_no_padding(self):
        self.assertEqual(make_contact_sheet([self.tile] * 10).shape, (4, 15, 3))

    def test_last_row_is_filled_up_to_the_full_width(self):
        self.assertEqual(make_contact_sheet([self.tile] * 6).shape, (4, 15, 3))

    def test_filled_up_tiles_are_black(self):
        sheet = make_contact_sheet([self.tile] * 6)
        self.assertEqual(sheet[2:, 3:].sum(), 0)

    def test_given_tiles_keep_their_place(self):
        sheet = make_contact_sheet([self.tile] * 6)
        np.testing.assert_array_equal(sheet[2:, :3], self.tile)


if __name__ == "__main__":
    unittest.main()
