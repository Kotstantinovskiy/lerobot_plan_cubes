"""Finding cubes on the top camera frame and drawing the target box; all frames are synthetic."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cube_boxes  # noqa: E402
from inference_pyramid_helpers import SPLIT_TOLERANCE, box_centered_at, make_frame  # noqa: E402

FRAME_SHAPE = (720, 1280, 3)
TARGET_BOX = [100, 100, 140, 140]
WHITE = 255


class DetectCubesTests(unittest.TestCase):
    def test_single_cube_gives_one_box(self):
        found = cube_boxes.detect_cubes(make_frame({"red": (500, 400)}))
        self.assertEqual(found["red"], [box_centered_at(500, 400)])

    def test_two_separate_cubes_of_a_color_give_two_boxes(self):
        frame = make_frame({"red": (500, 400)}, [("red", (700, 400))])
        found = cube_boxes.detect_cubes(frame)
        self.assertCountEqual(found["red"], [box_centered_at(500, 400), box_centered_at(700, 400)])

    def test_touching_cubes_side_by_side_give_a_box_each(self):
        frame = make_frame({"red": (500, 400)}, [("red", (540, 400))])
        found = cube_boxes.detect_cubes(frame)
        expected = [box_centered_at(500, 400), box_centered_at(540, 400)]
        np.testing.assert_allclose(sorted(found["red"]), expected, atol=SPLIT_TOLERANCE)

    def test_touching_cubes_one_above_the_other_give_a_box_each(self):
        frame = make_frame({"blue": (500, 400)}, [("blue", (500, 440))])
        found = cube_boxes.detect_cubes(frame)
        expected = [box_centered_at(500, 400), box_centered_at(500, 440)]
        np.testing.assert_allclose(sorted(found["blue"]), expected, atol=SPLIT_TOLERANCE)

    def test_detect_all_keeps_touching_cubes_as_one_box_for_labelling(self):
        frame = make_frame({"red": (500, 400)}, [("red", (540, 400))])
        self.assertEqual(cube_boxes.detect_all(frame)["red"], [[480, 380, 560, 420]])


class CubeShapeTests(unittest.TestCase):
    def test_cube_sized_filled_square_has_the_shape(self):
        self.assertTrue(cube_boxes.has_cube_shape(40, 40, 1600))

    def test_too_small_a_blob_does_not(self):
        self.assertFalse(cube_boxes.has_cube_shape(10, 10, 100))

    def test_too_large_a_blob_does_not(self):
        self.assertFalse(cube_boxes.has_cube_shape(100, 100, 9000))

    def test_long_thin_blob_does_not(self):
        self.assertFalse(cube_boxes.has_cube_shape(90, 10, 900))

    def test_blob_covering_little_of_its_box_does_not(self):
        self.assertFalse(cube_boxes.has_cube_shape(60, 60, 1000))


class TargetBoxDrawingTests(unittest.TestCase):
    def test_box_grows_by_the_margin(self):
        self.assertEqual(cube_boxes.expand([100, 100, 140, 140], (720, 1280), 8), (92, 92, 148, 148))

    def test_box_stays_inside_the_frame(self):
        self.assertEqual(cube_boxes.expand([2, 2, 1278, 718], (720, 1280), 8), (0, 0, 1280, 720))

    def test_outline_is_drawn_outside_the_cube(self):
        frame = np.zeros(FRAME_SHAPE, np.uint8)
        cube_boxes.draw_rgb(frame, TARGET_BOX)
        self.assertEqual(frame[92, 120].tolist(), [WHITE, WHITE, WHITE])

    def test_inside_of_the_box_stays_as_it_was(self):
        frame = np.zeros(FRAME_SHAPE, np.uint8)
        cube_boxes.draw_rgb(frame, TARGET_BOX)
        self.assertEqual(frame[120, 120].tolist(), [0, 0, 0])

    def test_tensor_gets_the_same_box_as_the_camera_frame(self):
        frame = np.zeros(FRAME_SHAPE, np.uint8)
        camera_view = cube_boxes.draw_rgb(frame, TARGET_BOX).astype(np.float32) / WHITE
        tensor_view = cube_boxes.draw_chw(torch.zeros(3, 720, 1280), TARGET_BOX).permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(camera_view, tensor_view)


if __name__ == "__main__":
    unittest.main()
