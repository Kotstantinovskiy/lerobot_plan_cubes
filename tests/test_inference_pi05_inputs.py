"""Frames shrunk as openpi shrinks them for pi0.5, with black bars. The tokenizer's download is not exercised."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import pi05_inputs  # noqa: E402

SIZE = pi05_inputs.IMAGE_SIZE


class ResizeTests(unittest.TestCase):
    def test_frame_comes_out_square_with_its_batch(self):
        self.assertEqual(pi05_inputs.resize(torch.ones(2, 3, 720, 1280)).shape, (2, 3, SIZE, SIZE))

    def test_wide_frame_gets_black_bars_above_and_below(self):
        self.assertEqual(pi05_inputs.resize(torch.ones(1, 3, 720, 1280))[0, 0, 0, SIZE // 2], 0.0)

    def test_picture_stays_in_the_middle(self):
        self.assertEqual(pi05_inputs.resize(torch.ones(1, 3, 720, 1280))[0, 0, SIZE // 2, SIZE // 2], 1.0)

    def test_tall_frame_gets_black_bars_left_and_right(self):
        self.assertEqual(pi05_inputs.resize(torch.ones(1, 3, 1280, 720))[0, 0, SIZE // 2, 0], 0.0)


class ShrinkCamerasTests(unittest.TestCase):
    FRAME = {"observation.images.top": np.full((720, 1280, 3), 255, np.uint8), "observation.state": np.zeros(6)}

    def test_camera_comes_out_as_square_uint8(self):
        image = pi05_inputs.shrink_cameras(self.FRAME)["observation.images.top"]
        self.assertEqual((image.shape, image.dtype), ((SIZE, SIZE, 3), np.uint8))

    def test_white_stays_white(self):
        self.assertEqual(pi05_inputs.shrink_cameras(self.FRAME)["observation.images.top"][SIZE // 2, SIZE // 2, 0], 255)

    def test_other_features_are_left_as_they_are(self):
        state = pi05_inputs.shrink_cameras(self.FRAME)["observation.state"]
        self.assertIs(state, self.FRAME["observation.state"])

    def test_given_frame_stays_as_it_was(self):
        pi05_inputs.shrink_cameras(self.FRAME)
        self.assertEqual(self.FRAME["observation.images.top"].shape, (720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
