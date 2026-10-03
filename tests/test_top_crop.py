import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import top_crop  # noqa: E402

FRAME_HEIGHT, FRAME_WIDTH = 720, 1280
VIEW_WIDTH, VIEW_HEIGHT = top_crop.CROP["size"]
OLD_CROP = {"camera": "observation.images.top", "box": [360, 0, 880, 600]}
ONE_LEVEL = 1 / 255
MARK_RADIUS = 8


def make_gradient_frame():
    """A 720x1280x3 uint8 frame whose every pixel differs from its neighbours."""
    rows, columns = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    return np.stack([columns % 256, rows % 256, (columns + rows) % 256], axis=2).astype(np.uint8)


def make_frame_with_mark(x, y):
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    frame[y - MARK_RADIUS : y + MARK_RADIUS, x - MARK_RADIUS : x + MARK_RADIUS] = 255
    return frame


def to_chw(frame):
    return torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255


class StraightenedViewTests(unittest.TestCase):
    def test_view_has_the_configured_size(self):
        view = top_crop.crop_hwc(make_gradient_frame())
        self.assertEqual(view.shape, (VIEW_HEIGHT, VIEW_WIDTH, 3))

    def test_quad_corner_lands_on_the_views_corner(self):
        x, y = top_crop.CROP["quad"][0]
        view = top_crop.crop_hwc(make_frame_with_mark(x, y))
        self.assertGreater(view[2, 2].min(), 200)

    def test_opposite_quad_corner_lands_on_the_opposite_corner(self):
        x, y = top_crop.CROP["quad"][2]
        view = top_crop.crop_hwc(make_frame_with_mark(x, y))
        self.assertGreater(view[VIEW_HEIGHT - 3, VIEW_WIDTH - 3].min(), 200)

    def test_point_outside_the_quad_is_not_in_the_view(self):
        view = top_crop.crop_hwc(make_frame_with_mark(100, 100))
        self.assertEqual(view.max(), 0)

    def test_tensor_view_matches_the_camera_view(self):
        frame = make_gradient_frame()
        camera_view = top_crop.crop_hwc(frame).astype(np.float32) / 255
        tensor_view = top_crop.crop_chw(to_chw(frame)).permute(1, 2, 0).numpy()
        self.assertLessEqual(np.abs(camera_view - tensor_view).max(), ONE_LEVEL + 1e-6)

    def test_tensor_view_keeps_the_leading_dimensions(self):
        batch = to_chw(make_gradient_frame()).expand(2, 3, -1, -1, -1)
        self.assertEqual(top_crop.crop_chw(batch).shape, (2, 3, 3, VIEW_HEIGHT, VIEW_WIDTH))

    def test_tensor_view_keeps_the_data_type(self):
        self.assertEqual(top_crop.crop_chw(to_chw(make_gradient_frame())).dtype, torch.float32)

    def test_outline_marks_the_quad(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        top_crop.draw_outline(frame)
        x, y = top_crop.CROP["quad"][0]
        self.assertGreater(frame[y, x].min(), 0)


class OlderCheckpointTests(unittest.TestCase):
    def test_box_crop_of_a_frame_still_slices(self):
        self.assertEqual(top_crop.crop_hwc(make_gradient_frame(), OLD_CROP).shape, (600, 520, 3))

    def test_box_crop_of_a_tensor_still_slices(self):
        self.assertEqual(top_crop.crop_chw(to_chw(make_gradient_frame()), OLD_CROP).shape, (3, 600, 520))

    def test_box_outline_marks_the_box(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        top_crop.draw_outline(frame, OLD_CROP)
        self.assertGreater(frame[300, 360].min(), 0)


class SavedCropTests(unittest.TestCase):
    def test_crop_saved_next_to_the_weights_gives_the_same_view(self):
        frame = make_gradient_frame()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "top_crop.json").write_text(json.dumps(top_crop.CROP))
            loaded = top_crop.load(directory)
        self.assertTrue(np.array_equal(top_crop.crop_hwc(frame, loaded), top_crop.crop_hwc(frame)))

    def test_checkpoint_without_a_crop_gets_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(top_crop.load(directory))


if __name__ == "__main__":
    unittest.main()
