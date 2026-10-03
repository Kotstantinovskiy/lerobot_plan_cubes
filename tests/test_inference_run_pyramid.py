"""run_pyramid.py as a user runs it, as a separate process, on frames of a pyramid growing step by step."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from inference_pyramid_helpers import INFERENCE, SLOT_BOXES, make_frame, step_lines


class ScriptTests(unittest.TestCase):
    """run_pyramid.py as a user runs it, on frames of a pyramid growing step by step."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.slot_boxes = self.directory / "slots.json"
        self.slot_boxes.write_text(json.dumps(SLOT_BOXES))

    def write_frame(self, name, cube_centers, more_cubes=()):
        path = self.directory / name
        cv2.imwrite(str(path), cv2.cvtColor(make_frame(cube_centers, more_cubes), cv2.COLOR_RGB2BGR))
        return str(path)

    def run_script(self, *arguments, env=None):
        command = [sys.executable, str(INFERENCE / "run_pyramid.py"), *arguments]
        return subprocess.run(command, capture_output=True, text=True, env=env)

    def test_loop_planner_builds_the_pyramid_over_frames(self):
        table = {"red": (450, 500), "blue": (600, 500), "yellow": (750, 500)}
        frames = [
            self.write_frame("0.png", table),
            self.write_frame("1.png", {**table, "red": (590, 330)}),
            self.write_frame("2.png", {**table, "red": (590, 330), "blue": (650, 330)}),
            self.write_frame("3.png", {"red": (590, 330), "blue": (650, 330), "yellow": (620, 330)}),
        ]
        result = self.run_script("--planner", "loop", "--frame", *frames, "--slot-boxes", str(self.slot_boxes))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            step_lines(result.stdout),
            [
                "step: cube 1 (red) to base_left",
                "step: cube 1 (blue) to base_right",
                "step: cube 1 (yellow) to top",
                "step: finish",
            ],
        )

    def test_loop_planner_builds_a_pyramid_with_a_pair_of_one_color(self):
        left_red, right_red = (590, 330), (650, 330)
        frames = [
            self.write_frame("0.png", {"red": (450, 500), "blue": (600, 500)}, [("red", (750, 500))]),
            self.write_frame("1.png", {"red": left_red, "blue": (600, 500)}, [("red", (750, 500))]),
            self.write_frame("2.png", {"red": left_red, "blue": (600, 500)}, [("red", right_red)]),
            self.write_frame("3.png", {"red": left_red}, [("red", right_red), ("blue", (620, 330))]),
        ]
        result = self.run_script(
            "--planner",
            "loop",
            "--colors",
            "red,red,blue",
            "--frame",
            *frames,
            "--slot-boxes",
            str(self.slot_boxes),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            step_lines(result.stdout),
            [
                "step: cube 1 (red) to base_left",
                "step: cube 2 (red) to base_right",
                "step: cube 1 (blue) to top",
                "step: finish",
            ],
        )

    def test_llm_planner_without_a_key_stops_with_the_reason(self):
        frame = self.write_frame("0.png", {"red": (450, 500)})
        result = self.run_script("--planner", "llm", "--goal", "build", "--frame", frame, env={"PATH": ""})
        self.assertEqual(result.returncode, 1)
        self.assertIn("OPENROUTER_API_KEY is not set", result.stderr)

    def test_llm_planner_needs_a_goal(self):
        frame = self.write_frame("0.png", {"red": (450, 500)})
        result = self.run_script("--planner", "llm", "--frame", frame)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--goal", result.stderr)

    def test_frame_of_another_size_is_refused(self):
        path = self.directory / "small.png"
        cv2.imwrite(str(path), np.zeros((360, 640, 3), np.uint8))
        result = self.run_script("--frame", str(path))
        self.assertEqual(result.returncode, 1)
        self.assertIn("expected a 1280x720", result.stderr)


if __name__ == "__main__":
    unittest.main()
