"""The language model planner: the picture and request it gets, the answers it may give, OpenRouter's transport.

Nothing here reaches the network: OpenRouter is replaced by a function or by a server on a local port.
"""

import base64
import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import llm_planner  # noqa: E402
import run_pyramid  # noqa: E402
import top_crop  # noqa: E402
from inference_pyramid_helpers import (  # noqa: E402
    FREE_ALL,
    PLACE_RED_LEFT,
    box_centered_at,
    completion,
    make_frame,
    make_state,
    step_lines,
)
from pyramid_plan import FINISH, NO_CHOICE, PICK_AND_PLACE, Cube  # noqa: E402


class CubeIdDrawingTests(unittest.TestCase):
    def setUp(self):
        self.box = box_centered_at(600, 400)
        self.cube = Cube("3", "red", self.box)
        self.frame = make_frame({"red": (600, 400)})

    def test_id_is_drawn_inside_the_box_of_its_cube(self):
        labelled = llm_planner.draw_cube_ids(self.frame, [self.cube])
        rows, columns = np.nonzero((labelled != self.frame).any(axis=2))
        self.assertGreater(len(rows), 0)
        x0, y0, x1, y1 = self.box
        self.assertTrue(x0 <= columns.min() and columns.max() < x1 and y0 <= rows.min() and rows.max() < y1)

    def test_drawing_ids_leaves_the_given_frame_as_it_was(self):
        original = self.frame.copy()
        llm_planner.draw_cube_ids(self.frame, [self.cube])
        np.testing.assert_array_equal(self.frame, original)

    def test_photo_with_ids_differs_from_the_photo_without_them(self):
        with_ids = llm_planner.encode_top_view(self.frame, [self.cube])
        without_ids = llm_planner.encode_top_view(self.frame, [])
        self.assertNotEqual(with_ids, without_ids)


class LlmRequestTests(unittest.TestCase):
    def setUp(self):
        self.jpeg = llm_planner.encode_top_view(make_frame({}), [])
        self.body = llm_planner.build_request_body("goal", make_state(["red", "red"]), [], self.jpeg, "some/model")

    def test_user_message_puts_the_text_before_the_image(self):
        kinds = [part["type"] for part in self.body["messages"][1]["content"]]
        self.assertEqual(kinds, ["text", "image_url"])

    def test_image_is_the_policy_crop_of_the_top_frame(self):
        url = self.body["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        jpeg = base64.b64decode(url.split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        width, height = top_crop.CROP["size"]
        self.assertEqual(image.shape[:2], (height, width))

    def test_answer_is_held_to_the_schema(self):
        self.assertEqual(
            self.body["response_format"]["json_schema"]["schema"]["required"], ["kind", "cube", "slot", "comment"]
        )

    def test_provider_must_not_collect_the_frame(self):
        self.assertEqual(self.body["provider"], {"require_parameters": True, "data_collection": "deny"})

    def test_scene_text_carries_goal_and_history(self):
        scene = json.loads(self.body["messages"][1]["content"][0]["text"])
        self.assertEqual(scene["goal"], "goal")
        self.assertEqual(scene["history"], [])

    def test_scene_text_gives_cube_ids_and_colors_without_boxes(self):
        scene = json.loads(self.body["messages"][1]["content"][0]["text"])
        self.assertEqual(scene["table"]["free_cubes"], [{"id": "1", "color": "red"}, {"id": "2", "color": "red"}])


class LlmAnswerTests(unittest.TestCase):
    def test_valid_answer_becomes_a_step(self):
        fields = {"kind": PICK_AND_PLACE, "cube": "1", "slot": "base_left", "comment": "red first"}
        self.assertEqual(llm_planner.parse_step(completion(fields)), PLACE_RED_LEFT)

    def test_cube_id_that_is_not_a_string_is_an_error(self):
        fields = {"kind": PICK_AND_PLACE, "cube": 1, "slot": "base_left", "comment": ""}
        with self.assertRaisesRegex(RuntimeError, "outside the schema"):
            llm_planner.parse_step(completion(fields))

    def test_answer_without_a_cube_is_an_error(self):
        fields = {"kind": PICK_AND_PLACE, "color": "red", "slot": "base_left", "comment": ""}
        with self.assertRaisesRegex(RuntimeError, "outside the schema"):
            llm_planner.parse_step(completion(fields))

    def test_answer_that_is_not_json_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "not JSON"):
            llm_planner.parse_step({"choices": [{"message": {"content": "sure, red first"}}]})

    def test_empty_answer_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "not JSON"):
            llm_planner.parse_step({"choices": [{"message": {"content": None}}]})

    def test_error_inside_a_normal_answer_is_raised(self):
        with self.assertRaisesRegex(RuntimeError, "provider down"):
            llm_planner.parse_step({"error": {"message": "provider down"}})

    def test_missing_api_key_says_what_to_do(self):
        with (
            patch.dict("os.environ", clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY is not set"),
        ):
            llm_planner.read_api_key()


class ChooseStepTests(unittest.TestCase):
    def test_rejected_step_goes_back_to_the_model_with_the_reason(self):
        bad_fields = {"kind": PICK_AND_PLACE, "cube": "4", "slot": "top", "comment": ""}
        good_fields = {"kind": PICK_AND_PLACE, "cube": "1", "slot": "base_left", "comment": "red first"}
        history = []
        with patch.object(
            llm_planner, "post_chat_completion", side_effect=[completion(bad_fields), completion(good_fields)]
        ) as post:
            step = llm_planner.choose_step("goal", make_frame({}), make_state(FREE_ALL), history, api_key="key")
        self.assertEqual(step, PLACE_RED_LEFT)
        second_scene = json.loads(post.call_args_list[1].args[0]["messages"][1]["content"][0]["text"])
        self.assertIn("both base cubes", second_scene["history"][0]["outcome"])

    def test_id_of_a_cube_that_is_not_there_goes_back_to_the_model(self):
        bad_fields = {"kind": PICK_AND_PLACE, "cube": "9", "slot": "base_left", "comment": ""}
        good_fields = {"kind": PICK_AND_PLACE, "cube": "1", "slot": "base_left", "comment": "red first"}
        history = []
        with patch.object(
            llm_planner, "post_chat_completion", side_effect=[completion(bad_fields), completion(good_fields)]
        ):
            step = llm_planner.choose_step("goal", make_frame({}), make_state(FREE_ALL), history, api_key="key")
        self.assertEqual(step, PLACE_RED_LEFT)
        self.assertIn("not free", history[0].outcome)

    def test_model_that_keeps_breaking_the_rules_is_stopped(self):
        bad_fields = {"kind": PICK_AND_PLACE, "cube": "4", "slot": "top", "comment": ""}
        with (
            patch.object(llm_planner, "post_chat_completion", return_value=completion(bad_fields)) as post,
            self.assertRaisesRegex(RuntimeError, "broke the rules"),
        ):
            llm_planner.choose_step("goal", make_frame({}), make_state(FREE_ALL), [], api_key="key")
        self.assertEqual(post.call_count, llm_planner.MAX_REJECTED_ANSWERS)


class FakeOpenRouter(BaseHTTPRequestHandler):
    """Answers like OpenRouter on a local port and keeps what it was sent; `status` decides the answer."""

    status = 200
    answer = {}
    seen_authorization = None
    seen_body = None

    def do_POST(self):
        FakeOpenRouter.seen_authorization = self.headers["Authorization"]
        FakeOpenRouter.seen_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps(FakeOpenRouter.answer).encode()
        self.send_response(FakeOpenRouter.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        """Keeps the test output clean."""


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        url = f"http://127.0.0.1:{self.server.server_port}/api/v1/chat/completions"
        url_patch = patch.object(llm_planner, "OPENROUTER_URL", url)
        url_patch.start()
        self.addCleanup(url_patch.stop)

    def run_llm_script(self, frame, goal):
        """(exit code, printed text) of run_pyramid.py with the llm planner on one frame."""
        with tempfile.TemporaryDirectory() as directory:
            frame_path = Path(directory) / "frame.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            arguments = ["run_pyramid.py", "--planner", "llm", "--goal", goal, "--frame", str(frame_path)]
            printed = io.StringIO()
            with (
                patch.object(sys, "argv", arguments),
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret-key"}),
                contextlib.redirect_stdout(printed),
            ):
                exit_code = run_pyramid.main()
        return exit_code, printed.getvalue()

    def test_request_reaches_the_server_with_the_key_and_the_body(self):
        FakeOpenRouter.status, FakeOpenRouter.answer = 200, {"ok": True}
        answer = llm_planner.post_chat_completion({"model": "some/model"}, "secret-key")
        self.assertEqual(answer, {"ok": True})
        self.assertEqual(FakeOpenRouter.seen_authorization, "Bearer secret-key")
        self.assertEqual(FakeOpenRouter.seen_body, {"model": "some/model"})

    def test_http_error_shows_the_server_message(self):
        FakeOpenRouter.status, FakeOpenRouter.answer = 401, {"error": {"message": "No auth credentials"}}
        with self.assertRaisesRegex(RuntimeError, "401.*No auth credentials"):
            llm_planner.post_chat_completion({}, "bad-key")

    def test_unreachable_server_is_a_clear_error(self):
        with (
            patch.object(llm_planner, "OPENROUTER_URL", "http://127.0.0.1:1/none"),
            self.assertRaisesRegex(RuntimeError, "Cannot reach OpenRouter"),
        ):
            llm_planner.post_chat_completion({}, "key")

    def test_script_with_the_llm_planner_prints_the_models_comment(self):
        fields = {"kind": FINISH, "cube": NO_CHOICE, "slot": NO_CHOICE, "comment": "Purple is not on the table"}
        FakeOpenRouter.status, FakeOpenRouter.answer = 200, completion(fields)
        exit_code, printed = self.run_llm_script(make_frame({"red": (450, 500)}), "Purple on top")
        self.assertEqual(exit_code, 0)
        self.assertIn("comment: Purple is not on the table", printed)
        self.assertEqual(FakeOpenRouter.seen_body["model"], llm_planner.DEFAULT_MODEL)

    def test_script_prints_the_exact_cube_the_model_chose_among_two_of_a_color(self):
        fields = {"kind": PICK_AND_PLACE, "cube": "2", "slot": "base_left", "comment": "The right red cube"}
        FakeOpenRouter.status, FakeOpenRouter.answer = 200, completion(fields)
        frame = make_frame({"red": (450, 500)}, [("red", (750, 500))])
        exit_code, printed = self.run_llm_script(frame, "Red on the left")
        self.assertEqual(exit_code, 0)
        self.assertEqual(step_lines(printed), ["step: cube 2 (red) to base_left"])
        self.assertIn(f"target box: {box_centered_at(750, 500)}", printed)
        scene = json.loads(FakeOpenRouter.seen_body["messages"][1]["content"][0]["text"])
        self.assertEqual(scene["table"]["free_cubes"], [{"id": "1", "color": "red"}, {"id": "2", "color": "red"}])


if __name__ == "__main__":
    unittest.main()
