import base64
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

INFERENCE = Path(__file__).resolve().parents[1] / "inference"
sys.path.insert(0, str(INFERENCE))

import cube_boxes  # noqa: E402
import llm_planner  # noqa: E402
from pyramid_plan import (  # noqa: E402
    FINISH,
    NO_CHOICE,
    PICK_AND_PLACE,
    Cube,
    PlannedStep,
    TableState,
    assign_colors,
    describe_table,
    find_free_cube,
    find_rule_violation,
    next_loop_step,
)
import run_pyramid  # noqa: E402
import top_crop  # noqa: E402

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


class TableStateTests(unittest.TestCase):
    def test_cube_inside_a_slot_box_is_in_that_slot(self):
        state = describe_table({"red": [box_centered_at(590, 330)]}, SLOT_BOXES)
        self.assertEqual(state.slots["base_left"], "red")

    def test_cube_outside_slot_boxes_is_free(self):
        box = box_centered_at(800, 500)
        state = describe_table({"red": [box]}, SLOT_BOXES)
        self.assertEqual(state.free_cubes, [Cube("1", "red", box)])

    def test_color_without_cubes_is_reported_as_not_seen(self):
        state = describe_table({"red": []}, SLOT_BOXES)
        self.assertEqual(state.not_seen, ["red"])

    def test_overlapping_slot_boxes_choose_the_nearest_center(self):
        state = describe_table({"yellow": [box_centered_at(620, 330)]}, SLOT_BOXES)
        self.assertEqual(state.slots["top"], "yellow")

    def test_two_free_cubes_of_a_color_get_their_own_ids(self):
        left_box, right_box = box_centered_at(450, 500), box_centered_at(750, 500)
        state = describe_table({"red": [right_box, left_box]}, SLOT_BOXES)
        self.assertEqual(state.free_cubes, [Cube("1", "red", left_box), Cube("2", "red", right_box)])

    def test_free_cubes_are_numbered_left_to_right_across_colors(self):
        state = describe_table(
            {"red": [box_centered_at(800, 500)], "blue": [box_centered_at(450, 500)]},
            SLOT_BOXES,
        )
        self.assertEqual([(cube.id, cube.color) for cube in state.free_cubes], [("1", "blue"), ("2", "red")])

    def test_cube_in_a_slot_leaves_its_twin_free(self):
        twin_box = box_centered_at(800, 500)
        state = describe_table({"red": [box_centered_at(590, 330), twin_box]}, SLOT_BOXES)
        self.assertEqual(state.slots["base_left"], "red")
        self.assertEqual(state.free_cubes, [Cube("1", "red", twin_box)])

    def test_cubes_of_one_color_can_fill_both_base_slots(self):
        state = describe_table({"red": [box_centered_at(590, 330), box_centered_at(650, 330)]}, SLOT_BOXES)
        self.assertEqual((state.slots["base_left"], state.slots["base_right"]), ("red", "red"))

    def test_detector_on_a_frame_gives_the_table_state(self):
        frame = make_frame({"red": (590, 330), "blue": (800, 500)})
        state = describe_table(cube_boxes.detect_cubes(frame), SLOT_BOXES)
        self.assertEqual(state.slots["base_left"], "red")
        self.assertEqual([cube.color for cube in state.free_cubes], ["blue"])
        self.assertCountEqual(state.not_seen, ["green", "yellow"])


class RuleTests(unittest.TestCase):
    def test_free_cube_on_an_empty_base_slot_is_allowed(self):
        self.assertIsNone(find_rule_violation(PLACE_RED_LEFT, make_state(FREE_ALL)))

    def test_top_is_refused_until_both_base_cubes_are_in_place(self):
        step = PlannedStep(PICK_AND_PLACE, "1", "top", "")
        state = make_state(["yellow", "blue"], base_left="red")
        self.assertIn("both base cubes", find_rule_violation(step, state))

    def test_top_is_allowed_on_a_full_base(self):
        step = PlannedStep(PICK_AND_PLACE, "1", "top", "")
        state = make_state(["yellow"], base_left="red", base_right="blue")
        self.assertIsNone(find_rule_violation(step, state))

    def test_occupied_slot_is_refused(self):
        state = make_state(["red"], base_left="blue")
        self.assertIn("already holds", find_rule_violation(PLACE_RED_LEFT, state))

    def test_id_without_a_free_cube_is_refused_with_the_valid_ids(self):
        step = PlannedStep(PICK_AND_PLACE, "9", "base_right", "")
        problem = find_rule_violation(step, make_state(["blue", "red"]))
        self.assertIn("not free", problem)
        self.assertIn("['1', '2']", problem)

    def test_pick_and_place_without_a_slot_is_refused(self):
        step = PlannedStep(PICK_AND_PLACE, "1", NO_CHOICE, "")
        self.assertIn("does not exist", find_rule_violation(step, make_state(FREE_ALL)))

    def test_finish_is_always_allowed(self):
        self.assertIsNone(find_rule_violation(FINISHED, make_state([])))


class LoopPlannerTests(unittest.TestCase):
    def test_first_step_fills_base_left(self):
        state = make_state(FREE_ALL)
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("red", "base_left"))

    def test_second_step_fills_base_right(self):
        state = make_state(["blue", "yellow"], base_left="red")
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("blue", "base_right"))

    def test_top_comes_last(self):
        state = make_state(["yellow"], base_left="red", base_right="blue")
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("yellow", "top"))

    def test_finishes_when_top_is_right_even_with_a_hidden_base(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state([], top="yellow"))
        self.assertEqual(step.kind, FINISH)

    def test_cube_knocked_off_its_slot_becomes_the_next_step(self):
        state = make_state(["red", "yellow"], base_right="blue")
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("red", "base_left"))

    def test_wrong_cube_in_a_slot_ends_the_plan(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state(["red"], base_left="green"))
        self.assertEqual(step.kind, FINISH)
        self.assertIn("clear it by hand", step.comment)

    def test_missing_cube_ends_the_plan(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state(["blue"]))
        self.assertEqual(step.kind, FINISH)
        self.assertIn("red", step.comment)

    def test_first_of_two_free_cubes_of_a_color_is_taken(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state(["blue", "red", "red"]))
        self.assertEqual(step.cube, "2")

    def test_repeated_color_fills_two_slots(self):
        self.assertEqual(
            assign_colors(["red", "red", "blue"]),
            {"base_left": "red", "base_right": "red", "top": "blue"},
        )

    def test_pair_of_one_color_takes_a_second_cube_of_it(self):
        colors_by_slot = assign_colors(["red", "red", "blue"])
        state = make_state(["blue", "red"], base_left="red")
        step = next_loop_step(colors_by_slot, state)
        self.assertEqual((step.cube, step.slot), ("2", "base_right"))

    def test_wrong_number_of_colors_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Give 3 colours"):
            assign_colors(["red", "blue"])

    def test_unknown_color_is_refused(self):
        with self.assertRaisesRegex(ValueError, "purple"):
            assign_colors(["red", "blue", "purple"])


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
        self.assertEqual(self.body["response_format"]["json_schema"]["schema"]["required"],
                         ["kind", "cube", "slot", "comment"])

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
        with patch.object(llm_planner, "post_chat_completion",
                          side_effect=[completion(bad_fields), completion(good_fields)]) as post:
            step = llm_planner.choose_step("goal", make_frame({}), make_state(FREE_ALL), history, api_key="key")
        self.assertEqual(step, PLACE_RED_LEFT)
        second_scene = json.loads(post.call_args_list[1].args[0]["messages"][1]["content"][0]["text"])
        self.assertIn("both base cubes", second_scene["history"][0]["outcome"])

    def test_id_of_a_cube_that_is_not_there_goes_back_to_the_model(self):
        bad_fields = {"kind": PICK_AND_PLACE, "cube": "9", "slot": "base_left", "comment": ""}
        good_fields = {"kind": PICK_AND_PLACE, "cube": "1", "slot": "base_left", "comment": "red first"}
        history = []
        with patch.object(llm_planner, "post_chat_completion",
                          side_effect=[completion(bad_fields), completion(good_fields)]):
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
        self.assertEqual(step_lines(result.stdout), [
            "step: cube 1 (red) to base_left",
            "step: cube 1 (blue) to base_right",
            "step: cube 1 (yellow) to top",
            "step: finish",
        ])

    def test_loop_planner_builds_a_pyramid_with_a_pair_of_one_color(self):
        left_red, right_red = (590, 330), (650, 330)
        frames = [
            self.write_frame("0.png", {"red": (450, 500), "blue": (600, 500)}, [("red", (750, 500))]),
            self.write_frame("1.png", {"red": left_red, "blue": (600, 500)}, [("red", (750, 500))]),
            self.write_frame("2.png", {"red": left_red, "blue": (600, 500)}, [("red", right_red)]),
            self.write_frame("3.png", {"red": left_red}, [("red", right_red), ("blue", (620, 330))]),
        ]
        result = self.run_script(
            "--planner", "loop", "--colors", "red,red,blue", "--frame", *frames, "--slot-boxes", str(self.slot_boxes),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(step_lines(result.stdout), [
            "step: cube 1 (red) to base_left",
            "step: cube 2 (red) to base_right",
            "step: cube 1 (blue) to top",
            "step: finish",
        ])

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
