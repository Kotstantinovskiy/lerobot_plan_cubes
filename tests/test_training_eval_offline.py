"""The pure parts of training/eval_offline.py, with a fake policy and no dataset. Needs LeRobot, as the script does."""

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import eval_offline  # noqa: E402

COLORS = ["red", "blue"]
TASKS = {"red": "pick red", "blue": "pick blue"}
TOP_CAMERA = "observation.images.top"
BOX_STYLE = {"camera": TOP_CAMERA, "rgb": [255, 255, 255], "thickness": 1, "margin": 0}
BOX_LABELS = {"0": {"boxes": {"red": [2, 2, 5, 5], "blue": None}}}
BLACK_FRAME_BATCH = {TOP_CAMERA: torch.zeros(1, 3, 8, 8)}
ACTION_SIZE = 2
CHUNK_SIZE = 4


def fake_setup(**overrides) -> eval_offline.EvalSetup:
    """A setup that runs on a CPU, with a fake policy whose chunk is the noise, without boxes or crop by default."""
    values = {
        "policy": make_noise_policy(),
        "preprocess": lambda observation: observation,
        "postprocess": lambda chunk: chunk,
        "device": "cpu",
        "horizon": CHUNK_SIZE,
        "observation_keys": [TOP_CAMERA],
        "tasks": TASKS,
        "color_of_task": {task: color for color, task in TASKS.items()},
        "robot_type": "so_follower",
        "box_style": None,
        "box_labels": None,
        "crop": None,
    }
    return eval_offline.EvalSetup(**{**values, **overrides})


def make_noise_policy() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config=types.SimpleNamespace(chunk_size=CHUNK_SIZE, max_action_dim=ACTION_SIZE),
        predict_action_chunk=lambda batch, noise: noise,
    )


def make_item(episode: int = 0, frame: int = 0, *, task: str = "pick red") -> dict:
    return {
        TOP_CAMERA: torch.zeros(3, 8, 8),
        "action": torch.zeros(CHUNK_SIZE, ACTION_SIZE),
        "action_is_pad": torch.zeros(CHUNK_SIZE, dtype=torch.bool),
        "episode_index": torch.tensor(episode),
        "frame_index": torch.tensor(frame),
        "task": task,
    }


def make_row(color: str, *, error: dict, shift: dict) -> dict:
    return {"episode": 0, "frame": 0, "color": color, "error": error, "shift": shift}


ROWS = [
    make_row("red", error={"red": 1.0, "blue": 5.0}, shift={"red": 0.0, "blue": 2.0}),
    make_row("red", error={"red": 3.0, "blue": 7.0}, shift={"red": 0.0, "blue": 4.0}),
    make_row("blue", error={"red": 9.0, "blue": 2.0}, shift={"red": 6.0, "blue": 0.0}),
]


class FrameIndicesTests(unittest.TestCase):
    table = [
        {"dataset_from_index": 0, "dataset_to_index": 5},
        {"dataset_from_index": 5, "dataset_to_index": 8},
    ]

    def test_every_stride_th_frame_of_each_episode(self):
        self.assertEqual(eval_offline.frame_indices(self.table, [0, 1], 2), [0, 2, 4, 5, 7])

    def test_episodes_keep_the_given_order(self):
        self.assertEqual(eval_offline.frame_indices(self.table, [1, 0], 3), [5, 0, 3])

    def test_no_episodes_give_no_frames(self):
        self.assertEqual(eval_offline.frame_indices(self.table, [], 1), [])


class EpisodeListTests(unittest.TestCase):
    def test_numbers_are_sorted_without_repeats(self):
        self.assertEqual(eval_offline.episode_list("3, 1,3"), [1, 3])

    def test_only_separators_give_an_empty_list(self):
        self.assertEqual(eval_offline.episode_list(","), [])

    def test_invalid_number_is_reported_by_argparse_under_the_name_of_the_parser(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            eval_offline.build_parser().parse_args(["--policy", "folder", "--episodes", "first"])
        self.assertIn("invalid episode_list value: 'first'", stderr.getvalue())


class CommandedViewTests(unittest.TestCase):
    def test_without_box_and_crop_the_observations_are_passed_as_they_are(self):
        view = eval_offline.commanded_view(BLACK_FRAME_BATCH, [make_item()], "red", fake_setup())
        self.assertIs(view, BLACK_FRAME_BATCH)

    def test_box_of_the_commanded_colour_is_drawn(self):
        setup = fake_setup(box_style=BOX_STYLE, box_labels=BOX_LABELS)
        view = eval_offline.commanded_view(BLACK_FRAME_BATCH, [make_item()], "red", setup)
        self.assertEqual(view[TOP_CAMERA][0, 0, 2, 2].item(), 1.0)

    def test_cube_the_detector_missed_is_left_unmarked(self):
        setup = fake_setup(box_style=BOX_STYLE, box_labels=BOX_LABELS)
        view = eval_offline.commanded_view(BLACK_FRAME_BATCH, [make_item()], "blue", setup)
        self.assertEqual(view[TOP_CAMERA].sum().item(), 0.0)

    def test_drawing_leaves_the_original_frames_black(self):
        setup = fake_setup(box_style=BOX_STYLE, box_labels=BOX_LABELS)
        eval_offline.commanded_view(BLACK_FRAME_BATCH, [make_item()], "red", setup)
        self.assertEqual(BLACK_FRAME_BATCH[TOP_CAMERA].sum().item(), 0.0)

    def test_crop_comes_after_the_box(self):
        setup = fake_setup(
            box_style=BOX_STYLE,
            box_labels=BOX_LABELS,
            crop={"camera": TOP_CAMERA, "box": [2, 2, 4, 4]},
        )
        view = eval_offline.commanded_view(BLACK_FRAME_BATCH, [make_item()], "red", setup)
        self.assertEqual(view[TOP_CAMERA].shape, (1, 3, 2, 2))
        self.assertEqual(view[TOP_CAMERA][0, 0, 0, 0].item(), 1.0)


class PredictChunksTests(unittest.TestCase):
    def test_every_colour_gets_the_same_noise(self):
        noise = torch.ones(1, CHUNK_SIZE, ACTION_SIZE)
        chunks = eval_offline.predict_chunks(BLACK_FRAME_BATCH, [make_item()], noise, fake_setup())
        np.testing.assert_array_equal(chunks["red"], chunks["blue"])

    def test_chunks_are_cut_to_the_horizon(self):
        noise = torch.ones(1, CHUNK_SIZE, ACTION_SIZE)
        chunks = eval_offline.predict_chunks(BLACK_FRAME_BATCH, [make_item()], noise, fake_setup(horizon=3))
        self.assertEqual(chunks["red"].shape, (1, 3, ACTION_SIZE))

    def test_each_colour_is_commanded_with_its_task_for_the_whole_batch(self):
        tasks_seen = []

        def remember_tasks(observation):
            tasks_seen.append(observation["task"])
            return observation

        noise = torch.ones(2, CHUNK_SIZE, ACTION_SIZE)
        items = [make_item(), make_item()]
        observations = {TOP_CAMERA: torch.zeros(2, 3, 8, 8)}
        eval_offline.predict_chunks(observations, items, noise, fake_setup(preprocess=remember_tasks))
        self.assertEqual(tasks_seen, [["pick red", "pick red"], ["pick blue", "pick blue"]])


class MakeRowsTests(unittest.TestCase):
    def setUp(self):
        self.items = [make_item(episode=4, frame=7)]
        self.chunks = {"red": np.full((1, 2, 1), 1.0), "blue": np.full((1, 2, 1), 3.0)}
        self.recorded_actions = np.zeros((1, 2, 1))
        self.all_steps_valid = np.ones((1, 2), dtype=bool)

    def test_row_says_where_the_frame_comes_from(self):
        row = eval_offline.make_rows(self.items, ["red"], self.chunks, self.recorded_actions, self.all_steps_valid)[0]
        self.assertEqual((row["episode"], row["frame"], row["color"]), (4, 7, "red"))

    def test_error_is_the_difference_from_the_recording(self):
        row = eval_offline.make_rows(self.items, ["red"], self.chunks, self.recorded_actions, self.all_steps_valid)[0]
        self.assertEqual(row["error"], {"red": 1.0, "blue": 3.0})

    def test_shift_is_the_difference_from_the_chunk_of_the_true_colour(self):
        row = eval_offline.make_rows(self.items, ["red"], self.chunks, self.recorded_actions, self.all_steps_valid)[0]
        self.assertEqual(row["shift"], {"red": 0.0, "blue": 2.0})

    def test_padded_steps_do_not_count(self):
        chunks = {"red": np.array([[[1.0], [100.0]]])}
        valid_steps = np.array([[True, False]])
        row = eval_offline.make_rows(self.items, ["red"], chunks, self.recorded_actions, valid_steps)[0]
        self.assertEqual(row["error"], {"red": 1.0})


class EvaluateFramesTests(unittest.TestCase):
    def setUp(self):
        self.dataset = [make_item(frame=frame, task="pick blue") for frame in range(5)]

    def evaluate(self, indices: list[int], batch_size: int) -> tuple[list[dict], str]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rows = eval_offline.evaluate_frames(self.dataset, indices, batch_size, 0, fake_setup())
        return rows, stdout.getvalue()

    def test_rows_follow_the_indices_across_batches(self):
        rows, _ = self.evaluate([4, 0, 2], batch_size=2)
        self.assertEqual([row["frame"] for row in rows], [4, 0, 2])

    def test_true_colour_comes_from_the_task_of_the_frame(self):
        rows, _ = self.evaluate([0], batch_size=1)
        self.assertEqual(rows[0]["color"], "blue")

    def test_progress_is_printed_after_every_batch(self):
        _, stdout = self.evaluate([0, 1, 2], batch_size=2)
        self.assertEqual(stdout, "2/3 кадров\n3/3 кадров\n")


class SummaryTests(unittest.TestCase):
    def test_mean_over_other_colors_skips_the_true_colour_of_each_row(self):
        self.assertEqual(eval_offline.mean_over_other_colors(ROWS, "error", COLORS), 7.0)

    def test_summary_counts_the_frames_of_a_colour(self):
        summary = eval_offline.summarize_by_color(ROWS, COLORS)
        self.assertEqual(summary["red"]["frames"], 2)

    def test_summary_averages_the_error_per_commanded_colour(self):
        summary = eval_offline.summarize_by_color(ROWS, COLORS)
        self.assertEqual(summary["red"]["error"], {"red": 2.0, "blue": 6.0})

    def test_summary_averages_the_shift_to_other_colours(self):
        summary = eval_offline.summarize_by_color(ROWS, COLORS)
        self.assertEqual(summary["red"]["shift_to_other_colors"], 3.0)

    def test_colour_without_frames_is_left_out(self):
        summary = eval_offline.summarize_by_color(ROWS[:2], COLORS)
        self.assertEqual(list(summary), ["red"])


class PrintReportTests(unittest.TestCase):
    results = {
        "policy": "outputs/train/run/checkpoints/020000/pretrained_model",
        "episodes": [0, 1],
        "horizon": 50,
        "every_s": 1.0,
        "summary": eval_offline.summarize_by_color(ROWS, COLORS),
        "frames": ROWS,
    }

    def print_report(self, *, uses_boxes: bool) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            eval_offline.print_report(self.results, COLORS, uses_boxes=uses_boxes)
        return stdout.getvalue()

    def test_report_is_printed_in_the_format_eval_checkpoints_greps(self):
        expected = (
            "\nПолитика: outputs/train/run/checkpoints/020000/pretrained_model\n"
            "Эпизоды: [0, 1], кадр каждые 1.00 с, горизонт 50\n"
            "Средняя |предсказание - запись| по суставам; строка — цвет в эпизоде, столбец — цвет в команде.\n"
            "         кадров      red     blue    сдвиг\n"
            "     red      2     2.00     6.00     3.00\n"
            "    blue      1     9.00     2.00     6.00\n"
            "\nОшибка с правильной командой 2.00, с чужим цветом 7.00; сдвиг пачки от смены цвета 4.00.\n"
        )
        self.assertEqual(self.print_report(uses_boxes=False), expected)

    def test_report_of_a_policy_with_boxes_says_the_box_moves_with_the_word(self):
        report = self.print_report(uses_boxes=True)
        self.assertIn("столбец — цвет в команде (слово и рамка на кубике этого цвета).\n", report)


if __name__ == "__main__":
    unittest.main()
