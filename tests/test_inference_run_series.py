"""run_series.py's schedule, prompts and results; the trials are replaced, so nothing starts run_policy.py."""

import contextlib
import csv
import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import run_series  # noqa: E402
from run_series import GRASPED, MISSED, PLACED, WRONG_CUBE, ScheduledTrial, TrialResult  # noqa: E402

SEAMS = {"count": 3, "median": 1.26, "p90": 2.34, "max": 4.5}
SUMMARY = {"target_box": [1, 2, 3, 4], "seams": SEAMS, "stats": {"steps": 12}, "video": "run.mp4"}


def make_result(color="red", outcome=PLACED) -> TrialResult:
    return TrialResult(
        trial=1,
        layout=1,
        color=color,
        outcome=outcome,
        target_box=None,
        seam_median=None,
        seam_p90=None,
        steps=10,
        video="run.mp4",
    )


class TallyTests(unittest.TestCase):
    def test_grasped_counts_the_placed_ones_too(self):
        results = [make_result(outcome=PLACED), make_result(outcome=GRASPED), make_result(outcome=MISSED)]
        self.assertEqual(run_series.count_outcomes(results), "placed 1/3, grasped 2/3, wrong cube 0/3")

    def test_wrong_cube_is_counted(self):
        results = [make_result(outcome=WRONG_CUBE), make_result(outcome=MISSED)]
        self.assertEqual(run_series.count_outcomes(results), "placed 0/2, grasped 0/2, wrong cube 1/2")

    def test_tally_gives_all_trials_then_each_color_in_order_of_appearance(self):
        results = [make_result("blue"), make_result("red", MISSED)]
        self.assertEqual(
            run_series.tally(results),
            "All: placed 1/2, grasped 1/2, wrong cube 0/2\n"
            "  blue: placed 1/1, grasped 1/1, wrong cube 0/1\n"
            "  red: placed 0/1, grasped 0/1, wrong cube 0/1",
        )


class ScheduleTests(unittest.TestCase):
    def test_each_layout_serves_every_color_once(self):
        self.assertEqual(
            run_series.make_schedule(["red", "blue"], 1),
            [ScheduledTrial(1, "red"), ScheduledTrial(1, "blue")],
        )

    def test_order_rotates_from_layout_to_layout(self):
        schedule = run_series.make_schedule(["red", "blue", "green"], 2)
        self.assertEqual([trial.color for trial in schedule[3:]], ["blue", "green", "red"])

    def test_no_layouts_mean_no_trials(self):
        self.assertEqual(run_series.make_schedule(["red"], 0), [])


class AskOutcomeTests(unittest.TestCase):
    def test_answer_is_read_without_spaces_and_case(self):
        with patch("builtins.input", return_value="  P "):
            self.assertEqual(run_series.ask_outcome(), "p")

    def test_unknown_letter_is_asked_again(self):
        with patch("builtins.input", side_effect=["x", "", "g"]) as ask:
            self.assertEqual(run_series.ask_outcome(), "g")
        self.assertEqual(ask.call_count, 3)

    def test_run_again_and_stop_are_answers_too(self):
        with patch("builtins.input", return_value="r"):
            self.assertEqual(run_series.ask_outcome(), run_series.RUN_AGAIN)


class ResultFileTests(unittest.TestCase):
    def make_result_from(self, summary):
        return run_series.make_result(
            2,
            ScheduledTrial(1, "red"),
            PLACED,
            summary,
        )

    def test_seams_are_rounded_to_one_digit(self):
        result = self.make_result_from(SUMMARY)
        self.assertEqual((result.seam_median, result.seam_p90), (1.3, 2.3))

    def test_run_without_seams_leaves_them_empty(self):
        result = self.make_result_from({**SUMMARY, "seams": None})
        self.assertIsNone(result.seam_median)

    def test_file_has_its_columns_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / run_series.RESULTS_FILE
            run_series.write_results(path, [make_result()])
            header = path.read_text().splitlines()[0]
        self.assertEqual(header, "trial,layout,color,outcome,target_box,seam_median,seam_p90,steps,video")

    def test_missing_values_are_empty_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / run_series.RESULTS_FILE
            run_series.write_results(path, [make_result()])
            with path.open(newline="") as file:
                row = next(csv.DictReader(file))
        self.assertEqual(row["seam_median"], "")

    def test_box_is_written_as_a_list(self):
        result = self.make_result_from(SUMMARY)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / run_series.RESULTS_FILE
            run_series.write_results(path, [result])
            with path.open(newline="") as file:
                row = next(csv.DictReader(file))
        self.assertEqual(row["target_box"], "[1, 2, 3, 4]")


class RunTrialTests(unittest.TestCase):
    """The child process is replaced by a function that writes, or does not write, the summary."""

    def run_trial(self, summary=None):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"

            def finish_child():
                if summary is not None:
                    summary_path.write_text(json.dumps(summary))

            child = MagicMock()
            child.wait.side_effect = finish_child
            with patch.object(run_series.subprocess, "Popen", return_value=child) as popen:
                result = run_series.run_trial(
                    "some/policy",
                    "red",
                    summary_path,
                    ["--seconds", "30"],
                )
        return result, popen, summary_path

    def test_trial_without_a_summary_did_not_run(self):
        result, _, _ = self.run_trial()
        self.assertIsNone(result)

    def test_trial_without_steps_did_not_run(self):
        result, _, _ = self.run_trial({"stats": {"steps": 0}})
        self.assertIsNone(result)

    def test_trial_with_steps_gives_its_summary(self):
        result, _, _ = self.run_trial(SUMMARY)
        self.assertEqual(result, SUMMARY)

    def test_policy_waits_for_enter_and_the_extra_arguments_go_through(self):
        _, popen, summary_path = self.run_trial()
        command = popen.call_args.args[0]
        self.assertEqual(
            command[2:],
            ["--policy", "some/policy", "--color", "red", "--wait", "--summary", str(summary_path), "--seconds", "30"],
        )

    def test_ctrl_c_is_ignored_here_during_the_trial_and_restored_after_it(self):
        with patch.object(run_series.signal, "signal", return_value="previous") as set_handler:
            self.run_trial()
        self.assertEqual(
            [call.args for call in set_handler.call_args_list],
            [(signal.SIGINT, signal.SIG_IGN), (signal.SIGINT, "previous")],
        )


class RunScheduleTests(unittest.TestCase):
    SCHEDULE = [ScheduledTrial(1, "red"), ScheduledTrial(1, "blue")]

    def run_schedule(self, trials, answers):
        results = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(run_series, "run_trial", side_effect=trials) as run_trial,
            patch("builtins.input", side_effect=answers),
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            run_series.run_schedule(
                self.SCHEDULE,
                "some/policy",
                [],
                Path(directory),
                results,
            )
        return results, run_trial, printed.getvalue()

    def test_every_trial_gets_a_result(self):
        results, _, _ = self.run_schedule([SUMMARY, SUMMARY], ["p", "w"])
        self.assertEqual([result.outcome for result in results], [PLACED, WRONG_CUBE])

    def test_stop_keeps_the_results_so_far(self):
        results, run_trial, _ = self.run_schedule([SUMMARY, SUMMARY], ["p", "q"])
        self.assertEqual((len(results), run_trial.call_count), (1, 2))

    def test_run_again_repeats_the_same_trial(self):
        _, run_trial, _ = self.run_schedule([SUMMARY, SUMMARY, SUMMARY], ["r", "p", "p"])
        self.assertEqual([call.args[1] for call in run_trial.call_args_list], ["red", "red", "blue"])

    def test_trial_that_did_not_run_can_be_run_again(self):
        results, _, _ = self.run_schedule([None, SUMMARY, SUMMARY], ["", "p", "p"])
        self.assertEqual(len(results), 2)

    def test_trial_that_did_not_run_can_end_the_series(self):
        results, _, _ = self.run_schedule([None], ["q"])
        self.assertEqual(results, [])

    def test_first_trial_of_a_layout_says_to_lay_out_the_cubes(self):
        _, _, printed = self.run_schedule([SUMMARY], ["q"])
        self.assertIn("=== Trial 1/2, layout 1: red. While the policy loads, lay the cubes out anew.", printed)

    def test_next_trial_says_to_put_back_what_moved(self):
        _, _, printed = self.run_schedule([SUMMARY, SUMMARY], ["p", "q"])
        self.assertIn("=== Trial 2/2, layout 1: blue. While the policy loads, put back what the arm moved.", printed)


if __name__ == "__main__":
    unittest.main()
