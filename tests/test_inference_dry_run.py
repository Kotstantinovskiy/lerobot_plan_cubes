"""run_policy.py --dry-run with a fake policy and recorded frames made up here: no checkpoint or dataset is touched."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "recording"))

import dry_run  # noqa: E402
from arm_control import PolicyView  # noqa: E402
from inference_arm_fakes import (  # noqa: E402
    SHOULDER,
    TOP_SHAPE,
    WRIST_SHAPE,
    make_args,
    make_chunk,
    make_facts,
)

EPISODE = dry_run.RECORDED_EPISODE


def recorded_observation(offset_s):
    return {
        SHOULDER: 50.04,
        "elbow.pos": 60.0,
        "top": np.zeros(TOP_SHAPE, np.uint8),
        "wrist": np.zeros(WRIST_SHAPE, np.uint8),
    }


def printed_by(function, *args):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        function(*args)
    return printed.getvalue()


class RecordedSourceTests(unittest.TestCase):
    def test_server_gives_the_frames_when_there_is_one(self):
        remote = MagicMock()
        dry_run.make_recorded_source(remote, None, "root")(offset_s=3)
        remote.recorded_observation.assert_called_once_with(episode=EPISODE, offset_s=3)

    def test_local_dataset_gives_them_otherwise(self):
        with patch.object(dry_run, "dataset_observation") as observation:
            dry_run.make_recorded_source(None, "metadata", "root")(offset_s=3)
        observation.assert_called_once_with(
            "metadata",
            "root",
            episode=EPISODE,
            offset_s=3,
        )


class OneInferenceTests(unittest.TestCase):
    def run_it(self, chunk):
        plan = MagicMock(return_value=chunk)
        return printed_by(
            dry_run.run_one_inference,
            plan,
            recorded_observation,
            make_facts(),
            PolicyView(None, None),
            "red",
        )

    def test_time_of_the_inference_is_printed(self):
        self.assertIn("Dry run: one inference 0.25s", self.run_it(make_chunk()))

    def test_joints_outside_the_envelope_are_counted(self):
        self.assertIn("2 joints clipped", self.run_it(make_chunk(120.0)))

    def test_state_is_printed_rounded(self):
        self.assertIn("state  [50.0, 60.0]", self.run_it(make_chunk()))

    def test_first_action_is_printed(self):
        self.assertIn("action [70.0, 70.0]", self.run_it(make_chunk(70.0)))


class CheckRtcTests(unittest.TestCase):
    def setUp(self):
        # At FPS 8 an inference of 0.25 s is a delay of 2 steps: the takeover goes from action 3 to the later chunk's 2.
        self.plan = MagicMock(side_effect=[make_chunk(50.0), make_chunk(60.0), make_chunk(54.0, seconds=0.5)])
        self.printed = printed_by(
            dry_run.check_rtc,
            self.plan,
            recorded_observation,
            make_facts(),
            PolicyView(None, None),
        )

    def test_seams_without_and_with_guidance_are_printed(self):
        self.assertIn("RTC check on frames 2 steps apart: seam 9.0 without, 3.0 with", self.printed)

    def test_guided_inference_time_is_printed(self):
        self.assertIn("guided inference 0.50s", self.printed)

    def test_guided_chunk_gets_the_delay(self):
        self.assertEqual(self.plan.call_args.kwargs["inference_delay"], 2)

    def test_guided_chunk_continues_after_the_delay(self):
        expected = make_chunk(50.0).raw_actions[2:]
        np.testing.assert_array_equal(self.plan.call_args.kwargs["prev_chunk_left_over"], expected)

    def test_delay_is_at_most_half_a_chunk(self):
        plan = MagicMock(side_effect=[make_chunk(50.0, length=6, seconds=9.0), make_chunk(), make_chunk()])
        printed = printed_by(
            dry_run.check_rtc,
            plan,
            recorded_observation,
            make_facts(),
            PolicyView(None, None),
        )
        self.assertIn("frames 3 steps apart", printed)


class RunDryTests(unittest.TestCase):
    def run_it(self, args):
        plan = MagicMock(side_effect=[make_chunk(), make_chunk(), make_chunk(), make_chunk()])
        with contextlib.redirect_stdout(io.StringIO()):
            dry_run.run_dry(
                args,
                plan,
                recorded_observation,
                make_facts(),
                PolicyView(None, None),
            )
        return plan

    def test_without_rtc_one_inference_is_enough(self):
        self.assertEqual(self.run_it(make_args()).call_count, 1)

    def test_with_rtc_the_check_adds_three_more(self):
        self.assertEqual(self.run_it(make_args(rtc=True)).call_count, 4)


if __name__ == "__main__":
    unittest.main()
