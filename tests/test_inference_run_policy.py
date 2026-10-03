"""run_policy.py around the control loop, with a fake robot, policy and video: no port or checkpoint is touched."""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "recording"))

import arm_control  # noqa: E402
import dry_run  # noqa: E402
import run_policy  # noqa: E402
from arm_control import PolicyView, RunState  # noqa: E402
from inference_arm_fakes import (  # noqa: E402
    FEATURES,
    FPS,
    JOINT_NAMES,
    MOVED_POSE,
    STILL_POSE,
    FakeClock,
    FakeRobot,
    make_args,
    make_chunk,
    make_facts,
    make_video,
)
from inference_pyramid_helpers import box_centered_at, make_frame  # noqa: E402

BOX_STYLE = {"camera": "observation.images.top", "rgb": [255, 255, 255], "thickness": 6, "margin": 8}
REST_POSE = dict.fromkeys(JOINT_NAMES, 1.0)
CONFIG = {"rest_pose": REST_POSE}
FOLD = "fold_to_rest"
ARM_RELEASED = "robot.disconnect"
IMPORTED_BY_OTHER_SCRIPTS = {"load_policy", "target_box_style", "compute_chunk", "dataset_observation", "saved_stats"}
CONFIG_FILE = {
    "tasks": {"red": "Pick up the red cube"},
    "repo_id": "local/cubes",
    "dataset_root": "data",
    "robot": {"type": "so101_follower"},
    "rest_pose": REST_POSE,
}


class ImportedNamesTests(unittest.TestCase):
    def test_names_other_scripts_import_from_here_exist(self):
        self.assertLessEqual(IMPORTED_BY_OTHER_SCRIPTS, set(vars(run_policy)))


class CommandLineTests(unittest.TestCase):
    def parse(self, *arguments):
        return run_policy.make_parser().parse_args(["--color", "red", *arguments])

    def test_defaults_stay_as_documented(self):
        args = self.parse()
        self.assertEqual(
            (args.policy, args.seconds, args.max_step, args.n_action_steps, args.start_threshold, args.device),
            ("lerobot/smolvla_base", 20, 5.0, 50, 3.0, "mps"),
        )

    def test_rtc_defaults_stay_as_documented(self):
        args = self.parse()
        self.assertEqual((args.rtc, args.rtc_horizon, args.rtc_weight), (False, 20, 10.0))

    def test_config_defaults_to_the_project_file(self):
        self.assertEqual(self.parse().config, PROJECT / "config.json")

    def test_rtc_without_replan_is_a_usage_error(self):
        printed = io.StringIO()
        with (
            patch.object(sys, "argv", ["run_policy.py", "--color", "red", "--rtc"]),
            contextlib.redirect_stderr(printed),
            self.assertRaises(SystemExit) as caught,
        ):
            run_policy.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("add --replan", printed.getvalue())

    def check_chunk_arguments(self, *arguments):
        """Run the check on these arguments; gives what it printed as a usage error, empty if it passed."""
        parser = run_policy.make_parser()
        printed = io.StringIO()
        with contextlib.redirect_stderr(printed), contextlib.suppress(SystemExit):
            run_policy.check_chunk_arguments(parser, self.parse(*arguments))
        return printed.getvalue()

    def test_replan_as_long_as_the_chunk_would_never_start(self):
        self.assertIn("--replan 50 would never start", self.check_chunk_arguments("--replan", "50"))

    def test_replan_longer_than_shortened_chunks_would_never_start(self):
        printed = self.check_chunk_arguments("--replan", "20", "--n-action-steps", "10")
        self.assertIn("--replan 20 would never start", printed)

    def test_replan_at_the_last_action_of_a_chunk_is_accepted(self):
        self.assertEqual(self.check_chunk_arguments("--replan", "49"), "")

    def test_no_replan_is_accepted_with_any_chunk_length(self):
        self.assertEqual(self.check_chunk_arguments("--n-action-steps", "1"), "")


class RunFactsTests(unittest.TestCase):
    INFO = {"features": FEATURES, "fps": FPS, "low": [0, 1], "high": [10, 11], "policy": "/srv/policy"}

    def make_dataset_meta(self):
        meta = MagicMock(features=FEATURES, fps=FPS, total_episodes=3)
        meta.stats = {"action": {"min": np.array([0.0, 1.0]), "max": np.array([10.0, 11.0])}}
        return meta

    def test_server_facts_name_the_server(self):
        facts = run_policy.facts_from_server(self.INFO, "gpu:1")
        self.assertEqual(facts.policy_name, "/srv/policy at gpu:1")

    def test_server_envelope_becomes_float_arrays(self):
        facts = run_policy.facts_from_server(self.INFO, "gpu:1")
        self.assertEqual(facts.joint_low.dtype, float)

    def test_joint_names_come_from_the_action_feature(self):
        facts = run_policy.facts_from_server(self.INFO, "gpu:1")
        self.assertEqual(facts.joint_names, JOINT_NAMES)

    def test_dataset_envelope_comes_from_its_action_statistics(self):
        facts = run_policy.facts_from_dataset(self.make_dataset_meta(), "some/policy", "cpu")
        self.assertEqual(facts.joint_high.tolist(), [10.0, 11.0])

    def test_dataset_facts_say_how_many_episodes_gave_the_envelope(self):
        facts = run_policy.facts_from_dataset(self.make_dataset_meta(), "some/policy", "cpu")
        self.assertEqual(facts.envelope_source, "3 episodes")


class MessagesTests(unittest.TestCase):
    def test_run_description_lists_the_joint_envelope(self):
        self.assertEqual(
            run_policy.describe_run("Pick up", make_facts()),
            "Task: Pick up\nPolicy: some/policy on cpu\n"
            "Joint envelope from 3 episodes: shoulder.pos [0, 100], elbow.pos [0, 100]",
        )

    def test_control_line_without_replan_gives_the_actions_per_chunk(self):
        self.assertIn("50 actions per chunk", run_policy.describe_control(make_args()))

    def test_control_line_with_replan_gives_the_background_steps(self):
        self.assertIn("in the background after 7 actions", run_policy.describe_control(make_args(replan=7)))

    def test_control_line_with_rtc_gives_its_settings(self):
        line = run_policy.describe_control(make_args(replan=7, rtc=True))
        self.assertIn("RTC over 20 actions, guidance up to 10", line)


class RunOnArmTests(unittest.TestCase):
    """What happens to the arm, the video and the summary however a run ends."""

    def setUp(self):
        self.robot = FakeRobot()
        self.video = make_video()
        self.printed = io.StringIO()
        clock = FakeClock()
        self.start_patch(patch.object(arm_control.time, "perf_counter", clock.perf_counter))
        self.start_patch(patch.object(arm_control, "precise_sleep", clock.sleep))
        self.start_patch(patch.object(arm_control, "open_video", return_value=self.video))
        self.start_patch(patch.object(arm_control, "fold_to_rest", side_effect=self.note_fold))

    def start_patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def note_fold(self, *args, **kwargs):
        self.robot.calls.append(FOLD)

    def run_on_arm(self, args, view=None, chunk=None):
        with contextlib.redirect_stdout(self.printed):
            run_policy.run_on_arm(
                args,
                CONFIG,
                self.robot,
                MagicMock(return_value=chunk or make_chunk()),
                make_facts(),
                view or PolicyView(None, None),
            )

    def interrupt_at_third_action(self):
        self.robot.failing_send, self.robot.send_error = 3, KeyboardInterrupt()

    def test_finished_run_folds_the_arm_before_releasing_it(self):
        self.run_on_arm(make_args())
        self.assertLess(self.robot.calls.index(FOLD), self.robot.calls.index(ARM_RELEASED))

    def test_ctrl_c_is_a_normal_end(self):
        self.interrupt_at_third_action()
        self.run_on_arm(make_args())
        self.assertIn("Stopped by Ctrl+C.", self.printed.getvalue())

    def test_ctrl_c_folds_the_arm_before_releasing_it(self):
        self.interrupt_at_third_action()
        self.run_on_arm(make_args())
        self.assertLess(self.robot.calls.index(FOLD), self.robot.calls.index(ARM_RELEASED))

    def test_error_is_raised_after_the_arm_is_released(self):
        self.robot.failing_send, self.robot.send_error = 3, RuntimeError("bus failed")
        with self.assertRaisesRegex(RuntimeError, "bus failed"):
            self.run_on_arm(make_args())
        self.assertIn(ARM_RELEASED, self.robot.calls)

    def test_error_still_closes_the_video(self):
        self.robot.failing_send, self.robot.send_error = 3, RuntimeError("bus failed")
        with self.assertRaisesRegex(RuntimeError, "bus failed"):
            self.run_on_arm(make_args())
        self.video.container.close.assert_called_once_with()

    def test_no_rest_leaves_the_arm_unfolded(self):
        self.run_on_arm(make_args(no_rest=True))
        self.assertNotIn(FOLD, self.robot.calls)

    def test_bus_that_never_connected_is_left_alone(self):
        self.robot.bus.connect = MagicMock(side_effect=RuntimeError("no port"))
        with self.assertRaisesRegex(RuntimeError, "no port"):
            self.run_on_arm(make_args())
        self.assertNotIn(ARM_RELEASED, self.robot.calls)

    def test_refused_arm_is_neither_folded_nor_released(self):
        self.robot.bus.torque = 1
        with self.assertRaisesRegex(RuntimeError, "torque disabled"):
            self.run_on_arm(make_args())
        self.assertNotIn(FOLD, self.robot.calls)
        self.assertNotIn(ARM_RELEASED, self.robot.calls)
        self.assertEqual(self.robot.calls[-1], "bus.disconnect(disable_torque=False)")

    def test_refusal_still_closes_the_video(self):
        self.robot.bus.torque = 1
        with self.assertRaises(RuntimeError):
            self.run_on_arm(make_args())
        self.video.container.close.assert_called_once_with()

    def test_ctrl_c_during_the_disconnect_still_disables_torque(self):
        self.robot.disconnect_error = KeyboardInterrupt()
        self.run_on_arm(make_args())
        self.assertEqual(self.robot.calls[-1], "bus.disconnect(disable_torque=True)")

    def test_ctrl_c_during_the_disconnect_still_closes_the_video_and_reports(self):
        self.robot.disconnect_error = KeyboardInterrupt()
        self.run_on_arm(make_args())
        self.video.container.close.assert_called_once_with()
        self.assertIn("Summary:", self.printed.getvalue())

    def test_failing_last_resort_still_closes_the_video_and_reports(self):
        self.robot.disconnect_error = KeyboardInterrupt()
        closes_the_port = self.robot.bus.disconnect

        def fail_to_disable_torque(disable_torque=True):
            if disable_torque:
                raise OSError("port gone")
            closes_the_port(disable_torque=False)

        self.robot.bus.disconnect = fail_to_disable_torque
        with self.assertRaisesRegex(OSError, "port gone"):
            self.run_on_arm(make_args())
        self.video.container.close.assert_called_once_with()
        self.assertIn("Summary:", self.printed.getvalue())

    def test_summary_is_printed_with_the_counters_by_name(self):
        self.run_on_arm(make_args())
        self.assertIn(f"Summary: {{'steps': {FPS}, 'chunks': 1,", self.printed.getvalue())

    def test_target_is_found_before_the_policy_takes_over(self):
        self.robot.top_frame = make_frame({"red": (500, 400)})
        view = PolicyView(BOX_STYLE, None)
        self.run_on_arm(make_args(), view)
        self.assertEqual(view.box, box_centered_at(500, 400))

    def run_with_summary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            self.run_on_arm(make_args(summary=summary_path))
            return json.loads(summary_path.read_text()), summary_path

    def test_summary_file_has_the_keys_run_series_reads(self):
        summary, _ = self.run_with_summary_file()
        self.assertEqual(list(summary), ["color", "target_box", "stats", "seams", "video", "args"])

    def test_summary_file_holds_the_counters(self):
        summary, _ = self.run_with_summary_file()
        self.assertEqual(summary["stats"]["steps"], FPS)

    def test_summary_file_writes_paths_as_text(self):
        summary, summary_path = self.run_with_summary_file()
        self.assertEqual(summary["args"]["summary"], str(summary_path))

    def test_background_chunks_work_with_a_real_thread(self):
        # A chunk shorter than the run makes the loop wait for the job, so that a takeover is certain.
        self.robot.poses = (STILL_POSE, MOVED_POSE)
        self.run_on_arm(make_args(replan=2, n_action_steps=4), chunk=make_chunk(length=4))
        self.assertIn("chunk changes", self.printed.getvalue())

    def test_seams_are_reported_when_chunks_took_over(self):
        run = MagicMock(args=make_args(), video=self.video, view=PolicyView(None, None))
        state = RunState(seams=[1.0, 2.0, 3.0])
        with contextlib.redirect_stdout(self.printed):
            run_policy.report_run(run, state)
        self.assertIn(
            "Seams over 3 chunk changes: median 2.0, 90th percentile 2.8, max 3.0 units", self.printed.getvalue()
        )


class MainOrderTests(unittest.TestCase):
    """What main does first and what it leaves out; everything slow or physical is replaced."""

    def setUp(self):
        self.calls = []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config_path = Path(directory.name) / "config.json"
        self.config_path.write_text(json.dumps(CONFIG_FILE))
        dataset_meta = MagicMock(features=FEATURES, fps=FPS, total_episodes=3)
        dataset_meta.stats = {"action": {"min": np.array([0.0, 0.0]), "max": np.array([9.0, 9.0])}}
        self.start_patch(patch.object(run_policy, "LeRobotDatasetMetadata", return_value=dataset_meta))
        self.start_patch(patch.object(run_policy, "get_safe_torch_device"))
        self.start_patch(patch.object(run_policy, "target_box_style", return_value=None))
        self.start_patch(patch.object(run_policy.top_crop, "load", return_value=None))
        self.start_patch(patch.object(run_policy, "load_policy", side_effect=self.note_policy_load))
        self.start_patch(patch.object(arm_control, "build_robot", side_effect=self.note_robot_build))
        self.run_dry = self.start_patch(patch.object(dry_run, "run_dry"))
        self.run_on_arm = self.start_patch(patch.object(run_policy, "run_on_arm"))

    def start_patch(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def note_policy_load(self, *args):
        self.calls.append("load_policy")
        return MagicMock(), MagicMock(), MagicMock()

    def note_robot_build(self, *args):
        self.calls.append("build_robot")
        return FakeRobot()

    def run_main(self, *arguments, color="red"):
        argv = ["run_policy.py", "--config", str(self.config_path), "--color", color, *arguments]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            run_policy.main()

    def test_robot_is_built_before_the_slow_model_load(self):
        self.run_main()
        self.assertEqual(self.calls, ["build_robot", "load_policy"])

    def test_policy_goes_on_to_the_arm(self):
        self.run_main()
        self.run_on_arm.assert_called_once()

    def test_dry_run_builds_no_robot(self):
        self.run_main("--dry-run")
        self.assertEqual(self.calls, ["load_policy"])

    def test_dry_run_stays_away_from_the_arm(self):
        self.run_main("--dry-run")
        self.run_dry.assert_called_once()
        self.run_on_arm.assert_not_called()

    def test_cancelled_wait_stays_away_from_the_arm(self):
        with patch("builtins.input", side_effect=EOFError):
            self.run_main("--wait")
        self.run_on_arm.assert_not_called()

    def test_enter_after_the_wait_goes_on_to_the_arm(self):
        with patch("builtins.input", return_value=""):
            self.run_main("--wait")
        self.run_on_arm.assert_called_once()

    def test_unknown_color_stops_before_anything_is_built(self):
        with self.assertRaises(KeyError):
            self.run_main(color="purple")
        self.assertEqual(self.calls, [])


class WaitForEnterTests(unittest.TestCase):
    def test_enter_starts_the_run(self):
        with patch("builtins.input", return_value=""):
            self.assertTrue(run_policy.wait_for_enter())

    def test_ctrl_c_cancels_the_run(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run_policy.wait_for_enter())

    def test_end_of_input_cancels_the_run(self):
        with patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run_policy.wait_for_enter())


class ServerConnectionTests(unittest.TestCase):
    def make_parser_and_args(self):
        parser = argparse.ArgumentParser()
        return parser, make_args(server="gpu:1", rtc=True)

    def test_rtc_is_refused_by_a_server_without_it(self):
        parser, args = self.make_parser_and_args()
        remote = MagicMock(info={"rtc": False})
        with (
            patch.object(run_policy, "RemotePolicy", return_value=remote),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            run_policy.connect_to_server(parser, args)
        self.assertEqual(caught.exception.code, 2)

    def test_rtc_is_accepted_by_a_server_with_it(self):
        parser, args = self.make_parser_and_args()
        remote = MagicMock(info={"rtc": True})
        with patch.object(run_policy, "RemotePolicy", return_value=remote):
            self.assertIs(run_policy.connect_to_server(parser, args), remote)

    def test_server_plans_for_the_task(self):
        remote = MagicMock(info={"target_box": None, "top_crop": None})
        plan, _ = run_policy.use_server_policy(remote, "Pick up")
        plan("frame", inference_delay=2)
        remote.plan.assert_called_once_with("frame", task="Pick up", inference_delay=2)

    def test_server_says_how_the_policy_sees(self):
        remote = MagicMock(info={"target_box": BOX_STYLE, "top_crop": None})
        _, view = run_policy.use_server_policy(remote, "Pick up")
        self.assertEqual((view.box_style, view.crop, view.box), (BOX_STYLE, None, None))


if __name__ == "__main__":
    unittest.main()
