"""The pure helpers of the recording scripts: command lines, checks of the config, interpolation and timing.

No port or camera is opened and no script's main runs.
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
import record  # noqa: E402
import replay  # noqa: E402
import rest  # noqa: E402
import teleoperate  # noqa: E402
import trim_start  # noqa: E402
import verify_recording  # noqa: E402

FOLLOWER_PORT = "/dev/tty.follower"
LEADER_PORT = "/dev/tty.leader"
CAMERAS = {"top": {"type": "opencv", "index_or_path": 0}}


def make_config():
    """A config as load_config returns it."""
    return {
        "robot": {"type": "so101_follower", "port": FOLLOWER_PORT, "id": "follower", "cameras": CAMERAS},
        "teleop": {"type": "so101_leader", "port": LEADER_PORT, "id": "leader"},
        "repo_id": "local/test",
        "dataset_root": "/datasets/test",
        "fps": 20,
        "episode_time_s": 40,
        "reset_time_s": 10,
        "tasks": {"red": "Pick up the red cube.", "blue": "Pick up the blue cube."},
    }


def record_command(colors, *, resume=False):
    """The recorder's command and dataset folder for five episodes of these colors."""
    return record.build_command(
        make_config(),
        colors,
        5,
        resume=resume,
    )


class RecordCommandTests(unittest.TestCase):
    def test_arm_settings_become_dotted_options(self):
        options = record.arm_options(make_config())
        self.assertEqual(options["robot.port"], FOLLOWER_PORT)
        self.assertEqual(options["teleop.port"], LEADER_PORT)

    def test_dicts_and_booleans_are_passed_as_json_and_the_rest_as_text(self):
        arguments = record.option_arguments({"robot.cameras": {"top": 0}, "resume": True, "fps": 20})
        self.assertEqual(arguments, ['--robot.cameras={"top": 0}', "--resume=true", "--fps=20"])

    def test_command_runs_the_live_video_script_with_the_dataset_options(self):
        command, root = record_command(["red"])
        self.assertEqual(command[1], str(record.PROJECT / "recording" / "live_video.py"))
        self.assertIn("--dataset.num_episodes=5", command)
        self.assertIn("--dataset.single_task=Pick up the red cube.", command)
        self.assertEqual(root, Path("/datasets/test"))

    def test_one_color_adds_no_task_cycle(self):
        command, _ = record_command(["red"])
        self.assertFalse(any(argument.startswith(record.TASK_CYCLE) for argument in command))

    def test_several_colors_add_their_tasks_in_turn_as_the_last_argument(self):
        command, _ = record_command(["red", "blue"], resume=True)
        expected_cycle = '[["red", "Pick up the red cube."], ["blue", "Pick up the blue cube."]]'
        self.assertEqual(command[-1], record.TASK_CYCLE + expected_cycle)

    def test_empty_task_is_refused(self):
        config = make_config()
        config["tasks"]["red"] = "  "
        with self.assertRaises(ValueError):
            record.build_command(
                config,
                ["red"],
                5,
                resume=False,
            )


class ValidationTests(unittest.TestCase):
    def test_filled_in_config_passes(self):
        record.validate_arms(make_config())
        record.validate_cameras(make_config())

    def test_placeholder_port_is_refused(self):
        config = make_config()
        config["robot"]["port"] = "SET_FOLLOWER_PORT"
        with self.assertRaisesRegex(ValueError, "robot.port"):
            record.validate_arms(config)

    def test_arms_on_the_same_port_are_refused(self):
        config = make_config()
        config["teleop"]["port"] = FOLLOWER_PORT
        with self.assertRaisesRegex(ValueError, "разные порты"):
            record.validate_arms(config)

    def test_no_cameras_are_refused(self):
        config = make_config()
        config["robot"]["cameras"] = {}
        with self.assertRaisesRegex(ValueError, "хотя бы одна камера"):
            record.validate_cameras(config)

    def test_camera_without_index_is_refused(self):
        config = make_config()
        config["robot"]["cameras"] = {"top": {"type": "opencv"}}
        with self.assertRaisesRegex(ValueError, "камеры top"):
            record.validate_cameras(config)


class TeleoperateCommandTests(unittest.TestCase):
    def test_without_cameras_they_are_emptied_after_both_arms_and_nothing_is_displayed(self):
        command = teleoperate.build_command_without_cameras(make_config())
        self.assertEqual(
            command[3:],
            [
                "--robot.type=so101_follower",
                f"--robot.port={FOLLOWER_PORT}",
                "--robot.id=follower",
                "--teleop.type=so101_leader",
                f"--teleop.port={LEADER_PORT}",
                "--teleop.id=leader",
                "--robot.cameras={}",
                "--fps=20",
                "--display_data=false",
            ],
        )

    def test_with_cameras_they_stay_in_the_robots_options_and_are_displayed(self):
        command = teleoperate.build_command_with_cameras(make_config())
        self.assertEqual(
            command[3:],
            [
                "--robot.type=so101_follower",
                f"--robot.port={FOLLOWER_PORT}",
                "--robot.id=follower",
                '--robot.cameras={"top": {"type": "opencv", "index_or_path": 0}}',
                "--teleop.type=so101_leader",
                f"--teleop.port={LEADER_PORT}",
                "--teleop.id=leader",
                "--fps=20",
                "--display_data=true",
            ],
        )

    def test_seconds_limit_the_session(self):
        command = teleoperate.build_command_without_cameras(make_config(), 30)
        self.assertEqual(command[-1], "--teleop_time_s=30")

    def test_missing_fps_falls_back_to_the_default(self):
        config = make_config()
        del config["fps"]
        self.assertIn("--fps=30", teleoperate.build_command_with_cameras(config))


class FoldToRestHelperTests(unittest.TestCase):
    def test_step_count_follows_the_largest_move_of_the_phase(self):
        start = {"a": 0.0, "b": 0.0}
        end = {"a": 2.5, "b": -7.5}
        step_count = rest.phase_step_count(
            start,
            end,
            ("a", "b"),
            1.0,
        )
        self.assertEqual(step_count, 8)

    def test_phase_with_nothing_to_move_still_takes_one_step(self):
        pose = {"a": 1.0}
        step_count = rest.phase_step_count(
            pose,
            pose,
            ("a",),
            1.0,
        )
        self.assertEqual(step_count, 1)

    def test_command_between_interpolates_every_motor(self):
        command = rest.command_between({"a": 0.0, "b": 10.0}, {"a": 4.0, "b": 10.0}, 0.25)
        self.assertEqual(command, {"a": 1.0, "b": 10.0})

    def test_largest_lag_names_the_joint_furthest_from_its_command(self):
        lag = rest.largest_lag({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": -5.0})
        self.assertEqual(lag, ("b", 5.0))

    def test_blocked_gripper_does_not_count_as_lag(self):
        lag = rest.largest_lag({"a": 0.0, "gripper": 0.0}, {"a": 1.0, "gripper": 50.0})
        self.assertEqual(lag, ("a", 1.0))

    def test_pose_is_formatted_with_one_decimal(self):
        self.assertEqual(rest.format_pose({"a": 1.25, "b": -3.0}), "a 1.2, b -3.0")


class ReplayHelperTests(unittest.TestCase):
    def test_full_speed_keeps_the_actions(self):
        actions = np.array([[0.0], [1.0], [4.0]])
        np.testing.assert_array_equal(replay.slow_trajectory(actions, 1.0), actions)

    def test_half_speed_interpolates_between_the_actions(self):
        actions = np.array([[0.0], [1.0], [4.0]])
        expected = np.array([[0.0], [0.5], [1.0], [2.5], [4.0]])
        np.testing.assert_array_equal(replay.slow_trajectory(actions, 0.5), expected)

    def test_speed_above_one_is_refused(self):
        with self.assertRaises(ValueError):
            replay.slow_trajectory(np.zeros((3, 1)), 1.5)

    def test_recorded_actions_are_stacked_by_frame_and_joint(self):
        frames = pd.DataFrame({"action": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]})
        actions = replay.recorded_actions(frames, ["a.pos", "b.pos"])
        np.testing.assert_array_equal(actions, [[1.0, 2.0], [3.0, 4.0]])

    def test_recorded_actions_with_another_joint_count_are_refused(self):
        frames = pd.DataFrame({"action": [np.array([1.0, 2.0])]})
        with self.assertRaises(ValueError):
            replay.recorded_actions(frames, ["a.pos"])

    def test_recorded_actions_that_are_not_finite_are_refused(self):
        frames = pd.DataFrame({"action": [np.array([np.nan])]})
        with self.assertRaises(ValueError):
            replay.recorded_actions(frames, ["a.pos"])


def decoded_frame(seconds):
    """What PyAV's decoder gives for a frame at this time, in the time base of an MP4 of the recordings."""
    time_base = Fraction(1, 15360)
    return SimpleNamespace(pts=round(seconds * 15360), time_base=time_base, seconds=seconds)


class TrimStartHelperTests(unittest.TestCase):
    def test_first_moving_frame_is_the_first_past_the_threshold(self):
        actions = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, -3.5], [9.0, 9.0]])
        self.assertEqual(trim_start.first_moving_frame(actions), 2)

    def test_actions_that_never_move_have_no_first_moving_frame(self):
        self.assertIsNone(trim_start.first_moving_frame(np.zeros((4, 2))))

    def test_frames_at_range_boundaries_are_chosen_by_frame_number(self):
        frames = [decoded_frame(seconds) for seconds in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)]
        kept = trim_start.frames_in_ranges(frames, [(1, 2), (4, 5)], fps=20)
        self.assertEqual([frame.seconds for frame in kept], [0.05, 0.2])

    def test_boundary_frame_read_just_below_its_time_is_not_kept(self):
        frame_at_50_65_s_read_as_50_6499 = SimpleNamespace(pts=777983, time_base=Fraction(1, 15360))
        kept = trim_start.frames_in_ranges([frame_at_50_65_s_read_as_50_6499], [(1000, 1013)], fps=20)
        self.assertEqual(list(kept), [])

    def test_decoding_stops_after_the_last_range(self):
        frames = iter([decoded_frame(0.0), decoded_frame(0.05), decoded_frame(0.1)])
        self.assertEqual(len(list(trim_start.frames_in_ranges(frames, [(0, 1)], fps=20))), 1)
        self.assertEqual(next(frames).seconds, 0.1)

    def test_distinct_indices_below_the_total_are_valid(self):
        self.assertTrue(trim_start.are_distinct_indices([2, 0], 3))

    def test_repeated_index_is_invalid(self):
        self.assertFalse(trim_start.are_distinct_indices([1, 1], 3))

    def test_index_at_the_total_is_invalid(self):
        self.assertFalse(trim_start.are_distinct_indices([3], 3))


class VerifyRecordingReportTests(unittest.TestCase):
    LOOPS = [
        verify_recording.LoopTiming(start=0.0, end=20.0, frames=400),
        verify_recording.LoopTiming(start=21.0, end=26.0, frames=None),
        verify_recording.LoopTiming(start=26.5, end=46.5, frames=380),
    ]

    def test_episodes_are_reported_and_resets_are_not(self):
        reports = verify_recording.episode_reports(self.LOOPS)
        self.assertEqual(
            reports,
            [
                {"wall_time_s": 20.0, "frames": 400, "actual_fps": 20.0},
                {"wall_time_s": 20.0, "frames": 380, "actual_fps": 19.0},
            ],
        )

    def test_pauses_are_the_time_between_consecutive_loops(self):
        self.assertEqual(verify_recording.pauses_between(self.LOOPS), [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
