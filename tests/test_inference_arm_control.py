"""The arm around the control loop, with a fake robot and video: no port, camera or checkpoint is touched."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "recording"))

import arm_control  # noqa: E402
from arm_control import (  # noqa: E402
    PolicyView,
    build_robot,
    close_video,
    connect_arm,
    disconnect_arm,
    find_target,
    fold_if_holding,
    make_model_frame,
    open_video,
    write_video_frame,
)
from inference_arm_fakes import (  # noqa: E402
    ELBOW,
    FEATURES,
    FPS,
    JOINT_NAMES,
    SHOULDER,
    STILL_POSE,
    TOP_SHAPE,
    FakeRobot,
    make_video,
    silently,
)
from inference_pyramid_helpers import box_centered_at, make_frame  # noqa: E402

BOX_STYLE = {"camera": "observation.images.top", "rgb": [255, 255, 255], "thickness": 2, "margin": 4}
OLD_CROP = {"camera": "observation.images.top", "box": [100, 100, 300, 200]}
QUAD_CROP = {
    "camera": "observation.images.top",
    "quad": [[100, 100], [300, 100], [300, 200], [100, 200]],
    "size": [200, 100],
}


class ModelFrameTests(unittest.TestCase):
    def make_observation(self):
        return dict(
            zip(JOINT_NAMES, STILL_POSE, strict=True),
            top=np.zeros(TOP_SHAPE, np.uint8),
            wrist=np.zeros((4, 4, 3), np.uint8),
        )

    def test_joints_become_the_state(self):
        frame = make_model_frame(self.make_observation(), PolicyView(None, None), FEATURES)
        self.assertEqual(frame["observation.state"].tolist(), list(STILL_POSE))

    def test_box_is_drawn_on_the_observation_so_the_video_shows_it(self):
        observation = self.make_observation()
        make_model_frame(observation, PolicyView(BOX_STYLE, None, box=[100, 100, 140, 140]), FEATURES)
        self.assertEqual(observation["top"][96, 120].tolist(), BOX_STYLE["rgb"])

    def test_frame_shares_no_memory_with_the_observation(self):
        observation = self.make_observation()
        frame = make_model_frame(observation, PolicyView(None, None), FEATURES)
        self.assertFalse(np.shares_memory(frame["observation.images.top"], observation["top"]))

    def test_slice_of_an_old_crop_shares_no_memory_with_the_observation(self):
        observation = self.make_observation()
        frame = make_model_frame(observation, PolicyView(None, OLD_CROP), FEATURES)
        self.assertFalse(np.shares_memory(frame["observation.images.top"], observation["top"]))

    def test_model_gets_the_crop(self):
        frame = make_model_frame(self.make_observation(), PolicyView(None, QUAD_CROP), FEATURES)
        self.assertEqual(frame["observation.images.top"].shape, (100, 200, 3))

    def test_observation_keeps_the_full_frame(self):
        observation = self.make_observation()
        make_model_frame(observation, PolicyView(None, QUAD_CROP), FEATURES)
        self.assertEqual(observation["top"].shape, TOP_SHAPE)


class FindTargetTests(unittest.TestCase):
    def test_frame_of_another_size_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "1280x720"):
            find_target(np.zeros((360, 640, 3), np.uint8), "red")

    def test_color_without_a_cube_is_refused(self):
        frame = make_frame({"blue": (500, 400)})
        with self.assertRaisesRegex(RuntimeError, "No red cube"):
            find_target(frame, "red")

    def test_box_of_the_cube_is_returned(self):
        frame = make_frame({"red": (500, 400), "blue": (700, 400)})
        self.assertEqual(silently(find_target, frame, "red"), box_centered_at(500, 400))


class BuildRobotTests(unittest.TestCase):
    def build(self, robot):
        with (
            patch.object(arm_control.draccus, "decode", return_value="robot config") as decode,
            patch.object(arm_control, "make_robot_from_config", return_value=robot),
        ):
            built = build_robot({"type": "so101_follower"}, 2.5)
        return built, decode

    def test_max_step_limits_the_robot(self):
        _, decode = self.build(FakeRobot())
        self.assertEqual(decode.call_args.args[1], {"type": "so101_follower", "max_relative_target": 2.5})

    def test_incomplete_calibration_is_refused(self):
        robot = FakeRobot()
        robot.calibration = {SHOULDER: None}
        with self.assertRaisesRegex(RuntimeError, "incomplete follower calibration"):
            self.build(robot)

    def test_torque_writes_are_retried(self):
        built, _ = self.build(FakeRobot())
        built.bus.enable_torque()
        self.assertEqual(built.calls, [f"bus.enable_torque(num_retry={arm_control.BUS_RETRIES})"])

    def test_position_reads_are_retried(self):
        built, _ = self.build(FakeRobot())
        self.assertEqual(built.bus.sync_read.keywords, {"num_retry": arm_control.BUS_RETRIES})


class ConnectArmTests(unittest.TestCase):
    def test_stale_goal_is_replaced_before_torque_comes_on(self):
        robot = FakeRobot()
        connect_arm(robot)
        self.assertEqual(
            robot.calls,
            [
                "bus.connect",
                "bus.sync_read(Torque_Enable)",
                "bus.sync_read(Present_Position)",
                "bus.sync_write(Goal_Position)",
                "bus.disconnect(disable_torque=False)",
                "robot.connect(calibrate=False)",
                "bus.enable_torque(num_retry=0)",
            ],
        )

    def test_calibration_mismatch_is_refused(self):
        robot = FakeRobot()
        robot.is_calibrated = False
        with self.assertRaisesRegex(RuntimeError, "Calibration mismatch"):
            connect_arm(robot)

    def test_motors_with_torque_on_are_refused(self):
        robot = FakeRobot()
        robot.bus.torque = 1
        with self.assertRaisesRegex(RuntimeError, "torque disabled"):
            connect_arm(robot)

    def test_refused_arm_keeps_its_torque_and_the_port_is_closed(self):
        robot = FakeRobot()
        robot.bus.torque = 1
        with self.assertRaises(RuntimeError):
            connect_arm(robot)
        self.assertEqual(
            robot.calls,
            ["bus.connect", "bus.sync_read(Torque_Enable)", "bus.disconnect(disable_torque=False)"],
        )


class LettingGoTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.robot.bus.torque = 1
        self.config = {"rest_pose": {SHOULDER: 1.0, ELBOW: 2.0}}
        patcher = patch.object(arm_control, "fold_to_rest")
        self.fold_to_rest = patcher.start()
        self.addCleanup(patcher.stop)

    def fold(self):
        silently(fold_if_holding, self.robot, self.config, FPS)

    def test_holding_arm_folds_to_the_rest_pose(self):
        self.fold()
        self.assertEqual(self.fold_to_rest.call_args.args, (self.robot.bus, self.config["rest_pose"], FPS))

    def test_arm_without_torque_stays_where_it_is(self):
        self.robot.bus.torque = 0
        self.fold()
        self.fold_to_rest.assert_not_called()

    def test_ctrl_c_during_the_fold_does_not_escape(self):
        self.fold_to_rest.side_effect = KeyboardInterrupt
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            fold_if_holding(self.robot, self.config, FPS)
        self.assertIn("Return to rest stopped: KeyboardInterrupt()", printed.getvalue())

    def test_config_without_a_rest_pose_does_not_escape(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            fold_if_holding(self.robot, {}, FPS)
        self.assertIn("Return to rest stopped: KeyError('rest_pose')", printed.getvalue())

    def test_failed_disconnect_still_disables_torque_on_the_bus(self):
        self.robot.bus.is_connected = True
        self.robot.disconnect_error = RuntimeError("camera gone")
        silently(disconnect_arm, self.robot)
        self.assertEqual(self.robot.calls, ["robot.disconnect", "bus.disconnect(disable_torque=True)"])

    def test_ctrl_c_during_the_disconnect_still_disables_torque_on_the_bus(self):
        self.robot.bus.is_connected = True
        self.robot.disconnect_error = KeyboardInterrupt()
        silently(disconnect_arm, self.robot)
        self.assertEqual(self.robot.calls, ["robot.disconnect", "bus.disconnect(disable_torque=True)"])

    def test_bus_closed_before_a_camera_failed_is_not_closed_again(self):
        self.robot.bus.is_connected = False
        self.robot.disconnect_error = RuntimeError("camera gone")
        silently(disconnect_arm, self.robot)
        self.assertEqual(self.robot.calls, ["robot.disconnect"])


class VideoTests(unittest.TestCase):
    def test_video_opens_at_the_path_with_the_fps(self):
        with patch.object(arm_control.av, "open") as av_open:
            video = open_video(Path("run.mp4"), FPS)
        self.assertEqual(av_open.call_args.args, ("run.mp4",))
        self.assertEqual(video.container.add_stream.call_args.kwargs["rate"], FPS)

    def test_video_has_the_size_of_two_cameras_side_by_side(self):
        with patch.object(arm_control.av, "open"):
            video = open_video(Path("run.mp4"), FPS)
        self.assertEqual((video.stream.width, video.stream.height), (1280, 360))

    def test_frame_holds_both_cameras_side_by_side(self):
        video = make_video()
        write_video_frame(video, {"top": np.zeros(TOP_SHAPE, np.uint8), "wrist": np.zeros((48, 64, 3), np.uint8)})
        self.assertEqual(video.stream.encode.call_args.args[0].to_ndarray().shape, (360, 1280, 3))

    def test_packets_of_a_frame_go_to_the_file(self):
        video = make_video()
        write_video_frame(video, {"top": np.zeros(TOP_SHAPE, np.uint8), "wrist": np.zeros((48, 64, 3), np.uint8)})
        video.container.mux.assert_called_once_with("packet")

    def test_closing_writes_what_the_encoder_still_holds(self):
        video = make_video()
        close_video(video)
        video.container.mux.assert_called_once_with("packet")

    def test_closing_closes_the_file(self):
        video = make_video()
        close_video(video)
        video.container.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
