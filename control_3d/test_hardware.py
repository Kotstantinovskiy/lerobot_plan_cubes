"""Hardware handshake and command checks using a fake serial bus."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from control_3d.hardware import JOINT_NAMES, SO101Hardware


class FakeBus:
    def __init__(self):
        self.is_connected = False
        self.torque = dict.fromkeys(JOINT_NAMES, 0)
        self.positions = dict.fromkeys(JOINT_NAMES, 0.0)
        self.events = []
        self.fail_enable = False
        self.fail_disable = False

    def connect(self):
        self.is_connected = True
        self.events.append("connect")

    def sync_read(self, register):
        self.events.append(("read", register))
        if register == "Torque_Enable":
            return self.torque.copy()
        if register == "Operating_Mode":
            return dict.fromkeys(JOINT_NAMES, 0)
        return self.positions.copy()

    def sync_write(self, register, positions):
        self.events.append(("write", register, positions.copy()))

    def enable_torque(self, num_retry):
        self.events.append("enable")
        self.torque[JOINT_NAMES[0]] = 1
        if isinstance(self.fail_enable, BaseException):
            raise self.fail_enable
        if self.fail_enable:
            raise OSError("enable failed")
        self.torque = dict.fromkeys(JOINT_NAMES, 1)

    def disable_torque(self, num_retry, motors=None):
        self.events.append("disable")
        names = [motors] if isinstance(motors, str) else list(JOINT_NAMES)
        self.events.append(("disable_motors", tuple(names)))
        if self.fail_disable is True or (
            isinstance(self.fail_disable, set) and any(name in self.fail_disable for name in names)
        ):
            raise OSError("disable failed")
        for name in names:
            self.torque[name] = 0

    def disconnect(self, disable_torque):
        self.events.append(("disconnect", disable_torque))
        self.is_connected = False


class FakeRobot:
    def __init__(self):
        self.bus = FakeBus()
        self.is_calibrated = True

    def send_action(self, action):
        self.bus.events.append(("action", action.copy()))
        return action


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        calibration = {
            name: {"id": index, "range_min": 1000, "range_max": 3000}
            for index, name in enumerate(JOINT_NAMES, start=1)
        }
        (root / "arm.json").write_text(json.dumps(calibration))
        config = {"robot": {"type": "so101_follower", "id": "arm", "port": "/dev/fake", "calibration_dir": "."}}
        self.config_path = root / "config.json"
        self.config_path.write_text(json.dumps(config))
        self.robot = FakeRobot()
        factory = patch("control_3d.hardware.make_robot", return_value=self.robot)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.hardware = SO101Hardware(self.config_path)
        self.addCleanup(self.hardware.close)

    def test_connect_only_reads_and_disables_cameras(self):
        self.hardware.connect()
        self.assertNotIn("enable", self.robot.bus.events)
        self.assertFalse(any(event[0] == "write" for event in self.robot.bus.events if isinstance(event, tuple)))
        fields = self.factory.call_args.args[0]
        self.assertEqual(fields["cameras"], {})
        self.assertTrue(fields["use_degrees"])
        self.assertEqual(fields["calibration_dir"], str(self.config_path.parent.resolve()))

    def test_rest_pose_is_converted_from_normalized_units_to_degrees(self):
        config = json.loads(self.config_path.read_text())
        config["rest_pose"] = {**dict.fromkeys(JOINT_NAMES, 50.0), "gripper": 40.0}
        self.config_path.write_text(json.dumps(config))
        hardware = SO101Hardware(self.config_path)
        half_span = hardware.joint_limits["elbow_flex"][1]
        self.assertAlmostEqual(hardware.rest_positions["elbow_flex"], half_span / 2)
        self.assertEqual(hardware.rest_positions["gripper"], 40.0)

    def test_config_without_a_rest_pose_has_no_rest_positions(self):
        self.assertIsNone(self.hardware.rest_positions)

    def test_default_goal_step_leaves_the_elbow_enough_force_to_lift_the_arm(self):
        self.hardware.connect()
        fields = self.factory.call_args.args[0]
        self.assertEqual(fields["max_relative_target"], 10.0)

    def test_arm_latches_latest_pose_before_torque(self):
        self.hardware.connect()
        self.robot.bus.positions["elbow_flex"] = 17.5
        positions = self.hardware.arm()
        self.assertEqual(positions["elbow_flex"], 17.5)
        write_index = next(i for i, event in enumerate(self.robot.bus.events) if isinstance(event, tuple) and event[:2] == ("write", "Goal_Position"))
        self.assertLess(write_index, self.robot.bus.events.index("enable"))
        self.assertEqual(self.robot.bus.events[write_index][2], positions)

    def test_existing_torque_is_preserved_on_refusal(self):
        self.robot.bus.torque["elbow_flex"] = 1
        with self.assertRaisesRegex(RuntimeError, "already enabled"):
            self.hardware.connect()
        self.assertNotIn("disable", self.robot.bus.events)
        self.assertEqual(self.robot.bus.torque["elbow_flex"], 1)
        self.assertFalse(self.robot.bus.is_connected)

    def test_calibration_mismatch_closes_port_without_writing(self):
        self.robot.is_calibrated = False
        with self.assertRaisesRegex(RuntimeError, "calibration differs"):
            self.hardware.connect()
        self.assertFalse(self.robot.bus.is_connected)
        self.assertNotIn("enable", self.robot.bus.events)

    def test_failed_enable_releases_partial_torque(self):
        self.hardware.connect()
        self.robot.bus.fail_enable = True
        with self.assertRaisesRegex(OSError, "enable failed"):
            self.hardware.arm()
        self.assertFalse(self.hardware.armed)
        self.assertFalse(any(self.robot.bus.torque.values()))

    def test_close_still_closes_port_when_disable_fails(self):
        self.hardware.connect()
        self.hardware.arm()
        self.robot.bus.fail_disable = True
        with self.assertRaisesRegex(OSError, "disable failed"):
            self.hardware.close()
        self.assertFalse(self.robot.bus.is_connected)
        self.assertFalse(self.hardware.armed)
        self.assertIsNone(self.hardware.robot)

    def test_release_attempts_other_motors_after_one_servo_fails(self):
        self.hardware.connect()
        self.hardware.arm()
        self.robot.bus.fail_disable = {"shoulder_pan"}
        with self.assertRaisesRegex(OSError, "disable failed"):
            self.hardware.close()
        self.assertEqual(self.robot.bus.torque["shoulder_pan"], 1)
        self.assertTrue(all(self.robot.bus.torque[name] == 0 for name in JOINT_NAMES[1:]))
        self.assertEqual(self.robot.bus.events.count("disable"), 6)
        self.assertFalse(self.hardware.armed)

    def test_reconnect_after_failed_release_requires_a_new_torque_enable(self):
        self.hardware.connect()
        self.hardware.arm()
        self.robot.bus.fail_disable = True
        with self.assertRaises(OSError):
            self.hardware.close()
        self.robot.bus.fail_disable = False
        self.robot.bus.torque = dict.fromkeys(JOINT_NAMES, 0)
        self.robot.bus.events.clear()
        self.hardware.connect()
        self.hardware.arm()
        self.assertIn("enable", self.robot.bus.events)

    def test_partial_torque_cannot_be_reused_as_an_armed_connection(self):
        self.hardware.connect()
        self.hardware.arm()
        self.robot.bus.torque["elbow_flex"] = 0
        with self.assertRaisesRegex(RuntimeError, "torque is inconsistent"):
            self.hardware.arm()

    def test_enable_error_keeps_both_enable_and_release_failures_visible(self):
        self.hardware.connect()
        self.robot.bus.fail_enable = True
        self.robot.bus.fail_disable = {"shoulder_pan"}
        try:
            with self.assertRaisesRegex(RuntimeError, "enable failed.*disable failed"):
                self.hardware.arm()
            self.assertTrue(all(self.robot.bus.torque[name] == 0 for name in JOINT_NAMES[1:]))
            with self.assertRaisesRegex(RuntimeError, "torque is inconsistent"):
                self.hardware.arm()
        finally:
            self.robot.bus.fail_disable = False

    def test_interrupt_during_enable_is_not_hidden_by_a_cleanup_failure(self):
        self.hardware.connect()
        self.robot.bus.fail_enable = KeyboardInterrupt()
        self.robot.bus.fail_disable = {"shoulder_pan"}
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.hardware.arm()
            self.assertEqual(self.robot.bus.events.count("disable"), 6)
        finally:
            self.robot.bus.fail_disable = False

    def test_commands_require_enable_and_calibrated_limits(self):
        self.hardware.connect()
        positions = self.robot.bus.positions.copy()
        with self.assertRaisesRegex(RuntimeError, "Enable movement"):
            self.hardware.send_positions(positions)
        self.hardware.arm()
        for invalid in (float("nan"), float("inf"), 180.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.hardware.send_positions(dict(positions, elbow_flex=invalid))
        self.assertFalse(any(event[0] == "action" for event in self.robot.bus.events if isinstance(event, tuple)))
        positions["gripper"] = 40
        sent = self.hardware.send_positions(positions)
        self.assertEqual(sent, positions)

    def test_tolerated_endpoint_readings_can_be_armed_and_held(self):
        self.hardware.connect()
        elbow_low = self.hardware.joint_limits["elbow_flex"][0]
        self.robot.bus.positions.update(gripper=100.05, elbow_flex=elbow_low - 0.05)
        positions = self.hardware.arm()
        self.assertEqual(positions["gripper"], 100.0)
        self.assertEqual(positions["elbow_flex"], elbow_low)
        self.assertEqual(self.hardware.send_positions(self.hardware.read_positions()), positions)
        self.robot.bus.positions["gripper"] = 100.2
        with self.assertRaisesRegex(ValueError, "outside its calibrated range"):
            self.hardware.read_positions()


if __name__ == "__main__":
    unittest.main()
