"""What recording/ scripts do to the follower's torque when they refuse, only read, or finish; on a fake follower."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))

import replay  # noqa: E402
import rest  # noqa: E402
from follower_bus import connect_free_bus  # noqa: E402

MOTORS = ("shoulder_pan", "gripper")
ACTION_NAMES = [f"{motor}.pos" for motor in MOTORS]
TWO_FRAMES = np.zeros((2, len(MOTORS)))
FAR_FROM_THE_TARGET = 100.0
TORQUE_KEPT = "bus.disconnect(disable_torque=False)"
TORQUE_RELEASED = "bus.disconnect(disable_torque=True)"
ROBOT_RELEASED = "robot.disconnect"


class FakeBus:
    """Writes its calls to the list it shares with its follower."""

    def __init__(self, calls: list[str], torque: int):
        self.calls = calls
        self.torque = torque
        self.motors = dict.fromkeys(MOTORS)
        self.is_connected = False
        self.read_error = None

    def connect(self):
        self.calls.append("bus.connect")
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.calls.append(f"bus.disconnect(disable_torque={disable_torque})")
        self.is_connected = False

    def sync_read(self, item):
        if self.read_error:
            raise self.read_error
        return dict.fromkeys(MOTORS, self.torque if item == "Torque_Enable" else 0.0)

    def sync_write(self, item, values):
        self.calls.append(f"bus.sync_write({item})")

    def enable_torque(self, num_retry=0):
        self.calls.append("bus.enable_torque")

    def disable_torque(self, num_retry=0):
        self.calls.append("bus.disable_torque")


class FakeFollower:
    def __init__(self, torque: int = 0):
        self.calls = []
        self.bus = FakeBus(self.calls, torque)
        self.calibration = dict.fromkeys(MOTORS)
        self.calibration_fpath = "calibration.json"
        self.action_features = dict.fromkeys(ACTION_NAMES)
        self.is_calibrated = True
        self.observed_position = 0.0

    def configure(self):
        self.calls.append("robot.configure")

    def disconnect(self):
        self.calls.append(ROBOT_RELEASED)
        self.bus.is_connected = False

    def get_observation(self):
        return dict.fromkeys(ACTION_NAMES, self.observed_position)

    def send_action(self, action):
        self.calls.append("robot.send_action")


class ConnectFreeBusTests(unittest.TestCase):
    def test_free_follower_stays_connected(self):
        robot = FakeFollower()
        connect_free_bus(robot)
        self.assertTrue(robot.bus.is_connected)

    def test_free_follower_is_not_touched_otherwise(self):
        robot = FakeFollower()
        connect_free_bus(robot)
        self.assertEqual(robot.calls, ["bus.connect"])

    def test_arm_under_torque_is_refused_and_says_how_to_go_on(self):
        with self.assertRaisesRegex(RuntimeError, "torque disabled.*recording/rest.py"):
            connect_free_bus(FakeFollower(torque=1))

    def test_refused_arm_under_torque_keeps_it(self):
        robot = FakeFollower(torque=1)
        with self.assertRaises(RuntimeError):
            connect_free_bus(robot)
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_calibration_mismatch_keeps_the_torque(self):
        robot = FakeFollower(torque=1)
        robot.is_calibrated = False
        with self.assertRaisesRegex(RuntimeError, "Calibration mismatch"):
            connect_free_bus(robot)
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_failing_read_keeps_the_torque(self):
        robot = FakeFollower()
        robot.bus.read_error = OSError("no status packet")
        with self.assertRaises(OSError):
            connect_free_bus(robot)
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_ctrl_c_while_checking_keeps_the_torque(self):
        robot = FakeFollower()
        robot.bus.read_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            connect_free_bus(robot)
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_port_that_does_not_open_needs_no_closing(self):
        robot = FakeFollower()
        robot.bus.connect = MagicMock(side_effect=ConnectionError("no port"))
        with self.assertRaises(ConnectionError):
            connect_free_bus(robot)
        self.assertEqual(robot.calls, [])


class ReplayTorqueTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(replay.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def replay_on(self, robot: FakeFollower) -> str:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            replay.replay_on_follower(
                robot,
                ACTION_NAMES,
                TWO_FRAMES,
                approach_s=0.1,
                fps=20,
                episode=0,
            )
        return printed.getvalue()

    def test_refused_arm_keeps_its_torque(self):
        robot = FakeFollower(torque=1)
        with self.assertRaisesRegex(RuntimeError, "torque disabled"):
            self.replay_on(robot)
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_refused_arm_gets_no_goal_and_no_configuration(self):
        robot = FakeFollower(torque=1)
        with self.assertRaises(RuntimeError):
            self.replay_on(robot)
        self.assertNotIn("robot.configure", robot.calls)
        self.assertNotIn("bus.sync_write(Goal_Position)", robot.calls)

    def test_stale_goal_is_replaced_before_torque_comes_on(self):
        robot = FakeFollower()
        self.replay_on(robot)
        self.assertLess(robot.calls.index("bus.sync_write(Goal_Position)"), robot.calls.index("robot.configure"))

    def test_finished_replay_releases_the_arm(self):
        robot = FakeFollower()
        printed = self.replay_on(robot)
        self.assertEqual(robot.calls[-1], ROBOT_RELEASED)
        self.assertIn("Follower disconnected; torque disabled.", printed)

    def test_failure_after_the_takeover_releases_the_arm(self):
        robot = FakeFollower()
        robot.observed_position = FAR_FROM_THE_TARGET
        with self.assertRaisesRegex(RuntimeError, "too far from next target"):
            self.replay_on(robot)
        self.assertEqual(robot.calls[-1], ROBOT_RELEASED)


class RestTorqueTests(unittest.TestCase):
    def run_rest(self, robot: FakeFollower, *arguments: str, fold=None) -> str:
        """rest.py's main() on a fake follower, with `fold` instead of the fold; gives what it printed."""
        config = {"rest_pose": dict.fromkeys(MOTORS, 1.0), "robot": {"type": "so101_follower"}}
        fold = fold or MagicMock(return_value=dict.fromkeys(MOTORS, 1.0))
        printed = io.StringIO()
        with (
            patch.object(sys, "argv", ["rest.py", *arguments]),
            patch.object(rest, "load_config", return_value=config),
            patch.object(rest.draccus, "decode"),
            patch.object(rest, "make_robot_from_config", return_value=robot),
            patch.object(rest, "fold_to_rest", fold),
            contextlib.redirect_stdout(printed),
        ):
            rest.main()
        return printed.getvalue()

    def test_dry_run_keeps_the_torque_of_an_arm_holding_its_pose(self):
        robot = FakeFollower(torque=1)
        self.run_rest(robot, "--dry-run")
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_dry_run_keeps_the_torque_off_when_it_was_off(self):
        robot = FakeFollower()
        self.run_rest(robot, "--dry-run")
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_dry_run_says_what_it_did_to_the_torque(self):
        printed = self.run_rest(FakeFollower(torque=1), "--dry-run")
        self.assertIn("Follower disconnected; torque left as it was.", printed)

    def test_dry_run_folds_nothing(self):
        fold = MagicMock()
        self.run_rest(FakeFollower(torque=1), "--dry-run", fold=fold)
        fold.assert_not_called()

    def test_calibration_mismatch_keeps_the_torque(self):
        robot = FakeFollower(torque=1)
        robot.is_calibrated = False
        with self.assertRaisesRegex(RuntimeError, "Calibration mismatch"):
            self.run_rest(robot)
        self.assertEqual(robot.calls, ["bus.connect", TORQUE_KEPT])

    def test_finished_fold_disables_the_torque(self):
        robot = FakeFollower(torque=1)
        printed = self.run_rest(robot)
        self.assertEqual(robot.calls[-1], TORQUE_RELEASED)
        self.assertIn("Follower disconnected; torque disabled.", printed)

    def test_ctrl_c_during_the_fold_disables_the_torque(self):
        robot = FakeFollower(torque=1)
        with self.assertRaises(KeyboardInterrupt):
            self.run_rest(robot, fold=MagicMock(side_effect=KeyboardInterrupt))
        self.assertEqual(robot.calls[-1], TORQUE_RELEASED)

    def test_arm_without_torque_gets_a_fresh_goal_before_the_fold(self):
        robot = FakeFollower()
        self.run_rest(robot)
        self.assertEqual(robot.calls[:3], ["bus.connect", "bus.sync_write(Goal_Position)", "robot.configure"])


if __name__ == "__main__":
    unittest.main()
