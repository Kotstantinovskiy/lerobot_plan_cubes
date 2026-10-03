"""A fake follower, video, clock and executor for the tests of a policy run. Holds no tests itself.

Nothing here opens a port or a camera, loads a checkpoint or uses the network.
"""

import argparse
import contextlib
import io
import sys
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "recording"))

from arm_control import PolicyFacts, RunVideo  # noqa: E402
from remote_policy import PlannedChunk  # noqa: E402

FPS = 8  # a power of two, so that the fake clock's sums of 1 / FPS come out exact
SHOULDER, ELBOW = "shoulder.pos", "elbow.pos"
JOINT_NAMES = [SHOULDER, ELBOW]
JOINT_LOW, JOINT_HIGH = np.array([0.0, 0.0]), np.array([100.0, 100.0])
STILL_POSE = (50.0, 50.0)
MOVED_POSE = (70.0, 50.0)
TOP_SHAPE = (720, 1280, 3)
WRIST_SHAPE = (48, 64, 3)
FEATURES = {
    "action": {"dtype": "float32", "shape": [2], "names": JOINT_NAMES},
    "observation.state": {"dtype": "float32", "shape": [2], "names": JOINT_NAMES},
    "observation.images.top": {"dtype": "video", "shape": list(TOP_SHAPE), "names": None},
    "observation.images.wrist": {"dtype": "video", "shape": list(WRIST_SHAPE), "names": None},
}


class FakeBus:
    """Writes its calls to the list it shares with its robot."""

    def __init__(self, calls: list[str]):
        self.calls = calls
        self.motors = dict.fromkeys(JOINT_NAMES)
        self.is_connected = False
        self.torque = 0

    def connect(self):
        self.calls.append("bus.connect")
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.calls.append(f"bus.disconnect(disable_torque={disable_torque})")
        self.is_connected = False

    def enable_torque(self, num_retry=0):
        self.calls.append(f"bus.enable_torque(num_retry={num_retry})")
        self.torque = 1

    def sync_read(self, item, num_retry=0):
        self.calls.append(f"bus.sync_read({item})")
        if item == "Torque_Enable":
            return dict.fromkeys(JOINT_NAMES, self.torque)
        return dict.fromkeys(JOINT_NAMES, 42.0)

    def sync_write(self, item, values):
        self.calls.append(f"bus.sync_write({item})")


class FakeRobot:
    """Goes through `poses` and stays at the last one. Raises `send_error`, if set, on send number `failing_send`.

    `before_observation`, if set, is called before each observation, to let time pass.
    """

    def __init__(self, poses=(STILL_POSE,)):
        self.calls = []
        self.bus = FakeBus(self.calls)
        self.calibration = dict.fromkeys(JOINT_NAMES)
        self.calibration_fpath = "calibration.json"
        self.is_calibrated = True
        self.poses = poses
        self.top_frame = np.zeros(TOP_SHAPE, np.uint8)
        self.observation_count = 0
        self.sent = []
        self.failing_send = None
        self.send_error = None
        self.disconnect_error = None
        self.before_observation = None

    def connect(self, calibrate=True):
        self.calls.append(f"robot.connect(calibrate={calibrate})")
        self.bus.is_connected = True
        self.bus.enable_torque()

    def disconnect(self):
        self.calls.append("robot.disconnect")
        if self.disconnect_error:
            raise self.disconnect_error
        self.bus.is_connected = False

    def get_observation(self):
        self.calls.append("robot.get_observation")
        if self.before_observation:
            self.before_observation()
        pose = self.poses[min(self.observation_count, len(self.poses) - 1)]
        self.observation_count += 1
        observation = dict(zip(JOINT_NAMES, pose, strict=True))
        observation["top"] = self.top_frame.copy()
        observation["wrist"] = np.zeros(WRIST_SHAPE, np.uint8)
        return observation

    def send_action(self, action):
        self.calls.append("robot.send_action")
        self.sent.append(action)
        if len(self.sent) == self.failing_send:
            raise self.send_error


class FakeClock:
    """Time that passes only when someone sleeps."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class InlineExecutor:
    """Computes a job at once, so that its result is there at the next step; keeps the arguments of the jobs."""

    def __init__(self):
        self.submitted_guidance = []

    def submit(self, function, frame, **guidance):
        self.submitted_guidance.append(guidance)
        job = Future()
        job.set_result(function(frame, **guidance))
        return job


def silently(function, *args, **kwargs):
    """The function's result, with what it prints kept out of the test output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def make_facts() -> PolicyFacts:
    return PolicyFacts(
        features=FEATURES,
        fps=FPS,
        joint_names=JOINT_NAMES,
        joint_low=JOINT_LOW,
        joint_high=JOINT_HIGH,
        envelope_source="3 episodes",
        policy_name="some/policy on cpu",
    )


def make_args(**overrides) -> argparse.Namespace:
    """run_policy.py's command line as the control loop reads it, one second long by default."""
    options = {
        "color": "red",
        "seconds": 1,
        "max_step": 5.0,
        "n_action_steps": 50,
        "replan": None,
        "start_threshold": 3.0,
        "rtc": False,
        "rtc_horizon": 20,
        "rtc_weight": 10.0,
        "no_rest": False,
        "summary": None,
    }
    return argparse.Namespace(**{**options, **overrides})


def make_chunk(first_value: float = 50.0, length: int = 50, seconds: float = 0.25) -> PlannedChunk:
    """A chunk whose joints all read first_value, first_value + 1, ... from step to step."""
    steps = first_value + np.arange(length, dtype=float)
    actions = np.column_stack([steps, steps])
    return PlannedChunk(actions, actions / 100, seconds)


def make_video() -> RunVideo:
    stream = MagicMock()
    stream.encode.return_value = ["packet"]
    return RunVideo(Path("run.mp4"), MagicMock(), stream)
