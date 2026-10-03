"""LiveVideoDataset against LeRobot's stock dataset on the same frames, the recorder's wait and the color cycle; no
camera or arm is opened."""

import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_record as recorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from live_video import LiveVideoDataset, TaskCycle, precise_sleep  # noqa: E402

REPO_ID = "local/test"
FPS = 20
VIDEO_CODEC = "h264"
IMAGE_WRITER_THREADS = 2
IMAGE_SHAPE = (240, 320, 3)  # wider than 300 px, so the statistics sample every 2nd pixel as on the real cameras
JOINT_COUNT = 2
PAN_PIXELS_PER_FRAME = 3
TASK = "Pick up the red cube and place it on the platform."
FEATURES = {
    "observation.images.top": {"dtype": "video", "shape": IMAGE_SHAPE, "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (JOINT_COUNT,), "names": ["a", "b"]},
    "action": {"dtype": "float32", "shape": (JOINT_COUNT,), "names": ["a", "b"]},
}


class Attempt(NamedTuple):
    images: list[np.ndarray]
    states: np.ndarray
    keep: bool  # False: discarded with the left arrow instead of saved


def make_attempt(frame_count, seed, *, keep=True):
    rng = np.random.default_rng(seed)
    scene = rng.integers(
        0,
        256,
        IMAGE_SHAPE,
        dtype=np.uint8,
    )
    # A pan, so consecutive frames differ.
    images = [np.roll(scene, PAN_PIXELS_PER_FRAME * i, axis=1) for i in range(frame_count)]
    return Attempt(images, rng.normal(size=(frame_count, JOINT_COUNT)).astype(np.float32), keep)


def new_dataset(cls, root):
    return cls.create(
        REPO_ID,
        FPS,
        features=FEATURES,
        root=root,
        image_writer_threads=IMAGE_WRITER_THREADS,
        vcodec=VIDEO_CODEC,
    )


def resumed_dataset(cls, root):  # as the recorder opens it with --resume
    dataset = cls(REPO_ID, root=root, vcodec=VIDEO_CODEC)
    dataset.start_image_writer(num_threads=IMAGE_WRITER_THREADS)
    return dataset


def record_attempts(dataset, attempts):
    """Records the attempts into the dataset and returns its files, as {path inside the dataset: bytes}."""
    try:
        for attempt in attempts:
            for image, state in zip(attempt.images, attempt.states, strict=False):
                frame = {"observation.images.top": image, "observation.state": state, "action": state, "task": TASK}
                dataset.add_frame(frame)
            if attempt.keep:
                dataset.save_episode()
            else:
                dataset.clear_episode_buffer()
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    return {path.relative_to(dataset.root): path.read_bytes() for path in dataset.root.rglob("*") if path.is_file()}


def record_resumed_session(cls, root):
    """Starts a dataset with the stock recorder and adds two episodes to it with `cls`; returns its files."""
    record_attempts(new_dataset(LeRobotDataset, root), [make_attempt(20, 0)])
    return record_attempts(resumed_dataset(cls, root), [make_attempt(15, 1), make_attempt(18, 2)])


class LiveVideoTests(unittest.TestCase):
    def assert_same_files(self, stock, live):
        self.assertEqual(sorted(stock), sorted(live))
        for path in stock:  # not assertEqual: its diff of two video files would take minutes
            self.assertTrue(stock[path] == live[path], f"{path} differs")

    def test_same_files_as_stock_recorder(self):
        attempts = [make_attempt(30, 0), make_attempt(25, 1)]  # the second video is appended to the first one's file
        with tempfile.TemporaryDirectory() as tmp:
            stock = record_attempts(new_dataset(LeRobotDataset, Path(tmp, "stock")), attempts)
            live = record_attempts(new_dataset(LiveVideoDataset, Path(tmp, "live")), attempts)
        self.assertIn(Path("videos/observation.images.top/chunk-000/file-000.mp4"), live)
        self.assert_same_files(stock, live)

    def test_resumed_dataset_same_as_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            stock = record_resumed_session(LeRobotDataset, Path(tmp, "stock"))
            live = record_resumed_session(LiveVideoDataset, Path(tmp, "live"))
        self.assertIn(Path("videos/observation.images.top/chunk-000/file-001.mp4"), live)
        self.assert_same_files(stock, live)

    def test_discarded_attempt_leaves_nothing(self):
        retry = make_attempt(12, 1)
        with tempfile.TemporaryDirectory() as tmp:
            stock = record_attempts(new_dataset(LeRobotDataset, Path(tmp, "stock")), [retry])
            live = record_attempts(
                new_dataset(LiveVideoDataset, Path(tmp, "live")),
                [make_attempt(30, 0, keep=False), retry],
            )
        self.assert_same_files(stock, live)


def measure_lateness_s(wait_s, repeats):
    """How late each of `repeats` waits of `wait_s` seconds ended."""
    late = []
    for _ in range(repeats):
        deadline = time.perf_counter() + wait_s
        precise_sleep(wait_s)
        late.append(time.perf_counter() - deadline)
    return late


class WaitTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "the recorder's wait is replaced on macOS only")
    def test_wait_ends_on_time(self):
        late = measure_lateness_s(0.03, repeats=20)
        self.assertGreaterEqual(min(late), 0)
        self.assertLess(statistics.median(late), 0.001)  # time.sleep wakes 4-8 ms late here


COLORS = ["blue", "yellow", "red", "green"]


def run_session(episodes, first, discarded=()):
    """Calls the cycle's wrappers in the order lerobot_record.record calls the recorder's names.

    discarded: attempts (counted from 0) ended with the left arrow. Returns the spoken cues and each attempt's task.
    """
    cues, tasks = [], []
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def record_loop(dataset=None, single_task=None, **_):
        if dataset is not None:
            events["rerecord_episode"] = len(tasks) in discarded
            tasks.append(single_task)

    with patch.multiple(
        recorder,
        record_loop=record_loop,
        log_say=lambda text, *_: cues.append(text),
        init_keyboard_listener=lambda: (None, events),
    ):
        wrapped = TaskCycle([[color, f"Pick up the {color} cube."] for color in COLORS]).patches()
    dataset = SimpleNamespace(num_episodes=first)
    wrapped["init_keyboard_listener"]()
    recorded = 0
    while recorded < episodes:
        wrapped["log_say"](f"Recording episode {dataset.num_episodes}", True)
        wrapped["record_loop"](dataset=dataset, single_task="the session's task")
        if recorded < episodes - 1 or events["rerecord_episode"]:
            wrapped["log_say"]("Reset the environment", True)
            wrapped["record_loop"](single_task="the session's task")
        if events["rerecord_episode"]:
            wrapped["log_say"]("Re-record episode", True)
            events["rerecord_episode"] = False
            continue
        dataset.num_episodes += 1
        recorded += 1
    return cues, tasks


class TaskCycleTests(unittest.TestCase):
    def test_colors_take_turns(self):
        cues, tasks = run_session(8, first=40)
        self.assertEqual(tasks, [f"Pick up the {color} cube." for color in COLORS * 2])
        self.assertEqual(
            cues[:4],
            [
                "Recording episode 40, blue",
                "Reset the environment. Next: yellow",
                "Recording episode 41, yellow",
                "Reset the environment. Next: red",
            ],
        )
        self.assertEqual(cues[-1], "Recording episode 47, green")

    def test_discarded_attempt_keeps_its_color(self):
        cues, tasks = run_session(3, first=0, discarded={1})
        self.assertEqual(
            tasks,
            ["Pick up the blue cube.", "Pick up the yellow cube.", "Pick up the yellow cube.", "Pick up the red cube."],
        )
        self.assertEqual(
            cues[2:6],
            [
                "Recording episode 1, yellow",
                "Reset the environment. Next: yellow",
                "Re-record episode",
                "Recording episode 1, yellow",
            ],
        )


if __name__ == "__main__":
    unittest.main()
