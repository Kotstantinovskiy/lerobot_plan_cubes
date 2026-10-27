"""LiveVideoDataset against LeRobot's stock dataset on the same frames, the recorder's wait and the color cycle; no
camera or arm is opened."""

from pathlib import Path
import sys
import statistics
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_record as recorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from live_video import LiveVideoDataset, TaskCycle, precise_sleep  # noqa: E402

# Wider than 300 px, so the statistics sample every 2nd pixel as they do on the real cameras.
FEATURES = {
    "observation.images.top": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}


def attempt(frames, seed, keep=True):
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    images = [np.roll(scene, 3 * i, axis=1) for i in range(frames)]  # a pan, so consecutive frames differ
    return images, rng.normal(size=(frames, 2)).astype(np.float32), keep


def new(cls, root):
    return cls.create("local/test", 20, features=FEATURES, root=root, image_writer_threads=2, vcodec="h264")


def resumed(cls, root):  # as the recorder opens it with --resume
    dataset = cls("local/test", root=root, vcodec="h264")
    dataset.start_image_writer(num_threads=2)
    return dataset


def record(dataset, attempts):
    try:
        for images, states, keep in attempts:
            for image, state in zip(images, states):
                dataset.add_frame({"observation.images.top": image, "observation.state": state, "action": state,
                                   "task": "Pick up the red cube and place it on the platform."})
            if keep:
                dataset.save_episode()
            else:
                dataset.clear_episode_buffer()
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    return {path.relative_to(dataset.root): path.read_bytes() for path in dataset.root.rglob("*") if path.is_file()}


class LiveVideoTests(unittest.TestCase):
    def assert_same_files(self, stock, live):
        self.assertEqual(sorted(stock), sorted(live))
        for path in stock:  # not assertEqual: its diff of two video files would take minutes
            self.assertTrue(stock[path] == live[path], f"{path} differs")

    def test_same_files_as_stock_recorder(self):
        attempts = [attempt(30, 0), attempt(25, 1)]  # the second video is appended to the first one's file
        with tempfile.TemporaryDirectory() as tmp:
            stock = record(new(LeRobotDataset, Path(tmp, "stock")), attempts)
            live = record(new(LiveVideoDataset, Path(tmp, "live")), attempts)
        self.assertIn(Path("videos/observation.images.top/chunk-000/file-000.mp4"), live)
        self.assert_same_files(stock, live)

    def test_resumed_dataset_same_as_stock(self):
        files = {}
        with tempfile.TemporaryDirectory() as tmp:
            for cls in (LeRobotDataset, LiveVideoDataset):
                root = Path(tmp, cls.__name__)
                record(new(LeRobotDataset, root), [attempt(20, 0)])
                files[cls] = record(resumed(cls, root), [attempt(15, 1), attempt(18, 2)])
        self.assertIn(Path("videos/observation.images.top/chunk-000/file-001.mp4"), files[LiveVideoDataset])
        self.assert_same_files(files[LeRobotDataset], files[LiveVideoDataset])

    def test_discarded_attempt_leaves_nothing(self):
        retry = attempt(12, 1)
        with tempfile.TemporaryDirectory() as tmp:
            stock = record(new(LeRobotDataset, Path(tmp, "stock")), [retry])
            live = record(new(LiveVideoDataset, Path(tmp, "live")), [attempt(30, 0, keep=False), retry])
        self.assert_same_files(stock, live)


class WaitTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "the recorder's wait is replaced on macOS only")
    def test_wait_ends_on_time(self):
        late = []
        for _ in range(20):
            deadline = time.perf_counter() + 0.03
            precise_sleep(0.03)
            late.append(time.perf_counter() - deadline)
        self.assertGreaterEqual(min(late), 0)
        self.assertLess(statistics.median(late), 0.001)  # time.sleep wakes 4-8 ms late here


COLORS = ["blue", "yellow", "red", "green"]


def session(episodes, first, discarded=()):
    """Calls the cycle's wrappers in the order lerobot_record.record calls the recorder's names.

    discarded: attempts (counted from 0) ended with the left arrow. Returns the spoken cues and each attempt's task.
    """
    cues, tasks = [], []
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def record_loop(dataset=None, single_task=None, **_):
        if dataset is not None:
            events["rerecord_episode"] = len(tasks) in discarded
            tasks.append(single_task)

    with patch.multiple(recorder, record_loop=record_loop, log_say=lambda text, *_: cues.append(text),
                        init_keyboard_listener=lambda: (None, events)):
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
        cues, tasks = session(8, first=40)
        self.assertEqual(tasks, [f"Pick up the {color} cube." for color in COLORS * 2])
        self.assertEqual(cues[:4], ["Recording episode 40, blue", "Reset the environment. Next: yellow",
                                    "Recording episode 41, yellow", "Reset the environment. Next: red"])
        self.assertEqual(cues[-1], "Recording episode 47, green")

    def test_discarded_attempt_keeps_its_color(self):
        cues, tasks = session(3, first=0, discarded={1})
        self.assertEqual(tasks, ["Pick up the blue cube.", "Pick up the yellow cube.", "Pick up the yellow cube.",
                                 "Pick up the red cube."])
        self.assertEqual(cues[2:6], ["Recording episode 1, yellow", "Reset the environment. Next: yellow",
                                     "Re-record episode", "Recording episode 1, yellow"])


if __name__ == "__main__":
    unittest.main()
