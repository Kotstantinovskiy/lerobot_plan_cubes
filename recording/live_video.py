#!/usr/bin/env python3
"""LeRobot's recorder with each episode's videos encoded while the episode is recorded.

The stock recorder writes every camera frame to a PNG and, once the reset after the episode is over, reads them back
and encodes the episode videos; with the 1080p wrist camera the arm stood still about as long as the episode had
lasted (23.5 s after a 20 s one) before the next episode began. Here each camera's frames go straight to an encoder
thread with the stock settings (H.264, g=2, crf=30, the defaults of encode_video_frames), so the videos are the bytes
the PNG route gives and are finished with the episode: the next one starts ~0.6 s after the reset. The episode
statistics read only every n-th pixel of each PNG, so just those pixels are still written, as small PNGs.

A discarded attempt (left arrow) leaves nothing behind. The stock recorder keeps the attempt's PNGs and encodes a
shorter retry with the attempt's extra frames at the end of its video.

On macOS the recorder also waits out each step on a timer the system does not delay (precise_sleep below), so the
20 frames a second the dataset declares are 20 frames a second of real time, not the stock recorder's 18.7.

record.py runs this script with the recorder's own arguments.
"""

from pathlib import Path
import queue
import select
import shutil
import sys
import tempfile
import threading
import time

import av
import numpy as np

from lerobot.datasets.compute_stats import auto_downsample_height_width
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_record as recorder


class VideoStream:
    """One camera's video of one episode, encoded on a background thread as the frames arrive."""

    def __init__(self, path, fps, vcodec):
        self.path = path
        self.frames = queue.Queue()
        self.error = None
        self.thread = threading.Thread(target=self._encode, args=(fps, vcodec), daemon=True)
        self.thread.start()

    def add(self, image):
        self._raise_error()
        self.frames.put(image)

    def close(self):
        """Encodes the frames still queued and finishes the file; returns its path."""
        self.frames.put(None)
        self.thread.join()
        self._raise_error()
        return self.path

    def discard(self):
        self.frames.put(None)
        self.thread.join()
        shutil.rmtree(self.path.parent, ignore_errors=True)

    def _encode(self, fps, vcodec):
        options = {"g": "2", "crf": "30", **({"preset": "12"} if vcodec == "libsvtav1" else {})}
        try:
            with av.open(str(self.path), "w") as output:
                stream = None
                while (image := self.frames.get()) is not None:
                    if stream is None:
                        stream = output.add_stream(vcodec, fps, options=options)
                        stream.pix_fmt = "yuv420p"
                        stream.height, stream.width = image.shape[:2]
                    output.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
                if stream is not None:
                    output.mux(stream.encode())
        except Exception as exc:  # reported to the recording loop by the next add() or by close()
            self.error = exc

    def _raise_error(self):
        if self.error is not None:
            raise RuntimeError(f"Видео {self.path.name} не закодировано") from self.error


class LiveVideoDataset(LeRobotDataset):
    """A LeRobotDataset that encodes the camera frames of the episode being recorded as they are added."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streams = {}  # camera key -> VideoStream of the episode being recorded

    @classmethod
    def create(cls, *args, **kwargs):
        dataset = super().create(*args, **kwargs)
        dataset.streams = {}
        return dataset

    def _save_image(self, image, fpath, compress_level=1):
        key = fpath.parent.parent.name  # images/<camera key>/episode-NNNNNN/frame-NNNNNN.png
        if key not in self.meta.video_keys:
            return super()._save_image(image, fpath, compress_level)
        if key not in self.streams:
            path = Path(tempfile.mkdtemp(dir=self.root)) / f"{key}_{fpath.parent.name}.mp4"
            self.streams[key] = VideoStream(path, self.fps, self.vcodec)
        self.streams[key].add(image)
        # The statistics load the PNG and keep every n-th pixel; a PNG of just those pixels is left as it is.
        sampled = auto_downsample_height_width(image.transpose(2, 0, 1)).transpose(1, 2, 0)
        super()._save_image(np.ascontiguousarray(sampled), fpath, compress_level)

    def save_episode(self, episode_data=None, parallel_encoding=True):
        # The parallel route encodes PNGs in worker processes; here the videos only need finishing.
        super().save_episode(episode_data, parallel_encoding=False)

    def _encode_temporary_episode_video(self, video_key, episode_index):
        path = self.streams.pop(video_key).close()
        shutil.rmtree(self._get_image_file_dir(episode_index, video_key))
        return path

    def clear_episode_buffer(self, delete_images=True):
        if delete_images and self.streams:  # an attempt being discarded
            self._wait_image_writer()
            for key, stream in self.streams.items():
                stream.discard()
                shutil.rmtree(self._get_image_file_dir(self.episode_buffer["episode_index"], key))
            self.streams.clear()
        super().clear_episode_buffer(delete_images)

    def finalize(self):
        for stream in self.streams.values():  # an episode cut short by an error
            stream.discard()
        self.streams.clear()
        super().finalize()


NOTE_NSECONDS = 0x4  # EVFILT_TIMER flags from macOS <sys/event.h>
NOTE_CRITICAL = 0x20


def precise_sleep(seconds):
    """Waits on a kqueue timer flagged critical, which macOS fires on time instead of coalescing it with others.

    LeRobot's precise_sleep sleeps until 3 ms before the deadline and spins from there, but macOS wakes a sleeping
    thread up to 5 ms late, 10 ms while the keyboard and mouse are idle: the recorder's 50 ms steps took 53.4 ms on
    average. With this timer they take 50.1 ms.
    """
    deadline = time.perf_counter() + seconds
    while (remaining := deadline - time.perf_counter()) > 0:
        timer = select.kevent(0, select.KQ_FILTER_TIMER, select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                              NOTE_NSECONDS | NOTE_CRITICAL, int(remaining * 1e9))
        select.kqueue().control([timer], 1)


# The names the recorder looks up when it runs that this script replaces.
RECORDER_PATCHES = {"LeRobotDataset": LiveVideoDataset}
if sys.platform == "darwin":  # elsewhere LeRobot's wait is time.sleep, which is on time there
    RECORDER_PATCHES["precise_sleep"] = precise_sleep


def main():
    for name, value in RECORDER_PATCHES.items():
        setattr(recorder, name, value)
    recorder.main()


if __name__ == "__main__":
    main()
