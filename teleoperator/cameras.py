"""Best-effort macOS camera previews, isolated from the robot control loop."""

import json
import logging
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import threading
import time

CAMERA_NAMES = ("wrist", "top")
MAX_FRAME_BYTES = 2 * 1024 * 1024
LOGGER = logging.getLogger(__name__)
# FFmpeg 8 logs the opened device under its input stream label instead of the demuxer name.
_OPENED_DEVICE = re.compile(r"^\[(?:avfoundation|AVFoundation indev|in#\d+) @ [^]]+\] '(.+)' opened$")


class JpegBuffer:
    """Keep the latest complete JPEG without accumulating a video queue."""

    def __init__(self):
        self.data = bytearray()

    def feed(self, chunk):
        self.data.extend(chunk)
        latest = None
        while self.data:
            start = self.data.find(b"\xff\xd8")
            if start < 0:
                self.data[:] = self.data[-1:] if self.data[-1] == 255 else b""
                break
            if start:
                del self.data[:start]
            end = self.data.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self.data) > MAX_FRAME_BYTES:
                    self.data.clear()
                    raise ValueError("Camera frame exceeds the preview limit")
                break
            if end + 2 > MAX_FRAME_BYTES:
                self.data.clear()
                raise ValueError("Camera frame exceeds the preview limit")
            latest = bytes(self.data[:end + 2])
            del self.data[:end + 2]
        return latest


def camera_settings(config_path):
    """Read only the two configured cameras; invalid entries stay unavailable."""
    try:
        config = json.loads(Path(config_path).read_text())
        cameras = config["robot"]["cameras"]
        if not isinstance(cameras, dict):
            return {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    settings = {}
    for name in CAMERA_NAMES:
        camera = cameras.get(name)
        if not isinstance(camera, dict) or camera.get("type", "opencv") != "opencv":
            continue
        source = camera.get("macos_device_name")
        if (
            not isinstance(source, str) or not 1 <= len(source) <= 512
            or any(char in source for char in ":\r\n\x00")
            or re.match(r"\s*[+-]?\d", source)
            or source.startswith(("default", "none", "Capture screen"))
        ):
            continue
        width, height = camera.get("width", 1280), camera.get("height", 720)
        fps = camera.get("fps", 30)
        if any(isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= 4096 for value in (width, height)):
            continue
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 120:
            continue
        settings[name] = {"source": source, "width": width, "height": height, "fps": float(fps)}
    return settings


def capture_command(ffmpeg, settings):
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "debug",
        "-f", "avfoundation", "-framerate", format(settings["fps"], ".9g"),
        "-video_size", f"{settings['width']}x{settings['height']}",
        "-pixel_format", "nv12", "-i", settings["source"] + ":none", "-an",
        "-vf", "fps=10,scale=640:-2", "-threads", "1", "-filter_threads", "1",
        "-c:v", "mjpeg", "-q:v", "5", "-f", "image2pipe", "pipe:1",
    ]


class CameraFeeds:
    """Run one owned FFmpeg process per camera and publish only fresh frames.

    The control loop never opens devices or waits for frames. A separate thread
    supervises each capture process, kills stalled captures, and retries them.
    """

    def __init__(self, config_path, *, startup_timeout=12.0, stale_after=2.0, retry_delay=3.0):
        self.settings = camera_settings(config_path)
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            self.ffmpeg = next((path for path in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.access(path, os.X_OK)), None)
        self.startup_timeout = startup_timeout
        self.stale_after = stale_after
        self.retry_delay = retry_delay
        self._frames = {name: None for name in CAMERA_NAMES}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []

    def frame(self, name):
        with self._lock:
            frame = self._frames.get(name)
            if self._stop.is_set() or frame is None or time.monotonic() - frame[0] > self.stale_after:
                return None
            return frame[1]

    def status(self):
        return {name: {"connected": self.frame(name) is not None} for name in CAMERA_NAMES}

    def _publish(self, name, frame):
        with self._lock:
            self._frames[name] = (time.monotonic(), frame) if frame is not None and not self._stop.is_set() else None

    def start(self):
        if self._threads:
            return
        if sys.platform != "darwin" or self.ffmpeg is None:
            LOGGER.warning("Camera preview unavailable: macOS and FFmpeg are required; arm control is unaffected.")
            return
        self._stop.clear()
        for name, settings in self.settings.items():
            thread = threading.Thread(target=self._watch, args=(name, settings), name=f"camera-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def _watch(self, name, settings):
        while not self._stop.is_set():
            try:
                self._capture(name, settings)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            finally:
                self._publish(name, None)
            self._stop.wait(self.retry_delay)

    def _capture(self, name, settings):
        process = subprocess.Popen(
            capture_command(self.ffmpeg, settings),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        parser = JpegBuffer()
        log_line = bytearray()
        verified_device = False
        started = time.monotonic()
        last_frame = None
        try:
            with selectors.DefaultSelector() as selector:
                for pipe in (process.stdout, process.stderr):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ)
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now - (last_frame or started) > (self.stale_after if last_frame is not None else self.startup_timeout):
                        break
                    events = selector.select(timeout=0.1)
                    # Verify the device identity before processing simultaneously ready frames.
                    events.sort(key=lambda event: event[0].fileobj is not process.stderr)
                    for key, _ in events:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if key.fileobj is process.stderr:
                            log_line.extend(chunk)
                            while b"\n" in log_line:
                                end = log_line.index(b"\n")
                                if end > 4096:
                                    raise ValueError("Camera diagnostic line exceeds the preview limit")
                                line = log_line[:end].decode("utf-8", errors="replace").rstrip("\r")
                                del log_line[:end + 1]
                                match = _OPENED_DEVICE.fullmatch(line)
                                if match is None:
                                    continue
                                actual_name = match[1]
                                if actual_name.startswith(("<", "Capture screen")):
                                    raise ValueError("Screen capture is not a camera preview")
                                if actual_name != settings["source"]:
                                    raise ValueError("FFmpeg opened a different camera")
                                verified_device = True
                            if len(log_line) > 4096:
                                raise ValueError("Camera diagnostic line exceeds the preview limit")
                            continue
                        frame = parser.feed(chunk)
                        if frame is not None and verified_device:
                            self._publish(name, frame)
                            last_frame = time.monotonic()
                    if process.poll() is not None and not events:
                        break
                    if not selector.get_map():
                        break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait(timeout=1)
            process.stdout.close()
            process.stderr.close()

    def stop(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        for name in CAMERA_NAMES:
            self._publish(name, None)
