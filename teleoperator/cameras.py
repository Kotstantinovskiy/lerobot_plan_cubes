"""Best-effort macOS camera previews, isolated from the robot control loop."""

import json
import logging
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

CAMERA_NAMES = ("wrist", "top")
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_LOG_LINE_BYTES = 4096
READ_CHUNK_BYTES = 65536
MAX_SOURCE_LENGTH = 512
MIN_FRAME_SIDE = 16
MAX_FRAME_SIDE = 4096
MIN_FPS = 1
MAX_FPS = 120
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
SELECT_TIMEOUT_SECONDS = 0.1
TERMINATE_TIMEOUT_SECONDS = 0.5
EXIT_TIMEOUT_SECONDS = 1
THREAD_JOIN_SECONDS = 2.0
FFMPEG_FALLBACK_PATHS = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
LOGGER = logging.getLogger(__name__)
# FFmpeg 8 logs the opened device under its input stream label instead of the demuxer name.
_OPENED_DEVICE = re.compile(r"^\[(?:avfoundation|AVFoundation indev|in#\d+) @ [^]]+\] '(.+)' opened$")


class JpegBuffer:
    """Keep the latest complete JPEG without accumulating a video queue."""

    def __init__(self):
        self.data = bytearray()

    def feed(self, chunk: bytes) -> bytes | None:
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
            latest = bytes(self.data[: end + 2])
            del self.data[: end + 2]
        return latest


class CameraSettings(NamedTuple):
    source: str
    width: int
    height: int
    fps: float


def _is_valid_source(source):
    """Accept only a device name: no device index, screen capture or extra stream specifier."""
    return (
        isinstance(source, str)
        and 1 <= len(source) <= MAX_SOURCE_LENGTH
        and not any(char in source for char in ":\r\n\x00")
        and not re.match(r"\s*[+-]?\d", source)
        and not source.startswith(("default", "none", "Capture screen"))
    )


def _is_valid_side(value):
    return not isinstance(value, bool) and isinstance(value, int) and MIN_FRAME_SIDE <= value <= MAX_FRAME_SIDE


def _is_valid_fps(fps):
    return (
        not isinstance(fps, bool) and isinstance(fps, (int, float)) and math.isfinite(fps) and MIN_FPS <= fps <= MAX_FPS
    )


def _read_camera_entry(camera):
    """Return the settings from one camera entry of the config, or None if it cannot be used."""
    if not isinstance(camera, dict) or camera.get("type", "opencv") != "opencv":
        return None
    source = camera.get("macos_device_name")
    width = camera.get("width", DEFAULT_WIDTH)
    height = camera.get("height", DEFAULT_HEIGHT)
    fps = camera.get("fps", DEFAULT_FPS)
    if (
        not _is_valid_source(source)
        or not _is_valid_side(width)
        or not _is_valid_side(height)
        or not _is_valid_fps(fps)
    ):
        return None
    return CameraSettings(source, width, height, float(fps))


def camera_settings(config_path: Path) -> dict[str, CameraSettings]:
    """Read only the two configured cameras; invalid entries stay unavailable."""
    try:
        config = json.loads(Path(config_path).read_text())
        cameras = config["robot"]["cameras"]
        if not isinstance(cameras, dict):
            return {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    entries = {name: _read_camera_entry(cameras.get(name)) for name in CAMERA_NAMES}
    return {name: entry for name, entry in entries.items() if entry is not None}


def capture_command(ffmpeg: str, settings: CameraSettings) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "debug",
        "-f",
        "avfoundation",
        "-framerate",
        format(settings.fps, ".9g"),
        "-video_size",
        f"{settings.width}x{settings.height}",
        "-pixel_format",
        "nv12",
        "-i",
        settings.source + ":none",
        "-an",
        "-vf",
        "fps=10,scale=640:-2",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-c:v",
        "mjpeg",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "pipe:1",
    ]


class DeviceLog:
    """Read FFmpeg's diagnostic output and check the device it says it opened."""

    def __init__(self, expected_name):
        self._expected_name = expected_name
        self._pending = bytearray()
        self.is_verified = False

    def feed(self, chunk):
        """Check each complete line; raise ValueError for an oversized line or the wrong device."""
        self._pending.extend(chunk)
        while b"\n" in self._pending:
            end = self._pending.index(b"\n")
            if end > MAX_LOG_LINE_BYTES:
                raise ValueError("Camera diagnostic line exceeds the preview limit")
            line = self._pending[:end].decode("utf-8", errors="replace").rstrip("\r")
            del self._pending[: end + 1]
            self._check_line(line)
        if len(self._pending) > MAX_LOG_LINE_BYTES:
            raise ValueError("Camera diagnostic line exceeds the preview limit")

    def _check_line(self, line):
        match = _OPENED_DEVICE.fullmatch(line)
        if match is None:
            return
        actual_name = match[1]
        if actual_name.startswith(("<", "Capture screen")):
            raise ValueError("Screen capture is not a camera preview")
        if actual_name != self._expected_name:
            raise ValueError("FFmpeg opened a different camera")
        self.is_verified = True


def _find_ffmpeg():
    """Return the FFmpeg path; an app started from the Finder may not have Homebrew on its PATH."""
    found = shutil.which("ffmpeg")
    if found is not None:
        return found
    return next((path for path in FFMPEG_FALLBACK_PATHS if os.access(path, os.X_OK)), None)


def _end_process(process):
    """Stop a capture process if it still runs and close its pipes."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=EXIT_TIMEOUT_SECONDS)
    process.stdout.close()
    process.stderr.close()


class CameraFeeds:
    """Run one owned FFmpeg process per camera and publish only fresh frames.

    The control loop never opens devices or waits for frames. A separate thread
    supervises each capture process, kills stalled captures, and retries them.
    """

    def __init__(
        self,
        config_path: Path,
        *,
        startup_timeout: float = 12.0,
        stale_after: float = 2.0,
        retry_delay: float = 3.0,
    ) -> None:
        self.settings = camera_settings(config_path)
        self.ffmpeg = _find_ffmpeg()
        self.startup_timeout = startup_timeout
        self.stale_after = stale_after
        self.retry_delay = retry_delay
        self._frames = {name: None for name in CAMERA_NAMES}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []

    def frame(self, name: str) -> bytes | None:
        with self._lock:
            frame = self._frames.get(name)
            if self._stop.is_set() or frame is None or time.monotonic() - frame[0] > self.stale_after:
                return None
            return frame[1]

    def status(self) -> dict[str, dict[str, bool]]:
        return {name: {"connected": self.frame(name) is not None} for name in CAMERA_NAMES}

    def _publish(self, name, frame):
        with self._lock:
            self._frames[name] = (time.monotonic(), frame) if frame is not None and not self._stop.is_set() else None

    def start(self) -> None:
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
                # A failed capture only costs the preview; the loop tries again after a delay.
                pass
            finally:
                self._publish(name, None)
            self._stop.wait(self.retry_delay)

    def _capture(self, name, settings):
        process = subprocess.Popen(
            capture_command(self.ffmpeg, settings),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        try:
            with selectors.DefaultSelector() as selector:
                for pipe in (process.stdout, process.stderr):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ)
                self._stream_frames(name, settings.source, process, selector)
        finally:
            _end_process(process)

    def _stream_frames(self, name, source, process, selector):
        """Publish frames until the capture stalls or ends, or until these feeds stop."""
        frames = JpegBuffer()
        device_log = DeviceLog(source)
        started = time.monotonic()
        last_frame = None
        while not self._stop.is_set():
            allowed_silence = self.stale_after if last_frame is not None else self.startup_timeout
            if time.monotonic() - (last_frame or started) > allowed_silence:
                return
            events = selector.select(timeout=SELECT_TIMEOUT_SECONDS)
            # Verify the device identity before processing simultaneously ready frames.
            events.sort(key=lambda event: event[0].fileobj is not process.stderr)
            for key, _ in events:
                frame = self._read_pipe(key.fileobj, process, selector, device_log, frames)
                if frame is not None and device_log.is_verified:
                    self._publish(name, frame)
                    last_frame = time.monotonic()
            has_ended = process.poll() is not None and not events
            if has_ended or not selector.get_map():
                return

    @staticmethod
    def _read_pipe(pipe, process, selector, device_log, frames):
        """Read what a ready pipe holds; return a frame if that completed one."""
        chunk = os.read(pipe.fileno(), READ_CHUNK_BYTES)
        if not chunk:
            selector.unregister(pipe)
            return None
        if pipe is process.stderr:
            device_log.feed(chunk)
            return None
        return frames.feed(chunk)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=THREAD_JOIN_SECONDS)
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        for name in CAMERA_NAMES:
            self._publish(name, None)
