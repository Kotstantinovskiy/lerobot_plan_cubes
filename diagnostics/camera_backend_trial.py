"""One measurement of the wrist camera through one capture backend: the worker of compare_camera_backends.py.

The report goes to report.json in the trial's folder and to stdout. Camera libraries are imported inside the functions
that use them, so that the process running the rounds stays free of them.
"""

import json
import os
import selectors
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]

LEROBOT = "lerobot"
FFMPEG = "ffmpeg"
BACKENDS = (LEROBOT, FFMPEG)
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
FFMPEG_LOG_NAME = "ffmpeg.log"
FFMPEG_MODE_WARNING_WORDS = ("fallback", "not support", "selected framerate", "selected video size")
NATIVE_PIXEL_FORMAT = int.from_bytes(b"420v", "big")
NATIVE_FPS_TOLERANCE = 0.01
RGB_CHANNELS = 3
MS_PER_S = 1000
READ_TIMEOUT_MS = 500
LONG_GAP_S = 0.5
FIRST_FRAME_TIMEOUT_S = 15
STALL_TIMEOUT_S = 8
SELECT_TIMEOUT_S = 0.1
READ_CHUNK_BYTES = 1048576
TOP_THREAD_JOIN_TIMEOUT_S = 2
FFMPEG_STOP_TIMEOUT_S = 5


@dataclass
class Trial:
    """A worker's measurement of the wrist camera: its report so far and what must be closed at the end."""

    out: Path
    seconds: int
    wrist_config: dict[str, Any]
    identity: dict[str, Any]
    result: dict[str, Any]
    started: float  # time.monotonic() that the report's startup times count from
    gaps: list[float] = field(default_factory=list)  # seconds between the wrist frames
    measuring: threading.Event = field(default_factory=threading.Event)
    top_stop: threading.Event = field(default_factory=threading.Event)
    top_stats: dict[str, Any] = field(
        default_factory=lambda: {"frames": 0, "timeouts": 0, "max_gap_ms": 0, "gaps_over_500ms": 0}
    )
    wrist_camera: Any = None
    top_camera: Any = None
    top_thread: threading.Thread | None = None
    ffmpeg_process: subprocess.Popen | None = None
    ffmpeg_log: Any = None


@dataclass
class FfmpegReading:
    """How far reading FFmpeg's raw frames has got; times are time.monotonic() values."""

    buffer: bytearray = field(default_factory=bytearray)
    first_arrival: float | None = None
    begin: float | None = None  # when measuring began, after the warm-up
    previous: float | None = None
    last_arrival: float | None = None
    last_frame: Any = None


def read_camera_config(name: str) -> dict[str, Any]:
    return json.loads((PROJECT / "config.json").read_text())["robot"]["cameras"][name]


def save_rgb_image(path: Path, frame) -> None:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def make_camera(config: dict[str, Any]):
    """Build LeRobot's camera from a camera's section of config.json."""
    import draccus
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    fields = dict(config)
    fields.pop("type")
    return OpenCVCamera(draccus.decode(OpenCVCameraConfig, fields))


def new_trial(backend: str, out: Path, seconds: int, *, with_top: bool) -> Trial:
    from lerobot.cameras.opencv.macos_devices import resolve_macos_camera

    wrist_config = read_camera_config("wrist")
    identity = resolve_macos_camera(wrist_config["macos_device_name"])
    result = {
        "backend": backend,
        "identity_before": identity,
        "config": wrist_config,
        "seconds_requested": seconds,
        "frames": 0,
        "timeouts": 0,
        "with_top": with_top,
    }
    return Trial(
        out,
        seconds,
        wrist_config,
        identity,
        result,
        started=time.monotonic(),
    )


def monitor_top(trial: Trial) -> None:
    """Read the top camera until told to stop, keeping statistics of the time the wrist camera is measured."""
    stats = trial.top_stats
    previous = None
    last = None
    while not trial.top_stop.is_set():
        try:
            frame = trial.top_camera.async_read(timeout_ms=READ_TIMEOUT_MS)
        except TimeoutError:
            if trial.measuring.is_set():
                stats["timeouts"] += 1
            continue
        except Exception as exc:
            stats["error"] = repr(exc)
            break
        if not trial.measuring.is_set():
            continue
        now = time.monotonic()
        if previous is not None:
            gap_ms = (now - previous) * MS_PER_S
            stats["max_gap_ms"] = max(stats["max_gap_ms"], gap_ms)
            stats["gaps_over_500ms"] += gap_ms > LONG_GAP_S * MS_PER_S
        previous = now
        last = frame
        stats["frames"] += 1
        if stats["frames"] == 1:
            save_rgb_image(trial.out / "top-first.jpg", frame)
    if last is not None:
        save_rgb_image(trial.out / "top-last.jpg", last)


def start_top_monitor(trial: Trial) -> None:
    """Connect the top camera and keep it streaming on a thread; startup times count from when it is running."""
    from lerobot.cameras.opencv.macos_devices import resolve_macos_camera

    top_config = read_camera_config("top")
    trial.result["top_identity"] = resolve_macos_camera(top_config["macos_device_name"])
    trial.top_camera = make_camera(top_config)
    trial.top_camera.connect()
    trial.top_thread = threading.Thread(target=monitor_top, args=(trial,), daemon=True)
    trial.top_thread.start()
    trial.started = time.monotonic()


def finish_measurement(trial: Trial, begin: float, last_frame) -> None:
    """Note in the report how long measuring took since `begin`, and save the last frame if there was one."""
    trial.result["elapsed_s"] = time.monotonic() - begin
    if last_frame is not None:
        save_rgb_image(trial.out / "last.jpg", last_frame)
        trial.result["shape"] = list(last_frame.shape)


def measure_with_lerobot(trial: Trial) -> None:
    """Read the wrist camera through LeRobot's camera class for trial.seconds."""
    result = trial.result
    trial.wrist_camera = make_camera(trial.wrist_config)
    trial.wrist_camera.connect()
    begin = previous = time.monotonic()
    trial.measuring.set()
    result["startup_s"] = begin - trial.started
    last_frame = None
    while time.monotonic() - begin < trial.seconds:
        try:
            last_frame = trial.wrist_camera.async_read(timeout_ms=READ_TIMEOUT_MS)
        except TimeoutError:
            result["timeouts"] += 1
            continue
        now = time.monotonic()
        trial.gaps.append(now - previous)
        previous = now
        result["frames"] += 1
        if result["frames"] == 1:
            save_rgb_image(trial.out / "first.jpg", last_frame)
    finish_measurement(trial, begin, last_frame)


def matching_frame_duration(device_format, fps: float):
    """The frame duration of the format's first frame rate range that tops out at `fps`, or None."""
    for rate in device_format.videoSupportedFrameRateRanges():
        if abs(rate.maxFrameRate() - fps) < NATIVE_FPS_TOLERANCE:
            return rate.minFrameDuration()
    return None


def native_frame_duration(identity: dict[str, Any], config: dict[str, Any]):
    """The duration the device advertises for the configured size in 420v at the configured frame rate."""
    # Match the device's advertised duration, also used by our native-mode helper.
    import AVFoundation
    import CoreMedia

    devices = [
        device
        for device in AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        if str(device.uniqueID()) == identity["uid"]
    ]
    wanted_mode = (config["width"], config["height"], NATIVE_PIXEL_FORMAT)
    duration = None
    for device_format in devices[0].formats():
        description = device_format.formatDescription()
        size = CoreMedia.CMVideoFormatDescriptionGetDimensions(description)
        mode = (size.width, size.height, int(CoreMedia.CMFormatDescriptionGetMediaSubType(description)))
        if mode != wanted_mode:
            continue
        format_duration = matching_frame_duration(device_format, config["fps"])
        if format_duration is not None:
            duration = format_duration
    if duration is None:
        raise RuntimeError("No matching native frame duration")
    return duration


def ffmpeg_command(config: dict[str, Any], duration) -> list[str]:
    """FFmpeg's command to stream the camera as raw RGB on its stdout, without duplicating or dropping frames."""
    return [
        FFMPEG_PATH,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-f",
        "avfoundation",
        "-framerate",
        f"{duration.timescale}/{duration.value}",
        "-video_size",
        f"{config['width']}x{config['height']}",
        "-pixel_format",
        "nv12",
        "-i",
        config["macos_device_name"] + ":none",
        "-an",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def count_ffmpeg_frame(trial: Trial, reading: FfmpegReading, raw_frame: bytes) -> None:
    """Note a frame's arrival; while measuring, it also counts in the report and becomes the last frame."""
    import numpy as np

    now = time.monotonic()
    reading.last_arrival = now
    if reading.first_arrival is None:
        reading.first_arrival = now
        trial.result["first_frame_s"] = now - trial.started
    if reading.begin is None:
        return
    trial.gaps.append(now - reading.previous)
    reading.previous = now
    trial.result["frames"] += 1
    frame_shape = (trial.wrist_config["height"], trial.wrist_config["width"], RGB_CHANNELS)
    reading.last_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(frame_shape)
    if trial.result["frames"] == 1:
        save_rgb_image(trial.out / "first.jpg", reading.last_frame)


def take_complete_frames(trial: Trial, reading: FfmpegReading, frame_bytes: int) -> None:
    while len(reading.buffer) >= frame_bytes:
        raw_frame = bytes(reading.buffer[:frame_bytes])
        del reading.buffer[:frame_bytes]
        count_ffmpeg_frame(trial, reading, raw_frame)


def begin_measuring_after_warmup(trial: Trial, reading: FfmpegReading, now: float) -> None:
    """Start measuring once the configured warm-up has passed since the first frame."""
    if reading.first_arrival is None or reading.begin is not None:
        return
    if now - reading.first_arrival < trial.wrist_config["warmup_s"]:
        return
    reading.begin = reading.previous = now
    trial.measuring.set()
    trial.result["startup_s"] = now - trial.started


def raise_if_stalled(reading: FfmpegReading, now: float) -> None:
    if reading.last_arrival is not None and now - reading.last_arrival > STALL_TIMEOUT_S:
        raise TimeoutError("FFmpeg stopped delivering frames for 8 seconds")


def read_ffmpeg_frames(trial: Trial, process: subprocess.Popen, reading: FfmpegReading) -> None:
    """Read FFmpeg's stdout until trial.seconds have been measured; raises if FFmpeg fails or goes silent."""
    frame_bytes = trial.wrist_config["width"] * trial.wrist_config["height"] * RGB_CHANNELS
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            now = time.monotonic()
            if reading.begin is not None and now - reading.begin >= trial.seconds:
                return
            if reading.first_arrival is None and now - trial.started > FIRST_FRAME_TIMEOUT_S:
                raise TimeoutError("FFmpeg produced no first frame within 15 seconds")
            begin_measuring_after_warmup(trial, reading, now)
            if process.poll() is not None:
                raise RuntimeError(f"FFmpeg exited early: {process.returncode}")
            if not selector.select(SELECT_TIMEOUT_S):
                raise_if_stalled(reading, now)
                continue
            chunk = os.read(process.stdout.fileno(), READ_CHUNK_BYTES)
            if not chunk:
                continue
            reading.buffer.extend(chunk)
            take_complete_frames(trial, reading, frame_bytes)


def measure_with_ffmpeg(trial: Trial) -> None:
    """Read the wrist camera through an FFmpeg process for trial.seconds after its warm-up."""
    duration = native_frame_duration(trial.identity, trial.wrist_config)
    command = ffmpeg_command(trial.wrist_config, duration)
    trial.result["command"] = command
    trial.ffmpeg_log = (trial.out / FFMPEG_LOG_NAME).open("w")
    trial.ffmpeg_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=trial.ffmpeg_log,
        bufsize=0,
    )
    os.set_blocking(trial.ffmpeg_process.stdout.fileno(), False)
    reading = FfmpegReading()
    read_ffmpeg_frames(trial, trial.ffmpeg_process, reading)
    finish_measurement(trial, reading.begin, reading.last_frame)


def gap_summary(gaps: list[float], frames: int, elapsed_s: float) -> dict[str, float]:
    """The frame rate, the longest gap between frames and how many gaps were long; `gaps` are in seconds."""
    return {
        "fps": frames / elapsed_s,
        "max_gap_ms": max(gaps) * MS_PER_S,
        "gaps_over_500ms": sum(gap > LONG_GAP_S for gap in gaps),
    }


def measure_wrist(trial: Trial, backend: str) -> None:
    if backend == LEROBOT:
        measure_with_lerobot(trial)
    else:
        measure_with_ffmpeg(trial)
    if trial.gaps:
        trial.result.update(gap_summary(trial.gaps, trial.result["frames"], trial.result["elapsed_s"]))


def stop_ffmpeg(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=FFMPEG_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    process.stdout.close()


def ffmpeg_mode_warnings(log_text: str) -> list[str]:
    """The lines of FFmpeg's log that say it did not get the requested mode or picked its own."""
    warnings = []
    for line in log_text.splitlines():
        if any(word in line.lower() for word in FFMPEG_MODE_WARNING_WORDS):
            warnings.append(line)
    return warnings


def finish_trial(trial: Trial) -> None:
    """Close what the trial opened, then write its report.json and print the report."""
    from lerobot.cameras.opencv.macos_devices import resolve_macos_camera

    result = trial.result
    trial.measuring.clear()
    trial.top_stop.set()
    if trial.top_thread is not None:
        trial.top_thread.join(timeout=TOP_THREAD_JOIN_TIMEOUT_S)
        result["top"] = trial.top_stats
    if trial.wrist_camera is not None and trial.wrist_camera.is_connected:
        trial.wrist_camera.disconnect()
    if trial.ffmpeg_process is not None:
        stop_ffmpeg(trial.ffmpeg_process)
    if trial.top_camera is not None and trial.top_camera.is_connected:
        trial.top_camera.disconnect()
    if trial.ffmpeg_log is not None:
        trial.ffmpeg_log.close()
        result["ffmpeg_mode_warnings"] = ffmpeg_mode_warnings((trial.out / FFMPEG_LOG_NAME).read_text())
    result["identity_after"] = resolve_macos_camera(trial.wrist_config["macos_device_name"])
    (trial.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def run_worker(backend: str, out: Path, seconds: int) -> None:
    """Measure the wrist camera alone; the outcome goes to out/report.json."""
    trial = new_trial(
        backend,
        out,
        seconds,
        with_top=False,
    )
    try:
        measure_wrist(trial, backend)
    except Exception as exc:
        trial.result["error"] = repr(exc)
    finally:
        finish_trial(trial)


def run_worker_with_top(backend: str, out: Path, seconds: int) -> None:
    """Measure the wrist camera while the top camera streams; the outcome goes to out/report.json."""
    trial = new_trial(
        backend,
        out,
        seconds,
        with_top=True,
    )
    try:
        start_top_monitor(trial)
        measure_wrist(trial, backend)
    except Exception as exc:
        trial.result["error"] = repr(exc)
    finally:
        finish_trial(trial)
