"""Test camera isolation with owned Python workers, without opening devices."""

import json
import subprocess
import sys
import time

import pytest

import cameras
from cameras import CameraFeeds, JpegBuffer, camera_settings, capture_command

JPEG = b"\xff\xd8preview\xff\xd9"
DISCONNECTED = {"wrist": {"connected": False}, "top": {"connected": False}}


@pytest.fixture
def config_path(tmp_path):
    def write(settings):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"robot": {"cameras": settings}}))
        return path

    return write


def camera_config(name="Sport Cam", **overrides):
    return {"type": "opencv", "macos_device_name": name, "index_or_path": 0, **overrides}


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "Camera worker did not reach the expected state"


def worker_script(device="Sport Cam", *, stream=False, ignore_termination=False):
    lines = ["import os, signal, time"]
    if ignore_termination:
        lines.append("signal.signal(signal.SIGTERM, signal.SIG_IGN)")
    if device is not None:
        opened = f"[AVFoundation indev @ 0x1] '{device}' opened\n".encode()
        lines.append(f"os.write(2, {opened!r})")
    if stream:
        lines.extend(["while True:", f"    os.write(1, {JPEG!r})", "    time.sleep(0.03)"])
    else:
        lines.append("time.sleep(60)")
    return "\n".join(lines)


@pytest.fixture
def fake_workers(monkeypatch):
    """Track and clean up only the subprocesses created by these tests."""
    processes = []
    original_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(cameras.subprocess, "Popen", popen)
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


def start_feeds(config_path, monkeypatch, command, **timeouts):
    monkeypatch.setattr(cameras, "capture_command", command)
    feeds = CameraFeeds(config_path, startup_timeout=0.25, stale_after=0.2, retry_delay=0.05, **timeouts)
    feeds.ffmpeg = sys.executable
    feeds.start()
    return feeds


def test_missing_or_empty_camera_config_is_optional(config_path, tmp_path):
    assert camera_settings(tmp_path / "missing.json") == {}
    assert camera_settings(config_path({})) == {}
    path = tmp_path / "broken.json"
    for text in ("not json", "null", "[]", "{}", '{"robot": null}', '{"robot":{"cameras":[]}}'):
        path.write_text(text)
        assert camera_settings(path) == {}


def test_settings_include_only_requested_camera_names_and_prefer_exact_names(config_path):
    path = config_path({"wrist": camera_config("Wrist Cam"), "top": camera_config(), "other": camera_config("Other")})
    settings = camera_settings(path)
    assert set(settings) == {"wrist", "top"}
    assert settings["wrist"].source == "Wrist Cam"
    assert settings["top"].source == "Sport Cam"
    assert settings["top"].width == 1280
    assert settings["top"].height == 720
    assert settings["top"].fps == 30.0


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"type": "realsense"},
        camera_config(""),
        camera_config("Camera:0"),
        camera_config("Camera\nName"),
        camera_config(42),
        camera_config(width=True),
        camera_config(width=0),
        camera_config(height=5000),
        camera_config(fps=True),
        camera_config(fps=float("nan")),
        camera_config(fps=0),
        camera_config(fps=121),
    ],
)
def test_invalid_camera_does_not_disable_the_other_one(config_path, invalid):
    settings = camera_settings(config_path({"top": invalid, "wrist": camera_config("Wrist Cam")}))
    assert set(settings) == {"wrist"}


@pytest.mark.parametrize("invalid_name", ["default", "none", "0", "Capture screen 0"])
def test_names_cannot_select_a_default_device_index_or_screen(config_path, invalid_name):
    assert camera_settings(config_path({"top": camera_config(invalid_name)})) == {}


def test_camera_identity_requires_a_name_instead_of_an_unstable_device_index(config_path):
    for value in (2, True, -1, "2", "/private/tmp/clip.mp4"):
        assert camera_settings(config_path({"top": {"index_or_path": value}})) == {}


def test_capture_command_uses_configured_name_without_numeric_fallback(config_path):
    settings = camera_settings(config_path({"top": camera_config("Absent Camera")}))["top"]
    command = capture_command("/fake/ffmpeg", settings)
    assert command[command.index("-i") + 1] == "Absent Camera:none"
    assert command[command.index("-f") + 1] == "avfoundation"
    assert command[-1] == "pipe:1"
    assert "0:none" not in command


@pytest.mark.parametrize(
    "log_prefix",
    ["[avfoundation @ 0x1]", "[AVFoundation indev @ 0x1]", "[in#0 @ 0x74ec20000]"],
)
def test_opened_device_line_is_recognized_across_ffmpeg_versions(log_prefix):
    match = cameras._OPENED_DEVICE.fullmatch(f"{log_prefix} 'Sport Cam' opened")
    assert match[1] == "Sport Cam"


def test_jpeg_parser_handles_fragmented_markers_noise_and_multiple_frames():
    parser = JpegBuffer()
    assert parser.feed(b"unrelated bytes\xff") is None
    assert parser.feed(b"\xd8first\xff") is None
    assert parser.feed(b"\xd9") == b"\xff\xd8first\xff\xd9"
    assert parser.feed(JPEG + b"\xff\xd8latest\xff\xd9") == b"\xff\xd8latest\xff\xd9"
    assert parser.data == b""
    assert parser.feed(JPEG + b"\xff\xd8partial") == JPEG
    assert parser.feed(b" rest\xff\xd9") == b"\xff\xd8partial rest\xff\xd9"


@pytest.mark.parametrize("complete", [False, True])
def test_jpeg_parser_bounds_incomplete_and_complete_frames(monkeypatch, complete):
    monkeypatch.setattr(cameras, "MAX_FRAME_BYTES", 32)
    parser = JpegBuffer()
    frame = b"\xff\xd8" + b"x" * 32 + (b"\xff\xd9" if complete else b"")
    with pytest.raises(ValueError, match="preview limit"):
        parser.feed(frame)
    assert parser.data == b""
    assert parser.feed(JPEG) == JPEG
    assert parser.feed(b"noise" * 1000) is None
    assert len(parser.data) <= 1


def test_only_fresh_frames_are_exposed_without_waiting_for_a_capture(config_path, monkeypatch):
    feeds = CameraFeeds(config_path({}), stale_after=1)
    monkeypatch.setattr(
        cameras.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("A snapshot must not start capture")
    )
    assert feeds.status() == DISCONNECTED
    feeds._publish("wrist", JPEG)
    assert feeds.frame("wrist") is JPEG
    assert feeds.status() == {"wrist": {"connected": True}, "top": {"connected": False}}
    with feeds._lock:
        feeds._frames["wrist"] = (time.monotonic() - 2, JPEG)
    assert feeds.frame("wrist") is None
    assert feeds.status() == DISCONNECTED
    assert feeds.frame("unknown") is None


def test_missing_ffmpeg_leaves_both_cameras_disconnected(config_path, fake_workers):
    feeds = CameraFeeds(config_path({"top": camera_config()}))
    feeds.ffmpeg = None
    feeds.start()
    feeds.stop()
    assert feeds.status() == DISCONNECTED
    assert not fake_workers


def test_hung_camera_retries_while_other_camera_keeps_streaming(config_path, monkeypatch, fake_workers):
    attempts = {"Sport Cam": 0, "Wrist Cam": 0}

    def command(ffmpeg, settings):
        source = settings.source
        attempts[source] += 1
        script = worker_script(source, stream=source == "Wrist Cam")
        return [sys.executable, "-u", "-c", script]

    feeds = start_feeds(
        config_path({"wrist": camera_config("Wrist Cam"), "top": camera_config()}), monkeypatch, command
    )
    try:
        wait_until(lambda: feeds.frame("wrist") == JPEG and attempts["Sport Cam"] >= 2)
        assert feeds.status() == {"wrist": {"connected": True}, "top": {"connected": False}}
        assert attempts["Wrist Cam"] == 1
    finally:
        feeds.stop()
    assert feeds.status() == DISCONNECTED
    assert all(process.poll() is not None for process in fake_workers)


def test_capture_recovers_after_a_failed_process(config_path, monkeypatch, fake_workers):
    attempts = []

    def command(ffmpeg, settings):
        attempts.append(settings.source)
        script = "raise SystemExit(1)" if len(attempts) == 1 else worker_script(stream=True)
        return [sys.executable, "-u", "-c", script]

    feeds = start_feeds(config_path({"top": camera_config()}), monkeypatch, command)
    try:
        wait_until(lambda: feeds.frame("top") == JPEG)
        assert attempts == ["Sport Cam", "Sport Cam"]
        assert feeds.frame("wrist") is None
    finally:
        feeds.stop()
    assert all(process.poll() is not None for process in fake_workers)


def test_unexpected_device_name_is_never_published_as_requested_camera(config_path, monkeypatch, fake_workers):
    attempts = []

    def command(ffmpeg, settings):
        attempts.append(settings.source)
        return [sys.executable, "-u", "-c", worker_script("Wrong Camera", stream=True)]

    feeds = start_feeds(config_path({"top": camera_config()}), monkeypatch, command)
    try:
        wait_until(lambda: len(attempts) >= 2)
        assert feeds.frame("top") is None
        assert feeds.status() == DISCONNECTED
        assert set(attempts) == {"Sport Cam"}
    finally:
        feeds.stop()


def test_stop_kills_an_unresponsive_owned_process_in_bounded_time(config_path, monkeypatch, fake_workers):
    def command(ffmpeg, settings):
        return [sys.executable, "-u", "-c", worker_script(ignore_termination=True)]

    feeds = start_feeds(config_path({"top": camera_config()}), monkeypatch, command)
    wait_until(lambda: len(fake_workers) == 1)
    # Give the child time to install its SIGTERM handler before stopping it.
    time.sleep(0.08)
    started = time.monotonic()
    feeds.stop()
    assert time.monotonic() - started < 1.5
    assert fake_workers[0].poll() is not None
    assert feeds.status() == DISCONNECTED
    feeds.stop()
