#!/usr/bin/env python3
"""Camera-only diagnostic. Never opens arm ports or sends motor commands."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.opencv.macos_devices import list_macos_cameras
from lerobot.utils.robot_utils import precise_sleep

PROJECT = Path(__file__).resolve().parents[1]

MS_PER_S = 1000
READ_TIMEOUT_MS = 500
LOOP_PERIOD_S = 0.04
STATUS_PERIOD_S = 15
WAIT_PERCENTILE = 95
NATIVE_PIXEL_FORMAT = "420v"
FOUR_CHARACTER_CODE_BYTES = 4
NATIVE_FPS_TOLERANCE = 0.01
NATIVE_WARMUP_S = 8
NATIVE_WARMUP_READ_TIMEOUT_MS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["wrist", "both"], default="both")
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument(
        "--native-mode", action="store_true", help="Explicitly select the AVFoundation mode after OpenCV opens"
    )
    parser.add_argument("--wrist-width", type=int)
    parser.add_argument("--wrist-height", type=int)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("Duration must be positive")
    if (args.wrist_width is None) != (args.wrist_height is None):
        parser.error("Specify both wrist width and height")
    if args.wrist_width is not None and min(args.wrist_width, args.wrist_height) <= 0:
        parser.error("Image size must be positive")
    return args


def load_camera_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """The cameras of config.json that --only selects, with the --wrist-width and --wrist-height applied."""
    configs = json.loads((PROJECT / "config.json").read_text())["robot"]["cameras"]
    configs = {name: config for name, config in configs.items() if args.only == "both" or name == "wrist"}
    if args.wrist_width is not None:
        configs["wrist"].update(width=args.wrist_width, height=args.wrist_height)
    return configs


def save_rgb_image(path: Path, frame: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def disconnect_if_connected(camera: OpenCVCamera) -> None:
    if camera.is_connected:
        camera.disconnect()


def describe_format(device_format) -> list:
    """An AVFoundation device format as [width, height, pixel format text]."""
    import CoreMedia

    description = device_format.formatDescription()
    size = CoreMedia.CMVideoFormatDescriptionGetDimensions(description)
    subtype = int(CoreMedia.CMFormatDescriptionGetMediaSubType(description))
    pixel_format = subtype.to_bytes(FOUR_CHARACTER_CODE_BYTES, "big").decode()
    return [size.width, size.height, pixel_format]


def find_native_device(device_name: str):
    import AVFoundation

    devices = [
        device
        for device in AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        if str(device.localizedName()) == device_name
    ]
    if len(devices) != 1:
        raise RuntimeError("Named native device missing or ambiguous")
    return devices[0]


def select_native_mode(name: str, config: dict[str, Any]) -> None:
    """Set the camera's device to the configured size in 420v at the configured frame rate."""
    device = find_native_device(config["macos_device_name"])
    before_format = describe_format(device.activeFormat())
    wanted_format = [config["width"], config["height"], NATIVE_PIXEL_FORMAT]
    matches = [device_format for device_format in device.formats() if describe_format(device_format) == wanted_format]
    if not matches:
        raise RuntimeError("Requested native 420v mode unavailable")
    native_format = matches[0]
    rates = [
        rate
        for rate in native_format.videoSupportedFrameRateRanges()
        if abs(rate.maxFrameRate() - config["fps"]) < NATIVE_FPS_TOLERANCE
    ]
    if not rates:
        raise RuntimeError("Requested native frame rate unavailable")
    is_locked, error = device.lockForConfiguration_(None)
    if not is_locked:
        raise RuntimeError(str(error))
    try:
        device.setActiveFormat_(native_format)
        duration = rates[0].minFrameDuration()
        device.setActiveVideoMinFrameDuration_(duration)
        device.setActiveVideoMaxFrameDuration_(duration)
    finally:
        device.unlockForConfiguration()
    print(
        name,
        "native format",
        before_format,
        "->",
        describe_format(device.activeFormat()),
        flush=True,
    )


def warm_up_native_mode(camera: OpenCVCamera) -> None:
    """Read frames for the warm-up time; TimeoutError if none arrived."""
    warmup_deadline = time.perf_counter() + NATIVE_WARMUP_S
    got_frame = False
    while time.perf_counter() < warmup_deadline:
        try:
            camera.async_read(timeout_ms=NATIVE_WARMUP_READ_TIMEOUT_MS)
            got_frame = True
        except TimeoutError:
            pass
    if not got_frame:
        raise TimeoutError("Native format did not produce a frame")


def connect_cameras(
    stack: ExitStack,
    configs: dict[str, dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    *,
    native_mode: bool,
) -> dict[str, OpenCVCamera]:
    """Connect the cameras one by one, adding a status to `statuses` for each camera that succeeds."""
    cameras = {}
    for name, config in configs.items():
        fields = dict(config)
        fields.pop("type")
        camera = OpenCVCamera(draccus.decode(OpenCVCameraConfig, fields))
        stack.callback(disconnect_if_connected, camera)
        camera.connect(warmup=not native_mode)
        if native_mode:
            select_native_mode(name, config)
            warm_up_native_mode(camera)
        cameras[name] = camera
        statuses[name] = {"frames": 0, "timeouts": 0, "errors": []}
    return cameras


def read_and_count_frame(
    name: str,
    camera: OpenCVCamera,
    status: dict[str, Any],
    waits_ms: list[float],
    out: Path,
    start: float,
) -> None:
    """Read one frame into the camera's status; errors are noted there, and raised only if its thread has died."""
    before = time.perf_counter()
    try:
        frame = camera.async_read(timeout_ms=READ_TIMEOUT_MS)
        waits_ms.append((time.perf_counter() - before) * MS_PER_S)
        status["frames"] += 1
        if status["frames"] == 1:
            save_rgb_image(out / f"{name}-first.jpg", frame)
        status["shape"] = list(frame.shape)
    except Exception as exc:
        status["timeouts"] += isinstance(exc, TimeoutError)
        status["errors"].append({"at_s": time.perf_counter() - start, "error": str(exc)})
        print(name, str(exc), flush=True)
        if not camera.thread or not camera.thread.is_alive():
            raise


def frame_counts(statuses: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """The frames and timeouts of each camera's status, for the periodic line."""
    counts = {}
    for name, status in statuses.items():
        counts[name] = {"frames": status["frames"], "timeouts": status["timeouts"]}
    return counts


def poll_cameras(
    cameras: dict[str, OpenCVCamera],
    statuses: dict[str, dict[str, Any]],
    out: Path,
    seconds: int,
) -> tuple[dict[str, list[float]], float]:
    """Read every camera in turn for `seconds`; returns each camera's frame waits in ms and the time this took."""
    waits_ms = {name: [] for name in cameras}
    start = time.perf_counter()
    last_status = start
    while time.perf_counter() - start < seconds:
        tick = time.perf_counter()
        for name, camera in cameras.items():
            read_and_count_frame(
                name,
                camera,
                statuses[name],
                waits_ms[name],
                out,
                start,
            )
        precise_sleep(max(0, LOOP_PERIOD_S - (time.perf_counter() - tick)))
        if time.perf_counter() - last_status >= STATUS_PERIOD_S:
            print(frame_counts(statuses), flush=True)
            last_status = time.perf_counter()
    return waits_ms, time.perf_counter() - start


def add_wait_statistics(
    cameras: dict[str, OpenCVCamera],
    statuses: dict[str, dict[str, Any]],
    waits_ms: dict[str, list[float]],
    out: Path,
) -> None:
    """Note each camera's frame waits in its status and save its latest frame."""
    for name, camera in cameras.items():
        if not waits_ms[name]:
            continue
        statuses[name].update(
            wait_p95_ms=float(np.percentile(waits_ms[name], WAIT_PERCENTILE)),
            wait_max_ms=max(waits_ms[name]),
        )
        save_rgb_image(out / f"{name}-last.jpg", camera.read_latest())


def measure_cameras(
    configs: dict[str, dict[str, Any]],
    report: dict[str, Any],
    out: Path,
    args: argparse.Namespace,
) -> None:
    """Connect the cameras, read them for args.seconds and note the results in `report`; they are closed after."""
    with ExitStack() as stack:
        cameras = connect_cameras(
            stack,
            configs,
            report["cameras"],
            native_mode=args.native_mode,
        )
        waits_ms, report["elapsed_s"] = poll_cameras(
            cameras,
            report["cameras"],
            out,
            args.seconds,
        )
        add_wait_statistics(
            cameras,
            report["cameras"],
            waits_ms,
            out,
        )


def write_report(report: dict[str, Any], out: Path) -> None:
    report["inventory_after"] = list_macos_cameras()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT", out / "report.json", flush=True)
    print(json.dumps(report.get("cameras"), indent=2), flush=True)
    if "fatal_error" in report:
        print(report["fatal_error"], flush=True)


def main():
    args = parse_args()
    configs = load_camera_configs(args)
    out = (
        PROJECT
        / "camera_check"
        / f"diagnostic-{args.only}-{'native' if args.native_mode else 'default'}-{time.time_ns()}"
    )
    out.mkdir(parents=True)
    report = {
        "mode": args.only,
        "native_mode": args.native_mode,
        "duration_requested_s": args.seconds,
        "arms_connected": False,
        "inventory_before": list_macos_cameras(),
        "config": configs,
        "cameras": {},
    }
    try:
        measure_cameras(
            configs,
            report,
            out,
            args,
        )
    except Exception as exc:
        report["fatal_error"] = repr(exc)
    finally:
        write_report(report, out)


if __name__ == "__main__":
    main()
