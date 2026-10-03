#!/usr/bin/env python3
"""Serve Teleoperator, local control of the SO-101 arm, starting in preview mode by default."""

import argparse
import contextlib
import secrets
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy

from cameras import CameraFeeds
from control_panel import ControlPanel
from controller import MAX_SPEED_DEGREES_PER_SECOND, MIN_SPEED_DEGREES_PER_SECOND, Controller
from hardware import SO101Hardware
from interaction import Interaction
from kinematics import MODEL_PATH, Kinematics
from pairing import local_address
from viewer import ViewerServer, guard_backend

DEFAULT_PORT = 8080
MAX_PORT = 65535
DEFAULT_SPEED_DEGREES_PER_SECOND = 20.0
DEFAULT_MIN_HEIGHT_MM = 15.0
MAX_MIN_HEIGHT_MM = 200
MILLIMETRES_PER_METRE = 1000
LOCAL_HOST = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"
CONTROL_PERIOD_SECONDS = 1 / 30
PANEL_PERIOD_SECONDS = 0.2


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Allow connecting the real arm from the UI")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config.json")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--lan", action="store_true", help="Allow paired iPhone access over the local network")
    parser.add_argument("--address", help="Mac private IPv4 address or .local hostname for the iPhone QR code")
    parser.add_argument(
        "--no-cameras", action="store_true", help="Leave both video previews disconnected without opening cameras"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED_DEGREES_PER_SECOND,
        help="Maximum joint speed in degrees per second",
    )
    parser.add_argument(
        "--min-height-mm",
        type=float,
        default=DEFAULT_MIN_HEIGHT_MM,
        help="Minimum tool-frame height above the base plane",
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate model and config without opening ports or a server"
    )
    parser.add_argument("--seconds", type=float, help="Optional server lifetime, useful for smoke checks")
    return parser


def parse_args():
    parser = _build_parser()
    args = parser.parse_args()
    if not 1 <= args.port <= MAX_PORT:
        parser.error("port must be between 1 and 65535")
    if args.address and not args.lan:
        parser.error("--address requires --lan")
    try:
        if args.lan:
            args.address = local_address(args.address)
    except ValueError as error:
        parser.error(str(error))
    is_speed_allowed = MIN_SPEED_DEGREES_PER_SECOND <= args.speed <= MAX_SPEED_DEGREES_PER_SECOND
    if not np.isfinite(args.speed) or not is_speed_allowed:
        parser.error("speed must be between 1 and 45 degrees per second")
    if not np.isfinite(args.min_height_mm) or not 0 <= args.min_height_mm <= MAX_MIN_HEIGHT_MM:
        parser.error("min-height-mm must be between 0 and 200")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be positive and finite")
    return args


def _print_check_report(urdf, controller):
    print(f"Model OK: {len(urdf.link_map)} links; five arm joints and one gripper.")
    print(f"Initial tool position (m): {controller.model.forward(controller.angles)[:3, 3].round(4)}")
    has_hardware = controller.hardware is not None
    print("Hardware configuration OK; no serial ports opened." if has_hardware else "Preview OK; no hardware needed.")


def _control_step(controller, interaction, panel, clients, dt):
    """Apply queued input and advance the motors by one step, keeping the widgets up to date."""
    if interaction.process(clients, dt):
        panel.update_controls()
    was_enabled = controller.enabled
    was_faulted = controller.faulted
    try:
        controller.tick(dt)
    except Exception as error:
        interaction.pause(f"Stopped: {error}")
    if (was_enabled and not controller.enabled) or controller.faulted != was_faulted:
        interaction.drag.clear()
        panel.update_controls()


def _run_control_loop(server, controller, interaction, panel, *, seconds):
    """Run the only loop that touches the motors, for the given seconds or until interrupted."""
    start = previous_tick = time.monotonic()
    last_panel_update = 0.0
    while seconds is None or time.monotonic() - start < seconds:
        tick = time.monotonic()
        _control_step(controller, interaction, panel, server.get_clients(), tick - previous_tick)
        previous_tick = tick
        tool_position = panel.show_arm()
        if tick - last_panel_update > PANEL_PERIOD_SECONDS:
            last_panel_update = tick
            panel.update_status(tool_position)
        time.sleep(max(0.0, CONTROL_PERIOD_SECONDS - (time.monotonic() - tick)))


def run(args):
    model = Kinematics(min_height=args.min_height_mm / MILLIMETRES_PER_METRE)
    hardware = SO101Hardware(args.config) if args.hardware else None
    controller = Controller(model, hardware, speed=args.speed)
    interaction = Interaction(controller)
    urdf = yourdfpy.URDF.load(str(MODEL_PATH), load_collision_meshes=False)
    if len(urdf.scene.geometry) == 0:
        raise RuntimeError("The robot meshes did not load")
    if args.check:
        _print_check_report(urdf, controller)
        return

    # Stops everything in reverse order on the way out, even if one of the steps fails.
    with contextlib.ExitStack() as cleanup:
        server = viser.ViserServer(host=LOCAL_HOST, port=0, label="Управление")
        cleanup.callback(server.stop)
        backend_key = guard_backend(server)
        panel = ControlPanel(server, urdf, controller, interaction, min_height_mm=args.min_height_mm)
        panel.update_controls()
        cameras = CameraFeeds(args.config)
        viewer = ViewerServer(
            websocket_port=server.get_port(),
            port=args.port,
            host=ALL_INTERFACES if args.lan else LOCAL_HOST,
            token=secrets.token_urlsafe(32) if args.lan else None,
            advertised_host=args.address if args.lan else None,
            backend_key=backend_key,
            cameras=cameras,
        )
        cleanup.callback(viewer.stop)
        cleanup.callback(cameras.stop)
        viewer.start()
        if not args.no_cameras:
            cameras.start()
        if args.lan:
            panel.add_pairing(viewer.pairing_url)
        print(f"Open {viewer.url} — {'hardware available, motors off' if hardware else 'preview only'}", flush=True)
        cleanup.callback(controller.disconnect)
        _run_control_loop(server, controller, interaction, panel, seconds=args.seconds)


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Server stopped; support the arm before switching off motors.")
