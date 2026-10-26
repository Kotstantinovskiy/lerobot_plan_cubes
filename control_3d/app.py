#!/usr/bin/env python3
"""Serve local SO-101 mouse control, starting in preview mode by default."""

import argparse
from html import escape
import json
from pathlib import Path
import secrets
import time

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

from cameras import CameraFeeds
from controller import Controller
from hardware import SO101Hardware
from interaction import Interaction, MODES
from kinematics import JOINT_NAMES, MODEL_PATH, Kinematics
from pairing import local_address, pairing_html
from viewer import ViewerServer, guard_backend

STATUS_TEXT = {
    "Preview: no motor commands": "Моторы отключены. Можно попробовать управление.",
    "Disconnected: preview only": "Рука отключена. Режим предпросмотра.",
    "Connected: movement disabled": "Рука подключена. Движение выключено.",
    "Movement enabled": "Рука следует за движениями.",
    "Torque released": "Моторы отпущены.",
    "Stopped; motors hold the current pose": "Движение остановлено. Моторы удерживают позу.",
    "Control mode changed; movement paused": "Способ управления изменён. Движение приостановлено.",
    "Target synchronized; movement paused": "Цель совмещена с захватом. Движение приостановлено.",
    "Operator disconnected; holding position": "Управляющая вкладка закрыта. Удержание позы.",
    "Operator heartbeat lost; holding position": "Пульт неактивен. Движение остановлено.",
    "Control released; movement paused": "Управление передано. Движение приостановлено.",
    "Control layout changed; movement paused": "Экран управления изменён. Движение приостановлено.",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Allow connecting the real arm from the UI")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lan", action="store_true", help="Allow paired iPhone access over the local network")
    parser.add_argument("--address", help="Mac private IPv4 address or .local hostname for the iPhone QR code")
    parser.add_argument("--no-cameras", action="store_true", help="Leave both video previews disconnected without opening cameras")
    parser.add_argument("--speed", type=float, default=20.0, help="Maximum joint speed in degrees per second")
    parser.add_argument("--min-height-mm", type=float, default=15.0, help="Minimum tool-frame height above the base plane")
    parser.add_argument("--check", action="store_true", help="Validate model and config without opening ports or a server")
    parser.add_argument("--seconds", type=float, help="Optional server lifetime, useful for smoke checks")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.address and not args.lan:
        parser.error("--address requires --lan")
    try:
        if args.lan:
            args.address = local_address(args.address)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.speed) or not 1 <= args.speed <= 45:
        parser.error("speed must be between 1 and 45 degrees per second")
    if not np.isfinite(args.min_height_mm) or not 0 <= args.min_height_mm <= 200:
        parser.error("min-height-mm must be between 0 and 200")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be positive and finite")
    return args


def run(args):
    model = Kinematics(min_height=args.min_height_mm / 1000)
    hardware = SO101Hardware(args.config) if args.hardware else None
    controller = Controller(model, hardware, speed=args.speed)
    interaction = Interaction(controller)
    urdf = yourdfpy.URDF.load(str(MODEL_PATH), load_collision_meshes=False)
    if len(urdf.scene.geometry) == 0:
        raise RuntimeError("The robot meshes did not load")
    if args.check:
        print(f"Model OK: {len(urdf.link_map)} links; five arm joints and one gripper.")
        print(f"Initial tool position (m): {model.forward(controller.angles)[:3, 3].round(4)}")
        print("Hardware configuration OK; no serial ports opened." if hardware else "Preview OK; no hardware needed.")
        return

    server = viser.ViserServer(host="127.0.0.1", port=0, label="Управление")
    try:
        backend_key = guard_backend(server)
    except Exception:
        server.stop()
        raise
    server.gui.configure_theme(
        dark_mode=True, control_width="medium", show_logo=False,
        show_share_button=False, brand_color=(208, 237, 158),
    )
    server.gui.main_panel.dock_right()
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/table", width=1.2, height=1.2, cell_size=0.05, section_size=0.25,
        cell_color=(42, 49, 58), section_color=(65, 75, 85),
        plane_color=(13, 17, 22), plane_opacity=1.0, shadow_opacity=0.35,
        fade_distance=1.5,
    )
    server.scene.add_frame("/arm", show_axes=False)
    visual = ViserUrdf(server, urdf, root_node_name="/arm")
    tool = server.scene.add_frame("/tool", axes_length=0.035, axes_radius=0.001, show_axes=False)
    nose = server.scene.add_icosphere(
        "/nose", radius=0.012, color=(24, 176, 210), opacity=0.7,
        position=model.forward(controller.angles)[:3, 3],
    )
    target = server.scene.add_transform_controls(
        "/target", scale=0.12, disable_rotations=True, depth_test=False,
        position=model.forward(controller.angles)[:3, 3],
        translation_limits=((-0.5, 0.5), (-0.5, 0.5), (model.min_height, 0.6)),
        visible=False,
    )
    server.gui.add_html(
        '<div class="control-brand"><h1>SO—101</h1></div>'
    )
    status = server.gui.add_html("")
    gripper_control = server.gui.add_vector2(
        "so101_gripper", (0.0, 50.0),
        min=(float(np.rad2deg(model.lower[4])), 0.0),
        max=(float(np.rad2deg(model.upper[4])), 100.0), step=0.5,
    )
    keyboard_control = server.gui.add_text("so101_keyboard", initial_value="", hint="0")
    session_control = server.gui.add_text("so101_session", initial_value="")
    stop_button = server.gui.add_button("Стоп", color="red", hint="Остановить движение и удерживать позу.")
    remote_control = server.gui.add_text("so101_remote", initial_value="")
    with server.gui.add_folder("Подключение", visible=args.hardware):
        connect_button = server.gui.add_button("Подключить руку", visible=args.hardware)
        enable_button = server.gui.add_button("Включить движение", visible=args.hardware)
        release_button = server.gui.add_button(
            "Отпустить моторы", visible=args.hardware, hint="Сначала поддержите руку: она перестанет удерживать вес."
        )
        disconnect_button = server.gui.add_button("Отключиться", visible=args.hardware)
    with server.gui.add_folder("Настройки", expand_by_default=False):
        mode_dropdown = server.gui.add_dropdown("Режим", MODES)
        speed_slider = server.gui.add_slider("Скорость, °/с", min=1.0, max=45.0, step=1.0, initial_value=args.speed)
        sync_button = server.gui.add_button("Цель к захвату")
        joint_sliders = []
        with server.gui.add_folder("Суставы", expand_by_default=False):
            for index, name in enumerate(JOINT_NAMES[:4]):
                joint_sliders.append(server.gui.add_slider(
                    name, min=float(np.rad2deg(model.lower[index])), max=float(np.rad2deg(model.upper[index])),
                    step=0.5, initial_value=0.0,
                ))
        with server.gui.add_folder("Положение", expand_by_default=False):
            telemetry = server.gui.add_html("")
        with server.gui.add_folder("Помощь", expand_by_default=False):
            server.gui.add_markdown(
                "Тяните носик захвата или голубую точку. Поверните вид, чтобы двигать в глубину. "
                "WASD поворачивает камеру, колесо меняет масштаб. "
                "Стрелки двигают захват параллельно столу относительно вида. K поднимает, M опускает. "
                "Вращайте круг для поворота запястья. Вертикальный регулятор открывает захват.\n\n"
                "Для настоящей руки: подключите её, проверьте совпадение позы и включите движение. "
                "Проверки столкновений нет."
            )

    # Callbacks only collect input. The control loop handles all motor access.
    submit = interaction.submit

    def bind_button(button, kind):
        @button.on_click
        async def on_click(event):
            submit(event.client_id, kind)

    for button, kind in (
        (connect_button, "connect"), (enable_button, "enable"), (stop_button, "stop"),
        (release_button, "release"), (disconnect_button, "disconnect"), (sync_button, "sync"),
    ):
        bind_button(button, kind)

    @target.on_update
    async def on_target(event):
        if not interaction.direct_mode:
            submit(event.client_id, "target", np.array(event.target.position))

    @mode_dropdown.on_update
    async def on_mode(event):
        submit(event.client_id, "mode", event.target.value)

    async def on_nose_drag(event):
        if interaction.direct_mode:
            submit(event.client_id, "drag", (
                event.target.name, event.phase,
                np.array(event.start_position), np.array(event.end_position),
            ))

    nose.on_drag(on_nose_drag)
    # ViserUrdf builds hierarchical mesh names from the URDF scene graph.
    # Use its public scene lookup to make the actual jaws draggable as well.
    for mesh_name in urdf.scene.graph.nodes_geometry:
        parent = urdf.scene.graph.transforms.parents[mesh_name]
        if parent not in {"gripper_link", "moving_jaw_so101_v1_link"}:
            continue
        frames = [mesh_name]
        while parent != urdf.scene.graph.base_frame:
            frames.append(parent)
            parent = urdf.scene.graph.transforms.parents[parent]
        handle = server.scene.get_handle_by_name("/arm/visual/" + "/".join(reversed(frames)))
        if handle is None:
            raise RuntimeError(f"Could not find gripper mesh: {mesh_name}")
        handle.on_drag(on_nose_drag)

    @gripper_control.on_update
    async def on_gripper_control(event):
        submit(event.client_id, "roll", event.target.value[0])
        submit(event.client_id, "gripper", event.target.value[1])

    @keyboard_control.on_update
    async def on_keyboard(event):
        submit(event.client_id, "keyboard", event.target.value)

    @session_control.on_update
    async def on_session(event):
        submit(event.client_id, "heartbeat", event.target.value)

    @remote_control.on_update
    async def on_remote(event):
        try:
            payload = json.loads(event.target.value)
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            if (
                set(payload) == {"action"} and isinstance(payload["action"], str)
                and payload["action"] in {"connect", "enable", "stop", "layout"}
            ):
                submit(event.client_id, payload["action"])
            return
        submit(event.client_id, "remote", payload)

    @speed_slider.on_update
    async def on_speed(event):
        submit(event.client_id, "speed", event.target.value)

    def bind_joint(slider, index):
        @slider.on_update
        async def on_joint(event):
            submit(event.client_id, "joint", (index, event.target.value))

    for index, slider in enumerate(joint_sliders):
        bind_joint(slider, index)

    @server.on_client_connect
    async def on_connect(client):
        client.camera.position = (0.60, 0.48, 0.42)
        client.camera.look_at = (0.16, 0.0, 0.14)

    def update_controls():
        keyboard_control.hint = str(interaction.keyboard_epoch)
        keyboard_control.disabled = not controller.motion_allowed
        remote_control.hint = json.dumps({
            "epoch": interaction.remote_epoch,
            "hardware": hardware is not None,
            "connected": controller.connected,
            "enabled": controller.enabled,
            "faulted": controller.faulted,
            "status": (
                "Ошибка связи" if controller.faulted else "Управление включено" if controller.enabled
                else "Пауза" if controller.connected else "Предпросмотр"
            ),
        }, ensure_ascii=False)
        mode_dropdown.value = interaction.mode
        speed_slider.value = controller.speed
        target.visible = not interaction.direct_mode
        tool.show_axes = not interaction.direct_mode
        nose.visible = interaction.direct_mode
        target.position = model.forward(controller.target_angles)[:3, 3]
        for index, slider in enumerate(joint_sliders):
            slider.min = float(np.rad2deg(model.lower[index]))
            slider.max = float(np.rad2deg(model.upper[index]))
            slider.value = float(np.rad2deg(controller.target_angles[index]))
        gripper_control.min = (float(np.rad2deg(model.lower[4])), 0.0)
        gripper_control.max = (float(np.rad2deg(model.upper[4])), 100.0)
        gripper_control.value = (float(np.rad2deg(controller.target_angles[4])), controller.target_gripper)

    update_controls()
    cameras = CameraFeeds(args.config)
    viewer = ViewerServer(
        websocket_port=server.get_port(), port=args.port,
        host="0.0.0.0" if args.lan else "127.0.0.1",
        token=secrets.token_urlsafe(32) if args.lan else None,
        advertised_host=args.address if args.lan else None,
        backend_key=backend_key,
        cameras=cameras,
    )
    try:
        viewer.start()
        if not args.no_cameras:
            cameras.start()
    except Exception:
        cameras.stop()
        viewer.stop()
        server.stop()
        raise
    if args.lan:
        with server.gui.add_folder("Подключить iPhone", expand_by_default=True):
            server.gui.add_html(pairing_html(viewer.pairing_url))
            handoff_button = server.gui.add_button(
                "Передать управление", hint="Остановить движение и освободить пульт для другого устройства.",
            )
            bind_button(handoff_button, "yield")
    print(f"Open {viewer.url} — {'hardware available, motors off' if hardware else 'preview only'}", flush=True)
    start = previous_tick = time.monotonic()
    last_panel = 0.0
    try:
        while args.seconds is None or time.monotonic() - start < args.seconds:
            tick = time.monotonic()
            dt = tick - previous_tick
            previous_tick = tick
            if interaction.process(server.get_clients(), dt):
                update_controls()
            was_enabled = controller.enabled
            was_faulted = controller.faulted
            try:
                controller.tick(dt)
            except Exception as exc:
                interaction.pause(f"Stopped: {exc}")
            if (was_enabled and not controller.enabled) or controller.faulted != was_faulted:
                interaction.drag.clear()
                update_controls()
            with server.atomic():
                display_angles = controller.measured_angles if controller.connected else controller.angles
                display_gripper = controller.measured_gripper if controller.connected else controller.gripper
                visual.update_cfg(model.visual_config(display_angles, display_gripper))
                pose = model.forward(display_angles)
                tool.position = pose[:3, 3]
                nose.position = pose[:3, 3]
            if tick - last_panel > 0.2:
                last_panel = tick
                if controller.faulted:
                    mode = "Ошибка связи"
                    state = "error"
                elif controller.enabled:
                    mode = "Управление включено"
                    state = "live"
                else:
                    mode = "Пауза" if controller.connected else "Предпросмотр"
                    if controller.connected and controller.status == "Torque released":
                        mode = "Моторы отпущены"
                    state = "paused" if controller.connected else "preview"
                detail = ""
                if controller.faulted or controller.status not in STATUS_TEXT:
                    detail = f'<p>{escape(STATUS_TEXT.get(controller.status, controller.status))}</p>'
                status.content = (
                    f'<div class="control-status is-{state}">'
                    f'<span class="control-status__label">{mode}</span>'
                    f'{detail}</div>'
                )
                position = pose[:3, 3] * 1000
                telemetry.content = (
                    '<div class="control-telemetry"><div class="control-telemetry__heading">ПОЗИЦИЯ / ММ</div>'
                    '<div class="control-telemetry__values">'
                    + ''.join(f'<div><span>{axis}</span><strong>{value:.0f}</strong></div>' for axis, value in zip("XYZ", position))
                    + f'</div><div class="control-telemetry__footer">Минимальная высота {args.min_height_mm:g} мм</div></div>'
                )
                connect_button.disabled = controller.connected
                enable_button.disabled = not controller.connected or controller.enabled
                release_button.disabled = hardware is None or not controller.connected or not hardware.armed
                disconnect_button.disabled = not controller.connected and not controller.faulted
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tick)))
    finally:
        try:
            controller.disconnect()
        finally:
            try:
                cameras.stop()
            finally:
                try:
                    viewer.stop()
                finally:
                    server.stop()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Server stopped; support the arm before switching off motors.")
