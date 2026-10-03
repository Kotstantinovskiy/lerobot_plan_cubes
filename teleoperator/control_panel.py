"""The 3D scene and control panel: builds the widgets, forwards input and shows the controller state."""

import json

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from controller import MAX_SPEED_DEGREES_PER_SECOND, MIN_SPEED_DEGREES_PER_SECOND, Controller
from interaction import MODES, Interaction
from kinematics import JOINT_NAMES, MAX_GRIPPER_OPENING, MAX_REACH, WRIST_ROLL_INDEX
from operator_text import remote_status_text, status_html, telemetry_html
from pairing import pairing_html

BRAND_COLOR = (208, 237, 158)
NOSE_COLOR = (24, 176, 210)
CAMERA_POSITION = (0.60, 0.48, 0.42)
CAMERA_LOOK_AT = (0.16, 0.0, 0.14)
TARGET_SIDE_LIMITS = (-0.5, 0.5)  # metres
START_GRIPPER_CONTROL = (0.0, 50.0)  # wrist roll in degrees, opening in percent
# The wrist roll has its own round control instead of a slider.
SLIDER_JOINTS = JOINT_NAMES[:WRIST_ROLL_INDEX]
GRIPPER_LINKS = {"gripper_link", "moving_jaw_so101_v1_link"}
REMOTE_ACTIONS = {"connect", "enable", "stop", "layout"}
HELP_TEXT = (
    "Тяните носик захвата или голубую точку. Поверните вид, чтобы двигать в глубину. "
    "WASD поворачивает камеру, колесо меняет масштаб. "
    "Стрелки двигают захват параллельно столу относительно вида. K поднимает, M опускает. "
    "Вращайте круг для поворота запястья. Вертикальный регулятор открывает захват.\n\n"
    "Для настоящей руки: подключите её, проверьте совпадение позы и включите движение. "
    "Проверки столкновений нет."
)


def _degrees(radians):
    return float(np.rad2deg(radians))


def _visual_path(graph, mesh_name):
    """Return the scene path ViserUrdf gives a mesh of the URDF scene graph."""
    frames = [mesh_name]
    parent = graph.transforms.parents[mesh_name]
    while parent != graph.base_frame:
        frames.append(parent)
        parent = graph.transforms.parents[parent]
    return "/arm/visual/" + "/".join(reversed(frames))


def _parse_remote_message(text):
    """Return (kind, value) of a message from the landscape remote, or None to ignore it."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return "remote", payload
    action = payload.get("action")
    if set(payload) != {"action", "sequence"} or not isinstance(action, str) or action not in REMOTE_ACTIONS:
        return None
    return action, None


class ControlPanel:
    """Everything the operator sees. Its callbacks only queue input; the control loop handles all motor access."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: yourdfpy.URDF,
        controller: Controller,
        interaction: Interaction,
        *,
        min_height_mm: float,
    ) -> None:
        self._server = server
        self._urdf = urdf
        self._controller = controller
        self._interaction = interaction
        self._model = controller.model
        self._min_height_mm = min_height_mm
        self._has_hardware = controller.hardware is not None
        self._build_scene()
        self._build_controls()
        self._bind_inputs()

    def _build_scene(self):
        server = self._server
        tool_position = self._model.forward(self._controller.angles)[:3, 3]
        server.gui.configure_theme(
            dark_mode=True,
            control_width="medium",
            show_logo=False,
            show_share_button=False,
            brand_color=BRAND_COLOR,
        )
        server.gui.main_panel.dock_right()
        server.scene.set_up_direction("+z")
        self._add_table()
        server.scene.add_frame("/arm", show_axes=False)
        self._visual = ViserUrdf(server, self._urdf, root_node_name="/arm")
        self._tool = server.scene.add_frame("/tool", axes_length=0.035, axes_radius=0.001, show_axes=False)
        self._nose = server.scene.add_icosphere(
            "/nose",
            radius=0.012,
            color=NOSE_COLOR,
            opacity=0.7,
            position=tool_position,
        )
        self._target = server.scene.add_transform_controls(
            "/target",
            scale=0.12,
            disable_rotations=True,
            depth_test=False,
            position=tool_position,
            translation_limits=(TARGET_SIDE_LIMITS, TARGET_SIDE_LIMITS, (self._model.min_height, MAX_REACH)),
            visible=False,
        )

    def _add_table(self):
        self._server.scene.add_grid(
            "/table",
            width=1.2,
            height=1.2,
            cell_size=0.05,
            section_size=0.25,
            cell_color=(42, 49, 58),
            section_color=(65, 75, 85),
            plane_color=(13, 17, 22),
            plane_opacity=1.0,
            shadow_opacity=0.35,
            fade_distance=1.5,
        )

    def _gripper_control_limits(self):
        """Return (lowest, highest) values of the control: wrist roll in degrees, opening in percent."""
        lowest = (_degrees(self._model.lower[WRIST_ROLL_INDEX]), 0.0)
        highest = (_degrees(self._model.upper[WRIST_ROLL_INDEX]), MAX_GRIPPER_OPENING)
        return lowest, highest

    def _build_controls(self):
        gui = self._server.gui
        gui.add_html('<div class="control-brand"><h1>Teleoperator</h1></div>')
        self._status = gui.add_html("")
        lowest, highest = self._gripper_control_limits()
        self._gripper_control = gui.add_vector2(
            "teleoperator_gripper", START_GRIPPER_CONTROL, min=lowest, max=highest, step=0.5
        )
        self._keyboard_control = gui.add_text("teleoperator_keyboard", initial_value="", hint="0")
        self._session_control = gui.add_text("teleoperator_session", initial_value="")
        self._stop_button = gui.add_button("Стоп", color="red", hint="Остановить движение и удерживать позу.")
        self._remote_control = gui.add_text("teleoperator_remote", initial_value="")
        self._build_connection_folder()
        self._build_settings_folder()

    def _build_connection_folder(self):
        gui = self._server.gui
        with gui.add_folder("Подключение", visible=self._has_hardware):
            self._connect_button = gui.add_button("Подключить руку", visible=self._has_hardware)
            self._enable_button = gui.add_button("Включить движение", visible=self._has_hardware)
            self._release_button = gui.add_button(
                "Отпустить моторы",
                visible=self._has_hardware,
                hint="Сначала поддержите руку: она перестанет удерживать вес.",
            )
            self._disconnect_button = gui.add_button("Отключиться", visible=self._has_hardware)

    def _build_settings_folder(self):
        gui = self._server.gui
        with gui.add_folder("Настройки", expand_by_default=False):
            self._mode_dropdown = gui.add_dropdown("Режим", MODES)
            self._speed_slider = gui.add_slider(
                "Скорость, °/с",
                min=float(MIN_SPEED_DEGREES_PER_SECOND),
                max=float(MAX_SPEED_DEGREES_PER_SECOND),
                step=1.0,
                initial_value=self._controller.speed,
            )
            self._sync_button = gui.add_button("Цель к захвату")
            with gui.add_folder("Суставы", expand_by_default=False):
                self._joint_sliders = [self._add_joint_slider(index, name) for index, name in enumerate(SLIDER_JOINTS)]
            with gui.add_folder("Положение", expand_by_default=False):
                self._telemetry = gui.add_html("")
            with gui.add_folder("Помощь", expand_by_default=False):
                gui.add_markdown(HELP_TEXT)

    def _add_joint_slider(self, index, name):
        return self._server.gui.add_slider(
            name,
            min=_degrees(self._model.lower[index]),
            max=_degrees(self._model.upper[index]),
            step=0.5,
            initial_value=0.0,
        )

    def _bind_inputs(self):
        for button, kind in (
            (self._connect_button, "connect"),
            (self._enable_button, "enable"),
            (self._stop_button, "stop"),
            (self._release_button, "release"),
            (self._disconnect_button, "disconnect"),
            (self._sync_button, "sync"),
        ):
            self._bind_button(button, kind)
        self._target.on_update(self._on_target)
        self._mode_dropdown.on_update(self._on_mode)
        self._nose.on_drag(self._on_tool_drag)
        self._make_jaws_draggable()
        self._gripper_control.on_update(self._on_gripper_control)
        self._keyboard_control.on_update(self._on_keyboard)
        self._session_control.on_update(self._on_session)
        self._remote_control.on_update(self._on_remote)
        self._speed_slider.on_update(self._on_speed)
        for index, slider in enumerate(self._joint_sliders):
            self._bind_joint(slider, index)
        self._server.on_client_connect(self._on_client_connect)

    def _make_jaws_draggable(self):
        graph = self._urdf.scene.graph
        for mesh_name in graph.nodes_geometry:
            if graph.transforms.parents[mesh_name] not in GRIPPER_LINKS:
                continue
            handle = self._server.scene.get_handle_by_name(_visual_path(graph, mesh_name))
            if handle is None:
                raise RuntimeError(f"Could not find gripper mesh: {mesh_name}")
            handle.on_drag(self._on_tool_drag)

    def _bind_button(self, button, kind):
        async def on_click(event):
            self._interaction.submit(event.client_id, kind)

        button.on_click(on_click)

    def _bind_joint(self, slider, index):
        async def on_joint(event):
            self._interaction.submit(event.client_id, "joint", (index, event.target.value))

        slider.on_update(on_joint)

    async def _on_target(self, event):
        if not self._interaction.direct_mode:
            self._interaction.submit(event.client_id, "target", np.array(event.target.position))

    async def _on_mode(self, event):
        self._interaction.submit(event.client_id, "mode", event.target.value)

    async def _on_tool_drag(self, event):
        if not self._interaction.direct_mode:
            return
        grab = np.array(event.start_position)
        pointer = np.array(event.end_position)
        self._interaction.submit(event.client_id, "drag", (event.target.name, event.phase, grab, pointer))

    async def _on_gripper_control(self, event):
        roll, opening = event.target.value
        self._interaction.submit(event.client_id, "roll", roll)
        self._interaction.submit(event.client_id, "gripper", opening)

    async def _on_keyboard(self, event):
        self._interaction.submit(event.client_id, "keyboard", event.target.value)

    async def _on_session(self, event):
        self._interaction.submit(event.client_id, "heartbeat", event.target.value)

    async def _on_remote(self, event):
        message = _parse_remote_message(event.target.value)
        if message is None:
            return
        kind, value = message
        self._interaction.submit(event.client_id, kind, value)

    async def _on_speed(self, event):
        self._interaction.submit(event.client_id, "speed", event.target.value)

    async def _on_client_connect(self, client):
        client.camera.position = CAMERA_POSITION
        client.camera.look_at = CAMERA_LOOK_AT

    def add_pairing(self, pairing_url: str) -> None:
        """Add the folder with the iPhone QR code and the button that hands control over."""
        gui = self._server.gui
        with gui.add_folder("Подключить iPhone", expand_by_default=True):
            gui.add_html(pairing_html(pairing_url))
            handoff_button = gui.add_button(
                "Передать управление",
                hint="Остановить движение и освободить пульт для другого устройства.",
            )
            self._bind_button(handoff_button, "yield")

    def _remote_state_json(self):
        controller = self._controller
        state = {
            "epoch": self._interaction.remote_epoch,
            "hardware": self._has_hardware,
            "connected": controller.connected,
            "enabled": controller.enabled,
            "faulted": controller.faulted,
            "status": remote_status_text(controller),
        }
        return json.dumps(state, ensure_ascii=False)

    def update_controls(self) -> None:
        """Make the input widgets show the controller's state, including limits that depend on calibration."""
        controller = self._controller
        interaction = self._interaction
        self._keyboard_control.hint = str(interaction.keyboard_epoch)
        self._keyboard_control.disabled = not controller.motion_allowed
        self._remote_control.hint = self._remote_state_json()
        self._mode_dropdown.value = interaction.mode
        self._speed_slider.value = controller.speed
        self._target.visible = not interaction.direct_mode
        self._tool.show_axes = not interaction.direct_mode
        self._nose.visible = interaction.direct_mode
        self._target.position = self._model.forward(controller.target_angles)[:3, 3]
        for index, slider in enumerate(self._joint_sliders):
            slider.min = _degrees(self._model.lower[index])
            slider.max = _degrees(self._model.upper[index])
            slider.value = _degrees(controller.target_angles[index])
        lowest, highest = self._gripper_control_limits()
        self._gripper_control.min = lowest
        self._gripper_control.max = highest
        self._gripper_control.value = (_degrees(controller.target_angles[WRIST_ROLL_INDEX]), controller.target_gripper)

    def show_arm(self) -> np.ndarray:
        """Move the 3D arm to the measured pose, or the commanded one in preview; return the tool position (metres)."""
        controller = self._controller
        angles = controller.measured_angles if controller.connected else controller.angles
        gripper = controller.measured_gripper if controller.connected else controller.gripper
        with self._server.atomic():
            self._visual.update_cfg(self._model.visual_config(angles, gripper))
            tool_position = self._model.forward(angles)[:3, 3]
            self._tool.position = tool_position
            self._nose.position = tool_position
        return tool_position

    def update_status(self, tool_position: np.ndarray) -> None:
        """Refresh the status, the position (metres) and which connection buttons can be pressed."""
        controller = self._controller
        hardware = controller.hardware
        self._status.content = status_html(controller)
        self._telemetry.content = telemetry_html(tool_position * 1000, self._min_height_mm)
        self._connect_button.disabled = controller.connected
        self._enable_button.disabled = not controller.connected or controller.enabled
        self._release_button.disabled = hardware is None or not controller.connected or not hardware.armed
        self._disconnect_button.disabled = not controller.connected and not controller.faulted
