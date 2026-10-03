"""Russian text and small HTML fragments the operator sees; pure functions of the controller state."""

from collections.abc import Sequence
from html import escape

from controller import PREVIEW_STATUS, RETURN_STOPPED_STATUS, RETURNING_STATUS, TORQUE_RELEASED_STATUS, Controller

STATUS_TEXT = {
    PREVIEW_STATUS: "Моторы отключены. Можно попробовать управление.",
    "Disconnected: preview only": "Рука отключена. Режим предпросмотра.",
    "Connected: movement disabled": "Рука подключена. Движение выключено.",
    "Movement enabled": "Рука следует за движениями.",
    TORQUE_RELEASED_STATUS: "Моторы отпущены.",
    "Stopped; motors hold the current pose": "Движение остановлено. Моторы удерживают позу.",
    "Control mode changed; movement paused": "Способ управления изменён. Движение приостановлено.",
    "Target synchronized; movement paused": "Цель совмещена с захватом. Движение приостановлено.",
    "Operator disconnected; holding position": "Управляющая вкладка закрыта. Удержание позы.",
    "Operator heartbeat lost; holding position": "Пульт неактивен. Движение остановлено.",
    "Control released; movement paused": "Управление передано. Движение приостановлено.",
    "Control layout changed; movement paused": "Экран управления изменён. Движение приостановлено.",
    RETURNING_STATUS: "Рука возвращается на место",
}
# Refusals that the operator can resolve; unlike STATUS_TEXT, the panel always shows these.
REFUSAL_TEXT = {
    "Control error: Pose exceeds model or calibrated joint limits": "Поза вне пределов: поднимите руку",
    "Control error: Gripper is below the minimum height": "Захват слишком низко: поднимите руку",
    "Paused: arm is not following the commanded pose": "Рука не успела за командой: включите снова",
    RETURN_STOPPED_STATUS: "Возврат остановлен: рука не успевает",
}
RISE_FIRST_TEXT = "Сначала поднимите захват вверх"
FAULT_TEXT = "Ошибка связи"
ENABLED_TEXT = "Управление включено"
PREVIEW_TEXT = "Предпросмотр"


def status_text(status: str) -> str:
    """Return the operator-facing Russian text for a controller status, or the status itself."""
    return REFUSAL_TEXT.get(status) or STATUS_TEXT.get(status, status)


def remote_status_text(controller: Controller) -> str:
    """Return the one line shown in the landscape remote header."""
    if controller.faulted:
        return FAULT_TEXT
    if controller.enabled:
        return RISE_FIRST_TEXT if controller.must_rise_first else ENABLED_TEXT
    return status_text(controller.status) if controller.connected else PREVIEW_TEXT


def _status_badge(controller):
    """Return (label, state name for the page style) of the status badge on the panel."""
    if controller.faulted:
        return FAULT_TEXT, "error"
    if controller.enabled:
        return ENABLED_TEXT, "live"
    state = "paused" if controller.connected else "preview"
    if controller.returning:
        return "Возврат на место", state
    if not controller.connected:
        return PREVIEW_TEXT, state
    if controller.status == TORQUE_RELEASED_STATUS:
        return "Моторы отпущены", state
    return "Пауза", state


def status_html(controller: Controller) -> str:
    """Return the panel's status badge with an explanation when the operator needs one."""
    label, state = _status_badge(controller)
    detail = ""
    if controller.faulted or controller.status not in STATUS_TEXT:
        detail = f"<p>{escape(status_text(controller.status))}</p>"
    elif controller.enabled and controller.must_rise_first:
        detail = f"<p>{RISE_FIRST_TEXT}</p>"
    return f'<div class="control-status is-{state}"><span class="control-status__label">{label}</span>{detail}</div>'


def telemetry_html(position_mm: Sequence[float], min_height_mm: float) -> str:
    """Return the panel's table of the tool position (three values, millimetres)."""
    values = "".join(
        f"<div><span>{axis}</span><strong>{value:.0f}</strong></div>"
        for axis, value in zip("XYZ", position_mm, strict=True)
    )
    return (
        '<div class="control-telemetry">'
        '<div class="control-telemetry__heading">ПОЗИЦИЯ / ММ</div>'
        f'<div class="control-telemetry__values">{values}</div>'
        f'<div class="control-telemetry__footer">Минимальная высота {min_height_mm:g} мм</div>'
        "</div>"
    )
