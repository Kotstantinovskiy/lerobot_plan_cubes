"""Operator-facing status text for refused poses and the landscape remote."""

from types import SimpleNamespace

import numpy as np
import pytest

from kinematics import Kinematics
from operator_text import RISE_FIRST_TEXT, remote_status_text, status_html, status_text, telemetry_html

# Joint angles in degrees: shoulder and elbow past the model limits, as at the folded rest pose.
PAST_LIMITS_DEGREES = (-2.5, -105.05, 103.8, 79.7, -56.9)
LOW_POSE_DEGREES = (-2.5, -95.0, 95.0, 79.7, -56.9)


def refusal_for(pose_degrees):
    with pytest.raises(ValueError) as error:
        Kinematics().validate(np.deg2rad(pose_degrees))
    return f"Control error: {error.value}"


def controller_state(**changes):
    state = {
        "faulted": False,
        "enabled": False,
        "connected": True,
        "must_rise_first": False,
        "status": "Connected: movement disabled",
        "returning": False,
    }
    return SimpleNamespace(**{**state, **changes})


def test_pose_past_the_limits_refusal_tells_the_operator_to_lift_the_arm():
    assert status_text(refusal_for(PAST_LIMITS_DEGREES)) == "Поза вне пределов: поднимите руку"


def test_low_gripper_refusal_tells_the_operator_to_lift_the_arm():
    assert status_text(refusal_for(LOW_POSE_DEGREES)) == "Захват слишком низко: поднимите руку"


def test_known_status_uses_its_translation():
    assert status_text("Torque released") == "Моторы отпущены."


def test_tracking_pause_tells_the_operator_to_enable_again():
    assert (
        status_text("Paused: arm is not following the commanded pose") == "Рука не успела за командой: включите снова"
    )


def test_return_status_is_translated_for_the_operator():
    assert status_text("Returning to rest") == "Рука возвращается на место"
    assert status_text("Return stopped: arm is not following") == "Возврат остановлен: рука не успевает"


def test_unknown_status_is_shown_unchanged():
    assert status_text("Control error: something new") == "Control error: something new"


def test_remote_says_to_lift_first_while_only_lifting_works():
    assert remote_status_text(controller_state(enabled=True, must_rise_first=True)) == RISE_FIRST_TEXT


def test_remote_reports_enabled_control_above_the_low_zone():
    assert remote_status_text(controller_state(enabled=True)) == "Управление включено"


def test_remote_shows_the_reason_for_a_pause():
    assert remote_status_text(controller_state(status="Torque released")) == "Моторы отпущены."


def test_remote_reports_a_fault_before_anything_else():
    assert remote_status_text(controller_state(enabled=True, faulted=True)) == "Ошибка связи"


def test_remote_reports_preview_without_the_arm():
    assert remote_status_text(controller_state(connected=False)) == "Предпросмотр"


def test_panel_says_the_arm_is_returning_while_it_walks_back():
    html = status_html(controller_state(returning=True, status="Returning to rest"))
    assert '<div class="control-status is-paused">' in html
    assert "Возврат на место" in html


def test_panel_marks_released_motors_as_a_pause_with_its_own_label():
    html = status_html(controller_state(status="Torque released"))
    assert '<div class="control-status is-paused">' in html
    assert "Моторы отпущены" in html


def test_panel_shows_preview_without_the_arm():
    html = status_html(controller_state(connected=False, status="Disconnected: preview only"))
    assert '<div class="control-status is-preview">' in html
    assert "Предпросмотр" in html


def test_panel_explains_an_unknown_status_and_escapes_it():
    html = status_html(controller_state(status="Paused: <odd> reason"))
    assert "<p>Paused: &lt;odd&gt; reason</p>" in html


def test_panel_asks_to_lift_first_while_only_lifting_works():
    html = status_html(controller_state(enabled=True, must_rise_first=True, status="Movement enabled"))
    assert '<div class="control-status is-live">' in html
    assert f"<p>{RISE_FIRST_TEXT}</p>" in html


def test_telemetry_rounds_the_position_and_shows_the_height_limit():
    html = telemetry_html((12.4, -3.6, 100.0), 15.0)
    assert "<div><span>X</span><strong>12</strong></div>" in html
    assert "<div><span>Y</span><strong>-4</strong></div>" in html
    assert "<div><span>Z</span><strong>100</strong></div>" in html
    assert "Минимальная высота 15 мм" in html
