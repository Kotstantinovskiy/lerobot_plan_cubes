"""Operator-facing status text for refused poses and the landscape remote."""

from types import SimpleNamespace

import numpy as np
import pytest

from app import RISE_FIRST_TEXT, remote_status_text, status_text
from kinematics import Kinematics

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
    }
    return SimpleNamespace(**{**state, **changes})


def test_pose_past_the_limits_refusal_tells_the_operator_to_lift_the_arm():
    assert status_text(refusal_for(PAST_LIMITS_DEGREES)) == "Поза вне пределов: поднимите руку"


def test_low_gripper_refusal_tells_the_operator_to_lift_the_arm():
    assert status_text(refusal_for(LOW_POSE_DEGREES)) == "Захват слишком низко: поднимите руку"


def test_known_status_uses_its_translation():
    assert status_text("Torque released") == "Моторы отпущены."


def test_tracking_pause_tells_the_operator_to_enable_again():
    assert status_text("Paused: arm is not following the commanded pose") == "Рука не успела за командой: включите снова"


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
