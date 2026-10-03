"""Serial failures and lost connections: the fault stays until someone deals with it."""

import numpy as np
import pytest

from fakes import connected_controller


def test_failed_serial_read_stops_further_updates():
    controller, hardware = connected_controller()
    controller.enable()

    def fail_read():
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="serial link lost"):
        controller.tick(0.05)
    assert not controller.enabled
    assert not hardware.sent


def test_lost_connection_cannot_turn_enabled_motion_into_preview_or_resume_it():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    hardware.connected = False
    with pytest.raises(RuntimeError, match="connection lost"):
        controller.tick(0.1)
    assert not controller.enabled
    assert controller.faulted
    frozen = controller.angles.copy()
    sent_count = len(hardware.sent)
    # A recovered serial link must not resume an old mouse target.
    hardware.connected = True
    controller.tick(0.1)
    assert np.array_equal(controller.angles, frozen)
    assert len(hardware.sent) == sent_count
    recovered_position = hardware.positions["shoulder_pan"]
    controller.enable()
    assert not controller.faulted
    controller.tick(0.1)
    assert hardware.sent[-1]["shoulder_pan"] == hardware.positions["shoulder_pan"]
    assert np.rad2deg(controller.target_angles[0]) == pytest.approx(recovered_position)


@pytest.mark.parametrize("enabled", [False, True])
def test_serial_failure_is_latched_instead_of_retried_on_every_tick(enabled):
    controller, hardware = connected_controller()
    if enabled:
        controller.enable()
    reads = []

    def fail_read():
        reads.append(True)
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="serial link lost"):
        controller.tick(0.1)
    first_attempts = len(reads)
    for _ in range(30):
        controller.tick(1 / 30)
    assert len(reads) == first_attempts
    assert controller.faulted
    assert not controller.enabled


def test_failed_hold_freezes_last_measured_pose_and_requires_explicit_reenable():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    # The command is ahead of the observation sampled at the beginning of tick.
    assert controller.angles[0] > controller.measured_angles[0]

    def fail_read():
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="hold failed"):
        controller.pause()
    assert controller.faulted
    assert not controller.enabled
    assert np.array_equal(controller.angles, controller.measured_angles)
    assert np.array_equal(controller.target_angles, controller.measured_angles)


def test_reenable_with_invalid_measured_pose_holds_instead_of_leaving_old_goal():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    hardware.positions["wrist_flex"] = 100
    with pytest.raises(ValueError, match="limits"):
        controller.enable()
    assert not controller.enabled
    assert hardware.armed
    assert hardware.sent[-1] == hardware.positions


def test_lost_connection_cannot_report_successful_torque_release():
    controller, hardware = connected_controller()
    controller.enable()
    hardware.connected = False
    with pytest.raises(RuntimeError, match="torque release not confirmed"):
        controller.release()
    assert controller.faulted
    assert not controller.enabled


@pytest.mark.parametrize("port_stays_open", [False, True])
def test_failed_disconnect_latches_fault_and_does_not_claim_release(port_stays_open):
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    close = hardware.close

    def fail_close():
        hardware.connected = port_stays_open
        raise OSError("torque disable failed")

    hardware.close = fail_close
    with pytest.raises(RuntimeError, match="torque release not confirmed"):
        controller.disconnect()
    assert controller.faulted
    assert not controller.enabled
    assert controller.connected == port_stays_open
    assert "Disconnect failed" in controller.status
    assert np.array_equal(controller.angles, controller.measured_angles)
    assert np.array_equal(controller.target_angles, controller.measured_angles)
    sent_count = len(hardware.sent)
    controller.tick(0.1)
    assert len(hardware.sent) == sent_count
    # Motor limits still apply while the failed disconnect leaves a live port.
    expected_limit = 90 if port_stays_open else 110
    assert np.rad2deg(controller.model.upper[0]) == pytest.approx(expected_limit, abs=0.001)
    hardware.close = close
    controller.disconnect()
    assert not controller.faulted
    assert not controller.connected
