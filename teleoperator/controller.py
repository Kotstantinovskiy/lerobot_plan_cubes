"""Single-threaded motion state shared by the browser and motor adapter."""

import math

import numpy as np
from numpy.typing import ArrayLike

from hardware import SO101Hardware
from height_rules import allowed_target, must_rise_first
from kinematics import JOINT_NAMES, MAX_GRIPPER_OPENING, BelowMinimumHeightError, Kinematics
from rest_return import ReturnWalk, plan_return

PREVIEW_STATUS = "Preview: no motor commands"
TORQUE_RELEASED_STATUS = "Torque released"
RETURNING_STATUS = "Returning to rest"
RETURN_STOPPED_STATUS = "Return stopped: arm is not following"
MIN_SPEED_DEGREES_PER_SECOND = 1
MAX_SPEED_DEGREES_PER_SECOND = 45
GRIPPER_SPEED_PER_SECOND = 25
MAX_TICK_SECONDS = 0.1
# Beyond these the arm is not following its command.
MAX_TRACKING_ERROR_DEGREES = 12
MAX_GRIPPER_TRACKING_ERROR = 20
ANGLE_TOLERANCE_RADIANS = 1e-6
# Start preview away from full extension so dragging has room to move.
PREVIEW_START_DEGREES = (0.0, -20.0, 45.0, -20.0, 0.0)
START_GRIPPER_OPENING = 50.0


def _angles_from(positions):
    """Return the arm joint angles in radians from a dict of positions in degrees."""
    return np.deg2rad([positions[name] for name in JOINT_NAMES])


class Controller:
    def __init__(self, model: Kinematics, hardware: SO101Hardware | None = None, speed: float = 20.0) -> None:
        self.model = model
        self.hardware = hardware
        self.speed = speed
        self._model_lower = model.lower.copy()
        self._model_upper = model.upper.copy()
        self.angles = np.deg2rad(PREVIEW_START_DEGREES)
        if model.forward(self.angles)[2, 3] < model.min_height:
            self.angles = np.zeros(len(JOINT_NAMES))
        self.gripper = START_GRIPPER_OPENING
        self.measured_angles = self.angles.copy()
        self.measured_gripper = self.gripper
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper
        self.enabled = False
        self._connection_expected = False
        self._faulted = False
        self.status = PREVIEW_STATUS
        self._return_walk = None

    @property
    def connected(self) -> bool:
        return self.hardware is not None and self.hardware.connected

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def motion_allowed(self) -> bool:
        return not self._faulted and (
            (self.connected and self.enabled) or (not self.connected and not self._connection_expected)
        )

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Joint speed must be between 1 and 45 degrees per second") from error
        if not math.isfinite(value) or not MIN_SPEED_DEGREES_PER_SECOND <= value <= MAX_SPEED_DEGREES_PER_SECOND:
            raise ValueError("Joint speed must be between 1 and 45 degrees per second")
        self._speed = value

    def _calibrated_limits(self):
        """Return an array of (lowest, highest) angles in radians for each arm joint."""
        return np.deg2rad([self.hardware.joint_limits[name] for name in JOINT_NAMES])

    def _sync(self, positions, gripper_command=None):
        self.angles = _angles_from(positions)
        self.gripper = float(positions["gripper"] if gripper_command is None else gripper_command)
        self._set_measured(positions)
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper

    def _set_measured(self, positions):
        self.measured_angles = _angles_from(positions)
        self.measured_gripper = float(positions["gripper"])

    def _read(self):
        positions = self.hardware.read_positions()
        self._set_measured(positions)
        return positions

    @staticmethod
    def _positions(angles, gripper):
        values = dict(zip(JOINT_NAMES, map(float, np.rad2deg(angles)), strict=False))
        values["gripper"] = float(gripper)
        return values

    def connect(self) -> None:
        if self.hardware is None:
            raise RuntimeError("Start with a robot configuration to use hardware")
        if self.connected:
            raise RuntimeError("The follower is already connected")
        try:
            positions = self.hardware.connect()
            limits = self._calibrated_limits()
            self.model.lower = np.maximum(self._model_lower, limits[:, 0])
            self.model.upper = np.minimum(self._model_upper, limits[:, 1])
            if np.any(self.model.lower >= self.model.upper):
                raise ValueError("Model and motor limits do not overlap")
            self._sync(positions)
            self.enabled = False
            self._connection_expected = True
            self._faulted = False
            self.status = "Connected: movement disabled"
        except Exception:
            self.disconnect()
            raise

    def enable(self) -> None:
        self.cancel_return()
        self.enabled = False
        if not self.connected:
            if self._connection_expected:
                self._communication_error(RuntimeError("Robot connection lost"))
            raise RuntimeError("Connect the follower first")
        self.status = "Movement disabled: checking the current pose"
        held_gripper = self.gripper if self.hardware.armed else None
        try:
            positions = self._read()
            self._sync(positions, gripper_command=held_gripper)
            self._check_start_pose()
            # The adapter reads once more and latches that exact pose before torque.
            positions = self.hardware.arm()
            self._sync(positions, gripper_command=held_gripper)
            self._check_start_pose()
        except ValueError as error:
            self._reject(error)
        except Exception as error:
            self._communication_error(error)
        self._faulted = False
        self.enabled = True
        self.status = "Movement enabled"

    def pause(self, reason: str = "Paused") -> None:
        """Hold measured arm joints and freeze the existing commanded grasp."""
        # A loaded jaw can be farther open than its command. Latching that
        # measured opening would relax the grasp precisely when stopping.
        gripper_command = self.gripper
        owned_torque = self.hardware is not None and self.hardware.armed
        self.enabled = False
        if self._connection_expected:
            self.angles = self.measured_angles.copy()
            if not owned_torque:
                self.gripper = self.measured_gripper
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper
        self.status = reason
        if not self.connected:
            if self._connection_expected:
                self._faulted = True
                self.status = f"{reason}; robot connection is unavailable; hold not confirmed"
                raise RuntimeError(self.status)
            return
        try:
            positions = self._read()
            self._sync(positions, gripper_command=gripper_command if owned_torque else None)
            if self.hardware.armed:
                hold = dict(positions)
                hold["gripper"] = self.gripper
                self.hardware.send_positions(hold)
        except Exception as error:
            self._faulted = True
            self.status = f"{reason}; hold failed: {error}"
            raise RuntimeError(self.status) from error

    def hold_motion(self) -> None:
        """Hold arm joints while preserving the commanded grasp and enable state."""
        was_enabled = self.enabled
        status = self.status
        target_gripper = self.target_gripper
        if self._faulted:
            self.enabled = False
            self.target_angles = self.measured_angles.copy()
            return
        self.pause("Keyboard movement stopped")
        self.target_gripper = target_gripper
        self.enabled = was_enabled and self.connected and not self._faulted
        self.status = status

    def release(self) -> None:
        """Release motor torque; the operator must support the physical arm."""
        self.cancel_return()
        self.enabled = False
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper
        self.status = TORQUE_RELEASED_STATUS
        if self._connection_expected and not self.connected:
            self._communication_error(RuntimeError("Robot connection lost; torque release not confirmed"))
        if self.connected:
            try:
                self.hardware.disarm()
                self._sync(self._read())
            except Exception as error:
                self._communication_error(error)

    def disconnect(self) -> None:
        self.cancel_return()
        self.enabled = False
        if self._connection_expected:
            self.angles = self.measured_angles.copy()
            self.gripper = self.measured_gripper
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper
        try:
            if self.hardware is not None:
                self.hardware.close()
        except Exception as error:
            self._connection_expected = self.connected
            self._faulted = True
            self.status = f"Disconnect failed: {error}; torque release not confirmed"
            raise RuntimeError(self.status) from error
        else:
            self._connection_expected = False
            self._faulted = False
            self.status = "Disconnected: preview only"
        finally:
            if not self.connected:
                self.model.lower = self._model_lower.copy()
                self.model.upper = self._model_upper.copy()

    def _reject(self, error):
        self.pause(f"Paused: {error}")
        raise ValueError(str(error)) from error

    @property
    def can_return_to_rest(self) -> bool:
        return self.connected and self.hardware.armed and self.hardware.rest_positions is not None

    @property
    def returning(self) -> bool:
        return self._return_walk is not None

    @property
    def return_stopped(self) -> bool:
        return self.status == RETURN_STOPPED_STATUS

    def start_return(self) -> None:
        """Hold the present pose, then walk to rest; tick_return() frees the port at the end."""
        self.pause(RETURNING_STATUS)
        steps = plan_return(self.model, self.measured_angles, self.hardware.rest_positions)
        self._return_walk = ReturnWalk(steps, self._positions(self.measured_angles, self.gripper))

    def cancel_return(self) -> None:
        self._return_walk = None

    def tick_return(self, dt: float) -> bool:
        """Advance the return by one bounded step; return True when it has just ended."""
        walk = self._return_walk
        if walk is None:
            return False
        if self._faulted:
            self.cancel_return()
            return True
        self.status = RETURNING_STATUS
        measured = self._positions(self.measured_angles, self.measured_gripper)
        if walk.lag_degrees(measured) > MAX_TRACKING_ERROR_DEGREES:
            self.cancel_return()
            self.pause(RETURN_STOPPED_STATUS)
            return True
        if walk.has_arrived:
            if not walk.has_settled(dt):
                return False
            self.cancel_return()
            self.disconnect()
            return True
        try:
            self.hardware.send_positions(walk.step(dt))
        except Exception as error:
            self.cancel_return()
            self._communication_error(error)
        return False

    def _check_start_pose(self):
        """Require only the calibrated range: the folded rest pose lies past the model limits."""
        limits = self._calibrated_limits()
        lowest, highest = limits[:, 0] - ANGLE_TOLERANCE_RADIANS, limits[:, 1] + ANGLE_TOLERANCE_RADIANS
        is_within_range = np.all(self.angles >= lowest) and np.all(self.angles <= highest)
        if not is_within_range:
            raise ValueError("Pose exceeds model or calibrated joint limits")

    @property
    def must_rise_first(self) -> bool:
        return must_rise_first(self.model, self.angles)

    def _show_preview_status(self):
        if not self._connection_expected:
            self.status = PREVIEW_STATUS

    def set_target(self, position: ArrayLike) -> None:
        try:
            position = np.array(position, dtype=float)
            if position.shape == (3,):
                position = allowed_target(self.model, self.angles, position)
                if position is None:
                    return
            # Lifting out of a low pose with the wrist still moves the shoulder and elbow together:
            # the elbow alone stalls raising the forearm against its weight.
            angles = self.model.solve(
                position,
                self.target_angles,
                current_angles=self.angles,
                keep_wrist_flex=self.must_rise_first,
            )
        except BelowMinimumHeightError:
            return
        except (ValueError, TypeError) as error:
            self._reject(error)
        self.target_angles = angles.copy()
        self._show_preview_status()

    def set_joints(self, angles: ArrayLike) -> None:
        try:
            angles = self.model.validate(angles, current_angles=self.angles)
        except BelowMinimumHeightError:
            return
        except (ValueError, TypeError) as error:
            self._reject(error)
        self.target_angles = angles.copy()
        self._show_preview_status()

    def set_gripper(self, value: float) -> None:
        try:
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= MAX_GRIPPER_OPENING:
                raise ValueError("Gripper opening must be between 0 and 100")
        except (ValueError, TypeError) as error:
            self._reject(error)
        self.target_gripper = value
        self._show_preview_status()

    def tick(self, dt: float) -> None:
        """Advance at bounded speed; a delayed loop cannot produce a large jump."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Control time step must be finite and nonnegative")
        dt = min(float(dt), MAX_TICK_SECONDS)
        if self._faulted:
            return
        if self._connection_expected and not self.connected:
            self._communication_error(RuntimeError("Robot connection lost"))
        if self.connected and not self.enabled:
            self._follow_measured_pose()
            return
        if self.connected and self._paused_for_lost_tracking():
            return
        self._step_towards_target(dt)

    def _follow_measured_pose(self):
        try:
            positions = self._read()
            self.angles = _angles_from(positions)
            if not self.hardware.armed:
                self.gripper = float(positions["gripper"])
        except Exception as error:
            self._communication_error(error)

    def _paused_for_lost_tracking(self):
        """Pause and return True when the arm has fallen too far behind its command."""
        try:
            positions = self._read()
            tracking_error = np.max(np.abs(np.rad2deg(self.measured_angles - self.angles)))
            is_gripper_behind = abs(positions["gripper"] - self.gripper) > MAX_GRIPPER_TRACKING_ERROR
            if tracking_error > MAX_TRACKING_ERROR_DEGREES or is_gripper_behind:
                self.pause("Paused: arm is not following the commanded pose")
                return True
        except Exception as error:
            self._communication_error(error)
        return False

    def _step_towards_target(self, dt):
        max_step = np.deg2rad(self.speed) * dt
        max_gripper_step = GRIPPER_SPEED_PER_SECOND * dt
        angles = self.angles + np.clip(self.target_angles - self.angles, -max_step, max_step)
        gripper = self.gripper + float(np.clip(self.target_gripper - self.gripper, -max_gripper_step, max_gripper_step))
        try:
            self.model.validate(angles, current_angles=self.angles)
        except BelowMinimumHeightError:
            # Hold at the limit and keep control, so the operator can simply move away.
            angles = self.angles
        except ValueError as error:
            self.pause(f"Paused: motion would leave the allowed workspace ({error})")
            return
        if self.connected:
            try:
                self.hardware.send_positions(self._positions(angles, gripper))
            except Exception as error:
                self._communication_error(error)
        # Keep the requested trajectory independent of driver clipping. Otherwise
        # a stalled joint stays within the driver's small cap forever and the
        # tracking-error check cannot detect that it stopped following.
        self.angles = angles
        self.gripper = gripper
        if not self.connected:
            self.measured_angles = angles.copy()
            self.measured_gripper = gripper

    def _communication_error(self, error):
        try:
            self.pause(f"Paused: hardware error ({error})")
        except Exception:
            # A failed serial link may also prevent a new hold command.
            self.status = f"Hardware error: {error}; check the arm before continuing"
        self._faulted = True
        raise RuntimeError(self.status) from error
