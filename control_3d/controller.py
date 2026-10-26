"""Single-threaded motion state shared by the browser and motor adapter."""

import math

import numpy as np

try:
    from .kinematics import JOINT_NAMES
except ImportError:
    from kinematics import JOINT_NAMES


class Controller:
    def __init__(self, model, hardware=None, speed=20.0):
        self.model = model
        self.hardware = hardware
        self.speed = speed
        self._model_lower = model.lower.copy()
        self._model_upper = model.upper.copy()
        # Start preview away from full extension so dragging has room to move.
        self.angles = np.deg2rad([0.0, -20.0, 45.0, -20.0, 0.0])
        if model.forward(self.angles)[2, 3] < model.min_height:
            self.angles = np.zeros(5)
        self.gripper = 50.0
        self.measured_angles = self.angles.copy()
        self.measured_gripper = self.gripper
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper
        self.enabled = False
        self._connection_expected = False
        self._faulted = False
        self.status = "Preview: no motor commands"

    @property
    def connected(self):
        return self.hardware is not None and self.hardware.connected

    @property
    def faulted(self):
        return self._faulted

    @property
    def motion_allowed(self):
        return not self._faulted and (
            (self.connected and self.enabled)
            or (not self.connected and not self._connection_expected)
        )

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Joint speed must be between 1 and 45 degrees per second") from error
        if not math.isfinite(value) or not 1 <= value <= 45:
            raise ValueError("Joint speed must be between 1 and 45 degrees per second")
        self._speed = value

    def _sync(self, positions, gripper_command=None):
        self.angles = np.deg2rad([positions[name] for name in JOINT_NAMES])
        self.gripper = float(positions["gripper"] if gripper_command is None else gripper_command)
        self._set_measured(positions)
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper

    def _set_measured(self, positions):
        self.measured_angles = np.deg2rad([positions[name] for name in JOINT_NAMES])
        self.measured_gripper = float(positions["gripper"])

    def _read(self):
        positions = self.hardware.read_positions()
        self._set_measured(positions)
        return positions

    @staticmethod
    def _positions(angles, gripper):
        values = dict(zip(JOINT_NAMES, map(float, np.rad2deg(angles))))
        values["gripper"] = float(gripper)
        return values

    def connect(self):
        if self.hardware is None:
            raise RuntimeError("Start with a robot configuration to use hardware")
        if self.connected:
            raise RuntimeError("The follower is already connected")
        try:
            positions = self.hardware.connect()
            limits = np.deg2rad([self.hardware.joint_limits[name] for name in JOINT_NAMES])
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

    def enable(self):
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
            self.model.validate(self.angles)
            # The adapter reads once more and latches that exact pose before torque.
            positions = self.hardware.arm()
            self._sync(positions, gripper_command=held_gripper)
            self.model.validate(self.angles)
        except ValueError as error:
            self._reject(error)
        except Exception as error:
            self._communication_error(error)
        self._faulted = False
        self.enabled = True
        self.status = "Movement enabled"

    def pause(self, reason="Paused"):
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

    def hold_motion(self):
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

    def release(self):
        """Release motor torque; the operator must support the physical arm."""
        self.enabled = False
        self.target_angles = self.angles.copy()
        self.target_gripper = self.gripper
        self.status = "Torque released"
        if self._connection_expected and not self.connected:
            self._communication_error(RuntimeError("Robot connection lost; torque release not confirmed"))
        if self.connected:
            try:
                self.hardware.disarm()
                self._sync(self._read())
            except Exception as error:
                self._communication_error(error)

    def disconnect(self):
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

    def set_target(self, position):
        try:
            angles = self.model.solve(position, self.target_angles)
        except (ValueError, TypeError) as error:
            self._reject(error)
        self.target_angles = angles.copy()
        if not self._connection_expected:
            self.status = "Preview: no motor commands"

    def set_joints(self, angles):
        try:
            angles = self.model.validate(angles)
        except (ValueError, TypeError) as error:
            self._reject(error)
        self.target_angles = angles.copy()
        if not self._connection_expected:
            self.status = "Preview: no motor commands"

    def set_gripper(self, value):
        try:
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Gripper opening must be between 0 and 100")
        except (ValueError, TypeError) as error:
            self._reject(error)
        self.target_gripper = value
        if not self._connection_expected:
            self.status = "Preview: no motor commands"

    def tick(self, dt):
        """Advance at bounded speed; a delayed loop cannot produce a large jump."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Control time step must be finite and nonnegative")
        dt = min(float(dt), 0.1)
        if self._faulted:
            return
        if self._connection_expected and not self.connected:
            self._communication_error(RuntimeError("Robot connection lost"))
        if self.connected and not self.enabled:
            try:
                positions = self._read()
                self.angles = np.deg2rad([positions[name] for name in JOINT_NAMES])
                if not self.hardware.armed:
                    self.gripper = float(positions["gripper"])
            except Exception as error:
                self._communication_error(error)
            return
        if self.connected:
            try:
                positions = self._read()
                error = np.max(np.abs(np.rad2deg(self.measured_angles - self.angles)))
                if error > 12 or abs(positions["gripper"] - self.gripper) > 20:
                    self.pause("Paused: arm is not following the commanded pose")
                    return
            except Exception as error:
                self._communication_error(error)
        max_step = np.deg2rad(self.speed) * dt
        angles = self.angles + np.clip(self.target_angles - self.angles, -max_step, max_step)
        gripper = self.gripper + float(np.clip(self.target_gripper - self.gripper, -25 * dt, 25 * dt))
        try:
            self.model.validate(angles)
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
