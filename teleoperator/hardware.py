"""Camera-free SO-101 access; angles are degrees and gripper opening is 0..100."""

import json
import math
from pathlib import Path
from typing import Any

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_RANGE = (0.0, 100.0)
MAX_ENCODER_POSITION = 4095
DEGREES_PER_TURN = 360
# Reading can land just outside a calibrated endpoint (degrees).
READ_TOLERANCE_DEGREES = 0.1
TORQUE_RETRIES = 2
# Largest goal step past the present position, in degrees. Servo force follows that position error,
# so a small step leaves the elbow too weak to lift the arm and it stalls under the arm's weight.
MAX_GOAL_STEP_DEGREES = 10.0


def make_robot(fields: dict[str, Any]) -> Any:
    """Import LeRobot only when a hardware connection is requested."""
    try:
        import draccus
        from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Run hardware mode with your LeRobot Python environment.") from exc
    return make_robot_from_config(draccus.decode(RobotConfig, fields))


def _joint_limits(calibration, calibration_path):
    """Return {motor name: (lowest, highest)} from a calibration file's contents."""
    if set(calibration) != set(MOTOR_NAMES):
        raise ValueError(f"Incomplete follower calibration: {calibration_path}")
    limits = {}
    for index, name in enumerate(MOTOR_NAMES, start=1):
        item = calibration[name]
        if item["id"] != index or not 0 <= item["range_min"] < item["range_max"] <= MAX_ENCODER_POSITION:
            raise ValueError(f"Invalid calibration for {name}.")
        span = (item["range_max"] - item["range_min"]) * DEGREES_PER_TURN / MAX_ENCODER_POSITION
        limits[name] = GRIPPER_RANGE if name == "gripper" else (-span / 2, span / 2)
    return limits


class SO101Hardware:
    """Call from one control thread; connecting never enables motor torque.

    No automatic calibration or motor configuration is performed. Existing
    calibration and position-control mode must already match the physical arm.
    Disarming releases the arm, so support it before disabling torque.
    """

    def __init__(self, config_path: Path, max_relative_target: float = MAX_GOAL_STEP_DEGREES) -> None:
        config_path = Path(config_path).expanduser().resolve()
        config = json.loads(config_path.read_text())
        fields = config["robot"]
        if fields.get("type") != "so101_follower":
            raise ValueError("Hardware mode requires robot.type=so101_follower.")
        if not math.isfinite(max_relative_target) or max_relative_target <= 0:
            raise ValueError("max_relative_target must be positive and finite.")
        calibration_dir = Path(fields["calibration_dir"]).expanduser()
        if not calibration_dir.is_absolute():
            calibration_dir = config_path.parent / calibration_dir
        calibration_dir = calibration_dir.resolve()
        calibration_path = calibration_dir / f"{fields['id']}.json"
        self.joint_limits = _joint_limits(json.loads(calibration_path.read_text()), calibration_path)
        self.rest_positions = self._rest_positions(config.get("rest_pose"))
        self.fields = {
            "type": fields["type"],
            "port": fields["port"],
            "id": fields["id"],
            "calibration_dir": str(calibration_dir),
            "cameras": {},
            "use_degrees": True,
            "max_relative_target": max_relative_target,
            "disable_torque_on_disconnect": False,
        }
        self.robot = None
        self.armed = False

    def _rest_positions(self, rest_pose):
        """Convert config rest_pose (normalized -100..100, gripper 0..100) to degrees, or None."""
        if rest_pose is None:
            return None
        return {
            name: rest_pose[name] if name == "gripper" else rest_pose[name] / 100 * self.joint_limits[name][1]
            for name in MOTOR_NAMES
        }

    @property
    def connected(self) -> bool:
        return self.robot is not None and self.robot.bus.is_connected

    def connect(self) -> dict[str, float]:
        """Read the arm without changing motor settings or enabling torque."""
        if self.connected:
            raise RuntimeError("The follower is already connected.")
        self.robot = make_robot(self.fields)
        try:
            self.robot.bus.connect()
            if not self.robot.is_calibrated:
                raise RuntimeError("Motor calibration differs from the saved file; calibrate separately.")
            if any(self.robot.bus.sync_read("Torque_Enable").values()):
                raise RuntimeError("Motor torque is already enabled; stop the other controller first.")
            if any(self.robot.bus.sync_read("Operating_Mode").values()):
                raise RuntimeError("All motors must already be configured for position control.")
            return self.read_positions()
        except BaseException:
            self.close()
            raise

    def read_positions(self) -> dict[str, float]:
        """Return all six joint positions, with no .pos suffix."""
        if not self.connected:
            raise RuntimeError("Connect the follower first.")
        positions = self.robot.bus.sync_read("Present_Position")
        return self._validate_positions(positions, tolerance=READ_TOLERANCE_DEGREES)

    def _validate_positions(self, positions, tolerance=0.0):
        if set(positions) != set(MOTOR_NAMES):
            raise ValueError("Positions must include exactly the six SO-101 joints.")
        result = {}
        for name in MOTOR_NAMES:
            value = float(positions[name])
            low, high = self.joint_limits[name]
            if not math.isfinite(value) or not low - tolerance <= value <= high + tolerance:
                raise ValueError(f"{name} is outside its calibrated range: {value}.")
            # Quantization just outside an endpoint must still produce a goal
            # accepted by the strict write path used for arming and holding.
            result[name] = min(high, max(low, value))
        return result

    def arm(self) -> dict[str, float]:
        """Latch the present pose before enabling torque; return that pose."""
        if not self.connected:
            raise RuntimeError("Connect the follower first.")
        if self.armed:
            torque = self.robot.bus.sync_read("Torque_Enable")
            if set(torque) != set(MOTOR_NAMES) or any(value != 1 for value in torque.values()):
                raise RuntimeError("Motor torque is inconsistent; release the motors before enabling again.")
            return self.read_positions()
        if any(self.robot.bus.sync_read("Torque_Enable").values()):
            raise RuntimeError("Motor torque was enabled by another controller.")
        positions = self.read_positions()
        self.robot.bus.sync_write("Goal_Position", positions)
        self.armed = True
        try:
            self.robot.bus.enable_torque(num_retry=TORQUE_RETRIES)
        except BaseException as error:
            self._release_after_failed_arm(error)
            raise
        return positions

    def _release_after_failed_arm(self, error):
        try:
            self.disarm()
        except Exception as cleanup_error:
            if not isinstance(error, Exception):
                raise error from cleanup_error
            raise RuntimeError(
                f"Enabling motor torque failed ({error}); releasing torque also failed ({cleanup_error})."
            ) from error

    def send_positions(self, positions: dict[str, float]) -> dict[str, float]:
        """Send a bounded target; the caller must also apply a speed limit."""
        if not self.connected or not self.armed:
            raise RuntimeError("Enable movement before sending a target.")
        positions = self._validate_positions(positions)
        action = self.robot.send_action({f"{name}.pos": value for name, value in positions.items()})
        return {name.removesuffix(".pos"): float(value) for name, value in action.items()}

    def disarm(self) -> None:
        """Release torque on motors enabled by this adapter."""
        if self.connected and self.armed:
            errors = []
            for name in MOTOR_NAMES:
                try:
                    # A failed servo must not prevent releasing the other five.
                    self.robot.bus.disable_torque(motors=name, num_retry=TORQUE_RETRIES)
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
        self.armed = False

    def close(self) -> None:
        """Release owned torque and close the port, including after errors."""
        try:
            self.disarm()
        finally:
            if self.connected:
                self.robot.bus.disconnect(disable_torque=False)
            self.robot = None
            self.armed = False
