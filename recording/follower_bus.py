"""Opening the follower's port without taking the arm from whoever may be holding it up."""

from typing import Any


def connect_free_bus(robot: Any) -> None:
    """Open the bus of a follower that is free to take over: calibrated, with torque off on every motor.

    Otherwise raises RuntimeError, with the port closed again and the torque as it was: an arm held up by another
    controller, or left under torque by one that stopped, must not drop because this script refused to run.
    """
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; refusing automatic calibration")
        if any(robot.bus.sync_read("Torque_Enable").values()):
            raise RuntimeError(
                "Expected motors at rest with torque disabled. Torque is left on: stop the other controller, "
                "or fold the arm with recording/rest.py"
            )
    except BaseException:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
        raise
