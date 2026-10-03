"""Compare robot kinematics to an independent URDF implementation."""

from pathlib import Path

import numpy as np
import pytest
from yourdfpy import URDF

from hardware import SO101Hardware
from kinematics import JOINT_NAMES, MODEL_PATH, TOOL_FRAME, BelowMinimumHeightError, Kinematics


@pytest.fixture
def kinematics():
    return Kinematics()


def test_random_forward_poses_match_yourdfpy(kinematics):
    reference = URDF.load(MODEL_PATH, load_meshes=False)
    random = np.random.default_rng(101)
    for angles in random.uniform(kinematics.lower, kinematics.upper, size=(150, 5)):
        reference.update_cfg(dict(zip(JOINT_NAMES, angles, strict=False)))
        expected = reference.get_transform(TOOL_FRAME, reference.base_link)
        np.testing.assert_allclose(kinematics.forward(angles), expected, atol=2e-12, rtol=0)


def test_nearby_reachable_targets_keep_the_current_branch(kinematics):
    random = np.random.default_rng(4321)
    checked = 0
    for _ in range(400):
        seed = random.uniform(kinematics.lower * 0.65, kinematics.upper * 0.65)
        target_angles = np.clip(seed + random.normal(0, 0.035, 5), kinematics.lower, kinematics.upper)
        target_angles[4] = seed[4]
        target = kinematics.forward(target_angles)[:3, 3]
        if min(target[2], kinematics.forward(seed)[2, 3]) < kinematics.min_height + 0.02:
            continue
        result = kinematics.solve(target, seed)
        assert np.linalg.norm(kinematics.forward(result)[:3, 3] - target) < 0.003
        assert np.max(np.abs(result - seed)) < 0.2
        assert result[4] == seed[4]
        kinematics.validate(result)
        checked += 1
        if checked == 60:
            break
    assert checked == 60


def test_existing_pose_is_not_moved_by_inverse_kinematics(kinematics):
    seed = np.deg2rad([15, -20, 30, -10, 25])
    target = kinematics.forward(seed)[:3, 3]
    np.testing.assert_allclose(kinematics.solve(target, seed), seed, atol=1e-10, rtol=0)


@pytest.mark.parametrize(
    "target",
    [
        [0.2, 0.0, 0.0],
        [0.2, 0.0, -0.1],
        [1.0, 0.0, 0.2],
        [float("nan"), 0.0, 0.2],
        [0.2, float("inf"), 0.2],
        [0.2, 0.0],
    ],
)
def test_invalid_cartesian_targets_are_rejected(kinematics, target):
    with pytest.raises(ValueError):
        kinematics.solve(target, np.zeros(5))


def test_unreachable_target_inside_workspace_guard_is_rejected(kinematics):
    with pytest.raises(ValueError, match="unreachable"):
        kinematics.solve([0.59, 0.0, 0.02], np.zeros(5))


@pytest.mark.parametrize(
    "angles",
    [
        [0.0] * 4,
        [0.0] * 6,
        [[0.0] * 5],
        [0.0, 0.0, float("nan"), 0.0, 0.0],
        [0.0, 0.0, float("inf"), 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_invalid_joint_poses_are_rejected(kinematics, angles):
    with pytest.raises(ValueError):
        kinematics.validate(angles)


@pytest.mark.parametrize("seed", [0, [], [0.0] * 4, [[0.0] * 5], [float("nan")] * 5])
def test_invalid_inverse_kinematics_seed_is_rejected(kinematics, seed):
    with pytest.raises(ValueError):
        kinematics.solve([0.2, 0.0, 0.2], seed)


def test_explicit_wrist_roll_is_kept_and_bounded(kinematics):
    seed = np.zeros(5)
    target = kinematics.forward(seed)[:3, 3]
    result = kinematics.solve(target, seed, wrist_roll=0.5)
    assert result[4] == 0.5
    assert np.linalg.norm(kinematics.forward(result)[:3, 3] - target) < 0.003
    for roll in (float("nan"), float("inf"), kinematics.upper[4] + 0.1):
        with pytest.raises(ValueError):
            kinematics.solve(target, seed, wrist_roll=roll)


def test_floor_guard_rejects_a_valid_joint_range_pose(kinematics):
    random = np.random.default_rng(5)
    for angles in random.uniform(kinematics.lower, kinematics.upper, size=(100, 5)):
        if kinematics.forward(angles)[2, 3] < kinematics.min_height:
            with pytest.raises(ValueError, match="minimum height"):
                kinematics.validate(angles)
            return
    pytest.fail("Expected at least one joint-limit-valid pose below the floor")


def test_pose_past_a_limit_may_stay_or_move_back_but_not_further(kinematics):
    past_limit = np.deg2rad([0, -105, 103, 80, 0])
    with pytest.raises(ValueError, match="limits"):
        kinematics.validate(past_limit)
    kinematics.validate(past_limit, current_angles=past_limit)
    moved_back = past_limit + np.deg2rad([0, 1, -1, 0, 0])
    kinematics.validate(moved_back, current_angles=past_limit)
    further_out = past_limit + np.deg2rad([0, -1, 0, 0, 0])
    with pytest.raises(ValueError, match="limits"):
        kinematics.validate(further_out, current_angles=past_limit)


def test_tool_below_the_floor_may_stay_or_rise_but_not_sink(kinematics):
    low = np.deg2rad([0, -85, 85, 80, 0])
    height = kinematics.forward(low)[2, 3]
    assert height < kinematics.min_height
    kinematics.validate(low, current_angles=low)
    risen = low + np.deg2rad([0, 2, -2, 0, 0])
    assert kinematics.forward(risen)[2, 3] > height
    kinematics.validate(risen, current_angles=low)
    sunk = low + np.deg2rad([0, -2, 2, 0, 0])
    assert kinematics.forward(sunk)[2, 3] < height
    with pytest.raises(BelowMinimumHeightError):
        kinematics.validate(sunk, current_angles=low)


def test_kept_wrist_flex_stays_at_its_seed_angle(kinematics):
    seed = np.deg2rad([0, -100, 100, 80, 0])
    tool = kinematics.forward(seed)[:3, 3]
    target = tool + [0, 0, 0.005]
    angles = kinematics.solve(target, seed, current_angles=seed, keep_wrist_flex=True)
    assert angles[3] == pytest.approx(seed[3], abs=1e-5)
    np.testing.assert_allclose(kinematics.forward(angles)[:3, 3], target, atol=0.001)


def test_solution_may_stay_as_far_past_a_limit_as_the_current_pose(kinematics):
    past_limit = np.deg2rad([0, -105, 103, 80, 0])
    tool = kinematics.forward(past_limit)[:3, 3]
    target = tool + [0, 0, 0.001]
    with pytest.raises(ValueError):
        kinematics.solve(target, past_limit)
    angles = kinematics.solve(target, past_limit, current_angles=past_limit)
    np.testing.assert_allclose(kinematics.forward(angles)[:3, 3], target, atol=0.001)
    assert angles[1] >= past_limit[1] - 1e-6
    assert angles[2] <= past_limit[2] + 1e-6


def test_degrees_round_trip_with_current_motor_calibration():
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    hardware = SO101Hardware(config_path)
    kinematics = Kinematics()
    for index, name in enumerate(JOINT_NAMES):
        low, high = np.deg2rad(hardware.joint_limits[name])
        kinematics.lower[index] = max(kinematics.lower[index], low)
        kinematics.upper[index] = min(kinematics.upper[index], high)
    random = np.random.default_rng(19)
    for angles in random.uniform(kinematics.lower, kinematics.upper, size=(50, 5)):
        degrees = dict(zip(JOINT_NAMES, np.rad2deg(angles), strict=False))
        degrees["gripper"] = 50.0
        validated = hardware._validate_positions(degrees)
        restored = np.deg2rad([validated[name] for name in JOINT_NAMES])
        np.testing.assert_allclose(restored, angles, atol=1e-15, rtol=0)


def test_visual_config_keeps_arm_radians_and_maps_jaw():
    angles = np.deg2rad([10, 20, -30, 40, 50])
    closed = Kinematics.visual_config(angles, 0)
    opened = Kinematics.visual_config(angles, 100)
    np.testing.assert_allclose([closed[name] for name in JOINT_NAMES], angles)
    assert closed["gripper"] == pytest.approx(np.deg2rad(-10))
    assert opened["gripper"] == pytest.approx(np.deg2rad(100))
