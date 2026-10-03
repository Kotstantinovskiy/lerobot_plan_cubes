# SO-101 model

The URDF and all 13 referenced STL meshes are unmodified copies from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101),
revision `a758567c3978dfeefe282ede0500085a48fe8f78` (6 October 2026).
They are distributed under the upstream Apache 2.0 license in `LICENSE`.
`provenance.json` records the source revision and SHA-256 of every copied file.
No network download is required when running the viewer.

Load `so101_new_calib.urdf` with its adjacent `assets/` directory. Mesh geometry
and joint translations use metres; URDF angles use radians. The end-effector
frame is `gripper_frame_link`. Its transform from `gripper_link` is fixed, so
opening the jaw does not move this frame.

| Joint | Lower limit, degrees | Upper limit, degrees |
| --- | ---: | ---: |
| `shoulder_pan` | -110 | 110 |
| `shoulder_lift` | -100 | 100 |
| `elbow_flex` | -96.83 | 96.83 |
| `wrist_flex` | -95 | 95 |
| `wrist_roll` | -157.21 | 162.79 |
| `gripper` | -10 | 100 |

LeRobot's `use_degrees=True` expresses the five arm joints in degrees, but the
gripper remains a percentage of its calibrated range, from 0 to 100. A visual
mapping of that percentage onto the URDF jaw range is approximate. It is not a
measurement of the actual opening or a conversion for commanding the motors.

The geometric model does not establish the physical table height, account for
camera mounts or cables, or guarantee collision clearance. The displayed base
frame and the motor calibration also do not establish surveyed physical
coordinates. Real motor limits should be restricted to the intersection of
the URDF limits and the specific arm's calibration ranges. Check model alignment
against the assembled arm before relying on Cartesian coordinates.
