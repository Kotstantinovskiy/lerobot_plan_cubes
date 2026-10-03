# SO-101 mouse control

Build a small local application in this folder. A Python process serves a Viser
browser interface and talks to the existing LeRobot motor driver over USB.
The first version targets macOS. iOS work follows later; the browser interface
and the Mac motor connection can be reused without a native iOS USB driver.

## Implementation

1. Bundle the official SO-101 model with its license and pinned provenance.
2. Implement forward and inverse kinematics with joint limits and continuous
   solutions near the current pose. Keep the wrist roll and gripper explicit.
3. Add a browser scene with direct dragging of the gripper nose by default,
   optional XYZ handles, gripper control, joint controls, and clear preview,
   enabled, and paused states. Preserve the original grab offset while the model
   follows the mouse in the camera-facing movement plane.
4. Connect through the existing project configuration. Check calibration, read
   the arm before enabling motors, and initialize goals to the measured pose.
5. Limit command speed and tracking error. Pause on lost operator connection,
   rejected targets, or communication failure. Keep motor access in one thread.
6. Verify kinematics, control transitions, and motor handshakes without hardware.
   Open the browser preview and document the first real-arm verification.

## Keep the scope small

- Use Viser's existing browser client; no separate frontend build or ROS.
- Use SciPy for position inverse kinematics; no neural network or GPU.
- Reuse calibration files without changing them or running calibration.
- One controlling browser at a time; localhost by default, paired LAN optional.
- Two optional camera previews share the viewer; failures never block motor control.
- No dataset recording or autonomous motions, except the return to the rest pose
  after the operator disconnects.
- Joint limits and a tool-height check do not replace collision planning.

## Verification boundary

The automated checks and browser preview can validate software without motors.
Real-arm direction, geometric alignment, loaded tracking, and physical clearance
must be checked on the assembled arm before normal operation.

## Current result

- Implemented the six steps above, including both mouse modes and a local launcher.
- Passed 340 Python tests and 66 browser-component tests, including independent kinematics comparisons,
  drag offsets, event ordering, operator ownership, stalled motors, torque
  handshakes, failed cleanup, and controller transitions.
- Checked model loading, gripper input, direct mouse dragging, and mode switching
  in the browser. A small nose drag changed the tip from (339, 0, 158) mm to
  (330, 6, 162) mm without a rejected target.
- Constructed the installed LeRobot driver and loaded the project calibration
  without opening serial ports.
- Added a dark control room, circular wrist dial, and vertical gripper-opening
  instrument using the standard Viser input protocol. The custom client is bundled;
  Node.js is only needed to rebuild the design.
- Added table-parallel keyboard jogging with arrows, vertical K/M movement,
  a 250 ms input lease, and held-key invalidation after control transitions.
- Added a paired LAN gateway, in-memory QR code, foreground operator heartbeat,
  and explicit control handoff. The Viser backend accepts only the private gateway.
- Prepared the native iOS project with QR/manual pairing, authenticated health
  preflight, and a restricted WebView. The portable Swift checks pass, including
  64 pairing URL checks and connection-model lifecycle scenarios.
- Added independent wrist and overhead camera previews with stale-frame rejection,
  automatic capture recovery, and shared authenticated access on the iPhone.
- Added a landscape two-thumb remote with central video previews, bounded
  Cartesian/wrist/gripper rates, and independent input expiry and ownership checks.
- Built the complete arm64 iPhone app with Xcode 26.6 and the iOS 26.5 SDK.
  The USB-connected iPhone 15 Pro on iOS 27.0.1 is paired and has Developer Mode.
- Verified a Personal Team signed build, installed it and launched it on the
  connected iPhone. The app remained running after launch. LAN connection and
  thumb controls on the phone still require a user trial; physical motion and
  geometric alignment remain unverified.
