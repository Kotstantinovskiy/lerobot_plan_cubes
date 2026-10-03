# Teleoperator

Teleoperator is a local browser scene for moving the SO-101 gripper with the mouse. Python
calculates the arm joint angles and, when explicitly enabled, sends bounded
commands through the existing LeRobot installation. The implementation uses
Viser, SciPy, and the bundled official SO-101 geometry. Normal launches need no
frontend build, ROS installation, neural network, or GPU.

`app.py` starts everything and runs the control loop; `control_panel.py` builds the
3D scene and the panel; `input_queue.py` collects browser input and `interaction.py`
orders it and selects the operator; `controller.py` limits motion and `height_rules.py`
holds its rules near the minimum height; `rest_return.py` plans the way back to rest;
`keyboard_jog.py` and `remote_jog.py` share the input lease in `jog_lease.py`;
`hardware.py` handles the USB connection. `cameras.py` independently supervises the two
optional video previews. Tests are in `tests/`.

The browser controls are in Russian. The implementation plan and scope are in
[PLAN.md](PLAN.md).

## Preview without a robot

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). From the
repository root, run:

```bash
uv run --project teleoperator python teleoperator/app.py
```

Alternatively, launch `teleoperator/run.sh` from any working directory. Open
`http://127.0.0.1:8080` in the browser on the Mac.

- Grab the nose of the gripper and drag it directly. The default mode has no
  axis handles: movement follows the mouse in a plane facing the current camera.
  Orbit the view to choose a different movement plane.
- Switch to the optional XYZ mode when a move constrained to a particular axis
  is more useful.
- Drag the background to orbit the view; scroll to zoom.
- WASD orbits the camera around the arm. A/D turns left/right, and W/S tilts
  up/down. Movement starts gently and stops on release.
  Camera panning and Q/E translation are disabled; scrolling still zooms.
- With the scene focused, arrow keys move the gripper parallel to the table,
  relative to the current view. K raises the gripper; M lowers it. Maximum
  Cartesian speed is 20 mm/s. Releasing a key holds the current arm position.
  Paused hardware requires explicit movement enablement before these keys work.
- Rotate the circular dial to turn the wrist. The dial does not jump when grabbed.
- Move the vertical control up to open the jaws, or down to close them.
- Both instruments support arrow keys; Shift makes larger changes. Home centers
  the wrist or closes the jaws; End moves to the upper limit. These local controls
  take precedence while focused; keyboard arm jogging stops when focus enters a control.
- Open **Настройки** for the movement mode, speed, joint controls, and position
  readout. These are collapsed by default to keep the main panel simple.
- The speed slider limits each joint's speed, initially to 20 degrees per second.
- **СТОП: удерживать** stops movement. **Совместить цель с захватом** puts the
  target at the current gripper position and pauses movement.

The robot preview follows targets smoothly. Unreachable targets, invalid joint
angles, and motions that put the tool frame below the configured minimum height
are rejected. A rejected target stops the current movement.

Validate the model without starting a web server or opening serial ports:

```bash
uv run --project teleoperator python teleoperator/app.py --check
```

## Camera previews

The control panel shows **Рука** and **Сверху** on the Mac and iPhone. Both cameras
remain attached to the Mac. Their device names, native resolution and frame rate
come from `robot.cameras.wrist` and `robot.cameras.top` in `config.json`. The current
configuration names the Innomaker wrist camera and Sport Cam overhead camera.
Each preview requires `macos_device_name` and checks that it matches exactly;
a missing camera never falls back to another device index.

Video capture uses [FFmpeg's AVFoundation input](https://ffmpeg.org/ffmpeg-devices.html#avfoundation)
on macOS. Install `ffmpeg` with Homebrew if it is absent (`brew install ffmpeg`)
and allow camera access to the launching application if macOS asks. No OpenCV or
additional Python camera packages are required by this viewer. The preview is
scaled to 640 pixels wide at up to 10 frames per second; this does not change the
native capture settings in the project configuration.

Each camera runs in a separate supervised process. Missing devices, capture
errors, or stalled frames show **Не подключена** in that camera's tile. Frames
older than two seconds are not served. Capture retries automatically, and one
camera can recover while the other remains absent. Neither camera is required
to start the app, connect the arm, or move it. FFmpeg being absent leaves both
tiles disconnected.

The same paired server supplies video on the iPhone; there is no second address
or extra pairing step. No video files are recorded. Camera requests pause while
the control page is hidden or disconnected. To leave cameras entirely unopened:

```bash
teleoperator/run.sh --no-cameras
```

`--check` also leaves all cameras unopened. Close other applications using a
camera if that camera cannot provide frames to this controller.

## Prepare the hardware environment

Use the Python interpreter from the existing, working LeRobot environment.
Create a separate environment that can import those packages, then install the
viewer dependencies into the separate environment:

```bash
uv venv --python /path/to/existing/lerobot/bin/python --system-site-packages teleoperator/.venv-robot
uv pip install --python teleoperator/.venv-robot/bin/python -r teleoperator/requirements.txt
```

Replace `/path/to/existing/lerobot/bin/python` with the actual interpreter path,
for example the Python inside an existing Conda environment. This preserves the
existing LeRobot environment and avoids installing another LeRobot version.
Python 3.10 or later is supported.

The follower port, robot ID, and calibration are read from the repository's
`config.json`. Calibration paths are resolved relative to that file. Use
`--config /path/to/config.json` to select another configuration. The arm must
already be configured for position control and calibrated with LeRobot.

First validate the configuration without opening the robot:

```bash
teleoperator/run.sh --hardware --check
```

This checks the saved configuration and bundled model. It does not establish
that a robot is connected, that its motors match the calibration, or that the
model is physically aligned. Those checks require the assembled arm.

## Operate the real arm

```bash
teleoperator/run.sh --hardware
```

The launcher does not install packages during a hardware launch. The server
still starts with motors disabled. In the browser:

1. Click **Подключить руку**. This opens the configured follower port, checks the
   motor calibration and position-control mode, and reads the current pose with
   torque off. It refuses a connection if torque is already enabled. No camera
   or leader-arm connection is opened.
2. Compare the displayed pose with the real arm. Verify joint directions and
   geometry with small movements before relying on the displayed coordinates.
3. Click **Включить движение**. The program validates the measured pose, writes
   that pose as the initial motor goal, then enables torque. Previous preview
   targets are discarded. If the current pose is outside the calibrated joint
   range, enabling is refused. The folded rest pose of `recording/rest.py` can be
   enabled, although its shoulder and elbow are past the model limits: a joint
   past a limit may only stay or move back. More than 10 mm below the minimum
   height, as at the rest pose, the tool only lifts straight up, because the model
   height says little about table clearance down there; the panel asks for this,
   and other input does nothing. Closer to the minimum, where the arm sags while
   holding, any target first brings the tool up to the minimum height.
4. Drag the gripper nose in small steps, or use the optional XYZ handles. The
   display shows measured motor positions
   while connected. Joint speed, gripper speed, calibrated joint ranges, and
   tracking error are bounded. An excessive tracking error pauses movement and
   commands the measured pose as a hold.

**СТОП: удерживать** pauses commands at the measured pose and keeps torque on.
It is a software hold, not a force-rated emergency stop. A failed USB connection
can prevent a new hold command from reaching the motors.

**Отпустить моторы** disables torque. Support the arm first: it will stop holding
its weight. **Отключиться**, Ctrl+C, and normal server exit attempt to release
owned torque and close the connection. A failed release is reported as unconfirmed.
Closing the controlling browser pauses movement
and retains torque; it does not release the arm. Only one browser client controls
the arm at a time, while STOP remains available to other connected clients. Changes
from a viewing client are discarded and the shared controls return to the
operator's values. STOP discards queued inputs, including a pending enable request.

A communication failure freezes the last measured pose and latches an error.
The program does not resume a previous trajectory or repeatedly poll a failed
connection. Check the arm and link, then explicitly enable movement again if the
connection is still open, or disconnect and reconnect. Old targets are discarded.

Keyboard input expires after 250 ms without a fresh heartbeat. STOP and mode or
connection changes invalidate held keys; release and press again to continue.
Keyboard release holds the arm while preserving an existing grasp. STOP also
preserves the commanded grip instead of adopting an object's measured jaw gap.

Automatic calibration is never run. Saved calibration files and motor EEPROM
settings are not rewritten by this application.

## iPhone over Wi-Fi

The iPhone is a remote control; the Mac retains USB, calibration and motion
control. The native app project and installation steps are in [ios/README.md](ios/README.md).

For a preview without hardware:

```bash
teleoperator/run.sh --lan
```

For the real follower:

```bash
teleoperator/run.sh --hardware --lan
```

Open the desktop URL printed by the server. Under **Подключить iPhone**, scan the
QR code with the iOS app or paste the displayed link. Keep both devices on the
same trusted Wi-Fi network and keep the Mac awake. The phone then uses the same
Connect and Enable buttons as the desktop. Verify the real pose before enabling.

The QR uses the Mac's `.local` hostname by default. If name resolution is not
available on the network, supply the Mac's private Wi-Fi address:

```bash
teleoperator/run.sh --hardware --lan --address 192.168.1.10
```

Replace that example address with the Mac's actual address. Each launch creates
a new pairing key; previous links stop working after restart. The link grants
access to the control room. Share it only with the intended operator.

If the desktop already controls the arm, click **Передать управление** there.
This pauses motion and keeps motor torque. On the phone, explicitly enable
movement again. Reading a QR code never silently takes control from another tab.

Normal launches remain loopback-only. `--lan` exposes one authenticated port for
the page, video snapshots and WebSocket; the raw Viser backend requires a separate private gateway
key. This first version uses HTTP/WebSocket on a trusted local network. It is not
an Internet-control endpoint and does not provide TLS.

The foreground controller sends a heartbeat. Moving its app into the background
or losing heartbeats for 500 ms pauses movement and discards queued commands.
Reconnection does not automatically enable motors. USB failure and power loss
still cannot guarantee a software hold.

## Landscape remote on iPhone

After pairing, turn the iPhone sideways to open the two-thumb remote. The shared
control page also has a **Пульт** button for trying this layout in a Mac browser.
**3D** returns to the normal scene.

- The left stick moves the tool parallel to the table, as seen from behind the arm.
  Up is forward, away from the base (positive model X); right is the arm's right
  (negative model Y). The camera images are not geometrically registered, so
  rotating a camera or the 3D scene does not redefine these directions.
- The right stick raises/lowers the tool and rotates the wrist. Up raises the tool;
  moving right increases wrist roll. Both sticks can be held together.
- Hold **Открыть** or **Закрыть** under the right thumb to change the gripper
  opening while the left thumb can keep moving over the table.
- The wrist and overhead previews remain between the sticks. Missing cameras
  show their disconnected state and do not disable any controls.
- **Стоп** remains available at the top. Real hardware still requires Connect
  and explicit movement enablement.

Stick displacement sets speed, with a neutral zone around the center. Maximum
translation speed remains 20 mm/s; wrist rotation is limited to 20 degrees/s
and the configured joint speed, and gripper opening to 25 percentage points/s.
Releasing a control removes its rate. Releasing every control holds the arm and
the commanded grasp. Input expires after 250 ms without a fresh packet.

Changing layout or rotating back clears held input and pauses movement for the
current operator. Hardware movement must then be explicitly enabled again.
An observer changing layout cannot interrupt another operator. Returning from
background or reconnecting never restarts held thumb commands.

The native shell identifies itself to the shared viewer and keeps the WebView
mounted while rotating. The native Disconnect button remains accessible in a
compact header. Source and portable checks are available here; a full iOS build
and touch testing on a physical iPhone still require Xcode.

## Limits of this version

- Cartesian coordinates come from the geometric model and motor calibration.
  There is no external position measurement or automatic alignment to the table.
- Minimum height applies to the tool frame, not the entire arm or its fingers.
  It is 15 mm above the model base plane by default. Set another value with
  `--min-height-mm` only after checking the physical setup. The tool stops at
  this height and motion stays enabled, so moving away needs no new command.
- There is no collision planner or contact detection. The model does not include
  this setup's wrist camera, cables, objects, or mounting changes.
- The gripper percentage is mapped approximately to the visual jaw angle. It
  does not measure the gap between the fingers or the force on an object.
- Dragging specifies position. Wrist roll and jaw opening are separate controls;
  the other joint angles are solved near the preceding pose. Arbitrary six-axis
  tool orientation is not supported.
- Hard process termination, power loss, and a broken serial link cannot guarantee
  software cleanup or a motor stop.
- **Отключиться** on a held arm, or 5 seconds with no page or phone connected,
  returns the arm to `rest_pose` at 10 degrees per second and then frees the serial
  port. From far out the tool rises to 100 mm, pulls back high, turns the base at a
  folded pose, and folds. This is the one automatic motion. It has no collision
  check, so keep the space above the table clear. Pressing **Стоп** or **Включить
  движение** ends it; a joint lagging more than 12 degrees stops it and holds the
  pose. Stopping the server itself still releases torque without returning.

The bundled model's source revision, license, checksums, and geometric caveats
are in [models/README.md](models/README.md).

The dark control room is bundled with the app. Editing its design and rebuilding
the optional browser client is described in [web/README.md](web/README.md).

## Useful commands

```bash
# Start on another local port at a lower joint speed.
teleoperator/run.sh --port 8081 --speed 10

# Run the model, controller, and fake motor-bus tests without hardware.
uv run --project teleoperator --group dev python -m pytest teleoperator -q
```

## Native iOS validation boundary

The iOS project contains a QR/manual connection screen and the existing 3D viewer
inside a restricted `WKWebView`. The arm64 iPhone app now builds successfully
with Xcode 26.6 and the iOS 26.5 SDK. A Personal Team signed build was installed
and launched on an iPhone 15 Pro with iOS 27.0.1. Portable pairing and
connection-lifecycle checks also pass. Phone networking, touch controls, camera
permissions and physical-arm operation still require checks on the installed
app. See [ios/README.md](ios/README.md).
