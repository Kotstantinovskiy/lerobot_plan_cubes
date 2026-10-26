# Custom control room

The app uses a dark Viser 1.1.1 viewer with one custom control pair:
a rotary wrist dial and a vertical gripper-opening control. The component uses
the standard Viser vector input and WebSocket protocol. It does not access the
motors directly.

The design uses flat shapes, large values, and short labels. Fine graduations,
decorative instrument details, and telemetry cards are omitted from the main
panel; additional controls are under the collapsed settings section.

WASD controls camera orbit, with a gentle start, no release drift, and bounded
frame steps. Translation by keyboard and mouse is
disabled; zoom and mouse orbit remain available. `keyboardCameraMotion.ts`
contains the angular-motion calculation; the build applies a small checked patch
to Viser's camera handler. Camera bindings remain separate from robot input.

`KeyboardControl.tsx` sends arrow/K/M input through a standard Viser text input.
Directions are projected into the robot's table plane, with K/M kept vertical.
The Python controller owns speed, reachability and an expiring input lease.
Epochs require a fresh press after STOP or a control transition. Editing fields,
instrument focus, blur and disconnect cancel the input stream.

`SessionHeartbeat.tsx` also gives the controlling tab a foreground lease. It
does not claim ownership. Background, stale or interrupted heartbeats pause the
operator and purge queued motion and enable requests; explicit handoff releases
ownership while retaining the physical connection.

`RemoteControl.tsx` mounts through the standard `so101_remote` text component.
Its normal panel contains the **Пульт** button and `CameraFeeds.tsx`; camera
requests never send control commands. Wrist and overhead JPEGs use independent,
sequential HTTP requests through the same authenticated gateway. Each feed
keeps one request in flight and replaces its object URL after a new frame,
at up to ten frames per second. A failed, empty, or timed-out response clears
the old image and shows "Не подключена"; retries run once per second. Hidden
pages and disconnected viewers abort their requests and release images.
The same component is used on macOS and in the iOS web view.

The landscape remote has a table-plane thumbstick on the left, two camera
previews in the center, and a height/wrist thumbstick on the right. Hold
**Открыть** or **Закрыть** to change the gripper opening. The left stick uses
fixed robot-world X/Y directions, independently of 3D camera orbit: right is
positive X and up is positive Y. On the right, up raises Z and right increases
wrist roll. Camera mounting orientation is not automatically calibrated.

Native iOS identifies its WebView with `SO101Control/1.0`; landscape widths of
560 pixels or more open the remote automatically. A manual **Пульт** button is
available for browser use. The interface switches camera placement without
duplicating streams. Initial layout selection sends no commands; later mode
changes pause an owned gesture and require fresh input. Physical motion still
requires explicit enablement.

Thumb directions use a radial dead zone and a 50 ms input heartbeat over the
existing authenticated WebSocket. The server owns limits and the 250 ms input
lease. Release, cancellation, blur, hidden pages, disconnect, server epoch
changes and layout changes clear held input. Two captured pointers can operate
independently; a third pointer or a pointer already assigned to another control
cannot start another gesture. `remoteMode.ts` also blocks keyboard jogging while
the remote is open, so the two input modes cannot compete. A native HTML modal
keeps the underlying 3D controls out of the focus and accessibility order;
Escape closes the remote and cancels held input.

`dist/index.html` is the self-contained production client. Normal launches only
need Python. `viewer.py` serves that file on the chosen application port and
connects it to the local Viser backend using its supported `websocket` parameter.
The raw Viser server stays on loopback and is guarded by a private gateway key.
The page and WebSocket use a single public port: loopback by default, or an
explicit paired LAN mode. Both servers are stopped together.

To edit the design, change `GripperControl.tsx` or `control-room.css`, then run
this from the repository root with Node.js 24 or newer and npm installed:

```bash
control_3d/.venv/bin/python control_3d/build_viewer.py
```

The build script copies the installed Viser client into the ignored
`.viewer-build` folder, overlays the custom files, runs the component and
keyboard-motion tests and TypeScript checks, and saves the bundled HTML. It leaves the installed
Python package unchanged. Dependencies stay in the build folder.

The bundled client is derived from Viser 1.1.1 under Apache 2.0. Changes are the
custom vector-input component, control-room styles, camera keyboard motion,
and application metadata.
Upstream license files are preserved beside the generated HTML.
