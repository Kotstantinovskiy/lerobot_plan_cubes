# SO-101 iPhone control

A small native iOS app for the existing Mac controller. It scans or accepts the
Mac pairing link, checks the authenticated `/health` endpoint, and displays the
same 3D control room in `WKWebView`. The Mac keeps the USB connection, calibration,
kinematics and motion limits. The app requires iOS 16 or newer.

## Build and install

1. Install the full Xcode application with an iOS SDK. Command Line Tools alone
   do not contain the iOS build tools or simulator.
2. Open `SO101Control.xcodeproj` in Xcode.
3. Under the SO101Control target, select your signing Team. Change
   `local.so101.control` if your team requires another bundle identifier.
4. Select an iPhone or iPhone Simulator and run the SO101Control scheme.

There are no package dependencies. The checked-in project and shared scheme are
complete. A distribution app icon and App Store metadata are not included in
this prototype.

## Connect

Start the Mac controller in its paired LAN mode. Keep the Mac and iPhone on the
same trusted Wi-Fi network, with the arm connected to the Mac by USB. Scan the
pairing QR code in the app, or paste the complete pairing URL and tap Connect.
The pairing link must contain its token. Tokens are kept in memory, not stored
in UserDefaults, Keychain or the WebView's persistent browser storage.

iOS requests local-network permission when checking the connection. Camera
permission is requested only for QR scanning. The simulator can use a pasted
link; `127.0.0.1` refers to the Mac from the iOS Simulator, while a real iPhone
needs the Mac's LAN address or `.local` name.

If the first local-network permission prompt makes the app inactive, the pending
connection is cancelled too. Allow the permission, then tap Connect again. The
app never lets a delayed health response open control while it is inactive.

When the app becomes inactive or goes into the background, it removes the
control WebView and replaces its document. This closes its WebSocket. The Mac's
foreground heartbeat timeout also stops stale input. Returning to the app
requires an explicit reconnect and never automatically enables the motors.

Manual Disconnect also clears the pairing link. After a server restart, scan
the new QR code because the previous token is no longer valid.

The usual sequence is to launch the Mac server, scan its code, then use Connect
and Enable in the iPhone control room. Opening the QR panel on the Mac does not
claim control. If the Mac browser already controls the arm, click **Передать
управление** on the Mac first, then explicitly enable movement on the phone.
Pairing never silently takes control from another operator.

The control room also displays the Mac's wrist and overhead camera previews.
They use the same pairing link and do not require access to the iPhone camera.
A missing or stalled camera shows **Не подключена** independently; arm control
and the other camera continue working. See the Mac README for FFmpeg setup and
the `--no-cameras` option. Video capture stays on the Mac, and no recordings are
created by the viewer.

## Landscape remote

Rotate the connected iPhone horizontally to open the two-thumb remote. The
native header becomes compact and retains **Отключиться**. Rotation resizes the
existing WebView without reloading it or changing the pairing link.

- The left stick moves the gripper parallel to the table.
- The right stick raises or lowers the gripper vertically and turns the wrist
  horizontally.
- Hold **Открыть** or **Закрыть** to change the jaw opening.
- The wrist and overhead camera previews stay between the two sticks. A missing
  camera continues to show **Не подключена** independently.
- **Стоп** pauses movement. **3D** returns to the model view; **Пульт** opens the
  remote again.

Switching layouts as the current operator pauses motion and clears held touches.
Initial landscape display does not claim control, and an observer rotating the
phone cannot interrupt another operator. Paused hardware needs explicit movement
enablement again. Releasing a stick stops its
motion; hiding the app, losing the connection, or rotating back to portrait also
clears input. Returning to the foreground never resumes a held command.

The WebView identifies the native app with the `SO101Control/1.0` user-agent
marker. This selects the landscape interface only; pairing and command
authorization still use the existing authenticated server connection.

## Network boundary

The first version uses authenticated HTTP/WebSocket on a trusted local network.
It is not an Internet remote-control service. Native pairing validation accepts
only private/link-local IPv4, loopback, or `.local` hosts, and strips unrelated
query parameters. The health check does not follow redirects. WebView navigation
is limited to the paired HTTP origin and the controller page paths.

ATS is not disabled globally. `NSAllowsLocalNetworking` permits local names;
private IPv4 and loopback CIDR exceptions cover the iOS 17 and newer IP-address
rules. See [Apple's local networking ATS key](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity/nsallowslocalnetworking).

## Validation

Run `./check.sh` to lint property lists, compile and run the Foundation-only
pairing validation checks, compile the connection model against the Mac SDK,
exercise its health checks against a local HTTP fixture, and parse every Swift
file. The model checks cover rejected tokens and redirects, suspension, and
cancelled connection attempts. When Xcode is available,
the script also builds the arm64 iPhone app without code signing. This validates
the actual iOS SDK and does not require downloading a simulator runtime.

If Xcode is not selected globally, point the check at its developer directory:

```bash
DEVELOPER_DIR=/path/to/Xcode.app/Contents/Developer ./check.sh
```

On this Mac, Xcode 26.6 is currently in `~/Downloads/Xcode.app`. The full arm64
iPhone build passed after fixing a SwiftUI color-name collision. A Personal Team
signed build was verified, installed and launched on the paired iPhone 15 Pro
running iOS 27.0.1. The application remained running after launch. Developer Mode
and trust for the development signature are enabled on that phone.

The Mac controller can now provide a paired LAN preview for the first remote
test. Local-network permission, installed-app touch controls, QR camera and
physical-arm operation remain separate checks; successful installation alone
does not validate them. No simulator runtime is required for the device build.
