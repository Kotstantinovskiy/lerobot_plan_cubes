#!/usr/bin/env python3
"""Build the optional custom browser client; normal app launches need no Node."""

from importlib import metadata
from pathlib import Path
import shutil
import subprocess

import viser

ROOT = Path(__file__).resolve().parent


def patch_camera_keyboard(client):
    path = client / "src" / "CameraControls.tsx"
    source = path.read_text()
    patches = [
        ('import { isFormElement } from "./utils/isFormElement";',
         'import { isFormElement } from "./utils/isFormElement";\nimport { KeyboardCameraMotion } from "./keyboardCameraMotion";', 1),
        ('  const heldKeysRef = useRef<Set<string>>(new Set());',
         '  const heldKeysRef = useRef<Set<string>>(new Set());\n  const keyboardMotionRef = useRef(new KeyboardCameraMotion());', 1),
        ('      if (!held.delete(event.code)) return;',
         '      if (!held.delete(event.code)) return;\n      if (held.size === 0) keyboardMotionRef.current.reset();', 1),
        ('      held.clear();', '      held.clear();\n      keyboardMotionRef.current.reset();', 2),
        ('  "KeyQ",\n  "KeyE",\n', '', 1),
        ('  "ArrowUp",\n  "ArrowDown",\n  "ArrowLeft",\n  "ArrowRight",\n', '', 1),
        ('  // Apply held-key camera movement each frame. Rates: linear 2.0/s, rotation\n  // 50 deg/s (`delta` is in seconds).',
         '  // WASD orbits the camera. Arrow keys belong to the robot controller.\n  // Camera translation is disabled for this workcell.', 1),
        ('    if (cameraControls === null || held.size === 0) return;',
         '    if (cameraControls === null || held.size === 0) {\n      keyboardMotionRef.current.reset();\n      return;\n    }', 1),
        ('    if (!cameraControls.enabled) return;',
         '    if (!cameraControls.enabled) {\n      keyboardMotionRef.current.reset();\n      return;\n    }', 1),
        ('''    const linear = 2.0 * delta;
    const angular = 50.0 * THREE.MathUtils.DEG2RAD * delta;
    if (held.has("KeyA")) cameraControls.truck(-linear, 0, false);
    if (held.has("KeyD")) cameraControls.truck(linear, 0, false);
    if (held.has("KeyW")) cameraControls.forward(linear, false);
    if (held.has("KeyS")) cameraControls.forward(-linear, false);
    if (held.has("KeyQ")) cameraControls.elevate(-linear, false);
    if (held.has("KeyE")) cameraControls.elevate(linear, false);
    if (held.has("ArrowLeft")) cameraControls.rotate(-angular, 0, true);
    if (held.has("ArrowRight")) cameraControls.rotate(angular, 0, true);
    if (held.has("ArrowUp")) cameraControls.rotate(0, -angular, true);
    if (held.has("ArrowDown")) cameraControls.rotate(0, angular, true);''',
         '''    const [azimuth, polar] = keyboardMotionRef.current.step(held, delta);
    if (azimuth !== 0 || polar !== 0) cameraControls.rotate(azimuth, polar, false);
    viewerMutable.requestRender();''', 1),
        ('        dollySpeed={0.3}', '        dollySpeed={0.3}\n        truckSpeed={0}', 1),
    ]
    for old, new, count in patches:
        if source.count(old) != count:
            raise RuntimeError("Pinned Viser keyboard source changed; review the camera patch")
        source = source.replace(old, new)
    path.write_text(source)


def main():
    version = metadata.version("viser")
    if version != "1.1.1":
        raise RuntimeError("Build this client with the project's pinned viser==1.1.1")
    package = Path(viser.__file__).resolve().parent
    work = ROOT / ".viewer-build"
    client = work / "client"
    shutil.copytree(
        package / "client", client, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("node_modules", "build", ".git"),
    )
    shutil.copytree(package / "_assets", work / "_assets", dirs_exist_ok=True)
    for name in (
        "Vector2.tsx", "GripperControl.tsx", "GripperControl.test.tsx", "control-room.css",
        "TextInput.tsx", "KeyboardControl.tsx", "KeyboardControl.test.tsx",
        "SessionHeartbeat.tsx", "SessionHeartbeat.test.tsx",
        "CameraFeeds.tsx", "CameraFeeds.test.tsx",
        "RemoteControl.tsx", "RemoteControl.test.tsx", "ThumbStick.tsx", "remoteMode.ts", "sequence.ts",
    ):
        shutil.copy2(ROOT / "web" / name, client / "src" / "components" / name)
    for name in ("keyboardCameraMotion.ts", "keyboardCameraMotion.test.ts"):
        shutil.copy2(ROOT / "web" / name, client / "src" / name)
    patch_camera_keyboard(client)
    shutil.copy2(ROOT / "web" / "vitest.config.mts", client / "so101.vitest.config.mts")
    index = client / "index.html"
    page = index.read_text().replace('<html lang="en">', '<html lang="ru">')
    page = page.replace("<title>Viser</title>", "<title>SO-101 / Control</title>")
    page = page.replace('content="Viser client"', 'content="SO-101 robot control"')
    index.write_text(page)
    if not (client / "node_modules").is_dir():
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=client, check=True)
    subprocess.run(["npm", "exec", "--", "vitest", "run", "--config", "so101.vitest.config.mts"], cwd=client, check=True)
    subprocess.run(["npm", "run", "build"], cwd=client, check=True)
    destination = ROOT / "web" / "dist"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(client / "build" / "index.html", destination / "index.html")
    distribution = metadata.distribution("viser")
    license_file = next(path for path in distribution.files if str(path).endswith("dist-info/licenses/LICENSE"))
    shutil.copy2(distribution.locate_file(license_file), destination / "LICENSE.viser")
    shutil.copy2(client / "src" / "vendor" / "instanced-mesh" / "LICENSE", destination / "LICENSE.instanced-mesh")
    print(f"Built custom Viser {version} client: {destination / 'index.html'}")


if __name__ == "__main__":
    main()
