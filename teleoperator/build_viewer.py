#!/usr/bin/env python3
"""Build the optional custom browser client; normal app launches need no Node."""

import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import viser

ROOT = Path(__file__).resolve().parent
PINNED_VISER_VERSION = "1.1.1"
# Files of ours that replace or join the components of the Viser client.
COMPONENT_FILES = (
    "Vector2.tsx",
    "GripperControl.tsx",
    "GripperControl.test.tsx",
    "control-room.css",
    "TextInput.tsx",
    "KeyboardControl.tsx",
    "KeyboardControl.test.tsx",
    "SessionHeartbeat.tsx",
    "SessionHeartbeat.test.tsx",
    "CameraFeeds.tsx",
    "CameraFeeds.test.tsx",
    "RemoteControl.tsx",
    "RemoteControl.test.tsx",
    "ThumbStick.tsx",
    "remoteMode.ts",
    "sequence.ts",
)
SOURCE_FILES = ("keyboardCameraMotion.ts", "keyboardCameraMotion.test.ts")
PAGE_REPLACEMENTS = (
    ('<html lang="en">', '<html lang="ru">'),
    ("<title>Viser</title>", "<title>Teleoperator</title>"),
    ('content="Viser client"', 'content="Teleoperator: SO-101 robot control"'),
)
# Text in the pinned CameraControls.tsx, what replaces it, and how many times it must be found.
CAMERA_KEYBOARD_PATCHES = (
    (
        'import { isFormElement } from "./utils/isFormElement";',
        'import { isFormElement } from "./utils/isFormElement";\nimport { KeyboardCameraMotion } from "./keyboardCameraMotion";',
        1,
    ),
    (
        "  const heldKeysRef = useRef<Set<string>>(new Set());",
        "  const heldKeysRef = useRef<Set<string>>(new Set());\n  const keyboardMotionRef = useRef(new KeyboardCameraMotion());",
        1,
    ),
    (
        "      if (!held.delete(event.code)) return;",
        "      if (!held.delete(event.code)) return;\n      if (held.size === 0) keyboardMotionRef.current.reset();",
        1,
    ),
    ("      held.clear();", "      held.clear();\n      keyboardMotionRef.current.reset();", 2),
    ('  "KeyQ",\n  "KeyE",\n', "", 1),
    ('  "ArrowUp",\n  "ArrowDown",\n  "ArrowLeft",\n  "ArrowRight",\n', "", 1),
    (
        "  // Apply held-key camera movement each frame. Rates: linear 2.0/s, rotation\n  // 50 deg/s (`delta` is in seconds).",
        "  // WASD orbits the camera. Arrow keys belong to the robot controller.\n  // Camera translation is disabled for this workcell.",
        1,
    ),
    (
        "    if (cameraControls === null || held.size === 0) return;",
        "    if (cameraControls === null || held.size === 0) {\n      keyboardMotionRef.current.reset();\n      return;\n    }",
        1,
    ),
    (
        "    if (!cameraControls.enabled) return;",
        "    if (!cameraControls.enabled) {\n      keyboardMotionRef.current.reset();\n      return;\n    }",
        1,
    ),
    (
        """    const linear = 2.0 * delta;
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
    if (held.has("ArrowDown")) cameraControls.rotate(0, angular, true);""",
        """    const [azimuth, polar] = keyboardMotionRef.current.step(held, delta);
    if (azimuth !== 0 || polar !== 0) cameraControls.rotate(azimuth, polar, false);
    viewerMutable.requestRender();""",
        1,
    ),
    ("        dollySpeed={0.3}", "        dollySpeed={0.3}\n        truckSpeed={0}", 1),
)


def patch_camera_keyboard(client):
    path = client / "src" / "CameraControls.tsx"
    source = path.read_text()
    for old, new, count in CAMERA_KEYBOARD_PATCHES:
        if source.count(old) != count:
            raise RuntimeError("Pinned Viser keyboard source changed; review the camera patch")
        source = source.replace(old, new)
    path.write_text(source)


def copy_client_sources(package, work, client):
    """Copy the Viser client out of its installed package and add our files to the copy."""
    shutil.copytree(
        package / "client",
        client,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("node_modules", "build", ".git"),
    )
    shutil.copytree(package / "_assets", work / "_assets", dirs_exist_ok=True)
    for name in COMPONENT_FILES:
        shutil.copy2(ROOT / "web" / name, client / "src" / "components" / name)
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / "web" / name, client / "src" / name)
    shutil.copy2(ROOT / "web" / "vitest.config.mts", client / "teleoperator.vitest.config.mts")


def rename_page(client):
    index = client / "index.html"
    page = index.read_text()
    for old, new in PAGE_REPLACEMENTS:
        page = page.replace(old, new)
    index.write_text(page)


def test_and_build_client(client):
    if not (client / "node_modules").is_dir():
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=client, check=True)
    subprocess.run(
        ["npm", "exec", "--", "vitest", "run", "--config", "teleoperator.vitest.config.mts"], cwd=client, check=True
    )
    subprocess.run(["npm", "run", "build"], cwd=client, check=True)


def publish_build(client, destination):
    """Copy the built page and the licenses it needs next to it."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(client / "build" / "index.html", destination / "index.html")
    distribution = metadata.distribution("viser")
    license_file = next(path for path in distribution.files if str(path).endswith("dist-info/licenses/LICENSE"))
    shutil.copy2(distribution.locate_file(license_file), destination / "LICENSE.viser")
    shutil.copy2(client / "src" / "vendor" / "instanced-mesh" / "LICENSE", destination / "LICENSE.instanced-mesh")


def main():
    version = metadata.version("viser")
    if version != PINNED_VISER_VERSION:
        raise RuntimeError("Build this client with the project's pinned viser==1.1.1")
    package = Path(viser.__file__).resolve().parent
    work = ROOT / ".viewer-build"
    client = work / "client"
    copy_client_sources(package, work, client)
    patch_camera_keyboard(client)
    rename_page(client)
    test_and_build_client(client)
    destination = ROOT / "web" / "dist"
    publish_build(client, destination)
    print(f"Built custom Viser {version} client: {destination / 'index.html'}")


if __name__ == "__main__":
    main()
