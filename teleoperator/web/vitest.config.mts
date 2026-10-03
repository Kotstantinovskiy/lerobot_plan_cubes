import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: [
      "src/components/GripperControl.test.tsx",
      "src/components/KeyboardControl.test.tsx",
      "src/components/SessionHeartbeat.test.tsx",
      "src/components/CameraFeeds.test.tsx",
      "src/components/RemoteControl.test.tsx",
      "src/keyboardCameraMotion.test.ts",
    ],
  },
});
