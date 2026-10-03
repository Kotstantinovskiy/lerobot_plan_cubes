// @vitest-environment jsdom
import * as React from "react";
import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import * as THREE from "three";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerContext } from "../ViewerContext";
import KeyboardControl, { keyboardDirection } from "./KeyboardControl";
import { setRemoteMode } from "./remoteMode";
import { resetSequenceForTests } from "./sequence";

vi.mock("../ViewerContext", async () => {
  const react = await import("react");
  return { ViewerContext: react.createContext(null) };
});
vi.mock("../WorldTransformUtils", () => ({
  computeT_threeworld_world: (viewer: { transform: THREE.Matrix4 }) => viewer.transform,
}));

const keys = (...values: string[]) => new Set(values);
function makeCamera(position: [number, number, number] = [0, -1, 1]) {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(...position);
  camera.up.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld(true);
  return camera;
}

function expectDirection(actual: number[], expected: number[]) {
  expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 10));
}

describe("table-parallel keyboard direction", () => {
  it("projects arrows into table XY relative to the current view", () => {
    const camera = makeCamera();
    const identity = new THREE.Matrix4();
    expectDirection(keyboardDirection(keys("ArrowUp"), camera, identity), [0, 1, 0]);
    expectDirection(keyboardDirection(keys("ArrowRight"), camera, identity), [1, 0, 0]);
    expectDirection(keyboardDirection(keys("ArrowDown"), camera, identity), [0, -1, 0]);
    expectDirection(keyboardDirection(keys("ArrowLeft"), camera, identity), [-1, 0, 0]);
    expectDirection(keyboardDirection(keys("ArrowRight"), makeCamera([1, 0, 1]), identity), [0, 1, 0]);
  });

  it("uses camera right at a top-down view without making arrows vertical", () => {
    const camera = makeCamera([0, 0, 1]);
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld(true);
    expectDirection(keyboardDirection(keys("ArrowUp"), camera, new THREE.Matrix4()), [0, 1, 0]);
    expectDirection(keyboardDirection(keys("ArrowRight"), camera, new THREE.Matrix4()), [1, 0, 0]);
  });

  it("converts the Three.js scene frame into the robot's Z-up frame", () => {
    const transform = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
    const camera = new THREE.PerspectiveCamera();
    camera.position.copy(new THREE.Vector3(0, -1, 1).applyMatrix4(transform));
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld(true);
    expectDirection(keyboardDirection(keys("ArrowUp"), camera, transform), [0, 1, 0]);
    expectDirection(keyboardDirection(keys("ArrowRight"), camera, transform), [1, 0, 0]);
  });

  it("keeps K/M vertical and normalizes simultaneous axes", () => {
    const camera = makeCamera();
    const identity = new THREE.Matrix4();
    expectDirection(keyboardDirection(keys("KeyK"), camera, identity), [0, 0, 1]);
    expectDirection(keyboardDirection(keys("KeyM"), camera, identity), [0, 0, -1]);
    expectDirection(keyboardDirection(keys("KeyK", "KeyM", "ArrowUp", "ArrowDown"), camera, identity), [0, 0, 0]);
    const direction = keyboardDirection(keys("ArrowUp", "KeyK"), camera, identity);
    expectDirection(direction, [0, Math.SQRT1_2, Math.SQRT1_2]);
    expect(Math.hypot(...direction)).toBeCloseTo(1, 10);
    expectDirection(keyboardDirection(keys("ArrowUp"), null, identity), [0, 0, 0]);
  });
});

describe("keyboard arm input lifecycle", () => {
  let host: HTMLDivElement;
  let root: Root;
  let connected: boolean;
  const onChange = vi.fn<(value: string) => void>();
  let viewer: {
    mutable: { current: { camera: THREE.Camera | null } };
    useGui: (select: (state: { websocketState: string }) => unknown) => unknown;
    useSceneTree: { get: () => object };
    transform: THREE.Matrix4;
  };

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    vi.useFakeTimers();
    resetSequenceForTests();
    vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    connected = true;
    viewer = {
      mutable: { current: { camera: makeCamera() } },
      useGui: (select) => select({ websocketState: connected ? "connected" : "disconnected" }),
      useSceneTree: { get: () => ({}) },
      transform: new THREE.Matrix4(),
    };
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    onChange.mockClear();
  });

  afterEach(() => {
    act(() => root.unmount());
    setRemoteMode(false);
    host.remove();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  function render(disabled = false, resetKey = "0") {
    act(() => root.render(<ViewerContext.Provider value={viewer as never}>
      <KeyboardControl disabled={disabled} resetKey={resetKey} onChange={onChange}/>
    </ViewerContext.Provider>));
  }

  function key(type: "keydown" | "keyup", code: string, options: KeyboardEventInit = {}, target: EventTarget = document.body) {
    const event = new KeyboardEvent(type, { bubbles: true, cancelable: true, code, ...options });
    act(() => target.dispatchEvent(event));
    return event;
  }

  function lastPayload(): number[] { return JSON.parse(onChange.mock.lastCall![0]); }

  it("does not claim control on mount or authoritative updates", () => {
    render();
    render(false, "3");
    render(true, "3");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("sends view-relative direction immediately and heartbeats with unique sequence numbers", () => {
    render(false, "7");
    const event = key("keydown", "ArrowUp");
    expect(event.defaultPrevented).toBe(true);
    expectDirection(lastPayload().slice(0, 3), [0, 1, 0]);
    expect(lastPayload().slice(3)).toEqual([1, 7]);
    act(() => vi.advanceTimersByTime(150));
    expect(onChange).toHaveBeenCalledTimes(4);
    expect(lastPayload().slice(3)).toEqual([4, 7]);
    key("keyup", "ArrowUp");
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(5);
  });

  it("does not depend on operating-system key repeat and ignores WASD", () => {
    render();
    key("keydown", "KeyW");
    expect(onChange).not.toHaveBeenCalled();
    key("keydown", "KeyK");
    key("keydown", "KeyK", { repeat: true });
    key("keydown", "KeyK", { repeat: true });
    expect(onChange).toHaveBeenCalledTimes(1);
    expectDirection(lastPayload().slice(0, 3), [0, 0, 1]);
    key("keydown", "KeyM");
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    key("keyup", "KeyK");
    expectDirection(lastPayload().slice(0, 3), [0, 0, -1]);
  });

  it("does not start an input stream before the camera is ready", () => {
    viewer.mutable.current.camera = null;
    render();
    key("keydown", "KeyK");
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("does not capture typing, instrument sliders, buttons, or system shortcuts", () => {
    render();
    for (const name of ["input", "textarea", "select", "button", "div"]) {
      const element = document.createElement(name);
      if (name === "div") element.setAttribute("role", "slider");
      host.append(element);
      expect(key("keydown", "ArrowUp", {}, element).defaultPrevented).toBe(false);
      key("keydown", "KeyK", {}, element);
    }
    key("keydown", "KeyK", { metaKey: true });
    key("keydown", "KeyK", { ctrlKey: true });
    key("keydown", "KeyK", { altKey: true });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("releases a held key even if focus moves into an input", () => {
    render();
    key("keydown", "ArrowRight");
    const input = document.createElement("input");
    host.append(input);
    key("keyup", "ArrowRight", {}, input);
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(2);
  });

  it("stops immediately when a held arrow moves focus to a form control", () => {
    render();
    key("keydown", "ArrowRight");
    const input = document.createElement("input");
    host.append(input);
    act(() => input.focus());
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    const count = onChange.mock.calls.length;
    act(() => vi.advanceTimersByTime(500));
    key("keydown", "ArrowRight", { repeat: true });
    expect(onChange).toHaveBeenCalledTimes(count);
  });

  it("stops on blur, visibility loss, disconnection, and disable", () => {
    render();
    key("keydown", "KeyK");
    act(() => window.dispatchEvent(new Event("blur")));
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    const afterBlur = onChange.mock.calls.length;
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(afterBlur);
    key("keydown", "KeyK");
    vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    key("keydown", "KeyK");
    connected = false;
    render();
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    connected = true;
    render();
    key("keydown", "KeyK");
    render(true);
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    const afterDisable = onChange.mock.calls.length;
    key("keydown", "ArrowUp");
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(afterDisable);
  });

  it("requires a fresh physical press after an epoch change", () => {
    render(false, "2");
    key("keydown", "ArrowRight");
    render(false, "3");
    expect(lastPayload().slice(3)).toEqual([2, 3]);
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    const afterReset = onChange.mock.calls.length;
    key("keydown", "ArrowRight", { repeat: true });
    act(() => vi.advanceTimersByTime(100));
    expect(onChange).toHaveBeenCalledTimes(afterReset);
    key("keyup", "ArrowRight");
    key("keydown", "ArrowRight");
    expectDirection(lastPayload().slice(0, 3), [1, 0, 0]);
    expect(lastPayload()[4]).toBe(3);
  });

  it("cancels held keys and blocks keyboard jogging while the thumb controller is open", () => {
    render();
    key("keydown", "ArrowRight");
    act(() => setRemoteMode(true));
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    const count = onChange.mock.calls.length;
    key("keydown", "KeyK");
    act(() => vi.advanceTimersByTime(100));
    expect(onChange).toHaveBeenCalledTimes(count);
    act(() => setRemoteMode(false));
    key("keydown", "ArrowRight", { repeat: true });
    expect(onChange).toHaveBeenCalledTimes(count);
    key("keydown", "ArrowRight");
    expectDirection(lastPayload().slice(0, 3), [1, 0, 0]);
  });

  it("sends a final neutral on unmount and then removes listeners", () => {
    render();
    key("keydown", "ArrowUp");
    act(() => root.render(<></>));
    expectDirection(lastPayload().slice(0, 3), [0, 0, 0]);
    const count = onChange.mock.calls.length;
    key("keydown", "KeyK");
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(count);
  });
});
