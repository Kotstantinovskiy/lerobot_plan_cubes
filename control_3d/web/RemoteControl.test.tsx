// @vitest-environment jsdom
import * as React from "react";
import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerContext } from "../ViewerContext";
import RemoteControl from "./RemoteControl";
import { setRemoteMode } from "./remoteMode";
import { stickValue } from "./ThumbStick";

vi.mock("../ViewerContext", async () => {
  const react = await import("react");
  return { ViewerContext: react.createContext(null) };
});
vi.mock("./CameraFeeds", () => ({ default: () => <div className="camera-feeds">
  <section aria-label="Камера: Рука">Рука: Не подключена</section>
  <section aria-label="Камера: Сверху">Сверху: Не подключена</section>
</div> }));

const preview = { epoch: 0, hardware: false, connected: false, enabled: false, faulted: false, status: "Предпросмотр" };

describe("landscape dual-thumb controller", () => {
  let host: HTMLDivElement;
  let root: Root;
  let connected: boolean;
  let hidden: boolean;
  let landscape: boolean;
  let orientationChanged: (() => void) | null;
  const onChange = vi.fn<(value: string) => void>();
  const showModal = vi.fn<() => void>();
  const closeModal = vi.fn<() => void>();
  const captured = new Map<HTMLElement, Set<number>>();

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    vi.useFakeTimers();
    connected = true;
    hidden = false;
    landscape = false;
    orientationChanged = null;
    captured.clear();
    setRemoteMode(false);
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    vi.spyOn(navigator, "userAgent", "get").mockReturnValue("test browser");
    vi.stubGlobal("matchMedia", () => ({
      get matches() { return landscape; },
      addEventListener: (_type: string, listener: () => void) => { orientationChanged = listener; },
      removeEventListener: () => { orientationChanged = null; },
    }));
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() => ({
      left: 0, top: 0, right: 200, bottom: 200, width: 200, height: 200, x: 0, y: 0, toJSON: () => ({}),
    }));
    HTMLElement.prototype.setPointerCapture = function(id) {
      if (!captured.has(this)) captured.set(this, new Set());
      captured.get(this)!.add(id);
    };
    HTMLElement.prototype.hasPointerCapture = function(id) { return captured.get(this)?.has(id) ?? false; };
    HTMLElement.prototype.releasePointerCapture = function(id) { captured.get(this)?.delete(id); };
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    onChange.mockReset();
    showModal.mockReset();
    closeModal.mockReset();
    HTMLDialogElement.prototype.showModal = function() { this.open = true; showModal(); };
    HTMLDialogElement.prototype.close = function() { this.open = false; closeModal(); };
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    setRemoteMode(false);
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function render(state: object = preview) {
    const viewer = { useGui: (select: (state: { websocketState: string }) => unknown) => select({
      websocketState: connected ? "connected" : "disconnected",
    }) };
    act(() => root.render(<ViewerContext.Provider value={viewer as never}>
      <RemoteControl hint={JSON.stringify(state)} onChange={onChange}/>
    </ViewerContext.Provider>));
  }

  function button(text: string) {
    return [...document.querySelectorAll<HTMLButtonElement>("button")].find((element) => element.textContent === text)!;
  }
  function click(text: string) { act(() => button(text).click()); }
  function control(name: string): HTMLElement { return document.querySelector(`[data-control="remote-${name}"]`)!; }
  function pointer(name: string, type: string, x = 100, y = 100, id = 1) {
    const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 });
    Object.defineProperty(event, "pointerId", { value: id });
    act(() => control(name).dispatchEvent(event));
  }
  function enter(state: object = preview) { render(state); click("Пульт"); onChange.mockClear(); }
  function last(): number[] { return JSON.parse(onChange.mock.lastCall![0]); }
  function values() { return last().slice(0, 5); }
  function advance(ms: number) { act(() => vi.advanceTimersByTime(ms)); }

  it("does not claim control on mount or status changes and keeps a single pair of camera feeds", () => {
    render();
    render({ ...preview, epoch: 2 });
    expect(onChange).not.toHaveBeenCalled();
    expect(document.querySelectorAll('[aria-label^="Камера:"]')).toHaveLength(2);
    click("Пульт");
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "layout" });
    expect(document.querySelectorAll('[aria-label^="Камера:"]')).toHaveLength(2);
    expect(document.querySelector('[aria-label="Пульт управления SO-101"]')).not.toBeNull();
    click("3D");
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "layout" });
    expect(document.querySelector('[aria-label="Пульт управления SO-101"]')).toBeNull();
    expect(document.querySelectorAll('[aria-label^="Камера:"]')).toHaveLength(2);
  });

  it("combines two thumbs, normalizes Cartesian motion, and releases each thumb independently", () => {
    enter();
    pointer("left", "pointerdown", 200, 100, 1);
    pointer("right", "pointerdown", 100, 0, 2);
    advance(50);
    expect(values()[0]).toBeCloseTo(Math.SQRT1_2);
    expect(values()[2]).toBeCloseTo(Math.SQRT1_2);
    expect(Math.hypot(...values().slice(0, 3))).toBeCloseTo(1);
    pointer("right", "pointermove", 200, 100, 2);
    advance(50);
    expect(values()).toEqual([1, 0, 0, 1, 0]);
    pointer("left", "pointerup", 200, 100, 1);
    expect(values()).toEqual([0, 0, 0, 1, 0]);
    pointer("right", "pointerup", 200, 100, 2);
    expect(values()).toEqual([0, 0, 0, 0, 0]);
    const count = onChange.mock.calls.length;
    advance(500);
    expect(onChange).toHaveBeenCalledTimes(count);
  });

  it("uses a quiet center and progressive analog speed without flooding pointer moves", () => {
    expect(stickValue(0.1, 0)).toEqual([0, 0]);
    expect(stickValue(0.56, 0)[0]).toBeCloseTo(0.5);
    expect(stickValue(10, 0)).toEqual([1, 0]);
    enter();
    pointer("left", "pointerdown", 170, 100);
    for (let x = 171; x <= 190; x++) pointer("left", "pointermove", x, 100);
    expect(onChange).toHaveBeenCalledTimes(1);
    advance(50);
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(values()[0]).toBeGreaterThan(0.8);
    expect(last().slice(5)).toEqual([2, 0]);
  });

  it("ignores touches outside the circular pads and never reassigns a held pointer", () => {
    enter();
    pointer("left", "pointerdown", 0, 0, 1);
    expect(onChange).not.toHaveBeenCalled();
    pointer("left", "pointerdown", 200, 100, 1);
    pointer("right", "pointerdown", 200, 100, 1);
    advance(50);
    expect(values()).toEqual([1, 0, 0, 0, 0]);
  });

  it("allows one thumb to move the arm while the other opens or closes the gripper", () => {
    enter();
    pointer("left", "pointerdown", 200, 100, 1);
    pointer("open", "pointerdown", 100, 100, 2);
    advance(50);
    expect(values()).toEqual([1, 0, 0, 0, 1]);
    pointer("close", "pointerdown", 100, 100, 3);
    advance(50);
    expect(values()).toEqual([1, 0, 0, 0, 1]);
    pointer("open", "pointerup", 100, 100, 2);
    expect(values()).toEqual([1, 0, 0, 0, 0]);
    pointer("close", "pointerdown", 100, 100, 2);
    advance(50);
    expect(values()).toEqual([1, 0, 0, 0, -1]);
    pointer("close", "pointerup", 100, 100, 2);
    expect(values()).toEqual([1, 0, 0, 0, 0]);
  });

  it("does not accept a third touch while both sticks are in use", () => {
    enter();
    pointer("left", "pointerdown", 200, 100, 1);
    pointer("right", "pointerdown", 200, 100, 2);
    pointer("open", "pointerdown", 100, 100, 3);
    advance(50);
    expect(values()).toEqual([1, 0, 0, 1, 0]);
  });

  it("cancels both thumbs after pointer cancellation or lost capture and requires a fresh touch", () => {
    enter();
    pointer("left", "pointerdown", 200, 100, 1);
    pointer("right", "pointerdown", 100, 0, 2);
    pointer("left", "pointercancel", 200, 100, 1);
    expect(values()).toEqual([0, 0, 0, 0, 0]);
    const count = onChange.mock.calls.length;
    pointer("right", "pointermove", 100, 0, 2);
    advance(200);
    expect(onChange).toHaveBeenCalledTimes(count);
    pointer("right", "pointerdown", 100, 0, 2);
    expect(values()).toEqual([0, 0, 1, 0, 0]);
    pointer("right", "lostpointercapture", 100, 0, 2);
    expect(values()).toEqual([0, 0, 0, 0, 0]);
  });

  it("stops on a new server epoch without restarting an old held gesture", () => {
    enter({ ...preview, epoch: 3 });
    pointer("left", "pointerdown", 200, 100);
    render({ ...preview, epoch: 4 });
    expect(values()).toEqual([0, 0, 0, 0, 0]);
    expect(last()[6]).toBe(4);
    const count = onChange.mock.calls.length;
    pointer("left", "pointermove", 200, 100);
    advance(250);
    expect(onChange).toHaveBeenCalledTimes(count);
    pointer("left", "pointerdown", 200, 100);
    expect(values()).toEqual([1, 0, 0, 0, 0]);
  });

  it("keeps hardware motion disabled until explicit enable and exposes ordinary connection actions", () => {
    enter({ ...preview, hardware: true });
    pointer("left", "pointerdown", 200, 100);
    pointer("open", "pointerdown", 100, 100, 2);
    expect(onChange).not.toHaveBeenCalled();
    click("Подключить");
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "connect" });
    render({ ...preview, hardware: true, connected: true, status: "Пауза" });
    click("Включить");
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "enable" });
    render({ ...preview, hardware: true, connected: true, enabled: true, status: "Управление включено" });
    pointer("left", "pointerdown", 200, 100);
    expect(values()).toEqual([1, 0, 0, 0, 0]);
  });

  it("clears input on stop, blur and visibility loss and never resumes old touches", () => {
    enter();
    pointer("left", "pointerdown", 200, 100);
    click("Стоп");
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "stop" });
    const count = onChange.mock.calls.length;
    pointer("left", "pointermove", 200, 100);
    advance(100);
    expect(onChange).toHaveBeenCalledTimes(count);
    pointer("left", "pointerdown", 200, 100);
    act(() => window.dispatchEvent(new Event("blur")));
    expect(values()).toEqual([0, 0, 0, 0, 0]);
    pointer("left", "pointerdown", 200, 100);
    hidden = true;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expect(values()).toEqual([0, 0, 0, 0, 0]);
    hidden = false;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    const afterShow = onChange.mock.calls.length;
    advance(250);
    expect(onChange).toHaveBeenCalledTimes(afterShow);
  });

  it("does not send input while disconnected or malformed and removes timers on unmount", () => {
    enter();
    pointer("left", "pointerdown", 200, 100);
    connected = false;
    render();
    const count = onChange.mock.calls.length;
    pointer("left", "pointermove", 200, 100);
    advance(200);
    expect(onChange).toHaveBeenCalledTimes(count);
    connected = true;
    render({});
    pointer("left", "pointerdown", 200, 100);
    expect(onChange).toHaveBeenCalledTimes(count);
    render();
    pointer("left", "pointerdown", 200, 100);
    act(() => root.render(<></>));
    expect(values()).toEqual([0, 0, 0, 0, 0]);
    const finalCount = onChange.mock.calls.length;
    advance(500);
    expect(onChange).toHaveBeenCalledTimes(finalCount);
    expect([...captured.values()].every((ids) => ids.size === 0)).toBe(true);
  });

  it("opens automatically in native landscape without sending a mount command and resets on rotation", () => {
    vi.spyOn(navigator, "userAgent", "get").mockReturnValue("WKWebView SO101Control/1.0");
    landscape = true;
    render();
    expect(document.querySelector('[aria-label="Пульт управления SO-101"]')).not.toBeNull();
    expect(onChange).not.toHaveBeenCalled();
    pointer("left", "pointerdown", 200, 100);
    landscape = false;
    act(() => orientationChanged!());
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "layout" });
    expect(document.querySelector('[aria-label="Пульт управления SO-101"]')).toBeNull();
    landscape = true;
    act(() => orientationChanged!());
    const count = onChange.mock.calls.length;
    advance(500);
    expect(onChange).toHaveBeenCalledTimes(count);
  });

  it("opens a native modal to isolate background controls and cancels held input on Escape", () => {
    enter();
    const modal = document.querySelector<HTMLDialogElement>("dialog")!;
    expect(showModal).toHaveBeenCalledOnce();
    expect(modal.open).toBe(true);
    expect(modal.getAttribute("aria-modal")).toBe("true");
    pointer("left", "pointerdown", 200, 100);
    const cancelEvent = new Event("cancel", { cancelable: true });
    act(() => modal.dispatchEvent(cancelEvent));
    expect(cancelEvent.defaultPrevented).toBe(true);
    expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ action: "layout" });
    expect(onChange.mock.calls.map(([value]) => JSON.parse(value))).toContainEqual([0, 0, 0, 0, 0, 2, 0]);
    expect(closeModal).toHaveBeenCalledOnce();
    expect(document.querySelector("dialog")).toBeNull();
    const count = onChange.mock.calls.length;
    advance(200);
    expect(onChange).toHaveBeenCalledTimes(count);
  });
});
