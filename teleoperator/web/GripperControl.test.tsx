// @vitest-environment jsdom
import * as React from "react";
import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import GripperControl from "./GripperControl";

type Pair = [number, number];

describe("SO-101 gripper controls", () => {
  let host: HTMLDivElement;
  let root: Root;
  const onChange = vi.fn<(value: Pair) => void>();

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    onChange.mockClear();
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.restoreAllMocks();
  });

  function render(value: Pair = [0, 50], disabled = false) {
    act(() => root.render(<GripperControl value={value} min={[-157.2, 0]}
      max={[162.789, 100]} disabled={disabled} onChange={onChange}/>));
    return {
      dial: host.querySelector<HTMLDivElement>('[data-control="wrist-dial"]')!,
      fader: host.querySelector<HTMLDivElement>('[data-control="gripper-fader"]')!,
    };
  }

  function preparePointer(element: HTMLDivElement, left = 0, top = 0, width = 200, height = 200) {
    const captured = new Set<number>();
    element.getBoundingClientRect = () => ({
      left, top, right: left + width, bottom: top + height, width, height, x: left, y: top,
      toJSON: () => ({}),
    });
    element.setPointerCapture = (id) => { captured.add(id); };
    element.hasPointerCapture = (id) => captured.has(id);
    element.releasePointerCapture = (id) => { captured.delete(id); };
  }

  function pointer(element: HTMLDivElement, type: string, x: number, y: number, pointerId = 1) {
    const event = new MouseEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0 });
    Object.defineProperty(event, "pointerId", { value: pointerId });
    act(() => element.dispatchEvent(event));
  }

  function key(element: HTMLDivElement, value: string, shiftKey = false) {
    act(() => element.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: value, shiftKey })));
  }

  it("does not emit commands on mount or authoritative value updates", () => {
    render();
    const { dial, fader } = render([-30, 75]);
    expect(onChange).not.toHaveBeenCalled();
    expect(dial.getAttribute("aria-valuenow")).toBe("-30");
    expect(fader.getAttribute("aria-valuenow")).toBe("75");
  });

  it("does not jump when the dial is first grabbed", () => {
    const { dial } = render([37, 50]);
    preparePointer(dial);
    pointer(dial, "pointerdown", 200, 100);
    expect(onChange).not.toHaveBeenCalled();
    pointer(dial, "pointermove", 200, 100);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("rotates clockwise relative to the grab and preserves jaw opening", () => {
    const { dial } = render([20, 65]);
    preparePointer(dial);
    pointer(dial, "pointerdown", 100, 0);
    pointer(dial, "pointermove", 200, 100);
    expect(onChange).toHaveBeenLastCalledWith([110, 65]);
  });

  it("preserves a newer authoritative jaw value during an active wrist gesture", () => {
    const { dial } = render([0, 50]);
    preparePointer(dial);
    pointer(dial, "pointerdown", 100, 0);
    render([0, 75]);
    pointer(dial, "pointermove", 200, 100);
    expect(onChange).toHaveBeenLastCalledWith([90, 75]);
  });

  it("wraps across the bottom of the dial without a full-turn jump", () => {
    const { dial } = render();
    preparePointer(dial);
    const atAngle = (angle: number): [number, number] => [
      100 + 90 * Math.sin(angle * Math.PI / 180),
      100 - 90 * Math.cos(angle * Math.PI / 180),
    ];
    pointer(dial, "pointerdown", ...atAngle(179));
    pointer(dial, "pointermove", ...atAngle(-179));
    expect(onChange).toHaveBeenLastCalledWith([2, 50]);
  });

  it("clamps after rounding so a noninteger motor limit is never exceeded", () => {
    const { dial } = render([160, 50]);
    preparePointer(dial);
    pointer(dial, "pointerdown", 100, 0);
    pointer(dial, "pointermove", 200, 100);
    expect(onChange).toHaveBeenLastCalledWith([162.789, 50]);
  });

  it("supports precise and coarse keyboard rotation", () => {
    const { dial } = render([0, 55]);
    key(dial, "ArrowRight");
    expect(onChange).toHaveBeenLastCalledWith([1, 55]);
    key(dial, "ArrowLeft", true);
    expect(onChange).toHaveBeenLastCalledWith([-4, 55]);
    key(dial, "Home");
    expect(onChange).toHaveBeenLastCalledWith([0, 55]);
    key(dial, "End");
    expect(onChange).toHaveBeenLastCalledWith([162.789, 55]);
  });

  it("maps the top of the vertical control to open and the bottom to closed", () => {
    const { fader } = render([22, 50]);
    preparePointer(fader, 0, 100, 54, 125);
    pointer(fader, "pointerdown", 27, 100);
    expect(onChange).toHaveBeenLastCalledWith([22, 100]);
    pointer(fader, "pointermove", 27, 225);
    expect(onChange).toHaveBeenLastCalledWith([22, 0]);
    pointer(fader, "pointermove", 27, 800);
    expect(onChange).toHaveBeenCalledTimes(2);
  });

  it("supports keyboard opening and closing without changing wrist rotation", () => {
    const { fader } = render([12, 50]);
    key(fader, "ArrowUp");
    expect(onChange).toHaveBeenLastCalledWith([12, 51]);
    key(fader, "ArrowDown", true);
    expect(onChange).toHaveBeenLastCalledWith([12, 46]);
    key(fader, "Home");
    expect(onChange).toHaveBeenLastCalledWith([12, 0]);
    key(fader, "End");
    expect(onChange).toHaveBeenLastCalledWith([12, 100]);
  });

  it("ignores late pointer movement after cancellation", () => {
    const { dial } = render();
    preparePointer(dial);
    pointer(dial, "pointerdown", 100, 0);
    pointer(dial, "pointercancel", 100, 0);
    pointer(dial, "pointermove", 200, 100);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("stops a drag on blur and does not emit keyboard or pointer commands while disabled", () => {
    const { dial } = render();
    preparePointer(dial);
    pointer(dial, "pointerdown", 100, 0);
    act(() => window.dispatchEvent(new Event("blur")));
    pointer(dial, "pointermove", 200, 100);
    render([0, 50], true);
    key(dial, "ArrowRight");
    pointer(dial, "pointerdown", 100, 0);
    pointer(dial, "pointermove", 200, 100);
    expect(onChange).not.toHaveBeenCalled();
  });
});
