// @vitest-environment jsdom
import * as React from "react";
import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerContext } from "../ViewerContext";
import SessionHeartbeat from "./SessionHeartbeat";

vi.mock("../ViewerContext", async () => {
  const react = await import("react");
  return { ViewerContext: react.createContext(null) };
});

describe("foreground operator heartbeat", () => {
  let host: HTMLDivElement;
  let root: Root;
  let connected: boolean;
  let hidden: boolean;
  const onChange = vi.fn<(value: string) => void>();

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    vi.useFakeTimers();
    connected = true;
    hidden = false;
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    onChange.mockClear();
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  function render(callback = onChange) {
    const viewer = {
      useGui: (select: (state: { websocketState: string }) => unknown) => select({
        websocketState: connected ? "connected" : "disconnected",
      }),
    };
    act(() => root.render(<ViewerContext.Provider value={viewer as never}>
      <SessionHeartbeat onChange={callback}/>
    </ViewerContext.Provider>));
  }

  function lastPayload(): [number, boolean] { return JSON.parse(onChange.mock.lastCall![0]); }
  function event(target: EventTarget, type: string) { act(() => target.dispatchEvent(new Event(type))); }

  it("sends immediately and keeps a foreground lease with unique sequence numbers", () => {
    render();
    expect(lastPayload()).toEqual([1, true]);
    expect(host.textContent).toBe("");
    act(() => vi.advanceTimersByTime(300));
    expect(onChange).toHaveBeenCalledTimes(4);
    expect(lastPayload()).toEqual([4, true]);
  });

  it("sends inactive on blur and resumes only after explicit focus", () => {
    render();
    event(window, "blur");
    expect(lastPayload()).toEqual([2, false]);
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(2);
    event(window, "focus");
    expect(lastPayload()).toEqual([3, true]);
    act(() => vi.advanceTimersByTime(100));
    expect(lastPayload()).toEqual([4, true]);
  });

  it("stops while hidden and starts again when the page becomes visible", () => {
    render();
    hidden = true;
    event(document, "visibilitychange");
    expect(lastPayload()).toEqual([2, false]);
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(2);
    hidden = false;
    event(document, "visibilitychange");
    expect(lastPayload()).toEqual([3, true]);
  });

  it("does not start a lease before connecting and stops on disconnect", () => {
    connected = false;
    render();
    expect(onChange).not.toHaveBeenCalled();
    connected = true;
    render();
    expect(lastPayload()).toEqual([1, true]);
    connected = false;
    render();
    expect(lastPayload()).toEqual([2, false]);
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(2);
  });

  it("does not treat reconnecting in a blurred window as foreground", () => {
    render();
    event(window, "blur");
    connected = false;
    render();
    connected = true;
    render();
    act(() => vi.advanceTimersByTime(500));
    expect(onChange).toHaveBeenCalledTimes(2);
    event(window, "focus");
    expect(lastPayload()).toEqual([3, true]);
  });

  it("uses the latest callback without restarting the heartbeat stream", () => {
    render();
    const replacement = vi.fn<(value: string) => void>();
    render(replacement);
    expect(replacement).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(100));
    expect(JSON.parse(replacement.mock.lastCall![0])).toEqual([2, true]);
  });

  it("sends inactive on unmount and removes timers and event listeners", () => {
    render();
    act(() => root.render(<></>));
    expect(lastPayload()).toEqual([2, false]);
    act(() => vi.advanceTimersByTime(500));
    event(window, "focus");
    event(document, "visibilitychange");
    expect(onChange).toHaveBeenCalledTimes(2);
  });
});
