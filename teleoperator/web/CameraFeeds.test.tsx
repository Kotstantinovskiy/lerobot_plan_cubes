// @vitest-environment jsdom
import * as React from "react";
import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerContext } from "../ViewerContext";
import CameraFeeds from "./CameraFeeds";

vi.mock("../ViewerContext", async () => {
  const react = await import("react");
  return { ViewerContext: react.createContext(null) };
});

function response(ok = true, type = "image/jpeg", body = "jpeg"): Response {
  return {
    ok, headers: new Headers({ "Content-Type": type }),
    blob: async () => new Blob([body], { type }),
  } as Response;
}

describe("independent camera feeds", () => {
  let host: HTMLDivElement;
  let root: Root;
  let connected: boolean;
  let hidden: boolean;
  let nextURL: number;
  const fetcher = vi.fn<typeof fetch>();
  const createObjectURL = vi.fn<(blob: Blob) => string>();
  const revokeObjectURL = vi.fn<(source: string) => void>();

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    vi.useFakeTimers();
    connected = true;
    hidden = false;
    nextURL = 0;
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    fetcher.mockReset().mockResolvedValue(response());
    createObjectURL.mockReset().mockImplementation(() => `blob:frame-${++nextURL}`);
    revokeObjectURL.mockReset();
    vi.stubGlobal("fetch", fetcher);
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = createObjectURL;
      static revokeObjectURL = revokeObjectURL;
    });
    window.history.replaceState(null, "", "/?token=pairing_secret&darkMode=&websocket=ws%3A%2F%2Fexample.local");
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  async function render() {
    const viewer = {
      useGui: (select: (state: { websocketState: string }) => unknown) => select({
        websocketState: connected ? "connected" : "disconnected",
      }),
    };
    await act(async () => root.render(<ViewerContext.Provider value={viewer as never}>
      <CameraFeeds/>
    </ViewerContext.Provider>));
  }

  async function advance(milliseconds: number) {
    await act(async () => vi.advanceTimersByTimeAsync(milliseconds));
  }

  async function visibility(value: boolean) {
    hidden = value;
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));
  }

  function feed(name: string): HTMLElement {
    return host.querySelector(`[aria-label="Камера: ${name}"]`)!;
  }

  it("requests both JPEGs from the same origin using only the pairing token", async () => {
    await render();
    expect(host.querySelectorAll("img")).toHaveLength(2);
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls.map(([url]) => new URL(String(url)).pathname))
      .toEqual(["/cameras/wrist/frame", "/cameras/top/frame"]);
    for (const [input, options] of fetcher.mock.calls) {
      const url = new URL(String(input));
      expect(url.origin).toBe(window.location.origin);
      expect([...url.searchParams.entries()]).toEqual([["token", "pairing_secret"]]);
      expect(options).toMatchObject({ cache: "no-store", credentials: "same-origin", redirect: "error" });
      expect(options!.signal).toBeInstanceOf(AbortSignal);
    }
    await advance(100);
    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:frame-1");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:frame-2");
  });

  it("supports local mode without a pairing token", async () => {
    window.history.replaceState(null, "", "/?darkMode=");
    await render();
    expect(fetcher.mock.calls.every(([url]) => new URL(String(url)).search === "")).toBe(true);
  });

  it("shows both missing cameras without interrupting or disabling the viewer", async () => {
    fetcher.mockResolvedValue(response(false));
    await render();
    expect(host.querySelectorAll("img")).toHaveLength(0);
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(2);
    expect(feed("Рука").textContent).toContain("Не подключена");
    expect(feed("Сверху").textContent).toContain("Не подключена");
    await advance(999);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await advance(1);
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it("clears a lost wrist image while the overhead video keeps updating and recovers automatically", async () => {
    await render();
    fetcher.mockImplementation(async (url) => response(!String(url).includes("/wrist/")));
    await advance(100);
    expect(feed("Рука").querySelector("img")).toBeNull();
    expect(feed("Рука").textContent).toContain("Не подключена");
    expect(feed("Сверху").querySelector("img")).not.toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:frame-1");
    await advance(900);
    expect(fetcher.mock.calls.filter(([url]) => String(url).includes("/wrist/"))).toHaveLength(2);
    fetcher.mockResolvedValue(response());
    await advance(100);
    expect(feed("Рука").querySelector("img")).not.toBeNull();
  });

  it("does not overlap requests when a camera is slow", async () => {
    const completions: ((value: Response) => void)[] = [];
    fetcher.mockImplementation(() => new Promise((resolve) => completions.push(resolve)));
    await render();
    expect(fetcher).toHaveBeenCalledTimes(2);
    await advance(500);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async () => completions[0](response()));
    expect(feed("Рука").querySelector("img")).not.toBeNull();
    await advance(99);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await advance(1);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it("times out hung requests, clears stale frames, and retries later", async () => {
    await render();
    fetcher.mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      options!.signal!.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    }));
    await advance(100);
    expect(host.querySelectorAll("img")).toHaveLength(2);
    await advance(2000);
    expect(host.querySelectorAll("img")).toHaveLength(0);
    expect(host.textContent).toContain("Не подключена");
    await advance(1000);
    expect(fetcher).toHaveBeenCalledTimes(6);
  });

  it("rejects non-JPEG, empty, and unauthorized responses", async () => {
    fetcher.mockResolvedValueOnce(response(true, "text/html")).mockResolvedValueOnce(response(true, "image/jpeg", ""));
    await render();
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(2);
    fetcher.mockResolvedValue(response(false));
    await advance(1000);
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it("hides a JPEG that the browser cannot decode", async () => {
    await render();
    const image = feed("Рука").querySelector("img")!;
    act(() => image.dispatchEvent(new Event("error")));
    expect(feed("Рука").textContent).toContain("Не подключена");
    expect(feed("Рука").querySelector("img")).toBeNull();
    expect(feed("Сверху").querySelector("img")).not.toBeNull();
    await advance(100);
    expect(feed("Рука").querySelector("img")).not.toBeNull();
  });

  it("releases images and stops polling while hidden, then starts fresh on return", async () => {
    await render();
    await visibility(true);
    expect(host.querySelectorAll("img")).toHaveLength(0);
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
    await advance(3000);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await visibility(false);
    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(host.querySelectorAll("img")).toHaveLength(2);
  });

  it("aborts pending requests and ignores their late results after hiding", async () => {
    const completions: ((value: Response) => void)[] = [];
    fetcher.mockImplementation(() => new Promise((resolve) => completions.push(resolve)));
    await render();
    await visibility(true);
    expect(fetcher.mock.calls.every(([, options]) => options!.signal!.aborted)).toBe(true);
    await act(async () => completions.forEach((resolve) => resolve(response())));
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(host.querySelectorAll("img")).toHaveLength(0);
    await advance(3000);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("stops on WebSocket disconnect and resumes only after reconnect", async () => {
    connected = false;
    await render();
    expect(fetcher).not.toHaveBeenCalled();
    connected = true;
    await render();
    expect(host.querySelectorAll("img")).toHaveLength(2);
    connected = false;
    await render();
    expect(host.querySelectorAll("img")).toHaveLength(0);
    await advance(3000);
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
  });

  it("aborts requests, releases images, and removes polling on unmount", async () => {
    await render();
    fetcher.mockImplementation(() => new Promise(() => {}));
    await advance(100);
    await act(async () => root.render(<></>));
    expect(fetcher.mock.calls.slice(2).every(([, options]) => options!.signal!.aborted)).toBe(true);
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
    await advance(4000);
    await visibility(false);
    expect(fetcher).toHaveBeenCalledTimes(4);
  });
});
