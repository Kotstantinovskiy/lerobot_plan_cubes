import * as React from "react";
import { ViewerContext } from "../ViewerContext";

type CameraName = "wrist" | "top";
type Frame = { source: string | null; connecting: boolean };

function frameURL(name: CameraName): string {
  const url = new URL(`/cameras/${name}/frame`, window.location.origin);
  const token = new URLSearchParams(window.location.search).get("token");
  if (token !== null) url.searchParams.set("token", token);
  return url.toString();
}

/** Poll each camera independently, keeping at most one network request in flight. */
function CameraFeed({ name, title, connected }: { name: CameraName; title: string; connected: boolean }) {
  const [frame, setFrame] = React.useState<Frame>({ source: null, connecting: true });
  const [failedSource, setFailedSource] = React.useState<string | null>(null);

  React.useEffect(() => {
    let disposed = false;
    let generation = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let requestTimeout: ReturnType<typeof setTimeout> | null = null;
    let request: AbortController | null = null;
    let source: string | null = null;

    const active = () => connected && !document.hidden && !disposed;
    const clearFrame = (connecting = false) => {
      if (source !== null) URL.revokeObjectURL(source);
      source = null;
      if (!disposed) setFrame({ source: null, connecting });
    };

    const poll = async (currentGeneration: number) => {
      if (!active() || currentGeneration !== generation) return;
      const controller = new AbortController();
      request = controller;
      const deadline = setTimeout(() => controller.abort(), 2000);
      requestTimeout = deadline;
      let delay = 1000;
      try {
        const response = await fetch(frameURL(name), {
          signal: controller.signal, cache: "no-store", credentials: "same-origin", redirect: "error",
        });
        if (!response.ok || !response.headers.get("Content-Type")?.startsWith("image/jpeg")) {
          throw new Error("Camera unavailable");
        }
        const blob = await response.blob();
        if (!blob.size) throw new Error("Empty camera frame");
        if (controller.signal.aborted) throw new Error("Camera timed out");
        if (!active() || currentGeneration !== generation) return;
        const nextSource = URL.createObjectURL(blob);
        if (source !== null) URL.revokeObjectURL(source);
        source = nextSource;
        setFrame({ source, connecting: false });
        delay = 100;
      } catch {
        if (active() && currentGeneration === generation) clearFrame();
      } finally {
        clearTimeout(deadline);
        if (request === controller) {
          request = null;
          requestTimeout = null;
        }
        if (active() && currentGeneration === generation) {
          timer = setTimeout(() => { void poll(currentGeneration); }, delay);
        }
      }
    };

    const update = () => {
      generation += 1;
      if (timer !== null) clearTimeout(timer);
      timer = null;
      request?.abort();
      request = null;
      if (requestTimeout !== null) clearTimeout(requestTimeout);
      requestTimeout = null;
      clearFrame(active());
      if (active()) void poll(generation);
    };

    document.addEventListener("visibilitychange", update);
    update();
    return () => {
      disposed = true;
      generation += 1;
      document.removeEventListener("visibilitychange", update);
      if (timer !== null) clearTimeout(timer);
      request?.abort();
      if (requestTimeout !== null) clearTimeout(requestTimeout);
      clearFrame();
    };
  }, [name, connected]);

  const visibleSource = frame.source !== failedSource ? frame.source : null;
  return <section className="camera-feed" aria-label={`Камера: ${title}`}>
    <h2>{title}</h2>
    <div className="camera-feed__frame">
      {visibleSource !== null
        ? <img src={visibleSource} alt={`Камера: ${title}`} draggable={false}
            onError={() => setFailedSource(visibleSource)}/>
        : <span className="camera-feed__placeholder" role="status">
            {frame.connecting ? "Подключение…" : "Не подключена"}
          </span>}
    </div>
  </section>;
}

export default function CameraFeeds() {
  const viewer = React.useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketState === "connected");
  return <div className="camera-feeds">
    <CameraFeed name="wrist" title="Рука" connected={connected}/>
    <CameraFeed name="top" title="Сверху" connected={connected}/>
  </div>;
}
