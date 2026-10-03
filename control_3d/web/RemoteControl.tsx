import * as React from "react";
import { createPortal } from "react-dom";
import { ViewerContext } from "../ViewerContext";
import CameraFeeds from "./CameraFeeds";
import ThumbStick, { StickValue } from "./ThumbStick";
import { setRemoteMode, useRemoteMode } from "./remoteMode";
import { nextSequence } from "./sequence";

type RemoteState = {
  epoch: number;
  hardware: boolean;
  connected: boolean;
  enabled: boolean;
  faulted: boolean;
  status: string;
};
type Props = { hint: string; onChange: (value: string) => void };
type Role = "left" | "right" | "gripper";

function readState(hint: string): RemoteState | null {
  try {
    const state = JSON.parse(hint) as RemoteState;
    if (!Number.isSafeInteger(state.epoch) || state.epoch < 0) return null;
    if ([state.hardware, state.connected, state.enabled, state.faulted].some((value) => typeof value !== "boolean")) return null;
    if (typeof state.status !== "string") return null;
    return state;
  } catch { return null; }
}

/** The Mac owns motion limits; this screen sends only leased thumb directions. */
export default function RemoteControl({ hint, onChange }: Props) {
  const viewer = React.useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketState === "connected");
  const active = useRemoteMode();
  const state = readState(hint);
  const allowed = connected && state !== null && !state.faulted && (!state.hardware || state.connected && state.enabled);
  const latest = React.useRef({ state, connected, active, allowed, onChange });
  latest.current = { state, connected, active, allowed, onChange };
  const pointers = React.useRef(new Map<number, Role>());
  const left = React.useRef<StickValue>([0, 0]);
  const right = React.useRef<StickValue>([0, 0]);
  const grip = React.useRef(0);
  const gripCapture = React.useRef<{ element: HTMLButtonElement; id: number } | null>(null);
  const timer = React.useRef<ReturnType<typeof setInterval> | null>(null);
  const dialog = React.useRef<HTMLDialogElement>(null);
  const [resetKey, setResetKey] = React.useState(0);
  const [gripping, setGripping] = React.useState(0);

  const send = React.useCallback((neutral = false) => {
    const current = latest.current;
    if (!current.connected || current.state === null) return;
    let [x, y, z] = neutral ? [0, 0, 0] : [left.current[0], left.current[1], right.current[1]];
    const length = Math.hypot(x, y, z);
    if (length > 1) { x /= length; y /= length; z /= length; }
    current.onChange(JSON.stringify([
      x, y, z, neutral ? 0 : right.current[0], neutral ? 0 : grip.current,
      nextSequence(), current.state.epoch,
    ]));
  }, []);

  const cancel = React.useCallback(() => {
    const held = pointers.current.size > 0;
    pointers.current.clear();
    const capture = gripCapture.current;
    gripCapture.current = null;
    if (capture?.element.hasPointerCapture(capture.id)) capture.element.releasePointerCapture(capture.id);
    left.current = [0, 0];
    right.current = [0, 0];
    grip.current = 0;
    if (timer.current !== null) clearInterval(timer.current);
    timer.current = null;
    setGripping(0);
    setResetKey((value) => value + 1);
    if (held) send(true);
  }, [send]);

  const action = React.useCallback((name: "connect" | "enable" | "stop" | "layout") => {
    cancel();
    // The number keeps a repeated press distinct: Viser ignores an unchanged value.
    if (latest.current.connected) latest.current.onChange(JSON.stringify({ action: name, sequence: nextSequence() }));
  }, [cancel]);

  const switchLayout = React.useCallback((next: boolean, initial = false) => {
    if (next === latest.current.active) return;
    if (!initial) action("layout");
    else cancel();
    setRemoteMode(next);
  }, [action, cancel]);

  React.useEffect(() => {
    cancel();
  }, [allowed, connected, state?.epoch, active, cancel]);

  React.useLayoutEffect(() => {
    if (!active || dialog.current === null) return;
    const modal = dialog.current;
    modal.showModal();
    return () => modal.close();
  }, [active]);

  React.useEffect(() => {
    const blur = () => cancel();
    const visibility = () => { if (document.hidden) cancel(); };
    window.addEventListener("blur", blur);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("blur", blur);
      document.removeEventListener("visibilitychange", visibility);
      cancel();
      setRemoteMode(false);
    };
  }, [cancel]);

  React.useEffect(() => {
    if (!navigator.userAgent.includes("SO101Control/1.0")) return;
    const landscape = window.matchMedia("(orientation: landscape) and (min-width: 560px)");
    switchLayout(landscape.matches, true);
    const changed = () => switchLayout(landscape.matches);
    landscape.addEventListener("change", changed);
    return () => landscape.removeEventListener("change", changed);
  }, [switchLayout]);

  const claim = (id: number, role: Role) => {
    if (!latest.current.allowed || !latest.current.active || document.hidden) return false;
    if (pointers.current.size >= 2 || pointers.current.has(id) || [...pointers.current.values()].includes(role)) return false;
    pointers.current.set(id, role);
    return true;
  };
  const release = (id: number) => { pointers.current.delete(id); };
  const emit = (immediate = false) => {
    if (!latest.current.allowed || !latest.current.active || document.hidden) { cancel(); return; }
    if (immediate || timer.current === null || pointers.current.size === 0) send();
    if (pointers.current.size === 0) {
      if (timer.current !== null) clearInterval(timer.current);
      timer.current = null;
    } else if (timer.current === null) {
      timer.current = setInterval(() => {
        if (!latest.current.allowed || !latest.current.active || document.hidden) cancel();
        else send();
      }, 50);
    }
  };

  const gripButton = (label: string, direction: number) => <button type="button"
    className={`remote-grip${gripping === direction ? " is-active" : ""}`}
    disabled={!allowed} data-control={direction > 0 ? "remote-open" : "remote-close"}
    onContextMenu={(event) => event.preventDefault()}
    onPointerDown={(event) => {
      if (event.button !== 0 || !claim(event.pointerId, "gripper")) return;
      try { event.currentTarget.setPointerCapture(event.pointerId); }
      catch { release(event.pointerId); return; }
      event.preventDefault();
      gripCapture.current = { element: event.currentTarget, id: event.pointerId };
      grip.current = direction;
      setGripping(direction);
      emit();
    }}
    onPointerUp={(event) => {
      if (pointers.current.get(event.pointerId) !== "gripper") return;
      release(event.pointerId);
      gripCapture.current = null;
      grip.current = 0;
      setGripping(0);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      emit(true);
    }}
    onPointerCancel={(event) => { if (pointers.current.has(event.pointerId)) cancel(); }}
    onLostPointerCapture={(event) => { if (pointers.current.has(event.pointerId)) cancel(); }}>
    {label}
  </button>;

  if (!active) return <>
    <button type="button" className="remote-entry" onClick={() => switchLayout(true)}>Пульт</button>
    <CameraFeeds/>
  </>;

  return createPortal(<dialog ref={dialog} className="remote-control" aria-label="Пульт управления SO-101"
    aria-modal="true" onCancel={(event) => { event.preventDefault(); switchLayout(false); }}>
    <header className="remote-control__header">
      <button type="button" className="remote-button remote-back" onClick={() => switchLayout(false)}>3D</button>
      <div className="remote-control__status"><strong>SO—101</strong>
        <span role="status">{connected ? state?.status ?? "Пульт недоступен" : "Нет связи"}</span></div>
      <div className="remote-control__actions">
        {state?.hardware && !state.connected && <button type="button" className="remote-button"
          disabled={!connected} onClick={() => action("connect")}>Подключить</button>}
        {state?.hardware && state.connected && !state.enabled && <button type="button" className="remote-button remote-enable"
          disabled={!connected} onClick={() => action("enable")}>Включить</button>}
        <button type="button" className="remote-button remote-stop" onClick={() => action("stop")}>Стоп</button>
      </div>
    </header>
    <div className="remote-control__body">
      <div className="remote-control__left">
        <ThumbStick label="По столу" name="remote-left" disabled={!allowed} resetKey={resetKey}
          claim={(id) => claim(id, "left")} release={release} onCancel={cancel}
          onChange={(value) => { left.current = value; emit(value[0] === 0 && value[1] === 0); }}/>
      </div>
      <div className="remote-control__cameras"><CameraFeeds/></div>
      <div className="remote-control__right">
        <ThumbStick label="Высота · поворот" name="remote-right" disabled={!allowed} resetKey={resetKey}
          claim={(id) => claim(id, "right")} release={release} onCancel={cancel}
          onChange={(value) => { right.current = value; emit(value[0] === 0 && value[1] === 0); }}/>
        <div className="remote-control__gripper">{gripButton("Открыть", 1)}{gripButton("Закрыть", -1)}</div>
      </div>
    </div>
  </dialog>, document.body);
}
