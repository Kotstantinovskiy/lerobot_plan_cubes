import * as React from "react";
import * as THREE from "three";
import { ViewerContext } from "../ViewerContext";
import { computeT_threeworld_world } from "../WorldTransformUtils";
import { useRemoteMode } from "./remoteMode";
import { nextSequence } from "./sequence";

type Direction = [number, number, number];
type Props = { disabled: boolean; resetKey: string; onChange: (value: string) => void };
const ARM_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyK", "KeyM"]);

/** Project screen-relative motion onto the table, keeping vertical motion separate. */
export function keyboardDirection(
  held: ReadonlySet<string>, camera: THREE.Camera | null, threeFromWorld: THREE.Matrix4,
): Direction {
  const horizontal = Number(held.has("ArrowRight")) - Number(held.has("ArrowLeft"));
  const forward = Number(held.has("ArrowUp")) - Number(held.has("ArrowDown"));
  const vertical = Number(held.has("KeyK")) - Number(held.has("KeyM"));
  if (camera === null) return [0, 0, 0];
  const worldFromThree = threeFromWorld.clone().invert();
  const ahead = camera.getWorldDirection(new THREE.Vector3()).transformDirection(worldFromThree);
  ahead.z = 0;
  if (ahead.lengthSq() < 1e-8) {
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).transformDirection(worldFromThree);
    right.z = 0;
    if (right.lengthSq() < 1e-8) return [0, 0, vertical];
    right.normalize();
    ahead.set(-right.y, right.x, 0);
  } else {
    ahead.normalize();
  }
  const direction = new THREE.Vector3(
    horizontal * ahead.y + forward * ahead.x,
    -horizontal * ahead.x + forward * ahead.y,
    vertical,
  );
  if (direction.lengthSq() > 1) direction.normalize();
  return [direction.x, direction.y, direction.z];
}

function editingControl(target: EventTarget | null) {
  return target instanceof Element && target.closest(
    'input, textarea, select, button, [role="slider"], [contenteditable]:not([contenteditable="false"])',
  ) !== null;
}

/** Invisible input transport: the server owns speed, reachability, and stopping. */
export default function KeyboardControl({ disabled, resetKey, onChange }: Props) {
  const viewer = React.useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketState === "connected");
  const remoteActive = useRemoteMode();
  const blocked = disabled || remoteActive;
  const latest = React.useRef({ disabled: blocked, connected, onChange, epoch: Number(resetKey) });
  latest.current = { disabled: blocked, connected, onChange, epoch: Number(resetKey) };
  const held = React.useRef(new Set<string>());
  const timer = React.useRef<ReturnType<typeof setInterval> | null>(null);

  const send = React.useCallback((direction: Direction) => {
    const state = latest.current;
    if (!Number.isSafeInteger(state.epoch) || state.epoch < 0) return;
    state.onChange(JSON.stringify([...direction, nextSequence(), state.epoch]));
  }, []);

  const cancel = React.useCallback(() => {
    const active = held.current.size > 0;
    held.current.clear();
    if (timer.current !== null) clearInterval(timer.current);
    timer.current = null;
    if (active) send([0, 0, 0]);
  }, [send]);

  React.useEffect(() => {
    // A lease change requires a fresh physical press, never an auto-repeat.
    cancel();
  }, [blocked, connected, resetKey, cancel]);

  React.useEffect(() => {
    const emit = () => {
      if (latest.current.disabled || !latest.current.connected || document.hidden) {
        cancel();
        return;
      }
      if (viewer.mutable.current.camera === null || viewer.useSceneTree.get("") === undefined) {
        cancel();
        return;
      }
      send(keyboardDirection(
        held.current, viewer.mutable.current.camera, computeT_threeworld_world(viewer),
      ));
    };
    const onDown = (event: KeyboardEvent) => {
      if (!ARM_KEYS.has(event.code) || editingControl(event.target)) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (latest.current.disabled || !latest.current.connected || document.hidden) return;
      if (viewer.mutable.current.camera === null || viewer.useSceneTree.get("") === undefined) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.repeat || held.current.has(event.code)) return;
      held.current.add(event.code);
      emit();
      if (held.current.size > 0 && timer.current === null) timer.current = setInterval(emit, 50);
    };
    const onUp = (event: KeyboardEvent) => {
      if (!held.current.delete(event.code)) return;
      event.preventDefault();
      event.stopPropagation();
      if (held.current.size === 0) {
        if (timer.current !== null) clearInterval(timer.current);
        timer.current = null;
        send([0, 0, 0]);
      } else {
        emit();
      }
    };
    const onVisibility = () => { if (document.hidden) cancel(); };
    const onFocus = (event: FocusEvent) => { if (editingControl(event.target)) cancel(); };
    document.addEventListener("keydown", onDown);
    document.addEventListener("keyup", onUp);
    document.addEventListener("focusin", onFocus);
    window.addEventListener("blur", cancel);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("keydown", onDown);
      document.removeEventListener("keyup", onUp);
      document.removeEventListener("focusin", onFocus);
      window.removeEventListener("blur", cancel);
      document.removeEventListener("visibilitychange", onVisibility);
      cancel();
    };
  }, [viewer, send, cancel]);

  return null;
}
