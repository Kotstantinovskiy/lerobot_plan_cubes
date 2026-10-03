import * as React from "react";
import "./control-room.css";

type Pair = [number, number];

type Props = {
  value: Pair;
  min: Pair | null;
  max: Pair | null;
  disabled: boolean;
  onChange: (value: Pair) => void;
};

type DialDrag = {
  pointerId: number;
  angle: number;
  element: HTMLDivElement;
};

function clamp(value: number, lower: number, upper: number) {
  return Math.min(upper, Math.max(lower, value));
}

function polar(angle: number, radius: number) {
  const radians = (angle * Math.PI) / 180;
  return [100 + Math.sin(radians) * radius, 100 - Math.cos(radians) * radius];
}

function arc(lower: number, upper: number, radius: number) {
  const start = polar(lower, radius);
  const end = polar(upper, radius);
  return `M ${start[0]} ${start[1]} A ${radius} ${radius} 0 ${upper - lower > 180 ? 1 : 0} 1 ${end[0]} ${end[1]}`;
}

function pointerAngle(event: React.PointerEvent<HTMLDivElement>) {
  const bounds = event.currentTarget.getBoundingClientRect();
  return (Math.atan2(event.clientX - bounds.left - bounds.width / 2,
    bounds.top + bounds.height / 2 - event.clientY) * 180) / Math.PI;
}

export default function GripperControl({ value, min, max, disabled, onChange }: Props) {
  const lower: Pair = min ?? [-157, 0];
  const upper: Pair = max ?? [163, 100];
  const latest = React.useRef<Pair>(value);
  latest.current = value;
  const dialDrag = React.useRef<DialDrag | null>(null);
  const faderDrag = React.useRef<{ pointerId: number; element: HTMLDivElement } | null>(null);
  const [active, setActive] = React.useState<"dial" | "fader" | null>(null);

  const cancel = React.useCallback(() => {
    for (const gesture of [dialDrag.current, faderDrag.current]) {
      if (gesture?.element.hasPointerCapture(gesture.pointerId)) {
        gesture.element.releasePointerCapture(gesture.pointerId);
      }
    }
    dialDrag.current = null;
    faderDrag.current = null;
    setActive(null);
  }, []);

  React.useEffect(() => {
    window.addEventListener("blur", cancel);
    const onVisibility = () => { if (document.hidden) cancel(); };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("blur", cancel);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [cancel]);

  React.useEffect(() => { if (disabled) cancel(); }, [disabled, cancel]);

  function emit(index: 0 | 1, next: number) {
    if (disabled || !Number.isFinite(next)) return;
    const bounded = clamp(Math.round(next * 10) / 10, lower[index], upper[index]);
    const pair: Pair = [...latest.current];
    if (pair[index] === bounded) return;
    pair[index] = bounded;
    latest.current = pair;
    onChange(pair);
  }

  function onDialDown(event: React.PointerEvent<HTMLDivElement>) {
    if (disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    dialDrag.current = { pointerId: event.pointerId, angle: pointerAngle(event), element: event.currentTarget };
    setActive("dial");
  }

  function onDialMove(event: React.PointerEvent<HTMLDivElement>) {
    const gesture = dialDrag.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    const angle = pointerAngle(event);
    // Incremental wrapped motion avoids a jump when the dial is grabbed.
    const delta = ((angle - gesture.angle + 540) % 360) - 180;
    gesture.angle = angle;
    emit(0, latest.current[0] + delta);
  }

  function moveFader(event: React.PointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const fraction = clamp((bounds.bottom - event.clientY) / bounds.height, 0, 1);
    emit(1, lower[1] + fraction * (upper[1] - lower[1]));
  }

  function onFaderDown(event: React.PointerEvent<HTMLDivElement>) {
    if (disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    faderDrag.current = { pointerId: event.pointerId, element: event.currentTarget };
    setActive("fader");
    moveFader(event);
  }

  function onKey(event: React.KeyboardEvent<HTMLDivElement>, index: 0 | 1) {
    const step = event.shiftKey ? 5 : 1;
    let next: number;
    if (event.key === "ArrowUp" || event.key === "ArrowRight") next = latest.current[index] + step;
    else if (event.key === "ArrowDown" || event.key === "ArrowLeft") next = latest.current[index] - step;
    else if (event.key === "Home") next = index === 0 ? clamp(0, lower[0], upper[0]) : lower[1];
    else if (event.key === "End") next = upper[index];
    else return;
    event.preventDefault();
    event.stopPropagation();
    emit(index, next);
  }

  const rotation = clamp(value[0], lower[0], upper[0]);
  const opening = clamp(value[1], lower[1], upper[1]);
  const fraction = (opening - lower[1]) / (upper[1] - lower[1]);
  const marker = polar(rotation, 84);

  return (
    <section className={`gripper-control ${disabled ? "is-disabled" : ""}`} aria-label="Управление захватом">
      <div className="gripper-control__instruments">
        <div className="dial-control">
          <div className={`wrist-dial ${active === "dial" ? "is-active" : ""}`}
            data-control="wrist-dial" role="slider" tabIndex={disabled ? -1 : 0}
            aria-label="Поворот захвата" aria-valuemin={lower[0]} aria-valuemax={upper[0]}
            aria-valuenow={rotation} aria-valuetext={`${Math.round(rotation)} градусов`} aria-disabled={disabled}
            onPointerDown={onDialDown} onPointerMove={onDialMove} onPointerUp={cancel}
            onPointerCancel={cancel} onLostPointerCapture={() => { dialDrag.current = null; setActive(null); }}
            onKeyDown={(event) => onKey(event, 0)}>
            <svg viewBox="0 0 200 200" className="wrist-dial__face" aria-hidden="true">
              <path className="wrist-dial__track" d={arc(lower[0], upper[0], 84)}/>
              <path className="wrist-dial__progress" d={arc(lower[0], Math.max(lower[0] + 0.01, rotation), 84)}/>
              <circle cx={marker[0]} cy={marker[1]} r="5" className="wrist-dial__marker"/>
            </svg>
            <div className="wrist-dial__readout"><strong>{Math.round(rotation)}<small>°</small></strong></div>
          </div>
          <div className="instrument-label">Поворот</div>
        </div>
        <div className="opening-control">
          <div className={`gripper-fader ${active === "fader" ? "is-active" : ""}`}
            data-control="gripper-fader" role="slider" tabIndex={disabled ? -1 : 0}
            aria-label="Открытие захвата" aria-orientation="vertical" aria-valuemin={lower[1]} aria-valuemax={upper[1]}
            aria-valuenow={opening} aria-valuetext={`${Math.round(opening)} процентов`} aria-disabled={disabled}
            onPointerDown={onFaderDown} onPointerMove={(event) => { if (faderDrag.current?.pointerId === event.pointerId) moveFader(event); }}
            onPointerUp={cancel} onPointerCancel={cancel} onLostPointerCapture={() => { faderDrag.current = null; setActive(null); }}
            onKeyDown={(event) => onKey(event, 1)}>
            <div className="gripper-fader__rail"><div className="gripper-fader__fill" style={{height: `${fraction * 100}%`}}/><div className="gripper-fader__thumb" style={{bottom: `${fraction * 100}%`}}/></div>
          </div>
          <div className="opening-control__readout">{Math.round(opening)}<small>%</small></div>
          <div className="instrument-label">Захват</div>
        </div>
      </div>
    </section>
  );
}
