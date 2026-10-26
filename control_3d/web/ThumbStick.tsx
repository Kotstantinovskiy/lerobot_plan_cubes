import * as React from "react";

export type StickValue = [number, number];
type Props = {
  label: string;
  name: string;
  disabled: boolean;
  resetKey: number;
  claim: (pointerId: number) => boolean;
  release: (pointerId: number) => void;
  onChange: (value: StickValue) => void;
  onCancel: () => void;
};

/** A radial dead zone gives the thumb a quiet center and a gradual speed ramp. */
export function stickValue(x: number, y: number): StickValue {
  const length = Math.hypot(x, y);
  if (!Number.isFinite(length) || length <= 0.12) return [0, 0];
  const speed = Math.min(1, (length - 0.12) / 0.88);
  return [x / length * speed, y / length * speed];
}

export default function ThumbStick({ label, name, disabled, resetKey, claim, release, onChange, onCancel }: Props) {
  const element = React.useRef<HTMLDivElement>(null);
  const pointer = React.useRef<number | null>(null);
  const [value, setValue] = React.useState<StickValue>([0, 0]);

  const reset = React.useCallback(() => {
    const id = pointer.current;
    pointer.current = null;
    if (id !== null && element.current?.hasPointerCapture(id)) {
      element.current.releasePointerCapture(id);
    }
    setValue([0, 0]);
  }, []);

  React.useEffect(reset, [disabled, resetKey, reset]);
  React.useEffect(() => {
    const node = element.current;
    return () => {
      const id = pointer.current;
      pointer.current = null;
      if (id !== null && node?.hasPointerCapture(id)) node.releasePointerCapture(id);
    };
  }, []);

  const read = (event: React.PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const radius = Math.min(box.width, box.height) / 2;
    if (radius <= 0) return { distance: Infinity, value: [0, 0] as StickValue };
    const x = (event.clientX - box.left - box.width / 2) / radius;
    const y = -(event.clientY - box.top - box.height / 2) / radius;
    return { distance: Math.hypot(x, y), value: stickValue(x, y) };
  };

  const change = (next: StickValue) => { setValue(next); onChange(next); };
  const finish = (event: React.PointerEvent<HTMLDivElement>, cancelled = false) => {
    if (pointer.current !== event.pointerId) return;
    const id = pointer.current;
    reset();
    if (cancelled) onCancel();
    else { release(id); onChange([0, 0]); }
  };

  return <div className="remote-stick-group">
    <div ref={element} className={`remote-stick${disabled ? " is-disabled" : ""}`}
      role="group" aria-label={label} aria-disabled={disabled} data-control={name}
      onContextMenu={(event) => event.preventDefault()}
      onPointerDown={(event) => {
        if (disabled || pointer.current !== null || event.button !== 0) return;
        const next = read(event);
        if (next.distance > 1 || !claim(event.pointerId)) return;
        try { event.currentTarget.setPointerCapture(event.pointerId); }
        catch { release(event.pointerId); return; }
        event.preventDefault();
        pointer.current = event.pointerId;
        change(next.value);
      }}
      onPointerMove={(event) => {
        if (pointer.current !== event.pointerId || disabled) return;
        event.preventDefault();
        change(read(event).value);
      }}
      onPointerUp={(event) => finish(event)}
      onPointerCancel={(event) => finish(event, true)}
      onLostPointerCapture={(event) => finish(event, true)}>
      <span className="remote-stick__cross" aria-hidden="true"/>
      <span className="remote-stick__thumb" aria-hidden="true"
        style={{ left: `${50 + value[0] * 30}%`, top: `${50 - value[1] * 30}%` }}/>
    </div>
    <div className="remote-stick__label">{label}</div>
  </div>;
}
