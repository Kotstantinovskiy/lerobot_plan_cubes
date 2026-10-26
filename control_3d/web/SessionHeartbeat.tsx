import * as React from "react";
import { ViewerContext } from "../ViewerContext";

type Props = { onChange: (value: string) => void };

/** Keep the operator lease alive only while this control screen is foreground. */
export default function SessionHeartbeat({ onChange }: Props) {
  const viewer = React.useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketState === "connected");
  const latest = React.useRef({ connected, onChange });
  latest.current = { connected, onChange };
  const sequence = React.useRef(0);
  const focused = React.useRef(true);

  React.useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    let active = false;

    const send = (foreground: boolean) => {
      latest.current.onChange(JSON.stringify([++sequence.current, foreground]));
    };
    const update = () => {
      const foreground = latest.current.connected && focused.current && !document.hidden;
      if (foreground) {
        send(true);
        active = true;
        if (timer === null) timer = setInterval(update, 100);
      } else {
        if (timer !== null) clearInterval(timer);
        timer = null;
        if (active) send(false);
        active = false;
      }
    };
    const onBlur = () => { focused.current = false; update(); };
    const onFocus = () => { focused.current = true; update(); };
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", update);
    update();
    return () => {
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", update);
      if (timer !== null) clearInterval(timer);
      if (active) send(false);
    };
  }, [connected]);

  return null;
}
