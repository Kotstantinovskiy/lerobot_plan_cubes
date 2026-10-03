import * as React from "react";

let active = false;
const listeners = new Set<() => void>();

export function setRemoteMode(next: boolean) {
  if (active === next) return;
  active = next;
  listeners.forEach((listener) => listener());
}

export function useRemoteMode(): boolean {
  return React.useSyncExternalStore((listener) => {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  }, () => active, () => false);
}
