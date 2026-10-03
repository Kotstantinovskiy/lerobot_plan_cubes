let lastSequence = 0;

/**
 * Number one message for the server. Viser sends only changed values, so each message needs a new
 * number. It lives outside the components because rotating the phone remounts the control panel,
 * and the server ignores every number it has already passed.
 */
export function nextSequence(): number {
  return ++lastSequence;
}

export function resetSequenceForTests() {
  lastSequence = 0;
}
