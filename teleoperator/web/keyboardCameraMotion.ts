type Orbit = [number, number];

/** Frame-independent camera orbit controlled by WASD. */
export class KeyboardCameraMotion {
  private velocity: Orbit = [0, 0];

  reset() {
    this.velocity = [0, 0];
  }

  step(held: ReadonlySet<string>, seconds: number): Orbit {
    if (!Number.isFinite(seconds) || seconds <= 0) {
      this.reset();
      return [0, 0];
    }
    const direction: Orbit = [
      Number(held.has("KeyD")) - Number(held.has("KeyA")),
      Number(held.has("KeyS")) - Number(held.has("KeyW")),
    ];
    const length = Math.hypot(...direction);
    if (length === 0) {
      this.reset();
      return [0, 0];
    }

    const speed = 50 * Math.PI / 180;
    const dt = Math.min(seconds, 0.05);
    const rampSeconds = 0.12;
    const decay = Math.exp(-dt / rampSeconds);
    const movement: Orbit = [0, 0];
    for (let index = 0; index < 2; index++) {
      const target = direction[index] / length * speed;
      // Releasing or reversing an axis must not leave it drifting.
      if (target === 0 || target * this.velocity[index] < 0) this.velocity[index] = 0;
      const previous = this.velocity[index];
      movement[index] = target * dt + (previous - target) * rampSeconds * (1 - decay);
      this.velocity[index] = target + (previous - target) * decay;
    }
    return movement;
  }
}
