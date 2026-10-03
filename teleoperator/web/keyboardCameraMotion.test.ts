import { describe, expect, it } from "vitest";
import { KeyboardCameraMotion } from "./keyboardCameraMotion";

const keys = (...values: string[]) => new Set(values);

function orbit(fps: number, held = keys("KeyW")) {
  const motion = new KeyboardCameraMotion();
  const angles = [0, 0];
  for (let frame = 0; frame < fps; frame++) {
    motion.step(held, 1 / fps).forEach((value, axis) => { angles[axis] += value; });
  }
  return angles;
}

describe("keyboard camera orbit", () => {
  it("maps WASD to camera orbit without claiming arm-control keys", () => {
    expect(orbit(60, keys("KeyW"))[1]).toBeLessThan(0);
    expect(orbit(60, keys("KeyS"))[1]).toBeGreaterThan(0);
    expect(orbit(60, keys("KeyA"))[0]).toBeLessThan(0);
    expect(orbit(60, keys("KeyD"))[0]).toBeGreaterThan(0);
    const motion = new KeyboardCameraMotion();
    expect(motion.step(keys("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyK", "KeyM", "KeyQ", "KeyE"), 1 / 60)).toEqual([0, 0]);
    expect(motion.step(keys("KeyW"), 1 / 60)).toHaveLength(2);
  });

  it("starts gently and bounds total angular speed to fifty degrees per second", () => {
    const motion = new KeyboardCameraMotion();
    const speed = 50 * Math.PI / 180;
    const first = Math.abs(motion.step(keys("KeyW"), 1 / 60)[1]);
    const second = Math.abs(motion.step(keys("KeyW"), 1 / 60)[1]);
    expect(first).toBeGreaterThan(0);
    expect(first).toBeLessThan(speed / 60 * 0.1);
    expect(second).toBeGreaterThan(first);
    expect(second).toBeLessThan(speed / 60);
    expect(Math.hypot(...orbit(60))).toBeLessThan(speed);
  });

  it("covers the same angle at thirty, sixty and 144 frames per second", () => {
    expect(orbit(30)[1]).toBeCloseTo(orbit(60)[1], 12);
    expect(orbit(60)[1]).toBeCloseTo(orbit(144)[1], 12);
  });

  it("normalizes diagonal rotation instead of making it faster", () => {
    const diagonal = orbit(60, keys("KeyW", "KeyD"));
    expect(Math.hypot(...diagonal)).toBeCloseTo(Math.hypot(...orbit(60)), 12);
    expect(diagonal[0]).toBeCloseTo(-diagonal[1], 12);
  });

  it("stops immediately on release and restarts from rest", () => {
    const motion = new KeyboardCameraMotion();
    for (let frame = 0; frame < 60; frame++) motion.step(keys("KeyW"), 1 / 60);
    expect(motion.step(keys(), 1 / 60)).toEqual([0, 0]);
    expect(motion.step(keys("KeyW"), 1 / 60)).toEqual(new KeyboardCameraMotion().step(keys("KeyW"), 1 / 60));
  });

  it("cancels opposing directions without stopping the other axis", () => {
    const motion = new KeyboardCameraMotion();
    for (let frame = 0; frame < 60; frame++) motion.step(keys("KeyW", "KeyD"), 1 / 60);
    const [azimuth, polar] = motion.step(keys("KeyW", "KeyS", "KeyD"), 1 / 60);
    expect(azimuth).toBeGreaterThan(0);
    expect(polar).toBe(0);
    expect(motion.step(keys("KeyA", "KeyD"), 1 / 60)).toEqual([0, 0]);
  });

  it("does not coast in the previous direction after reversal", () => {
    const motion = new KeyboardCameraMotion();
    for (let frame = 0; frame < 60; frame++) motion.step(keys("KeyW"), 1 / 60);
    expect(motion.step(keys("KeyS"), 1 / 60)[1]).toBeGreaterThan(0);
  });

  it("bounds delayed frames and resets after blur or camera locks", () => {
    const motion = new KeyboardCameraMotion();
    expect(motion.step(keys("KeyW"), 8)).toEqual(new KeyboardCameraMotion().step(keys("KeyW"), 0.05));
    for (let frame = 0; frame < 60; frame++) motion.step(keys("KeyW"), 1 / 60);
    motion.reset();
    expect(motion.step(keys("KeyW"), 1 / 60)).toEqual(new KeyboardCameraMotion().step(keys("KeyW"), 1 / 60));
    for (const dt of [-1, 0, NaN, Infinity]) {
      expect(motion.step(keys("KeyW"), dt)).toEqual([0, 0]);
    }
  });
});
