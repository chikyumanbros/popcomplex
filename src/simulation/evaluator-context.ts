/**
 * Shared types and pure utility functions used across the extracted
 * phase / action modules.  Keeping these in one place avoids circular
 * imports and lets each extracted file import only what it needs.
 */

/** Snapshot of a cell's position and mutable energy state, used during rule evaluation. */
export interface CellCtx {
  x: number;
  y: number;
  idx: number;
  energy: number;
  orgId: number;
}

/** Clamp a number to [0, 1]. */
export function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

/** Linear interpolation. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
