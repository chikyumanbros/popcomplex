import { CellType, GRID_WIDTH, GRID_HEIGHT } from '../constants';
import { morphAbsorbAffinity } from '../metabolic-edge';
import type { World } from '../world';

export const DEFAULT_GROUP_SCAN_CAP = 256;

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export function computeJamStrengthOnEdge(jamTicks: Uint8Array, aIdx: number, bIdx: number): number {
  const jt = Math.max(jamTicks[aIdx] ?? 0, jamTicks[bIdx] ?? 0);
  return clamp01(jt / 255);
}

/**
 * Same-org connected component size around `seedIdx`, bounded by `cap`.
 * `dirs` should be orthogonal or 8-neighbor deltas (caller decides neighbor mode).
 */
export function sameOrgConnectedGroupSize(
  world: World,
  seedIdx: number,
  orgId: number,
  dirs: ReadonlyArray<readonly [number, number]>,
  cap = DEFAULT_GROUP_SCAN_CAP,
): number {
  if (orgId <= 0) return 1;
  const seen = new Set<number>();
  const q: number[] = [seedIdx];
  seen.add(seedIdx);
  while (q.length > 0 && seen.size < cap) {
    const idx = q.pop()!;
    const x = idx % GRID_WIDTH;
    const y = (idx - x) / GRID_WIDTH;
    for (const [dx, dy] of dirs) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const ni = ny * GRID_WIDTH + nx;
      if (seen.has(ni)) continue;
      if (world.getOrganismIdByIdx(ni) !== orgId) continue;
      if (world.getCellTypeByIdx(ni) === CellType.Empty) continue;
      seen.add(ni);
      q.push(ni);
      if (seen.size >= cap) break;
    }
  }
  return seen.size;
}

/** 0..1, rises quickly then saturates with group size. */
export function computeGroupBoostFromSize(groupSize: number): number {
  return clamp01(1 - 1 / Math.sqrt(Math.max(1, groupSize)));
}

export interface AbsorbEdgeScalars {
  morphAffinity: number;  // 0..1
  jamStrength: number;    // 0..1 raw edge jam (TTL)
  jamEff: number;         // 0..1 jam after group boost
  breakFrac: number;      // 0..1 fraction suppressing jamEff
  I: number;              // 0..1 overall coupling intensity (1 - effectiveJam)
}

export interface HgtDriveDefense {
  drive: number;   // 0..1
  defense: number; // 0..1
  stealFactor: number; // 0..1 (for debugging/telemetry if needed)
  stomachFactor: number; // 0..1
}

export function computeHgtDriveDefense(
  contactFlux: number,
  maxSteal: number,
  trust: number,
  stomach: number,
  contactPressure: number,
  repairDefense: number,
  stomachK: number,
): HgtDriveDefense {
  const stealFactor = clamp01(Math.min(1, contactFlux / Math.max(0.01, maxSteal)));
  const stomachFactor = stomach / (stomach + stomachK);
  const interfaceGain = 0.2 + 0.8 * clamp01(contactPressure);
  const drive = clamp01(Math.max(0, Math.min(1, (0.35 + trust) * stomachFactor * stealFactor * interfaceGain)));
  const defense = repairDefense / (1 + repairDefense);
  return { drive, defense, stealFactor, stomachFactor };
}

/**
 * Compute ABSORB edge scalars without mutating state.
 * Breaker presence is a continuous function of mean morphB (no tuned scale): mB/(1+|mB|).
 */
export function computeAbsorbEdgeScalars(
  dA: number,
  dB: number,
  jamStrength: number,
  groupBoost: number,
  meanMorphB: number,
): AbsorbEdgeScalars {
  const morphAffinity = morphAbsorbAffinity(dA, dB);
  const jamEff = clamp01(jamStrength * (1 + groupBoost));
  const breakerPresence = meanMorphB / (1 + Math.abs(meanMorphB));
  const breakFrac = clamp01(Math.min(1, breakerPresence * groupBoost));
  const effectiveJamStrength = clamp01(jamEff * (1 - breakFrac));
  const I = clamp01(1 - effectiveJamStrength);
  return { morphAffinity, jamStrength, jamEff, breakFrac, I };
}

