import {
  DISTRESS_FIRE_CHANCE, DISTRESS_ENERGY_THRESH,
  BASE_METABOLIC, CROWD_METABOLIC,
  DOMINANCE_SHARE_SOFT, DOMINANCE_METABOLIC_COEFF, DOMINANCE_METABOLIC_CAP,
  DOMINANCE_SHARE_REP, DOMINANCE_REP_FAIL_COEFF, DOMINANCE_REP_FAIL_CAP,
  REPRO_CROWD_START_ORGS, REPRO_CROWD_FAIL_COEFF, REPRO_CROWD_FAIL_CAP,
  DISSOLVE_CROWD_START, DISSOLVE_CROWD_BONUS, DISSOLVE_CROWD_CAP,
  SPLIT_CROWD_COOLDOWN_START_ORGS, SPLIT_CROWD_COOLDOWN_COEFF, SPLIT_CROWD_COOLDOWN_CAP,
  SPLIT_CROWD_MIN_FRAGMENT_START_ORGS, SPLIT_CROWD_MIN_FRAGMENT_CELLS, SPLIT_MIN_FRAGMENT_CELLS,
} from '../sim-constants';

const EIGHT_DIRS: [number, number][] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
  [1, -1],
  [1, 1],
  [-1, 1],
  [-1, -1],
];

export interface StressCell {
  x: number;
  y: number;
  orgId: number;
}

export function getDistressFireChance(distressFireChanceScale: number): number {
  return Math.min(1, DISTRESS_FIRE_CHANCE * Math.max(0, distressFireChanceScale));
}

export function shouldTriggerDistressFire(cellEnergy: number, distressChance: number, random01: number): boolean {
  return cellEnergy < DISTRESS_ENERGY_THRESH && random01 < distressChance;
}

export function getDominanceReproduceFailChance(share: number, suppressionEnabled: boolean): number {
  if (!suppressionEnabled || share <= DOMINANCE_SHARE_REP) return 0;
  return Math.min(DOMINANCE_REP_FAIL_CAP, (share - DOMINANCE_SHARE_REP) * DOMINANCE_REP_FAIL_COEFF);
}

export function getCrowdingReproduceFailChance(organismCount: number, suppressionEnabled: boolean): number {
  if (!suppressionEnabled || organismCount <= REPRO_CROWD_START_ORGS) return 0;
  return Math.min(REPRO_CROWD_FAIL_CAP, (organismCount - REPRO_CROWD_START_ORGS) * REPRO_CROWD_FAIL_COEFF);
}

export function getDominanceMetabolicMultiplier(share: number, suppressionEnabled: boolean): number {
  if (!suppressionEnabled) return 1;
  return (
    1 +
    Math.min(
      DOMINANCE_METABOLIC_CAP - 1,
      Math.max(0, share - DOMINANCE_SHARE_SOFT) * DOMINANCE_METABOLIC_COEFF,
    )
  );
}

export function getPerCellMetabolicBase(
  organismCellCount: number,
  dominanceMultiplier: number,
  metabolicScale: number,
): number {
  return (BASE_METABOLIC + CROWD_METABOLIC * Math.sqrt(organismCellCount)) * dominanceMultiplier * metabolicScale;
}

export function getCrowdingDissolveBonus(organismCount: number, suppressionEnabled: boolean): number {
  if (!suppressionEnabled || organismCount <= DISSOLVE_CROWD_START) return 0;
  return Math.min(DISSOLVE_CROWD_CAP, (organismCount - DISSOLVE_CROWD_START) * DISSOLVE_CROWD_BONUS);
}

export function getSplitMinFragmentCells(organismCount: number, suppressionEnabled: boolean): number {
  if (suppressionEnabled && organismCount >= SPLIT_CROWD_MIN_FRAGMENT_START_ORGS) {
    return SPLIT_CROWD_MIN_FRAGMENT_CELLS;
  }
  return SPLIT_MIN_FRAGMENT_CELLS;
}

export function getSplitCrowdExtraCooldown(organismCount: number, suppressionEnabled: boolean): number {
  if (!suppressionEnabled || organismCount <= SPLIT_CROWD_COOLDOWN_START_ORGS) return 0;
  return Math.min(
    SPLIT_CROWD_COOLDOWN_CAP,
    (organismCount - SPLIT_CROWD_COOLDOWN_START_ORGS) * SPLIT_CROWD_COOLDOWN_COEFF,
  );
}

export function computeForeignContactPressure(
  cell: StressCell,
  getOrganismIdAt: (x: number, y: number) => number,
  gridWidth: number,
  gridHeight: number,
): number {
  let occupied = 0;
  let foreign = 0;
  for (const [dx, dy] of EIGHT_DIRS) {
    const nx = cell.x + dx;
    const ny = cell.y + dy;
    if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
    const nOrg = getOrganismIdAt(nx, ny);
    if (nOrg === 0) continue;
    occupied++;
    if (nOrg !== cell.orgId) foreign++;
  }
  if (occupied === 0) return 0;
  return foreign / occupied;
}
