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

const DISTRESS_FIRE_CHANCE = 0.2;
const DISTRESS_ENERGY_THRESH = 3;

const BASE_METABOLIC = 0.25;
const CROWD_METABOLIC = 0.008;
const DOMINANCE_SHARE_SOFT = 0.08;
const DOMINANCE_METABOLIC_COEFF = 5.5;
const DOMINANCE_METABOLIC_CAP = 2.4;

const DOMINANCE_SHARE_REP = 0.1;
const DOMINANCE_REP_FAIL_COEFF = 2.2;
const DOMINANCE_REP_FAIL_CAP = 0.72;
const REPRO_CROWD_START_ORGS = 700;
const REPRO_CROWD_FAIL_COEFF = 0.00014;
const REPRO_CROWD_FAIL_CAP = 0.58;

const DISSOLVE_CROWD_START = 2500;
const DISSOLVE_CROWD_BONUS = 0.000045;
const DISSOLVE_CROWD_CAP = 0.045;

const SPLIT_CROWD_COOLDOWN_START_ORGS = 300;
const SPLIT_CROWD_COOLDOWN_COEFF = 0.03;
const SPLIT_CROWD_COOLDOWN_CAP = 36;
const SPLIT_CROWD_MIN_FRAGMENT_START_ORGS = 450;
const SPLIT_CROWD_MIN_FRAGMENT_CELLS = 4;
const SPLIT_MIN_FRAGMENT_CELLS = 2;

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
