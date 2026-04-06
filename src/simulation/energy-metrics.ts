/**
 * Full-grid energy bookkeeping (English identifiers / logs).
 * Gray–Scott u is open-system (feed term) — totals are not expected to match INITIAL_TOTAL_ENERGY.
 */

import { TOTAL_CELLS } from './constants';
import type { World } from './world';
import { GRID_WIDTH, GRID_HEIGHT } from './constants';
import { U32_PER_CELL } from './world';
import type { OrganismManager, Organism } from './organism';
import type { RuleEvaluator } from './rule-evaluator';

export interface EnergyBookkeeping {
  envU: number;
  cellEnergy: number;
  stomach: number;
  morphA: number;
  morphB: number;
  occupiedCells: number;
}

export function measureEnergyBookkeeping(world: World, ruleEval: RuleEvaluator): EnergyBookkeeping {
  let envU = 0;
  for (let i = 0; i < TOTAL_CELLS; i++) {
    envU += ruleEval.envEnergy[i];
  }

  let cellEnergy = 0;
  let stomach = 0;
  let morphA = 0;
  let morphB = 0;
  let occupiedCells = 0;

  for (let i = 0; i < TOTAL_CELLS; i++) {
    if (world.getOrganismIdByIdx(i) === 0) continue;
    occupiedCells++;
    cellEnergy += world.getCellEnergyByIdx(i);
    stomach += world.getStomachByIdx(i);
    morphA += world.getMorphogenA(i);
    morphB += world.getMorphogenB(i);
  }

  return { envU, cellEnergy, stomach, morphA, morphB, occupiedCells };
}

export function biomassReservoirTotal(b: EnergyBookkeeping): number {
  return b.envU + b.cellEnergy + b.stomach;
}

export function diffBookkeeping(a: EnergyBookkeeping, b: EnergyBookkeeping): EnergyBookkeeping {
  return {
    envU: b.envU - a.envU,
    cellEnergy: b.cellEnergy - a.cellEnergy,
    stomach: b.stomach - a.stomach,
    morphA: b.morphA - a.morphA,
    morphB: b.morphB - a.morphB,
    occupiedCells: b.occupiedCells - a.occupiedCells,
  };
}

export interface PopulationMetrics {
  uniqueLineages: number;
  topLineageShare: number;
  simpsonDiversity: number;
  /** Shannon entropy of lineage cell-fractions, natural log (nats). */
  shannonLineageNats: number;
  /** Pielou J = H / ln(S); 0 if S ≤ 1. */
  pielouEvenness: number;
  /** Gini coefficient of per-lineage occupied-cell counts (0 = equal, 1 = one dominates). */
  giniLineageSizes: number;
  /** Mean `cells.size` over all registered organisms (includes 0-cell entries as 0). */
  meanCellsPerOrganism: number;
  perimeterRatio: number;
  avgComponents: number;
}

export function measurePopulationMetrics(world: World, organisms: OrganismManager): PopulationMetrics {
  let occupied = 0;
  let perimeterFaces = 0;
  const lineageCounts = new Map<number, number>();

  for (let i = 0; i < TOTAL_CELLS; i++) {
    if (world.getOrganismIdByIdx(i) === 0) continue;
    occupied++;

    const lineage = world.cellData[i * U32_PER_CELL + 7] & 0xffffff;
    lineageCounts.set(lineage, (lineageCounts.get(lineage) ?? 0) + 1);

    const x = i % GRID_WIDTH;
    const y = (i - x) / GRID_WIDTH;
    if (x === 0 || world.getOrganismId(x - 1, y) === 0) perimeterFaces++;
    if (x === GRID_WIDTH - 1 || world.getOrganismId(x + 1, y) === 0) perimeterFaces++;
    if (y === 0 || world.getOrganismId(x, y - 1) === 0) perimeterFaces++;
    if (y === GRID_HEIGHT - 1 || world.getOrganismId(x, y + 1) === 0) perimeterFaces++;
  }

  let topCount = 0;
  let simpson = 0;
  let shannonNats = 0;
  if (occupied > 0) {
    for (const count of lineageCounts.values()) {
      if (count > topCount) topCount = count;
      const p = count / occupied;
      simpson += p * p;
      shannonNats -= p * Math.log(p);
    }
  }

  const S = lineageCounts.size;
  const pielouEvenness =
    S > 1 && shannonNats > 0 ? shannonNats / Math.log(S) : 0;

  const giniLineageSizes = giniFromPositiveCounts(lineageCounts.values());

  let componentSum = 0;
  let componentN = 0;
  let cellsSumAllOrgs = 0;
  const nOrgsSlot = organisms.organisms.size;
  for (const org of organisms.organisms.values()) {
    cellsSumAllOrgs += org.cells.size;
    if (org.cells.size === 0) continue;
    componentSum += countOrgComponents(org);
    componentN++;
  }

  return {
    uniqueLineages: S,
    topLineageShare: occupied > 0 ? topCount / occupied : 0,
    simpsonDiversity: occupied > 0 ? 1 - simpson : 0,
    shannonLineageNats: occupied > 0 ? shannonNats : 0,
    pielouEvenness,
    giniLineageSizes,
    meanCellsPerOrganism: nOrgsSlot > 0 ? cellsSumAllOrgs / nOrgsSlot : 0,
    perimeterRatio: occupied > 0 ? perimeterFaces / (occupied * 4) : 0,
    avgComponents: componentN > 0 ? componentSum / componentN : 0,
  };
}

/** Gini for a multiset of positive counts (e.g. cells per lineage). */
function giniFromPositiveCounts(values: Iterable<number>): number {
  const arr: number[] = [];
  for (const v of values) {
    if (v > 0) arr.push(v);
  }
  arr.sort((a, b) => a - b);
  const n = arr.length;
  if (n === 0) return 0;
  let sum = 0;
  for (const v of arr) sum += v;
  if (sum === 0) return 0;
  let weighted = 0;
  for (let i = 0; i < n; i++) {
    weighted += (2 * (i + 1) - n - 1) * arr[i]!;
  }
  return weighted / (n * sum);
}

function countOrgComponents(org: Organism): number {
  if (org.cells.size === 0) return 0;
  const dirs: Array<[number, number]> = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ];

  const unvisited = new Set<number>(org.cells);
  let components = 0;

  while (unvisited.size > 0) {
    const start = unvisited.values().next().value as number;
    const stack = [start];
    unvisited.delete(start);
    components++;

    while (stack.length > 0) {
      const idx = stack.pop() as number;
      const x = idx % GRID_WIDTH;
      const y = (idx - x) / GRID_WIDTH;
      for (const [dx, dy] of dirs) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
        const ni = ny * GRID_WIDTH + nx;
        if (!unvisited.has(ni)) continue;
        unvisited.delete(ni);
        stack.push(ni);
      }
    }
  }

  return components;
}
