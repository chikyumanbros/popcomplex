import type { OrganismManager } from '../organism';
import type { World } from '../world';
import type { SuppressionMode } from '../runtime-config';

export interface MetabolicPhaseDeps {
  organisms: OrganismManager;
  world: World;
  envEnergy: Float32Array;
  liveCellCount: number;
  suppressionMode: SuppressionMode;
  metabolicScale: number;
  sameOrgNeighborRatioByIdx: (idx: number, orgId: number) => number;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
  randomF32: () => number;
  // from stress-signals
  getDominanceMetabolicMultiplier: (share: number, suppressionOn: boolean) => number;
  getPerCellMetabolicBase: (orgCells: number, dominanceMult: number, metabolicScale: number) => number;
  // constants
  isolationMetabolicPenalty: number;
  lowEnergyLeakMax: number;
}

export function applyMetabolicCostPhase(d: MetabolicPhaseDeps) {
  const liveN = Math.max(1, d.liveCellCount);
  for (const org of d.organisms.organisms.values()) {
    const share = org.cells.size / liveN;
    const dominanceMult = d.getDominanceMetabolicMultiplier(share, d.suppressionMode === 'on');
    const perCell = d.getPerCellMetabolicBase(org.cells.size, dominanceMult, d.metabolicScale);
    for (const idx of org.cells) {
      const sameRatio = d.sameOrgNeighborRatioByIdx(idx, org.id);
      const isolationMult = 1 + d.isolationMetabolicPenalty * (1 - sameRatio);
      const e = d.world.getCellEnergyByIdx(idx);
      const cost = Math.min(e, perCell * isolationMult);
      const newE = e - cost;
      d.setCellEnergyCappedByIdx(idx, newE, org.id);
      d.envEnergy[idx] += cost;
      if (newE < 2) {
        const leak = Math.min(newE, d.lowEnergyLeakMax);
        if (leak > 0) {
          d.setCellEnergyCappedByIdx(idx, newE - leak, org.id);
          d.envEnergy[idx] += leak;
        }
        if (d.randomF32() < 0.02) org.tape.applyReadDegradation(0, org.age);
      }
    }
  }
}

export interface OverheadPhaseDeps {
  organisms: OrganismManager;
  world: World;
  envEnergy: Float32Array;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
  overheadPerTick: number;
}

export function applyOrganismOverheadPhase(d: OverheadPhaseDeps) {
  for (const org of d.organisms.organisms.values()) {
    if (org.cells.size === 0) continue;
    const indices = [...org.cells];
    let totalE = 0;
    for (const idx of indices) totalE += d.world.getCellEnergyByIdx(idx);
    if (totalE < 1e-8) continue;
    const takeTotal = Math.min(totalE, d.overheadPerTick);
    for (const idx of indices) {
      const e = d.world.getCellEnergyByIdx(idx);
      const t = takeTotal * (e / totalE);
      d.setCellEnergyCappedByIdx(idx, e - t, org.id);
      d.envEnergy[idx] += t;
    }
  }
}

