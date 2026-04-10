import { TOTAL_CELLS } from '../constants';
import type { World } from '../world';
import type { BudgetMode } from '../runtime-config';
import { BUDGET_RESCALE_THRESHOLD } from '../sim-constants';

export interface EnergyBudgetDeps {
  world: World;
  envEnergy: Float32Array;
  ecosystemEnergyBudget: number;
  budgetMode: BudgetMode;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
}

/** Rescale env so sum(env) = ecosystemEnergyBudget − biomass (fixes diffusion boundary drift + float). */
export function runEnforceClosedEnergyBudget(d: EnergyBudgetDeps): void {
  let bio = 0;
  let es = 0;
  for (let i = 0; i < TOTAL_CELLS; i++) {
    es += d.envEnergy[i]!;
    if (d.world.getOrganismIdByIdx(i) === 0) continue;
    bio += d.world.getCellEnergyByIdx(i) + d.world.getStomachByIdx(i);
  }

  let targetEnv = d.ecosystemEnergyBudget - bio;
  if (targetEnv < -1e-4) {
    runScaleDownBiomass(d, -targetEnv, bio);
    let bio2 = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      if (d.world.getOrganismIdByIdx(i) === 0) continue;
      bio2 += d.world.getCellEnergyByIdx(i) + d.world.getStomachByIdx(i);
    }
    targetEnv = d.ecosystemEnergyBudget - bio2;
  }

  if (targetEnv <= 0) {
    for (let i = 0; i < TOTAL_CELLS; i++) d.envEnergy[i] = 0;
    return;
  }

  if (es <= 1e-12) {
    const per = targetEnv / TOTAL_CELLS;
    for (let i = 0; i < TOTAL_CELLS; i++) d.envEnergy[i] = per;
    return;
  }

  const errBeforeRescale = targetEnv - es;
  if (d.budgetMode === 'local') {
    if (Math.abs(errBeforeRescale) <= BUDGET_RESCALE_THRESHOLD) {
      const anchor = findBudgetAnchorIndex(d);
      d.envEnergy[anchor] = Math.max(0, d.envEnergy[anchor]! + errBeforeRescale);
      return;
    }
  }

  const sc = targetEnv / es;
  for (let i = 0; i < TOTAL_CELLS; i++) {
    d.envEnergy[i] = Math.max(0, d.envEnergy[i]! * sc);
  }

  let s2 = 0;
  for (let i = 0; i < TOTAL_CELLS; i++) s2 += d.envEnergy[i]!;
  const err = targetEnv - s2;
  if (Math.abs(err) > 1e-5) {
    const anchor = findBudgetAnchorIndex(d);
    d.envEnergy[anchor] = Math.max(0, d.envEnergy[anchor]! + err);
  }
}

function findBudgetAnchorIndex(d: EnergyBudgetDeps): number {
  for (let i = 0; i < TOTAL_CELLS; i++) {
    if (d.world.getOrganismIdByIdx(i) !== 0) return i;
  }
  return 0;
}

export function runScaleDownBiomass(d: EnergyBudgetDeps, reduceBioBy: number, knownBio?: number): void {
  let bio: number;
  if (knownBio !== undefined) {
    bio = knownBio;
  } else {
    bio = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      if (d.world.getOrganismIdByIdx(i) === 0) continue;
      bio += d.world.getCellEnergyByIdx(i) + d.world.getStomachByIdx(i);
    }
  }
  if (bio <= 1e-12) return;
  const newBio = Math.max(0, bio - reduceBioBy);
  const f = newBio / bio;
  for (let i = 0; i < TOTAL_CELLS; i++) {
    if (d.world.getOrganismIdByIdx(i) === 0) continue;
    const e = d.world.getCellEnergyByIdx(i);
    const st = d.world.getStomachByIdx(i);
    d.setCellEnergyCappedByIdx(i, e * f);
    d.world.setStomachByIdx(i, st * f);
  }
}
