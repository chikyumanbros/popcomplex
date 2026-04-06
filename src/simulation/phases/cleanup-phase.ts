import type { OrganismManager } from '../organism';
import type { World } from '../world';
import type { SuppressionMode } from '../runtime-config';

export interface CleanupPhaseDeps {
  organisms: OrganismManager;
  world: World;
  envEnergy: Float32Array;
  suppressionMode: SuppressionMode;
  randomF32: () => number;
  getCrowdingDissolveBonus: (orgCount: number, suppressionOn: boolean) => number;
  dissolveBase: number;
  dissolveSingleMult: number;
}

export function cleanupDeadOrganismsPhase(d: CleanupPhaseDeps, u32PerCell: number) {
  const nOrgs = d.organisms.count;
  const crowdDiss = d.getCrowdingDissolveBonus(nOrgs, d.suppressionMode === 'on');
  const baseDiss = d.dissolveBase + crowdDiss;

  const dead: number[] = [];
  for (const org of d.organisms.organisms.values()) {
    const toDissolve: number[] = [];
    let p = org.cells.size === 1 ? baseDiss * d.dissolveSingleMult : baseDiss;
    p = Math.min(0.22, p);
    for (const idx of org.cells) {
      if (d.world.getOrganismIdByIdx(idx) !== org.id) { toDissolve.push(idx); continue; }
      if (d.world.getCellEnergyByIdx(idx) <= 0 && d.randomF32() < p) toDissolve.push(idx);
    }
    for (const idx of toDissolve) {
      const e = d.world.getCellEnergyByIdx(idx);
      const s = d.world.getStomachByIdx(idx);
      if (e > 0) d.envEnergy[idx] += e;
      if (s > 0) d.envEnergy[idx] += s;
      const base = idx * u32PerCell;
      for (let i = 0; i < u32PerCell; i++) d.world.cellData[base + i] = 0;
      org.cells.delete(idx);
    }
    if (org.cells.size === 0) dead.push(org.id);
  }
  for (const id of dead) d.organisms.remove(id);
}

