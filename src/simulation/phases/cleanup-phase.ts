import type { OrganismManager } from '../organism';
import type { World } from '../world';
import type { SuppressionMode } from '../runtime-config';

export interface CleanupPhaseDeps {
  organisms: OrganismManager;
  world: World;
  envEnergy: Float32Array;
  suppressionMode: SuppressionMode;
  metabolicScale: number;
  getCrowdingDissolveBonus: (orgCount: number, suppressionOn: boolean) => number;
  dissolveBase: number;
  dissolveSingleMult: number;
  sameOrgConnectedGroupSize: (seedIdx: number, orgId: number) => number;
  gridWidth: number;
  gridHeight: number;
  neighborModeEight: boolean;
}

export function cleanupDeadOrganismsPhase(d: CleanupPhaseDeps, u32PerCell: number) {
  const nOrgs = d.organisms.count;
  const crowdDiss = d.getCrowdingDissolveBonus(nOrgs, d.suppressionMode === 'on');
  const baseDiss = d.dissolveBase + crowdDiss;

  // Deterministic rot progression: time-to-dissolve ~= 1/baseDiss ticks at size=1 (like the old expected lifetime).
  // Larger connected components slow rot (network self-maintenance). No hard "death confirmation" threshold besides rot>=1.
  const ROT_RECOVERY_PER_TICK = 0.08; // when alive, rot bleeds off quickly
  const STOMACH_LEAK_FRAC = 0.10; // dead tissue leaks gut contents each tick

  const dead: number[] = [];
  for (const org of d.organisms.organisms.values()) {
    const toDissolve: number[] = [];
    for (const idx of org.cells) {
      if (d.world.getOrganismIdByIdx(idx) !== org.id) {
        toDissolve.push(idx);
        d.world.rot[idx] = 0;
        continue;
      }

      const e = d.world.getCellEnergyByIdx(idx);
      if (e > 0) {
        d.world.rot[idx] = Math.max(0, d.world.rot[idx] - ROT_RECOVERY_PER_TICK);
        continue;
      }

      // Dead-tissue stomach leak: distribute gut contents into local environment (self + neighbors).
      const s0 = d.world.getStomachByIdx(idx);
      if (s0 > 1e-6) {
        const leak = Math.min(s0, s0 * STOMACH_LEAK_FRAC);
        d.world.setStomachByIdx(idx, s0 - leak);
        const x = idx % d.gridWidth;
        const y = (idx - x) / d.gridWidth;
        let n = 1;
        if (x > 0) n++;
        if (x + 1 < d.gridWidth) n++;
        if (y > 0) n++;
        if (y + 1 < d.gridHeight) n++;
        if (d.neighborModeEight) {
          if (x > 0 && y > 0) n++;
          if (x + 1 < d.gridWidth && y > 0) n++;
          if (x > 0 && y + 1 < d.gridHeight) n++;
          if (x + 1 < d.gridWidth && y + 1 < d.gridHeight) n++;
        }
        const share = leak / n;
        d.envEnergy[idx] += share;
        if (x > 0) d.envEnergy[idx - 1] += share;
        if (x + 1 < d.gridWidth) d.envEnergy[idx + 1] += share;
        if (y > 0) d.envEnergy[idx - d.gridWidth] += share;
        if (y + 1 < d.gridHeight) d.envEnergy[idx + d.gridWidth] += share;
        if (d.neighborModeEight) {
          if (x > 0 && y > 0) d.envEnergy[idx - d.gridWidth - 1] += share;
          if (x + 1 < d.gridWidth && y > 0) d.envEnergy[idx - d.gridWidth + 1] += share;
          if (x > 0 && y + 1 < d.gridHeight) d.envEnergy[idx + d.gridWidth - 1] += share;
          if (x + 1 < d.gridWidth && y + 1 < d.gridHeight) d.envEnergy[idx + d.gridWidth + 1] += share;
        }
      }

      // Rot progression: baseDiss becomes per-tick rot delta (expected time ~ 1/baseDiss).
      // Network slows rot via connected component size.
      const groupSize = Math.max(1, d.sameOrgConnectedGroupSize(idx, org.id));
      const netSlow = 1 + 0.6 * Math.log2(groupSize);
      const p = Math.min(0.22, (org.cells.size === 1 ? baseDiss * d.dissolveSingleMult : baseDiss));
      const harshMult = 0.8 + 1.2 * Math.max(0.2, Math.min(1.4, d.metabolicScale));
      d.world.rot[idx] = d.world.rot[idx] + (p * harshMult / netSlow);
      if (d.world.rot[idx] >= 1) toDissolve.push(idx);
    }
    for (const idx of toDissolve) {
      const e = d.world.getCellEnergyByIdx(idx);
      const s = d.world.getStomachByIdx(idx);
      if (e > 0) d.envEnergy[idx] += e;
      if (s > 0) d.envEnergy[idx] += s;
      const base = idx * u32PerCell;
      for (let i = 0; i < u32PerCell; i++) d.world.cellData[base + i] = 0;
      org.cells.delete(idx);
      d.world.ruleRoutes[idx] = 0;
      d.world.rot[idx] = 0;
    }
    if (org.cells.size === 0) dead.push(org.id);
  }
  for (const id of dead) d.organisms.remove(id);
}

