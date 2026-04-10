import type { OrganismManager } from '../organism';
import type { World } from '../world';

export interface DigestPhaseDeps {
  organisms: OrganismManager;
  world: World;
  envEnergy: Float32Array;
  digestRuleBoost: Float32Array;
  markerDigestSlot: number;
  sameOrgNeighborRatioByIdx: (idx: number, orgId: number) => number;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
  // constants
  passiveDigestRate: number;
  digestionHeatLoss: number;
  digestNetworkBase: number;
  digestNetworkCoeff: number;
  /** Fraction of each digested unit that accumulates as intracellular toxin (dimensionless). */
  toxinDigestRate: number;
  /** toxin=1 suppresses passiveDigestRate by this fraction (0..1). */
  toxinSuppress: number;
}

/** Single pipeline per tick: stomach -> cell energy + heat to env. */
export function runDigestPhase(d: DigestPhaseDeps) {
  for (const org of d.organisms.organisms.values()) {
    if (!org.tape.isDigestModuleIntact()) continue;
    for (const idx of org.cells) {
      const stomach = d.world.getStomachByIdx(idx);
      if (stomach < 0.01) continue;
      const cellE = d.world.getCellEnergyByIdx(idx);
      const enzymeEff = Math.max(0, Math.min(1, cellE / 3));
      const specBonus = 1 + d.world.getMarkerByIdx(idx, d.markerDigestSlot as 0 | 1 | 2 | 3) / 255;
      const boostMul = 1 + d.digestRuleBoost[idx];
      const sameRatio = d.sameOrgNeighborRatioByIdx(idx, org.id);
      const networkMul = d.digestNetworkBase + d.digestNetworkCoeff * sameRatio;
      // Quadratic toxin suppression: low loads (≤0.3 typical) have negligible effect;
      // only genuinely overloaded cells (toxin > 0.7) experience meaningful suppression.
      const toxinLoad = d.world.toxin[idx] ?? 0;
      const effectiveDigestRate = d.passiveDigestRate * (1 - toxinLoad * toxinLoad * d.toxinSuppress);
      const digested = stomach * effectiveDigestRate * enzymeEff * specBonus * boostMul * networkMul;
      const heat = digested * d.digestionHeatLoss;
      d.world.setStomachByIdx(idx, stomach - digested);
      d.setCellEnergyCappedByIdx(idx, cellE + digested - heat, org.id);
      d.envEnergy[idx] += heat;
      // Accumulate toxin byproduct from digestion (not energy — dimensionless modifier state).
      if (digested > 1e-6) {
        d.world.toxin[idx] = Math.min(1, toxinLoad + digested * d.toxinDigestRate);
      }
    }
  }
}

