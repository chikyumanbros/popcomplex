/**
 * Foreign-edge predicates (neighbor grid): morph absorb branch, passive intake, kin GIVE, and JAM-gated cooperation.
 * Spatial graph is still the neighbor grid; predation steal intentionally bypasses some cooperation gates — see `foreignAbsorbInteraction`.
 */

/** Max |Δmorph| for bidirectional relax branch of ABSORB (symbiotic interface). */
export const MORPH_ABSORB_MATCH_A = 1.7;
export const MORPH_ABSORB_MATCH_B = 1.7;

export function morphChannelsCompatible(absDeltaA: number, absDeltaB: number): boolean {
  return absDeltaA <= MORPH_ABSORB_MATCH_A && absDeltaB <= MORPH_ABSORB_MATCH_B;
}

export type ForeignAbsorbInteraction = 'bidirectional_relax' | 'predation_steal';

/**
 * ABSORB at a heterospecific face: morph match + not jammed → relax; else predation to actor stomach.
 * Jam blocks the symbiotic branch only; predation still applies (legacy behavior).
 */
export function foreignAbsorbInteraction(morphCompatible: boolean, edgeJammed: boolean): ForeignAbsorbInteraction {
  if (morphCompatible && !edgeJammed) return 'bidirectional_relax';
  return 'predation_steal';
}

/**
 * Passive env→stomach intake gate. Unifies “some biological activity” with EAT’s low-energy case:
 * either cell energy remains or there is digestible buffer in the gut.
 */
export function canPassiveIntakeFromEnv(cellEnergy: number, stomach: number): boolean {
  return cellEnergy > 0 || stomach >= 0.01;
}

/**
 * Foreign-cell GIVE (kin-trust path): trust must meet minimum. Caller should skip the edge when JAM is active.
 */
export function allowsForeignKinGive(kinTrust: number, minTrust: number): boolean {
  return kinTrust >= minTrust;
}

/**
 * Cross-lineage cooperative coupling (foreign kin GIVE, repair quorum weight, HGT): off when JAM is active on the edge.
 * Predation steal (ABSORB) intentionally ignores this — see `foreignAbsorbInteraction`.
 */
export function foreignKinCooperationEdgeOpen(edgeJammed: boolean): boolean {
  return !edgeJammed;
}
