/**
 * Foreign-edge predicates (neighbor grid): ABSORB coupling affinity, passive intake, kin GIVE, and JAM-gated cooperation.
 * Spatial graph is still the neighbor grid.
 */

/** Max |Δmorph| for bidirectional relax branch of ABSORB (symbiotic interface). */
export const MORPH_ABSORB_MATCH_A = 1.7;
export const MORPH_ABSORB_MATCH_B = 1.7;

export function morphChannelsCompatible(absDeltaA: number, absDeltaB: number): boolean {
  return absDeltaA <= MORPH_ABSORB_MATCH_A && absDeltaB <= MORPH_ABSORB_MATCH_B;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

/**
 * 0..1 morph affinity for ABSORB coupling, based on both channels' absolute deltas.
 * Uses a smooth exponential kernel so the interaction varies continuously rather than switching modes.
 *
 * Typical usage: `affinity≈1` means strong symmetric relax component; `affinity≈0` means strong stomach-steal component.
 */
export function morphAbsorbAffinity(absDeltaA: number, absDeltaB: number): number {
  const a = MORPH_ABSORB_MATCH_A;
  const b = MORPH_ABSORB_MATCH_B;
  // exp(-((dA/a)^2 + (dB/b)^2)) : 1 at perfect match, smoothly decays with mismatch.
  const x = (absDeltaA / a);
  const y = (absDeltaB / b);
  const g = Math.exp(-(x * x + y * y));
  // Avoid denorm/NaN surprises.
  return Number.isFinite(g) ? clamp01(g) : 0;
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
