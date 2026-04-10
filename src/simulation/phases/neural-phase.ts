import { GRID_WIDTH, GRID_HEIGHT } from '../constants';
import type { World } from '../world';
import type { OrganismManager } from '../organism';
import { clamp01 } from '../evaluator-context';
import { TOXIN_STRESS_WEIGHT, XENO_NN_SCALE } from '../sim-constants';
import { NN_OUTPUT } from '../organism';

export interface NeuralPhaseDeps {
  world: World;
  envEnergy: Float32Array;
  organisms: OrganismManager;
  /** 0..1: how much reactive stress modulates NN input sensitivity. */
  stressNnMix: number;
  /** 0..1: how much territorial-claim proxy modulates NN input sensitivity. */
  claimNnMix: number;
}

const EIGHT_DIRS: [number, number][] = [
  [0, -1], [1, 0], [0, 1], [-1, 0],
  [1, -1], [1, 1], [-1, 1], [-1, -1],
];

function isOuterCell(idx: number, orgCells: Set<number>): boolean {
  const x = idx % GRID_WIDTH;
  const y = (idx - x) / GRID_WIDTH;
  for (const [dx, dy] of EIGHT_DIRS) {
    const nx = x + dx, ny = y + dy;
    if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) return true;
    if (!orgCells.has(ny * GRID_WIDTH + nx)) return true;
  }
  return false;
}

function dominantMarker(world: World, idx: number): number {
  const [eat, digest, signal, move] = world.getMarkersByIdx(idx);
  const max = Math.max(eat, digest, signal, move);
  if (max === 0) return 0;
  if (max === eat)    return 64;
  if (max === digest) return 128;
  if (max === signal) return 192;
  return 255;
}

export function runUpdateNeuralNetworks(d: NeuralPhaseDeps): void {
  const dirsLen = EIGHT_DIRS.length;
  for (const org of d.organisms.organisms.values()) {
    if (org.cells.size === 0) continue;

    let totalE = 0;
    let totalS = 0;
    let totalEnv = 0;
    let boundaryCells = 0;
    let foreignNeighbors = 0;
    let markerDominance = 0;
    let envGradient = 0;
    let rotMax = 0;

    for (const idx of org.cells) {
      totalE += d.world.getCellEnergyByIdx(idx);
      totalS += d.world.getStomachByIdx(idx);
      totalEnv += d.envEnergy[idx]!;
      rotMax = Math.max(rotMax, d.world.rot[idx] ?? 0);

      const x = idx % GRID_WIDTH;
      const y = (idx - x) / GRID_WIDTH;
      if (isOuterCell(idx, org.cells)) boundaryCells++;

      for (const [dx, dy] of EIGHT_DIRS) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
        const nOrg = d.world.getOrganismId(nx, ny);
        if (nOrg !== 0 && nOrg !== org.id) foreignNeighbors++;
      }

      markerDominance += dominantMarker(d.world, idx) / 255;
      const c = d.envEnergy[idx]!;
      let localMax = c;
      for (const [dx, dy] of EIGHT_DIRS) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
        localMax = Math.max(localMax, d.envEnergy[ny * GRID_WIDTH + nx]!);
      }
      envGradient += Math.max(0, localMax - c);
    }

    const n = org.cells.size;
    const avgEnergy01 = Math.min(1, (totalE / n) / 255);
    const boundary01  = Math.min(1, boundaryCells / n);
    const foreign01   = Math.min(1, foreignNeighbors / (n * dirsLen));
    const marker01    = Math.min(1, markerDominance / n);
    const grad01      = Math.min(1, (envGradient / n) / 20);

    let totalToxin = 0;
    for (const idx of org.cells) totalToxin += d.world.toxin[idx] ?? 0;
    const avgToxin = totalToxin / n;

    const stress01 = clamp01(
      0.45 * (1 - avgEnergy01) +
      0.25 * foreign01 +
      0.20 * boundary01 +
      0.10 * rotMax +
      TOXIN_STRESS_WEIGHT * avgToxin,
    );
    org.nnStress01 = stress01;

    const claim01 = clamp01(
      0.65 * marker01 +
      0.20 * (1 - foreign01) +
      0.15 * (1 - grad01),
    );
    org.nnClaim01 = claim01;

    const xenoMod01 = clamp01(foreign01 * (1 + XENO_NN_SCALE * org.xenoTolerance));

    // Inputs 0-7: original signals
    org.nnInput[0] = avgEnergy01;
    org.nnInput[1] = Math.min(1, (totalS / n) / 255);
    org.nnInput[2] = Math.min(1, (totalEnv / n) / 50);
    org.nnInput[3] = Math.min(1, n / 64);
    org.nnInput[4] = boundary01;
    org.nnInput[5] = xenoMod01;    // experience-weighted foreign signal
    org.nnInput[6] = marker01;
    org.nnInput[7] = grad01;
    // Inputs 8-9: new direct perceptions added with NN_INPUT=10 expansion
    org.nnInput[8] = avgToxin;                      // metabolic toxin load (0..1)
    org.nnInput[9] = clamp01(org.stage / 3);        // developmental stage (0..1)

    // Stress→NN coupling: modulate input *values* (not a separate gain layer) by stress/claim.
    // This preserves the original effect of stressNnMix/claimNnMix without requiring inputGain.
    if (d.stressNnMix > 0) {
      const mix = d.stressNnMix;
      const s = stress01;
      const mod = (idx: number, scale: number) => {
        org.nnInput[idx] = clamp01(org.nnInput[idx]! * Math.max(0.10, Math.min(3.0, 1 + scale * mix * s)));
      };
      mod(0, -0.35);  // energy: downweight under stress
      mod(4,  0.55);  // boundary: upweight under stress
      mod(5,  0.70);  // foreign: upweight under stress
      mod(7,  0.45);  // gradient: upweight under stress
    }
    if (d.claimNnMix > 0) {
      const mix = d.claimNnMix;
      const c = claim01;
      const mod = (idx: number, scale: number) => {
        org.nnInput[idx] = clamp01(org.nnInput[idx]! * Math.max(0.10, Math.min(3.0, 1 + scale * mix * c)));
      };
      mod(6,  0.70);  // marker: upweight under claim
      mod(7, -0.55);  // gradient: downweight under claim
      mod(5, -0.35);  // foreign: downweight under claim
    }

    org.nn.forward(org.nnInput, org.nnOutput, org.nnPrimitives);

    let best = 0;
    for (let i = 1; i < NN_OUTPUT; i++) {
      if (org.nnOutput[i]! > org.nnOutput[best]!) best = i;
    }
    org.nnDominant = best;
  }
}
