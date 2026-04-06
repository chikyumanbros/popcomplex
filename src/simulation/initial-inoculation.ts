import type { World } from './world';
import type { OrganismManager } from './organism';
import type { RuleEvaluator } from './rule-evaluator';
import { GRID_HEIGHT, GRID_WIDTH } from './constants';
import type { Tape } from './tape';
import { LINEAGE_BYTE_AUX, syncGeneticKinFromPublic } from './tape';

export interface InoculationSite {
  x: number;
  y: number;
  /** Written to `LINEAGE_BYTE_AUX`; mixed into public kin tag so clades are face-distinct with identical rules. */
  cladeAux: number;
}

/** Three separated sites (256² grid): breaks single-origin symmetry without changing proto rule thresholds. */
export const DEFAULT_TRICLADE_SITES: InoculationSite[] = [
  { x: 56, y: 56, cladeAux: 0 },
  { x: 200, y: 56, cladeAux: 1 },
  { x: 128, y: 192, cladeAux: 2 },
];
export const DEFAULT_INOCULATION_PER_SITE = 10;
const DEFAULT_SITE_SPAWN_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [0, 0],
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
  [2, 0],
];

function clampSpawnCoord(v: number, max: number): number {
  return Math.max(2, Math.min(max - 3, v));
}

/**
 * Small regional env bumps at inoculation sites so multi-origin runs are not trivially energy-starved
 * relative to a single central colony (closed budget; sum increases slightly).
 */
export function addRegionalEnvBumps(
  env: Float32Array,
  sites: ReadonlyArray<{ x: number; y: number }>,
  delta: number,
  radius: number,
): void {
  const r2 = radius * radius;
  for (const s of sites) {
    const cx = Math.floor(s.x);
    const cy = Math.floor(s.y);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > r2) continue;
        const x = cx + dx;
        const y = cy + dy;
        if (x < 0 || y < 0 || x >= GRID_WIDTH || y >= GRID_HEIGHT) continue;
        env[y * GRID_WIDTH + x] += delta;
      }
    }
  }
}

/**
 * Add local nutrient spots while preserving total environmental energy.
 * Implementation: add bumps, then globally rescale so `sum(env)` stays unchanged.
 */
export function addRegionalEnvBumpsConservative(
  env: Float32Array,
  sites: ReadonlyArray<{ x: number; y: number }>,
  delta: number,
  radius: number,
): void {
  let before = 0;
  for (let i = 0; i < env.length; i++) before += env[i];
  if (before <= 0) return;
  addRegionalEnvBumps(env, sites, delta, radius);
  let after = 0;
  for (let i = 0; i < env.length; i++) after += env[i];
  if (after <= 0) return;
  const scale = before / after;
  for (let i = 0; i < env.length; i++) env[i] *= scale;
}

/** Spawn several protos from clones of `baseTape`, differing only in lineage aux / packed tag. */
export function spawnTricladProtos(
  world: World,
  organisms: OrganismManager,
  ruleEval: RuleEvaluator,
  baseTape: Tape,
  spawnEnergyEach: number,
  sites: ReadonlyArray<InoculationSite> = DEFAULT_TRICLADE_SITES,
  perSiteCount = DEFAULT_INOCULATION_PER_SITE,
): void {
  if (perSiteCount <= 0) return;
  if (perSiteCount > DEFAULT_SITE_SPAWN_OFFSETS.length) {
    throw new Error(`spawnTricladProtos: perSiteCount=${perSiteCount} exceeds supported offsets=${DEFAULT_SITE_SPAWN_OFFSETS.length}`);
  }
  for (const s of sites) {
    const cx = clampSpawnCoord(s.x, GRID_WIDTH);
    const cy = clampSpawnCoord(s.y, GRID_HEIGHT);
    for (let i = 0; i < perSiteCount; i++) {
      const off = DEFAULT_SITE_SPAWN_OFFSETS[i];
      const x = clampSpawnCoord(cx + off[0], GRID_WIDTH);
      const y = clampSpawnCoord(cy + off[1], GRID_HEIGHT);
      if (!world.isEmpty(x, y)) {
        throw new Error(`spawnTricladProtos: cell not empty (${x},${y})`);
      }
      if (!ruleEval.withdrawEnvUniform(spawnEnergyEach)) {
        throw new Error('spawnTricladProtos: env withdraw failed');
      }
      const tape = baseTape.clone();
      tape.data[LINEAGE_BYTE_AUX] = s.cladeAux & 0xff;
      syncGeneticKinFromPublic(tape.data);
      const id = world.spawnProto(x, y, tape.getPublicKinTagPacked(), spawnEnergyEach);
      organisms.register(id, tape, { parentId: null, birthTick: 0 });
      organisms.get(id)!.cells.add(y * GRID_WIDTH + x);
    }
  }
}
