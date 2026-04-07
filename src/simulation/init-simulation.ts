import { GRID_WIDTH, GRID_HEIGHT, INITIAL_ENV_ENERGY_PER_CELL } from './constants';
import {
  addRegionalEnvBumps,
  addRegionalEnvBumpsConservative,
  DEFAULT_TRICLADE_SITES,
  type InoculationSite,
  spawnTricladProtos,
} from './initial-inoculation';
import type { RuleEvaluator } from './rule-evaluator';
import type { Tape } from './tape';
import { createProtoTape } from './tape';
import type { World } from './world';
import type { OrganismManager } from './organism';

export interface InitSimulationOptions {
  /** If provided, used directly; otherwise a new env array is allocated. Length must be GRID_WIDTH*GRID_HEIGHT. */
  env?: Float32Array;
  /** Initial tape for all inoculated protos (cloned per site). */
  protoTape?: Tape;
  spawnEnergy: number;
  /**
   * `culture=true`: add nutrient hotspots while preserving total env energy (conservative bumps).
   * `multiOrigin=true`: inoculate multiple separated sites (triclad) using the same genome.
   *
   * Note: both flags currently imply triclad inoculation sites; `culture` is a distributional mode,
   * `multiOrigin` is an origin-count mode.
   */
  culture: boolean;
  multiOrigin: boolean;
  /** Override inoculation sites; defaults to 3 separated sites on 256² grid. */
  sites?: ReadonlyArray<InoculationSite>;
  /** Bump strength and radius (for culture/multiOrigin env shaping). */
  bumpDelta?: number;
  bumpRadius?: number;
}

export interface InitSimulationResult {
  env: Float32Array;
  protoTape: Tape;
  /** Number of organisms registered immediately after inoculation. */
  initialOrganisms: number;
}

function assertEnvLength(env: Float32Array) {
  const expected = GRID_WIDTH * GRID_HEIGHT;
  if (env.length !== expected) {
    throw new Error(`initSimulation: env length ${env.length} != ${expected}`);
  }
}

function spawnSingleCenterProto(
  world: World,
  organisms: OrganismManager,
  ruleEval: RuleEvaluator,
  tape: Tape,
  spawnEnergy: number,
): void {
  const cx = Math.floor(GRID_WIDTH / 2);
  const cy = Math.floor(GRID_HEIGHT / 2);
  if (!world.isEmpty(cx, cy)) {
    throw new Error(`initSimulation: center cell not empty (${cx},${cy})`);
  }
  if (!ruleEval.withdrawEnvUniform(spawnEnergy)) {
    throw new Error('initSimulation: env withdraw failed (single origin)');
  }
  const id = world.spawnProto(cx, cy, tape.getPublicKinTagPacked(), spawnEnergy);
  organisms.register(id, tape, { parentId: null, birthTick: 0 });
  organisms.get(id)!.cells.add(cy * GRID_WIDTH + cx);
}

/**
 * Canonical initialization helper used by browser, headless runs, and tests.
 *
 * Order of operations is intentional:
 * - Build env field (optionally with bumps)
 * - `setEnvEnergy(env)` then `snapClosedEnergyBudgetFromWorld()` before spawning
 * - Spawn withdraws energy from env uniformly, then creates cell biomass of equal magnitude (closed budget stays consistent)
 */
export function initSimulation(
  world: World,
  organisms: OrganismManager,
  ruleEval: RuleEvaluator,
  opts: InitSimulationOptions,
): InitSimulationResult {
  const env = opts.env ?? new Float32Array(GRID_WIDTH * GRID_HEIGHT);
  assertEnvLength(env);
  env.fill(INITIAL_ENV_ENERGY_PER_CELL);

  const sites = opts.sites ?? DEFAULT_TRICLADE_SITES;
  const bumpDelta = opts.bumpDelta ?? 1.0;
  const bumpRadius = opts.bumpRadius ?? 14;

  if (opts.culture) {
    addRegionalEnvBumpsConservative(env, sites, bumpDelta, bumpRadius);
  } else if (opts.multiOrigin) {
    addRegionalEnvBumps(env, sites, bumpDelta, bumpRadius);
  }

  ruleEval.setEnvEnergy(env);
  ruleEval.snapClosedEnergyBudgetFromWorld();

  const protoTape = opts.protoTape ?? createProtoTape();

  if (opts.culture || opts.multiOrigin) {
    // Multi-site inoculation: clones differ only in aux clade tag.
    // NOTE: `spawnTricladProtos` clones the tape and syncs public->genetic kin tags for each clade.
    spawnTricladProtos(world, organisms, ruleEval, protoTape, opts.spawnEnergy, sites);
  } else {
    // Single-origin baseline: one proto at center using the provided tape.
    spawnSingleCenterProto(world, organisms, ruleEval, protoTape, opts.spawnEnergy);
  }

  return { env, protoTape, initialOrganisms: organisms.count };
}

