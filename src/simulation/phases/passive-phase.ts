/**
 * Passive per-cell phase helpers.
 * These run once per cell per tick in the passive loop of `RuleEvaluator.evaluate()`.
 * They do NOT move energy between reservoirs (conservation guaranteed) except for
 * slight rot modifications which are purely gauge changes.
 */
import type { World } from '../world';
import type { Organism } from '../organism';
import type { CellCtx } from '../evaluator-context';
import {
  DEV_SENESCENT_ROT_PER_TICK,
  TOXIN_PASSIVE_DECAY,
  TOXIN_ROT_BOOST,
} from '../sim-constants';

// ==================== DEVELOPMENT PHASE ====================

/** Apply per-tick soft-aging effect for SENESCENT (stage 3) organisms. */
export function runApplyDevelopmentPhase(world: World, cell: CellCtx, org: Organism): void {
  if (org.stage === 3) {
    world.rot[cell.idx] = Math.min(1, (world.rot[cell.idx] ?? 0) + DEV_SENESCENT_ROT_PER_TICK);
  }
}

// ==================== TOXIN PASSIVE PHASE ====================

/**
 * Per-cell toxin passive phase.
 * - Natural dissipation toward zero (metabolic clearance).
 * - Minor rot acceleration on energy-depleted cells (toxin accelerates necrosis).
 * Does NOT move energy; toxin is a dimensionless modifier state.
 */
export function runApplyToxinPassivePhase(world: World, cell: CellCtx): void {
  const idx = cell.idx;
  const t = world.toxin[idx] ?? 0;
  if (t <= 0) return;
  world.toxin[idx] = Math.max(0, t * (1 - TOXIN_PASSIVE_DECAY));
  if (cell.energy <= 0 && t > 0.01) {
    world.rot[idx] = Math.min(1, (world.rot[idx] ?? 0) + t * TOXIN_ROT_BOOST);
  }
}
