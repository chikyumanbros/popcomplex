/**
 * Action implementations: DIV (divide), REPRODUCE.
 *
 * These actions manage organism lifecycle events (cell addition, new-organism birth)
 * and are deeply entangled with RuleEvaluator's organism registry, tape transcription,
 * and energy accounting.  The deps interface below captures the full contract.
 *
 * This file serves as the architectural boundary declaration — future extraction
 * of each action as a free function should land here.
 */
import type { World } from '../world';
import type { OrganismManager } from '../organism';
import type { CellCtx } from '../evaluator-context';

/** Shared context required by divide / reproduce action implementations. */
export interface OrganismActionDeps {
  world: World;
  envEnergy: Float32Array;
  organisms: OrganismManager;
  simTick: number;
  metabolicScale: number;
  suppressionEnabled: boolean;
  dominanceLiveCellCount: number;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
  withdrawEnvUniform: (amount: number) => boolean;
  cellAt: (idx: number, orgId: number) => CellCtx | null;
}

// Implementations remain in RuleEvaluator.ts pending full extraction.
// Each method signature:
//   actionDivide(cell, org, tape, divCost): boolean
//   actionReproduce(cell, org, tape, childFraction): boolean
