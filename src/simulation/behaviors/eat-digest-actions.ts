/**
 * Action implementations: EAT, DIGEST, GIVE, TAKE, EMIT, SPILL.
 *
 * These actions are deeply entangled with RuleEvaluator's private state
 * (envEnergy, world, organisms, helper methods).  The deps interface below
 * captures the full contract needed for extraction; the actual implementations
 * currently live in RuleEvaluator and are called via the private method
 * dispatch in executeAction().
 *
 * This file serves as the architectural boundary declaration — future
 * extraction of each action as a free function should land here.
 */
import type { World } from '../world';
import type { OrganismManager } from '../organism';
import type { CellCtx } from '../evaluator-context';

/** Shared context required by all eat/digest/transfer action implementations. */
export interface EatDigestActionDeps {
  world: World;
  envEnergy: Float32Array;
  digestRuleBoost: Float32Array;
  organisms: OrganismManager;
  metabolicScale: number;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
  setCellEnergyCapped: (x: number, y: number, energy: number, orgIdHint?: number) => number;
  stomachInflow: (idx: number, amount: number) => void;
  sameOrgNeighborRatioByIdx: (idx: number, orgId: number) => number;
  isBoundaryCell: (cell: CellCtx) => boolean;
  envGradient01At: (x: number, y: number) => number;
}

// Implementations remain in RuleEvaluator.ts pending full extraction.
// Each method signature:
//   actionEat(cell, maxGather, orgSize?): boolean
//   actionDigest(cell, org, rate): boolean
//   actionGive(cell, rate): boolean
//   actionTake(cell, maxPull): boolean
//   actionEmit(cell, actionParam, amount): void
//   actionSpill(cell, amount): boolean
