/**
 * Action implementations: ABSORB, JAM, FIRE/SIG, MOVE, REPAIR, APOPTOSE.
 *
 * These actions handle foreign-cell interactions, neural signalling, organism
 * movement, tape immune repair, and apoptotic dismantling.  They are deeply
 * entangled with RuleEvaluator's internal state and many helper methods.
 * The deps interface below captures the full contract.
 *
 * This file serves as the architectural boundary declaration — future extraction
 * of each action as a free function should land here.
 */
import type { World } from '../world';
import type { OrganismManager } from '../organism';
import type { CellCtx } from '../evaluator-context';

/** Shared context required by combat/social action implementations. */
export interface CombatSocialActionDeps {
  world: World;
  envEnergy: Float32Array;
  organisms: OrganismManager;
  jamTicks: Uint8Array;
  groupSizeCache: Map<number, number>;
  movedThisTick: Set<number>;
  metabolicScale: number;
  suppressionEnabled: boolean;
  setCellEnergyCappedByIdx: (idx: number, energy: number, orgIdHint?: number) => number;
  setCellEnergyCapped: (x: number, y: number, energy: number, orgIdHint?: number) => number;
  stomachInflow: (idx: number, amount: number) => void;
  sameOrgNeighborRatioByIdx: (idx: number, orgId: number) => number;
  sameOrgConnectedGroupSize: (seed: number, orgId: number) => number;
  localSignalCohesion: (cell: CellCtx) => number;
  kinTrustForeign: (selfOrgId: number, foreignKinTag: number, foreignSignalMarker: number, foreignMorphA: number) => number;
  nudgeXenoTolerance: (orgId: number, delta: number) => void;
}

// Implementations remain in RuleEvaluator.ts pending full extraction.
// Each method signature:
//   actionAbsorb(cell, maxSteal): boolean
//   actionJam(cell, intensity): boolean
//   actionFire(cell, org?): boolean
//   actionMove(cell, org): boolean
//   actionRepair(cell, org, tape, intensity): boolean
//   actionApoptose(cell, org, tape, rotBoost, energyDumpFrac, targetSelf): boolean
