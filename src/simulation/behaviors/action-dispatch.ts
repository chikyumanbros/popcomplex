/**
 * Rule-table opcode → side effects (markers, feedback, delegated actions).
 * Signal / morph / FIRE nuances (including distress FIRE vs table FIRE): `docs/signal-morph-actions.md`.
 */
import { type Organism } from '../organism';
import { ActionOpcode, type ConditionRule, type ReadCtx, type Tape } from '../tape';

export interface DispatchCell {
  x: number;
  y: number;
  idx: number;
  energy: number;
  orgId: number;
}

type MarkerSlot = 0 | 1 | 2 | 3;

export interface MarkerConfig {
  eat: MarkerSlot;
  digest: MarkerSlot;
  signal: MarkerSlot;
  bump: number;
}

export interface ActionDispatchDeps {
  movedThisTick: Set<number>;
  markers: MarkerConfig;
  buildReadCtx: (cell: DispatchCell, org: Organism) => ReadCtx;
  writeFeedback: (org: Organism, slot: number, value: number) => void;
  bumpMarker: (idx: number, marker: MarkerSlot, amount: number) => void;
  collectNeighborEnvSum: (cell: DispatchCell) => number;
  actionEat: (cell: DispatchCell, strength: number, orgCellCount: number) => boolean;
  actionGive: (cell: DispatchCell, intensity01: number) => boolean;
  actionTake: (cell: DispatchCell, intensity: number) => boolean;
  actionDivide: (cell: DispatchCell, org: Organism, tape: Tape, minEnergy: number) => boolean;
  actionFire: (cell: DispatchCell, org: Organism) => boolean;
  actionMove: (cell: DispatchCell, org: Organism) => boolean;
  actionReproduce: (cell: DispatchCell, org: Organism, tape: Tape, childFraction: number) => boolean;
  actionAbsorb: (cell: DispatchCell, maxSteal: number) => boolean;
  actionDigest: (cell: DispatchCell, org: Organism, rate: number) => boolean;
  actionEmit: (cell: DispatchCell, actionParam: number, amount: number) => void;
  actionRepair: (cell: DispatchCell, org: Organism, tape: Tape, intensity: number) => boolean;
  actionSpill: (cell: DispatchCell, amount: number) => boolean;
  actionJam: (cell: DispatchCell, intensity: number) => boolean;
  actionApoptose: (cell: DispatchCell, rotBoost: number, energyDumpFrac: number, targetSelf: boolean) => boolean;
}

export function dispatchAction(
  rule: ConditionRule,
  cell: DispatchCell,
  org: Organism,
  tape: Tape,
  deps: ActionDispatchDeps,
): boolean {
  const ctx = deps.buildReadCtx(cell, org);
  const mod = tape.readModifier(rule.actionParam, ctx);
  switch (rule.actionOpcode) {
    case ActionOpcode.EAT: {
      const strength = (mod / 255) * 5;
      const ok = deps.actionEat(cell, strength, org.cells.size);
      if (!ok) return false;
      deps.bumpMarker(cell.idx, deps.markers.eat, deps.markers.bump);
      deps.writeFeedback(org, 0, strength * 51);
      return true;
    }
    case ActionOpcode.GIVE: {
      const ok = deps.actionGive(cell, mod / 255);
      if (ok) deps.writeFeedback(org, 3, mod);
      return ok;
    }
    case ActionOpcode.TAKE: {
      const ok = deps.actionTake(cell, (mod / 255) * 3);
      if (ok) deps.writeFeedback(org, 3, mod);
      return ok;
    }
    case ActionOpcode.DIV: {
      const ok = deps.actionDivide(cell, org, tape, Math.max(10, mod));
      if (ok) deps.writeFeedback(org, 4, mod);
      return ok;
    }
    case ActionOpcode.FIRE:
    case ActionOpcode.SIG: {
      if (!deps.actionFire(cell, org)) return false;
      deps.bumpMarker(cell.idx, deps.markers.signal, deps.markers.bump);
      deps.writeFeedback(org, 5, deps.collectNeighborEnvSum(cell) * 25.5);
      return true;
    }
    case ActionOpcode.MOVE: {
      if (deps.movedThisTick.has(org.id)) return false;
      if (!deps.actionMove(cell, org)) return false;
      deps.movedThisTick.add(org.id);
      deps.writeFeedback(org, 6, Math.min(255, deps.collectNeighborEnvSum(cell) * 12.75));
      return true;
    }
    case ActionOpcode.REPRODUCE: {
      const frac = Math.max(0.1, (mod / 255) * 0.5);
      const ok = deps.actionReproduce(cell, org, tape, frac);
      if (ok) deps.writeFeedback(org, 7, Math.min(255, cell.energy * frac * 2.55));
      return ok;
    }
    case ActionOpcode.ABSORB: {
      if (!deps.actionAbsorb(cell, Math.max(1, mod / 64))) return false;
      deps.bumpMarker(cell.idx, deps.markers.eat, deps.markers.bump);
      deps.writeFeedback(org, 2, mod);
      return true;
    }
    case ActionOpcode.DIGEST: {
      const rate = (mod / 255) * 0.6;
      if (!deps.actionDigest(cell, org, rate)) return false;
      deps.bumpMarker(cell.idx, deps.markers.digest, deps.markers.bump);
      deps.writeFeedback(org, 1, mod);
      return true;
    }
    case ActionOpcode.EMIT: {
      deps.actionEmit(cell, rule.actionParam, (mod / 255) * 8);
      deps.bumpMarker(cell.idx, deps.markers.signal, deps.markers.bump);
      return true;
    }
    case ActionOpcode.REPAIR: {
      if (!deps.actionRepair(cell, org, tape, (mod / 255) * 0.7)) return false;
      deps.bumpMarker(cell.idx, deps.markers.signal, deps.markers.bump);
      deps.writeFeedback(org, 1, Math.min(255, 36 + mod / 4));
      return true;
    }
    case ActionOpcode.SPILL: {
      const ok = deps.actionSpill(cell, (mod / 255) * 0.8);
      if (ok) deps.writeFeedback(org, 2, mod);
      return ok;
    }
    case ActionOpcode.JAM: {
      const ok = deps.actionJam(cell, mod / 255);
      if (ok) deps.writeFeedback(org, 5, mod);
      return ok;
    }
    case ActionOpcode.APOPTOSE: {
      // actionParam % 2: 0 → self-targeting, odd → weakest same-org neighbor
      const targetSelf = rule.actionParam % 2 === 0;
      const rotBoost      = 0.12 + (mod / 255) * 0.28; // 0.12..0.40
      const energyDumpFrac = 0.30 + (mod / 255) * 0.40; // 0.30..0.70
      const ok = deps.actionApoptose(cell, rotBoost, energyDumpFrac, targetSelf);
      if (ok) deps.writeFeedback(org, 2, mod); // slot 2: shared with SPILL/apoptose
      return ok;
    }
    default:
      return false;
  }
}
