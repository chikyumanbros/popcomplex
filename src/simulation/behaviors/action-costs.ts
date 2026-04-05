import { ActionOpcode } from '../tape';

// Action energy costs (heat -> environment)
const ACTION_COST_FIRE = 0.4;
const ACTION_COST_MOVE = 0.5;
const ACTION_COST_REPRODUCE = 5.0;
const ACTION_COST_ABSORB = 0.3;
const ACTION_COST_DIGEST = 0.2;
const ACTION_COST_TAKE = 0.1;
const ACTION_COST_GIVE = 0.05;
const ACTION_COST_EMIT = 0.15;
const ACTION_COST_REPAIR = 0.28;
const ACTION_COST_SPILL = 0.08;
const ACTION_COST_JAM = 0.06;

export function getActionCostForOpcode(opcode: ActionOpcode): number {
  switch (opcode) {
    case ActionOpcode.FIRE:
      return ACTION_COST_FIRE;
    case ActionOpcode.SIG:
      return ACTION_COST_FIRE;
    case ActionOpcode.MOVE:
      return 0; // cost handled by distributeMoveCost
    case ActionOpcode.REPRODUCE:
      return 0; // cost handled internally
    case ActionOpcode.ABSORB:
      return ACTION_COST_ABSORB;
    case ActionOpcode.DIGEST:
      return ACTION_COST_DIGEST;
    case ActionOpcode.TAKE:
      return ACTION_COST_TAKE;
    case ActionOpcode.GIVE:
      return ACTION_COST_GIVE;
    case ActionOpcode.EMIT:
      return ACTION_COST_EMIT;
    case ActionOpcode.REPAIR:
      return ACTION_COST_REPAIR;
    case ActionOpcode.SPILL:
      return ACTION_COST_SPILL;
    case ActionOpcode.JAM:
      return ACTION_COST_JAM;
    default:
      return 0;
  }
}

export const MOVE_COST_PER_CELL = ACTION_COST_MOVE;
export const REPRODUCE_ACTION_COST = ACTION_COST_REPRODUCE;
