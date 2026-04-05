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
    case ActionOpcode.NOP:
      return 0;
    case ActionOpcode.DIV:
      return 0; // `actionDivide` pays `divCost` from cell energy on success
    case ActionOpcode.DIGEST:
      return ACTION_COST_DIGEST;
    case ActionOpcode.EAT:
      return 0; // paid as env draw + scan tax + wear; no extra heat on success
    case ActionOpcode.GIVE:
      return ACTION_COST_GIVE;
    case ActionOpcode.FIRE:
      return ACTION_COST_FIRE;
    case ActionOpcode.REPRODUCE:
      return 0; // `REPRODUCE_ACTION_COST` + child energy inside `actionReproduce`
    case ActionOpcode.ABSORB:
      return ACTION_COST_ABSORB;
    case ActionOpcode.SIG:
      return ACTION_COST_FIRE;
    case ActionOpcode.MOVE:
      return 0; // `distributeMoveCost` after shift
    case ActionOpcode.TAKE:
      return ACTION_COST_TAKE;
    case ActionOpcode.EMIT:
      return ACTION_COST_EMIT;
    case ActionOpcode.REPAIR:
      return ACTION_COST_REPAIR;
    case ActionOpcode.SPILL:
      return ACTION_COST_SPILL;
    case ActionOpcode.JAM:
      return ACTION_COST_JAM;
    default: {
      const _x: never = opcode;
      return _x;
    }
  }
}

export const MOVE_COST_PER_CELL = ACTION_COST_MOVE;
export const REPRODUCE_ACTION_COST = ACTION_COST_REPRODUCE;
