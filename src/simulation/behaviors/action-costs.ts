import { ActionOpcode } from '../tape';
import {
  ACTION_COST_FIRE, ACTION_COST_MOVE, ACTION_COST_REPRODUCE,
  ACTION_COST_ABSORB, ACTION_COST_DIGEST, ACTION_COST_TAKE,
  ACTION_COST_GIVE, ACTION_COST_EMIT, ACTION_COST_REPAIR,
  ACTION_COST_SPILL, ACTION_COST_JAM,
} from '../sim-constants';

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
    case ActionOpcode.APOPTOSE:
      return 0; // energy is recycled internally; no additional env heat cost
    default: {
      const _x: never = opcode;
      return _x;
    }
  }
}

export const MOVE_COST_PER_CELL = ACTION_COST_MOVE;
export const REPRODUCE_ACTION_COST = ACTION_COST_REPRODUCE;
