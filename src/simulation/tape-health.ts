import { ActionOpcode, CONDITIONS_OFFSET, MAX_RULES, MAX_VALID_ACTION_OPCODE, RULE_SIZE } from './tape';

/** Degradation thresholds (u8 0..255) for UI/reporting. */
export const TAPE_WEAR_WARN = 32;
export const TAPE_WEAR_DANGER = 96;

export type TapeWearLevel = 'ok' | 'warn' | 'danger';

export function wearLevelFromDegradation(d: number): TapeWearLevel {
  const x = d & 0xff;
  if (x > TAPE_WEAR_DANGER) return 'danger';
  if (x > TAPE_WEAR_WARN) return 'warn';
  return 'ok';
}

export function tapeDegradationSum(deg: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < deg.length; i++) sum += deg[i]!;
  return sum;
}

export function tapeDegradationPercent(deg: Uint8Array): number {
  const max = deg.length * 255;
  if (max <= 0) return 0;
  return (tapeDegradationSum(deg) / max) * 100;
}

export interface RuleOpcodeValidityCounts {
  invalid: number;
  nop: number;
  valid: number;
}

/** Raw rule table bytes: counts NOP / valid / invalid (invalid = opcode > MAX_VALID_ACTION_OPCODE). */
export function countInvalidRuleOpcodes(data: Uint8Array): RuleOpcodeValidityCounts {
  let invalid = 0;
  let nop = 0;
  let valid = 0;
  for (let r = 0; r < MAX_RULES; r++) {
    const op = data[CONDITIONS_OFFSET + r * RULE_SIZE + 2]!;
    if (op === ActionOpcode.NOP) nop++;
    else if (op > MAX_VALID_ACTION_OPCODE) invalid++;
    else valid++;
  }
  return { invalid, nop, valid };
}

