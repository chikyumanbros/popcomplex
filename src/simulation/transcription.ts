import {
  Tape,
  TAPE_SIZE,
  CONDITIONS_OFFSET,
  RULE_SIZE,
  MAX_RULES,
  REPLICATION_KEY_OFFSET,
  REPLICATION_KEY_LEN,
  RULE_TABLE_END,
  LINEAGE_BYTE_0,
  LINEAGE_BYTE_1,
  LINEAGE_BYTE_2,
  LINEAGE_BYTE_AUX,
  GENETIC_KIN_BYTE_0,
  GENETIC_KIN_BYTE_1,
  GENETIC_KIN_BYTE_2,
  GENETIC_KIN_BYTE_AUX,
} from './tape';
import { randomF32, randomInt } from './rng';
import {
  recordChannelBitflip,
  recordChannelSwap,
  recordDataDuplication,
  recordReadMisfetch,
  recordRuleDuplication,
  recordRuleSwap,
  recordStillbirth,
  recordWriteRandomization,
} from './telemetry';

const WRITE_ERROR_RATE = 0.0008; // 0.08% per byte (full random replace)
/** Per-byte XOR bit-flip in length-preserving channel (replaces RLE: entropy / runs no longer affect transcript stability). */
const CHANNEL_BITFLIP_RATE = 0.0011;
/** Bernoulli trial each transcript: then acceptance uses region/key/rule multipliers below. */
export const CHANNEL_SWAP_BASE_PROB = 0.018;
/**
 * Random channel swap can still jump arbitrary pairs; these scale **acceptance** (<1 = rarer).
 * Cross-region swaps (data vs CA vs rules vs NN) are damped; touching replication key or rule table is damped further.
 */
export const CHANNEL_SWAP_ACCEPT_CROSS_REGION_MULT = 0.22;
export const CHANNEL_SWAP_ACCEPT_REPL_KEY_MULT = 0.32;
export const CHANNEL_SWAP_ACCEPT_RULE_TABLE_MULT = 0.48;
const READ_FETCH_ERROR = 0.00008; // 0.008% adjacent mis-fetch

const RULE_DUP_RATE  = 0.03;  // 3% chance per rule to duplicate to another slot
const RULE_SWAP_RATE = 0.015; // 1.5% chance to swap two rules
const DATA_DUP_RATE  = 0.02;  // 2% chance per data byte to copy to random data slot
const LINEAGE_DRIFT_RATE = 0.02; // per public / private kin byte after transcription
const GENETIC_KIN_DRIFT_RATE = 0.017; // slightly slower than face — mimicry can persist longer

/** Extra write/read noise on replication-key bytes scales with parent degradation on those bytes. */
const KEY_WEAR_WRITE_MULT = 3.2;
const KEY_WEAR_CHANNEL_MULT = 3.2;
const KEY_WEAR_READ_MULT = 2.5;

/** Stillbirth after transcribe: copy can be noisy but sometimes fails entirely (higher with key damage + parent wear). */
const REPL_ABORT_BASE = 0.026;
const REPL_ABORT_BIT = 0.4;
const REPL_ABORT_DEG = 0.2;
const REPL_ABORT_CROSS = 0.38;
const REPL_ABORT_CAP = 0.6;

/** Full genomic copy for non-reproduction uses (e.g. future clone paths). */
export function transcribe(parent: Tape): Tape {
  const data = transcribeCore(parent);
  return new Tape(data);
}

/**
 * Copy for reproduction: same noise as `transcribe`, then viability gate on replication-key fidelity vs parent wear.
 * Returns null = mitosis failed after spending attempt cost in caller.
 */
export function transcribeForReproduction(parent: Tape): Tape | null {
  const data = transcribeCore(parent);
  if (replicationAborted(parent, data)) return null;
  return new Tape(data);
}

function transcribeCore(parent: Tape): Uint8Array {
  const deg = parent.degradation;
  let data = writePhase(parent.data, deg);
  data = channelPhase(data, deg);
  data = readPhase(data, deg);

  if (data.length !== TAPE_SIZE) {
    if (data.length > TAPE_SIZE) data = data.slice(0, TAPE_SIZE);
    else {
      const padded = new Uint8Array(TAPE_SIZE);
      padded.set(data);
      data = padded;
    }
  }

  structuralMutations(data);
  applyLineageDrift(data);
  return data;
}

function applyLineageDrift(data: Uint8Array) {
  const publicKinBytes = [LINEAGE_BYTE_0, LINEAGE_BYTE_1, LINEAGE_BYTE_2, LINEAGE_BYTE_AUX];
  for (const idx of publicKinBytes) {
    if (randomF32() < LINEAGE_DRIFT_RATE) {
      data[idx] ^= 1 << randomInt(8);
    }
  }
  const geneticKinBytes = [
    GENETIC_KIN_BYTE_0,
    GENETIC_KIN_BYTE_1,
    GENETIC_KIN_BYTE_2,
    GENETIC_KIN_BYTE_AUX,
  ];
  for (const idx of geneticKinBytes) {
    if (randomF32() < GENETIC_KIN_DRIFT_RATE) {
      data[idx] ^= 1 << randomInt(8);
    }
  }
}

function popcountByte(b: number): number {
  let x = b & 0xff;
  let n = 0;
  while (x) {
    n += x & 1;
    x >>>= 1;
  }
  return n;
}

function replicationAborted(parent: Tape, childData: Uint8Array): boolean {
  let mismatchBits = 0;
  let degSum = 0;
  for (let k = 0; k < REPLICATION_KEY_LEN; k++) {
    const idx = REPLICATION_KEY_OFFSET + k;
    mismatchBits += popcountByte(parent.data[idx] ^ childData[idx]);
    degSum += parent.degradation[idx];
  }
  const degMean = degSum / (REPLICATION_KEY_LEN * 255);
  const bitPen = mismatchBits / (REPLICATION_KEY_LEN * 8);
  const failP = Math.min(
    REPL_ABORT_CAP,
    REPL_ABORT_BASE
      + REPL_ABORT_BIT * bitPen
      + REPL_ABORT_DEG * degMean
      + REPL_ABORT_CROSS * bitPen * degMean,
  );
  const aborted = randomF32() < failP;
  if (aborted) recordStillbirth();
  return aborted;
}

function structuralMutations(data: Uint8Array) {
  // Rule duplication: copy one rule's bytes to another slot
  for (let r = 0; r < MAX_RULES; r++) {
    if (randomF32() < RULE_DUP_RATE) {
      const target = randomInt(MAX_RULES);
      if (target === r) continue;
      const srcOff = CONDITIONS_OFFSET + r * RULE_SIZE;
      const dstOff = CONDITIONS_OFFSET + target * RULE_SIZE;
      for (let b = 0; b < RULE_SIZE; b++) {
        data[dstOff + b] = data[srcOff + b];
      }
      recordRuleDuplication();
    }
  }

  // Rule swap: exchange two rules entirely
  if (randomF32() < RULE_SWAP_RATE) {
    const a = randomInt(MAX_RULES);
    const b = randomInt(MAX_RULES);
    if (a !== b) {
      const offA = CONDITIONS_OFFSET + a * RULE_SIZE;
      const offB = CONDITIONS_OFFSET + b * RULE_SIZE;
      for (let i = 0; i < RULE_SIZE; i++) {
        const tmp = data[offA + i];
        data[offA + i] = data[offB + i];
        data[offB + i] = tmp;
      }
      recordRuleSwap();
    }
  }

  // Data region byte duplication: copy one data byte to another data slot.
  // Protect only byte 4 (maxCells). Bytes 5-6 are active operation nodes and must remain evolvable.
  for (let i = 0; i < 32; i++) {
    if (randomF32() < DATA_DUP_RATE) {
      const target = randomInt(32);
      if (isProtectedDataDupTarget(target)) continue;
      data[target] = data[i];
      recordDataDuplication();
    }
  }
}

export function isProtectedDataDupTarget(target: number): boolean {
  return target === 4;
}

function keyWearBoost(degradation: Uint8Array, i: number): number {
  if (i < REPLICATION_KEY_OFFSET || i >= REPLICATION_KEY_OFFSET + REPLICATION_KEY_LEN) return 1;
  return 1 + (degradation[i] / 255) * KEY_WEAR_WRITE_MULT;
}

/** 0=data 0–31, 1=CA 32–63, 2=rules 64–127, 3=NN 128–255 */
function tapeCoarseRegion(i: number): number {
  if (i < 32) return 0;
  if (i < 64) return 1;
  if (i < 128) return 2;
  return 3;
}

function touchesReplicationKey(i: number): boolean {
  return i >= REPLICATION_KEY_OFFSET && i < REPLICATION_KEY_OFFSET + REPLICATION_KEY_LEN;
}

function touchesRuleTable(i: number): boolean {
  return i >= CONDITIONS_OFFSET && i < RULE_TABLE_END;
}

function channelSwapAcceptanceMultiplier(a: number, b: number): number {
  let m = 1;
  if (tapeCoarseRegion(a) !== tapeCoarseRegion(b)) m *= CHANNEL_SWAP_ACCEPT_CROSS_REGION_MULT;
  if (touchesReplicationKey(a) || touchesReplicationKey(b)) m *= CHANNEL_SWAP_ACCEPT_REPL_KEY_MULT;
  if (touchesRuleTable(a) || touchesRuleTable(b)) m *= CHANNEL_SWAP_ACCEPT_RULE_TABLE_MULT;
  return m;
}

function writePhase(source: Uint8Array, degradation: Uint8Array): Uint8Array {
  const copy = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i++) {
    copy[i] = source[i];
    const rate = WRITE_ERROR_RATE * keyWearBoost(degradation, i);
    if (randomF32() < rate) {
      copy[i] = randomInt(256);
      recordWriteRandomization();
    }
  }
  return copy;
}

function channelPhase(data: Uint8Array, degradation: Uint8Array): Uint8Array {
  const out = new Uint8Array(data);
  const n = TAPE_SIZE;
  for (let i = 0; i < n; i++) {
    let rate = CHANNEL_BITFLIP_RATE;
    if (i >= REPLICATION_KEY_OFFSET && i < REPLICATION_KEY_OFFSET + REPLICATION_KEY_LEN) {
      rate *= 1 + (degradation[i] / 255) * KEY_WEAR_CHANNEL_MULT;
    }
    if (randomF32() < rate) {
      out[i] ^= 1 << randomInt(8);
      recordChannelBitflip();
    }
  }
  if (randomF32() < CHANNEL_SWAP_BASE_PROB) {
    let a = randomInt(n);
    let b = randomInt(n);
    if (a === b) b = (b + 1 + randomInt(n - 1)) % n;
    const mult = channelSwapAcceptanceMultiplier(a, b);
    if (randomF32() < mult) {
      const t = out[a];
      out[a] = out[b];
      out[b] = t;
      recordChannelSwap(true);
    } else {
      recordChannelSwap(false);
    }
  }
  return out;
}

function readPhase(data: Uint8Array, degradation: Uint8Array): Uint8Array {
  const result = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    let fetchErr = READ_FETCH_ERROR;
    if (i >= REPLICATION_KEY_OFFSET && i < REPLICATION_KEY_OFFSET + REPLICATION_KEY_LEN) {
      fetchErr *= 1 + (degradation[i] / 255) * KEY_WEAR_READ_MULT;
    }
    if (randomF32() < fetchErr && i + 1 < data.length) {
      result[i] = data[i + 1]; // read adjacent byte
      recordReadMisfetch();
    } else {
      result[i] = data[i];
    }
  }
  return result;
}
