import { randomF32, randomInt } from './rng';
import { recordTapeCorruption } from './telemetry';

export const TAPE_SIZE = 256;

/** Save / handoff wire: `data[256]` then `degradation[256]` — fixed length, no RLE, easy parse. */
export const TAPE_SNAPSHOT_BYTES = TAPE_SIZE * 2;

/**
 * Tape layout (`data` is 256 bytes; `degradation` mirrors each index):
 * - **0–31** — Data region: operation nodes 0–15, literal pool 16–31; **byte 4** = maxCells (structural).
 * - **28–30** — Lineage pigment (evolvable); **31** = lineage aux (mixed into `getLineagePacked` 24-bit tag).
 * - **32–63** — CA band: refractory = low nibble of 32; **48–59** = energy storage code bank; **60–63** replication key.
 * - **64–127** — Condition→action rules: 16 slots × 4 bytes (flags, threshold, opcode, actionParam node index).
 * - **128–255** — NN parameters (pairs of bytes → float; see `getNNWeights`).
 *
 * Offspring from `transcribe` / `transcribeForReproduction` get **fresh** `degradation` (zeros); parent wear only biases copy noise.
 *
 * **`TAPE_SIZE` is not plug-and-play**: region sizes (`CA_RULES_SIZE`, rule table, NN block) are fixed numbers that must sum to `TAPE_SIZE`. Raising `TAPE_SIZE` without reallocating offsets breaks layout and snapshot codecs.
 */
export const CA_RULES_OFFSET = 32;
const CA_RULES_SIZE = 32;
export const CONDITIONS_OFFSET = 64;
const CONDITIONS_SIZE = 64;
const NN_WEIGHTS_OFFSET = 128;
const NN_WEIGHTS_SIZE = 128;

/** Exclusive end of rule bytes in `data` (same as start of NN weight bytes). */
export const RULE_TABLE_END = CONDITIONS_OFFSET + CONDITIONS_SIZE;

/** u16 big-endian from two tape bytes is centered here, then divided by `NN_TAPE_WEIGHT_SCALE`. */
export const NN_TAPE_WEIGHT_CENTER = 32768;
/** Maps full u16 span to roughly **±2.0** float NN params (`getNNWeights`). Tune for experiments. */
export const NN_TAPE_WEIGHT_SCALE = 16384;

/** LCG seed for canonical proto-tape NN bytes (same initial life every page load). */
export const PROTO_TAPE_NN_SEED = 0x50fc0d65;
export const RULE_SIZE = 4;
export const MAX_RULES = CONDITIONS_SIZE / RULE_SIZE;

/** 3 bytes in data region (literal pool) — evolvable “pigment / kinship” for rendering; near relatives share similar values. */
export const LINEAGE_BYTE_0 = 28;
export const LINEAGE_BYTE_1 = 29;
export const LINEAGE_BYTE_2 = 30;
/** Fourth lineage byte: XOR-mixed into the 24-bit packed lineage hue (byte 31 in data region). */
export const LINEAGE_BYTE_AUX = 31;

/**
 * Replication “key” in the CA padding band (byte 32 low nibble is refractory; 48–59 is energy-cap bank).
 * Offspring must pass a viability roll after transcribe; parent wear on these bytes increases copy noise and abort odds.
 */
export const REPLICATION_KEY_OFFSET = 60;
export const REPLICATION_KEY_LEN = 4;
export const ENERGY_CAP_BANK_OFFSET = 48;
export const ENERGY_CAP_BANK_LEN = 12;
export const ENERGY_CAP_BASE = 8;
export const ENERGY_CAP_PER_MODULE = 8;
export const ENERGY_CAP_MAX = 160;

export enum ActionOpcode {
  NOP = 0x00,
  DIV = 0x01,
  DIGEST = 0x02,
  EAT = 0x03,
  GIVE = 0x04,
  FIRE = 0x05,
  REPRODUCE = 0x06,
  ABSORB = 0x07,
  SIG = 0x08,
  MOVE = 0x09,
  TAKE = 0x0A,
  EMIT = 0x0B,     // emit morphogen (channel = actionParam % 2)
  REPAIR = 0x0C,   // immune: mend tape; neighbor quorum boosts success (same-org + partial foreign “kin” if lineage+signal+morph align)
  SPILL = 0x0D,    // spill local stomach content into nearby environment (1-hop redistribution)
  JAM = 0x0E,      // short-lived boundary disconnection against cross-lineage coupling
}

/** Highest defined `ActionOpcode` value (inclusive). Raw bytes above this are invalid until REPAIR or mutation. */
export const MAX_VALID_ACTION_OPCODE = ActionOpcode.JAM;
export const ENERGY_CAP_MODULE_ORDER: readonly ActionOpcode[] = [
  ActionOpcode.DIV,
  ActionOpcode.DIGEST,
  ActionOpcode.EAT,
  ActionOpcode.GIVE,
  ActionOpcode.FIRE,
  ActionOpcode.REPRODUCE,
  ActionOpcode.ABSORB,
  ActionOpcode.SIG,
  ActionOpcode.MOVE,
  ActionOpcode.TAKE,
  ActionOpcode.EMIT,
  ActionOpcode.REPAIR,
];
const ENERGY_CAP_MODULE_CODE_XOR = 0xA5;

function energyCapModuleCode(op: ActionOpcode): number {
  return ((op & 0xff) ^ ENERGY_CAP_MODULE_CODE_XOR) & 0xff;
}

/** Map corrupt / unknown opcode bytes to NOP so rule evaluation matches “dead row” semantics. */
export function normalizeActionOpcode(byte: number): ActionOpcode {
  const b = byte & 0xff;
  if (b >= ActionOpcode.NOP && b <= MAX_VALID_ACTION_OPCODE) return b as ActionOpcode;
  return ActionOpcode.NOP;
}

/** Decode one NN weight from adjacent high/low tape bytes (see `NN_TAPE_WEIGHT_*`). */
export function decodeNNWeightBytes(hi: number, lo: number): number {
  const u = ((hi & 0xff) << 8) | (lo & 0xff);
  return (u - NN_TAPE_WEIGHT_CENTER) / NN_TAPE_WEIGHT_SCALE;
}

/**
 * Multiplier on per-tick / wear corruption **probability** (lower = sturdier byte).
 * Ecological tuning: structural + rule-opcode + replication key bits erode more slowly than generic data.
 */
export const TAPE_DEGRAD_SENS_MAXCELLS = 0.32;
export const TAPE_DEGRAD_SENS_ENERGY_CAP_BANK = 0.22;
export const TAPE_DEGRAD_SENS_REPLICATION_KEY = 0.28;
export const TAPE_DEGRAD_SENS_RULE_OPCODE = 0.4;
export const TAPE_DEGRAD_SENS_REFRACTORY = 0.52;
export const TAPE_DEGRAD_SENS_NN = 0.82;
export const TAPE_DEGRAD_SENS_DEFAULT = 1.0;

export function tapeByteDegradationSensitivity(index: number): number {
  const i = index | 0;
  if (i === 4) return TAPE_DEGRAD_SENS_MAXCELLS;
  if (i >= ENERGY_CAP_BANK_OFFSET && i < ENERGY_CAP_BANK_OFFSET + ENERGY_CAP_BANK_LEN) {
    return TAPE_DEGRAD_SENS_ENERGY_CAP_BANK;
  }
  if (i >= REPLICATION_KEY_OFFSET && i < REPLICATION_KEY_OFFSET + REPLICATION_KEY_LEN) {
    return TAPE_DEGRAD_SENS_REPLICATION_KEY;
  }
  if (i >= CONDITIONS_OFFSET && i < RULE_TABLE_END) {
    const rel = (i - CONDITIONS_OFFSET) % RULE_SIZE;
    if (rel === 2) return TAPE_DEGRAD_SENS_RULE_OPCODE;
  }
  if (i === CA_RULES_OFFSET) return TAPE_DEGRAD_SENS_REFRACTORY;
  if (i >= NN_WEIGHTS_OFFSET) return TAPE_DEGRAD_SENS_NN;
  return TAPE_DEGRAD_SENS_DEFAULT;
}

export interface ConditionRule {
  conditionFlags: number;
  thresholdValue: number;
  actionOpcode: ActionOpcode;
  actionParam: number;
}

export const FEEDBACK_SLOTS = 8;

export interface ReadCtx {
  cellEnergy: number;
  stomach: number;
  orgCells: number;
  feedback: Uint8Array;
}

export class Tape {
  data: Uint8Array;
  degradation: Uint8Array;

  constructor(data?: Uint8Array) {
    this.data = data ? new Uint8Array(data) : new Uint8Array(TAPE_SIZE);
    this.degradation = new Uint8Array(TAPE_SIZE);
  }

  // Compute a parameter by interpreting the data byte at `slot` as an
  // operation node: [op:3][arg:5].  The arg indexes into the full data
  // region (0-31), where bytes 0-15 are operation nodes and 16-31 are
  // a literal pool.
  //
  //  op 0 LITERAL   – data[arg]  (backward-compatible static value)
  //  op 1 ADD       – (data[arg] + data[(arg+1)%32]) & 0xFF
  //  op 2 SELF_E    – min(255, cellEnergy)
  //  op 3 SCALE_E   – data[arg] * cellEnergy / 255
  //  op 4 SCALE_S   – data[arg] * stomach / 255
  //  op 5 IF_RICH   – cellEnergy > data[arg] ? data[(arg+1)%32] : data[(arg+2)%32]
  //  op 6 FEEDBACK  – feedback[arg % 8]
  //  op 7 SCALE_N   – data[arg] * orgCells / 64
  readModifier(actionParam: number, ctx?: ReadCtx): number {
    const slot = actionParam % 32;
    const opByte = this.data[slot];
    const op = (opByte >> 5) & 0x07;
    const arg = opByte & 0x1F;

    if (!ctx) return this.data[arg]; // fallback: pure literal lookup

    switch (op) {
      case 0: return this.data[arg];
      case 1: return (this.data[arg] + this.data[(arg + 1) % 32]) & 0xFF;
      case 2: return Math.min(255, Math.floor(ctx.cellEnergy));
      case 3: return Math.floor(this.data[arg] * Math.min(255, ctx.cellEnergy) / 255) & 0xFF;
      case 4: return Math.floor(this.data[arg] * Math.min(255, ctx.stomach) / 255) & 0xFF;
      case 5: return ctx.cellEnergy > this.data[arg]
        ? this.data[(arg + 1) % 32]
        : this.data[(arg + 2) % 32];
      case 6: return ctx.feedback[arg % FEEDBACK_SLOTS];
      case 7: return Math.min(255, Math.floor(this.data[arg] * ctx.orgCells / 64));
      default: return this.data[arg];
    }
  }

  // Structural reads from specific tape regions (not evolvable "parameters")
  getRefractoryPeriod(): number {
    return Math.max(1, this.data[CA_RULES_OFFSET] & 0x0F);
  }

  getMaxCells(): number {
    return Math.max(2, this.data[4]);
  }

  /**
   * Per-cell storage cap decoded from a dedicated tape bank.
   *
   * Encoding (CA band):
   * - Bytes 48..59: each slot has one canonical byte for one module kind.
   *   If the byte equals that module's expected code, the module is "surviving"; otherwise broken/empty.
   *
   * Capacity = ENERGY_CAP_BASE + survivingModules * ENERGY_CAP_PER_MODULE (capped at ENERGY_CAP_MAX).
   * Runtime combines this with "recently functioning" modules in `RuleEvaluator`.
   */
  getMaxCellEnergy(): number {
    let surviving = 0;
    const mask = this.getSurvivingEnergyModuleMask();
    let m = mask;
    while (m !== 0) {
      surviving++;
      m &= m - 1;
    }
    const cap = ENERGY_CAP_BASE + surviving * ENERGY_CAP_PER_MODULE;
    return Math.min(ENERGY_CAP_MAX, cap);
  }

  /** Bit i is 1 when energy-cap module slot i still has its canonical code (surviving). */
  getSurvivingEnergyModuleMask(): number {
    let mask = 0;
    for (let i = 0; i < ENERGY_CAP_MODULE_ORDER.length; i++) {
      const expected = energyCapModuleCode(ENERGY_CAP_MODULE_ORDER[i]!);
      const actual = this.data[ENERGY_CAP_BANK_OFFSET + i] & 0xff;
      if (actual === expected) mask |= 1 << i;
    }
    return mask >>> 0;
  }

  /** 24-bit kinship tag from bytes 28–31 (aux byte 31 XOR-mixed); transcribe + drift → similar hue for clades. */
  getLineagePacked(): number {
    const a = this.data[LINEAGE_BYTE_0] & 0xff;
    const b = this.data[LINEAGE_BYTE_1] & 0xff;
    const c = this.data[LINEAGE_BYTE_2] & 0xff;
    const aux = this.data[LINEAGE_BYTE_AUX] & 0xff;
    let base = ((a << 16) | (b << 8) | c) >>> 0;
    base ^= Math.imul(aux, 0x9e3779b9) >>> 0;
    return base & 0xffffff;
  }

  getRule(index: number): ConditionRule {
    if (index >= MAX_RULES) throw new Error(`Rule index ${index} out of range`);
    const off = CONDITIONS_OFFSET + index * RULE_SIZE;
    return {
      conditionFlags: this.data[off],
      thresholdValue: this.data[off + 1],
      actionOpcode: normalizeActionOpcode(this.data[off + 2]),
      actionParam: this.data[off + 3],
    };
  }

  getRuleCount(): number { return MAX_RULES; }

  /**
   * Decodes the full NN parameter block (64 floats):
   * [IH weights][HO weights][input gains][hidden bias][output bias].
   */
  getNNWeights(): Float32Array {
    const count = NN_WEIGHTS_SIZE / 2;
    const weights = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const hi = this.data[NN_WEIGHTS_OFFSET + i * 2];
      const lo = this.data[NN_WEIGHTS_OFFSET + i * 2 + 1];
      weights[i] = decodeNNWeightBytes(hi, lo);
    }
    return weights;
  }

  // avgEnergy: average cell energy for this organism (0-255 scale)
  // age: organism age in ticks
  // Both parameters scale the degradation rate:
  //   - low energy → faster degradation (entropy increases when starving)
  //   - old age → faster degradation (accumulated wear)
  //   - high existing degradation → cascade (fragile bytes break more)
  applyReadDegradation(avgEnergy = 50, age = 0) {
    const BASE_RATE = 0.0002;
    const energyFactor = 2 - Math.min(1, avgEnergy / 50);   // 1.0 (healthy) → 2.0 (starving)
    const ageFactor = 1 + Math.min(2, age * 0.0005);        // 1.0 (young) → 3.0 (age 4000+)
    const rate = BASE_RATE * energyFactor * ageFactor;

    for (let i = 0; i < TAPE_SIZE; i++) {
      const fragility = this.degradation[i] > 64 ? 1.5 : 1; // cascade: damaged bytes break more
      const sens = tapeByteDegradationSensitivity(i);
      if (randomF32() < rate * fragility * sens) this.corruptByte(i);
    }
  }

  applyActionWear(ruleIndex: number) {
    const RATE = 0.0003;
    const off = CONDITIONS_OFFSET + ruleIndex * RULE_SIZE;
    for (let b = 0; b < RULE_SIZE; b++) {
      const idx = off + b;
      if (randomF32() < RATE * tapeByteDegradationSensitivity(idx)) this.corruptByte(idx);
    }
    // Wear also propagates to the referenced data byte
    const actionParam = this.data[off + 3];
    const slot = actionParam % 32;
    if (randomF32() < RATE * 0.5 * tapeByteDegradationSensitivity(slot)) this.corruptByte(slot);
  }

  private corruptByte(index: number) {
    const bitPos = randomInt(8);
    this.data[index] ^= (1 << bitPos);
    this.degradation[index] = Math.min(255, this.degradation[index] + 16);
    recordTapeCorruption();
  }

  clone(): Tape {
    const t = new Tape(new Uint8Array(this.data));
    t.degradation.set(this.degradation);
    return t;
  }
}

const COND_TARGET = ['self', 'org', 'nb', 'env'];
const COND_ITEM: string[][] = [
  ['e', 'mood', 'mA', 'mB'],
  ['cells', 'totE', 'age', 'avgA'],
  ['same', 'for', 'emp', 'out'],
  ['here', 'grad', 'maxE', 'dom'],
];
const COND_CMP = ['>', '<', '≈', '≠'];

function formatConditionDisasm(flags: number, thr: number): string {
  const target = flags & 0x03;
  const item = (flags >> 2) & 0x03;
  const comp = (flags >> 4) & 0x03;
  const chain = (flags & 0x40) !== 0;
  const t = COND_TARGET[target]!;
  const it = COND_ITEM[target]![item]!;
  const c = COND_CMP[comp] ?? '?';
  return `${t}.${it} ${c} ${thr}${chain ? ' [chain]' : ''}`;
}

function opcodeDisasmName(op: ActionOpcode): string {
  switch (op) {
    case ActionOpcode.NOP: return 'NOP';
    case ActionOpcode.DIV: return 'DIV';
    case ActionOpcode.DIGEST: return 'DIGEST';
    case ActionOpcode.EAT: return 'EAT';
    case ActionOpcode.GIVE: return 'GIVE';
    case ActionOpcode.FIRE: return 'FIRE';
    case ActionOpcode.REPRODUCE: return 'REPRO';
    case ActionOpcode.ABSORB: return 'ABSORB';
    case ActionOpcode.SIG: return 'SIG';
    case ActionOpcode.MOVE: return 'MOVE';
    case ActionOpcode.TAKE: return 'TAKE';
    case ActionOpcode.EMIT: return 'EMIT';
    case ActionOpcode.REPAIR: return 'REPAIR';
    case ActionOpcode.SPILL: return 'SPILL';
    case ActionOpcode.JAM: return 'JAM';
    default: return `0x${Number(op).toString(16)}`;
  }
}

/** Shift+inspector: human-readable rule table (normalized opcode; raw byte shown if corrupt). */
export function formatTapeRulesInspectorHtml(tape: Tape): string {
  const rows: string[] = [];
  for (let i = 0; i < MAX_RULES; i++) {
    const off = CONDITIONS_OFFSET + i * RULE_SIZE;
    const rawOp = tape.data[off + 2] & 0xff;
    const norm = normalizeActionOpcode(rawOp);
    const flags = tape.data[off];
    const thr = tape.data[off + 1];
    const param = tape.data[off + 3];
    const cond = formatConditionDisasm(flags, thr);
    const rawNote =
      rawOp !== norm
        ? ` <span class="ins-rawop">raw 0x${rawOp.toString(16).padStart(2, '0')}</span>`
        : '';
    rows.push(
      `<div class="ins-rule-line"><span class="ins-ri">${String(i).padStart(2, '0')}</span> ` +
        `if ${cond} → <b>${opcodeDisasmName(norm)}</b> n${param}${rawNote}</div>`,
    );
  }
  return `<div class="ins-rules"><div class="ins-rules-hdr">── rules (disasm) ──</div>${rows.join('')}</div>`;
}

/** Base64 of 512 bytes: indices 0..255 = tape data, 256..511 = degradation (u8). */
export function tapeSnapshotBase64(tape: Tape): string {
  const buf = new Uint8Array(TAPE_SNAPSHOT_BYTES);
  buf.set(tape.data, 0);
  buf.set(tape.degradation, TAPE_SIZE);
  let bin = '';
  for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
  return btoa(bin);
}

/** Inverse of `tapeSnapshotBase64`; returns null if length or encoding is wrong. */
export function decodeTapeSnapshotBase64(b64: string): { data: Uint8Array; degradation: Uint8Array } | null {
  try {
    const bin = atob(b64.trim());
    if (bin.length !== TAPE_SNAPSHOT_BYTES) return null;
    const data = new Uint8Array(TAPE_SIZE);
    const degradation = new Uint8Array(TAPE_SIZE);
    for (let i = 0; i < TAPE_SIZE; i++) {
      data[i] = bin.charCodeAt(i) & 0xff;
      degradation[i] = bin.charCodeAt(TAPE_SIZE + i) & 0xff;
    }
    return { data, degradation };
  } catch {
    return null;
  }
}

/** Rebuild a `Tape` from a snapshot (e.g. after `decodeTapeSnapshotBase64`). */
export function tapeFromSnapshot(data: Uint8Array, degradation: Uint8Array): Tape {
  const t = new Tape(new Uint8Array(data));
  t.degradation.set(degradation);
  return t;
}

// Helper: encode a data-region operation node byte
function opNode(op: number, arg: number): number {
  return ((op & 0x07) << 5) | (arg & 0x1F);
}

function encodeEnergyCapBank(
  data: Uint8Array,
  activeModules: number,
): void {
  const modules = Math.max(0, Math.min(ENERGY_CAP_MODULE_ORDER.length, activeModules));
  for (let i = 0; i < ENERGY_CAP_BANK_LEN; i++) {
    const idx = ENERGY_CAP_BANK_OFFSET + i;
    if (i < modules) {
      data[idx] = energyCapModuleCode(ENERGY_CAP_MODULE_ORDER[i]!);
    } else {
      data[idx] = 0; // empty capacity slot
    }
  }
}

export function createProtoTape(): Tape {
  const tape = new Tape();
  const d = tape.data;

  // === Data region (bytes 0-31) ===
  //
  // Bytes 0-15: operation nodes [op:3][arg:5]
  //   op=0 LITERAL   arg→data[arg]
  //   op=3 SCALE_E   arg→data[arg] * cellEnergy / 255
  //   op=4 SCALE_S   arg→data[arg] * stomach / 255
  //   op=5 IF_RICH   cellEnergy > data[arg] ? data[arg+1] : data[arg+2]
  //   op=6 FEEDBACK  feedback[arg % 8]
  //   op=7 SCALE_N   arg→data[arg] * orgCells / 64
  //
  // Bytes 16-31: literal value pool (raw constants referenced by op nodes)

  // -- structural slots (not operation nodes) --
  d[4]  = 100;  // maxCells (structural, read directly by getMaxCells())

  // -- operation nodes (slots 0-15, referenced by rule actionParam) --
  d[0]  = opNode(0, 16);  // LITERAL → d[16]=140 (EAT power)
  d[1]  = opNode(4, 17);  // SCALE_S → d[17]=128 * stomach/255 (DIGEST: hungry=strong)
  d[2]  = opNode(7, 18);  // SCALE_N → d[18]=25 * orgCells/64 (DIV: grows with colony)
  d[3]  = opNode(0, 19);  // LITERAL → d[19]=90 (REPRODUCE fraction)
  // d[4] reserved for maxCells
  d[5]  = opNode(0, 20);  // LITERAL → d[20]=16 (ABSORB amount)
  d[6]  = opNode(3, 21);  // SCALE_E → d[21]=50 * energy/255 (TAKE: desperate=weak)
  d[7]  = opNode(5, 22);  // IF_RICH → e>d[22]=40 ? d[23]=100 : d[24]=30 (GIVE: generous if rich)
  d[8]  = opNode(0, 25);  // LITERAL → d[25]=180 (EMIT morphA)
  d[9]  = opNode(0, 26);  // LITERAL → d[26]=100 (EMIT morphB)
  d[10] = opNode(6, 0);   // FEEDBACK → feedback[0] (last EAT result)
  d[11] = opNode(0, 27);  // LITERAL → d[27]=77 (SHR rate for GIVE)
  d[12] = opNode(3, 16);  // SCALE_E → d[16]=140 * energy/255 (outer EAT: rich eat harder)

  // -- literal value pool (slots 16-31) --
  d[16] = 140;  // EAT power / also referenced by SCALE_E ops
  d[17] = 128;  // DIGEST base rate
  d[18] = 25;   // DIV cost base
  d[19] = 90;   // REPRODUCE fraction
  d[20] = 16;   // ABSORB amount
  d[21] = 50;   // TAKE amount
  d[22] = 40;   // IF_RICH threshold for GIVE
  d[23] = 100;  // IF_RICH true branch (generous GIVE)
  d[24] = 30;   // IF_RICH false branch (frugal GIVE)
  d[25] = 180;  // EMIT morphA strength
  d[26] = 100;  // EMIT morphB strength
  d[27] = 77;   // SHR/GIVE rate
  // Lineage / kinship pigment (render + human-readable clades; mutates with tape like any other bytes)
  d[LINEAGE_BYTE_0] = 0x6e; // R-ish channel for HSV mix
  d[LINEAGE_BYTE_1] = 0xa2;
  d[LINEAGE_BYTE_2] = 0x38;
  d[LINEAGE_BYTE_AUX] = 0; // aux mixer for packed lineage (proto: same hue as 28–30 only)

  // === CA region (bytes 32-63) ===
  d[CA_RULES_OFFSET] = 0x03; // refractoryPeriod = 3
  for (let i = 1; i < CA_RULES_SIZE; i++) d[CA_RULES_OFFSET + i] = 128;
  // Energy storage capability subsystem in CA free band:
  // 11 active modules => cap = 8 + 11*8 = 96 (recommended default).
  encodeEnergyCapBank(d, 11);
  // Replication key (overwrites tail of CA padding); transcribe + wear erode fidelity; mismatch can abort birth
  d[REPLICATION_KEY_OFFSET + 0] = 0x52;
  d[REPLICATION_KEY_OFFSET + 1] = 0x45;
  d[REPLICATION_KEY_OFFSET + 2] = 0x50;
  d[REPLICATION_KEY_OFFSET + 3] = 0x4b;

  // === Rule table (bytes 64-127): 16 condition→action pairs ===
  //
  // flags: [7:res][6:chain][5:4:comparison][3:2:item][1:0:target]
  // target: 0=self, 1=org, 2=neighbor, 3=env
  // self:   0=energy, 1=moodProb, 2=morphA, 3=morphB
  // org:    0=cells, 1=totalE, 2=age, 3=avgMorphA
  // neigh:  0=same, 1=foreign, 2=empty, 3=is_outer(0/255)
  // env:    0=here, 1=gradient, 2=max_neighbor, 3=dominant_marker
  // comp:   0=GT, 1=LT, 2=EQ±5, 3=NEQ±5

  function wr(idx: number, flags: number, thr: number, act: ActionOpcode, param: number) {
    const off = CONDITIONS_OFFSET + idx * RULE_SIZE;
    d[off] = flags; d[off+1] = thr; d[off+2] = act; d[off+3] = param;
  }

  // actionParam now points to operation node slots (0-15)
  wr(0,  0b00_00_00_11, 1,   ActionOpcode.EAT,        0); // env.here>1 → EAT via node0 (LITERAL→140)
  wr(1,  0b00_00_00_00, 25,  ActionOpcode.GIVE,        7); // self.e>25 → GIVE via node7 (IF_RICH)
  wr(2,  0b00_00_00_00, 42,  ActionOpcode.DIV,         2); // self.e>42 → DIV via node2 (SCALE_N)
  wr(3,  0b00_00_01_11, 3,   ActionOpcode.FIRE,        0); // env.grad>3 → FIRE
  // NOTE: "self.mood > 0" is effectively always true with softmax outputs, so keep a meaningful gate.
  // mood is dominant-probability scaled to 0..255; 144 ~= 0.56 confidence.
  wr(4,  0b00_00_01_00, 144, ActionOpcode.MOVE,        0); // self.mood>144 → MOVE
  wr(5,  0b00_00_00_00, 75,  ActionOpcode.REPRODUCE,   3); // self.e>75 → REPRO via node3 (LITERAL→90)
  wr(6,  0b00_00_01_10, 0,   ActionOpcode.ABSORB,      5); // neigh.foreign>0 → ABSORB via node5 (LITERAL→16)
  wr(7,  0b00_00_11_10, 128, ActionOpcode.EAT,        12); // neigh.outer>128 → EAT via node12 (SCALE_E)
  wr(8,  0b00_00_10_10, 2,   ActionOpcode.DIV,         2); // neigh.empty>2 → DIV via node2 (SCALE_N)
  wr(9,  0b00_00_00_01, 0,   ActionOpcode.DIGEST,      1); // org.cells>0 → DIGEST via node1 (SCALE_S)
  wr(10, 0b00_00_00_10, 2,   ActionOpcode.REPAIR,        1); // neigh.same>2 → REPAIR (cluster bias in evaluator)
  wr(11, 0b00_01_00_00, 20,  ActionOpcode.DIGEST,      1); // self.e<20 → hungry DIGEST via node1 (SCALE_S)
  wr(12, 0b00_00_11_10, 128, ActionOpcode.EMIT,        8); // outer → EMIT morphA via node8
  wr(13, 0b00_01_10_00, 30,  ActionOpcode.EMIT,        9); // self.morphA<30 → EMIT morphB via node9
  // Chain bit demo: rule 14 is a chain condition, rule 15 fires only if 14 passes
  wr(14, 0b01_01_00_00, 15,  ActionOpcode.TAKE,        6); // chain=1, self.e<15 → (chain: condition only)
  wr(15, 0b00_00_00_11, 0,   ActionOpcode.TAKE,        6); // env.here>0 AND chain → TAKE via node6 (SCALE_E)

  // === NN params (bytes 128-255) — fixed, not random (reproducible normal starter)
  let s = PROTO_TAPE_NN_SEED | 0;
  for (let i = 0; i < NN_WEIGHTS_SIZE; i++) {
    s = Math.imul(s, 1664525) + 1013904223 | 0;
    d[NN_WEIGHTS_OFFSET + i] = (s >>> 16) & 0xff;
  }

  return tape;
}
