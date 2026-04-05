import { CellType, GRID_WIDTH, GRID_HEIGHT, TOTAL_CELLS, ENV_DIFFUSION_RATE } from './constants';
import {
  ActionOpcode,
  CONDITIONS_OFFSET,
  MAX_RULES,
  MAX_VALID_ACTION_OPCODE,
  RULE_SIZE,
  TAPE_SIZE,
  type ConditionRule,
  type Tape,
  type ReadCtx,
  FEEDBACK_SLOTS,
} from './tape';
import { transcribeForReproduction } from './transcription';
import { type World, U32_PER_CELL } from './world';
import { type Organism, type OrganismManager, NN_OUTPUT, NN_MOVE } from './organism';
import { randomF32 } from './rng';
import type { BudgetMode, NeighborMode, SuppressionMode } from './runtime-config';
import {
  recordActionExecution,
  recordBirthFromReproduce,
  recordBirthFromSplit,
  recordReproductionAttempt,
  recordReproductionSuccess,
  recordReproduceFailCrowding,
  recordReproduceFailDominance,
  recordSocialCohesion,
  recordSplitEvent,
  recordXenoTransferAttempt,
} from './telemetry';
import { getActionCostForOpcode, MOVE_COST_PER_CELL, REPRODUCE_ACTION_COST } from './behaviors/action-costs';
import { dispatchAction } from './behaviors/action-dispatch';
import {
  computeForeignContactPressure,
  getCrowdingDissolveBonus,
  getCrowdingReproduceFailChance,
  getDistressFireChance,
  getDominanceMetabolicMultiplier,
  getDominanceReproduceFailChance,
  getPerCellMetabolicBase,
  getSplitCrowdExtraCooldown,
  getSplitMinFragmentCells,
  shouldTriggerDistressFire,
} from './behaviors/stress-signals';
import {
  applyJamToForeignBoundary,
  computeJamTtl,
  countRepulsionFacesForShift,
} from './behaviors/exclusion-actions';
import { spillStomachToNearbyEnv } from './behaviors/vent-actions';

const ORTH_DIRS: [number, number][] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
const EIGHT_DIRS: [number, number][] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
  [1, -1],
  [1, 1],
  [-1, 1],
  [-1, -1],
];

function popcount24(v: number): number {
  let x = v & 0xffffff;
  let n = 0;
  while (x) {
    n += x & 1;
    x >>>= 1;
  }
  return n;
}

// Tape REPAIR (immune): same-org neighbor quorum boosts success (collective error correction bias)
const REPAIR_NEIGHBOR_COEFF = 0.16; // success mult += coeff * sameOrthogonalNeighbors (0..4)
const REPAIR_BASE_P         = 0.36; // base success before bias × intensity
const REPAIR_DEG_HEAL       = 36;   // degradation subtracted on success

// Cross-org “kin” trust: pigment (cell lineage field) + signal marker + morph A — all must roughly match (pigment-only mimicry stays weak).
const KIN_TRUST_CAP_FOREIGN      = 0.52;
const KIN_TRUST_FOREIGN_SCALE    = 1.12;
const KIN_GIVE_MIN_TRUST         = 0.36;
const KIN_GIVE_RATE_CAP          = 0.24;   // vs same-org GIVE strength
const KIN_GIVE_EXTRA_HEAT        = 0.2;    // mis-altruism waste scales with imperfect trust
const REPAIR_FOREIGN_KIN_WEIGHT  = 0.4;    // foreign neighbor adds trust*this to repair quorum (capped trust above)

// Structural constants (NOT evolvable — facts of physics)
const ISOLATION_METABOLIC_PENALTY = 0.12; // isolated cell pays up to +12% metabolic vs well-connected tissue
const PASSIVE_DIGEST_RATE     = 0.15;
const DIGESTION_HEAT_LOSS     = 0.25;
const LOW_ENERGY_LEAK_MAX     = 0.08;
/**
 * Connectivity bonus in digestion: isolated cells digest less efficiently, well-connected tissues digest better.
 * This biases survival toward networked morphologies without adding non-conservative energy.
 */
const DIGEST_NETWORK_BASE     = 0.82; // same-neighbor ratio 0.0 -> 82% of baseline digest throughput
const DIGEST_NETWORK_COEFF    = 0.32; // ratio 1.0 (4 same orth-neighbors) -> +32%
/**
 * Max extra digest multiplier from DIGEST opcodes this tick per cell (applied in digestPhase only).
 * Neighbor stomachs are not used — avoids order-dependent “parasitic” multi-cell stripping and double-counting
 * with a passive+active split. Final boost is cap-clamped; rule-table order can still affect which DIGEST rows
 * pay cost once the cap is full.
 */
const DIGEST_RULE_BOOST_CAP   = 1.0;
const PASSIVE_ABSORB_RATE     = 0.15;
// Rule-table scan tax: tiny universal scan cost + extra penalty for dead rows.
const SCAN_TAX_ALL            = 0.00005;
const SCAN_TAX_NOP_EXTRA      = 0.00035;
const SCAN_TAX_INVALID_EXTRA  = 0.00100;

/** Orthogonal contact with another organism: passive interface “tension” (energy → env, closed system). */
const FOREIGN_INTERFACE_METABOLISM = 0.055;
/** MOVE score penalty per foreign face after a rigid shift (discourages sliding along / into heterospecifics). */
const MOVE_REPEL_FOREIGN_FACE = 14;
/** MOVE score penalty per world-edge face after shift (soft repulsion from map boundary). */
const MOVE_REPEL_MAP_EDGE_FACE = 4;
const DIAGONAL_MOVE_COST_MULT = Math.SQRT2;
/** Base diagonal bonus to counter axis-aligned lattice preference. */
const DIAGONAL_MOVE_SCORE_BIAS_PER_CELL_BASE = 0.24;
/** Extra diagonal bonus scaled by NN move drive (keeps the preference in the NN path). */
const DIAGONAL_MOVE_SCORE_BIAS_PER_CELL_NN = 0.16;

/**
 * ABSORB at a heterospecific face: if both morphogen channels are close on the two cells, relax cell
 * energies symmetrically toward their average (bidirectional “coupling”). Otherwise one-way drain to
 * actor’s stomach (predation). Thresholds align with kinTrust morph scale (~O(1) field).
 */
const MORPH_ABSORB_MATCH_A = 1.7;
const MORPH_ABSORB_MATCH_B = 1.7;
const ABSORB_BIDIR_RELAX = 0.26;
const XENO_TAPE_TRANSFER_STOMACH_K = 2.4;  // stomach-gated integration: higher gut load raises transfer fixation odds
const XENO_TAPE_TRANSFER_HEAT = 0.08;      // integration stress: small heat fee on successful transfer
const XENO_TRANSFER_CONTACT_SCALE = 0.50;  // contact-route transfer intensity scale (smaller => stronger contact route)

// Reproduction requires a minimum network size (must DIV first)
const MIN_CELLS_TO_REPRODUCE  = 2;

// Anti-monoculture: slow "rational apex" that tiles the whole grid
const REPRODUCE_COOLDOWN_TICKS   = 40;   // min ticks between successful REPRODUCE per org

// Selection pressure: fragments do not amortize organism-level cost (real culling + fewer org spam)
const ORG_OVERHEAD_PER_TICK      = 0.042; // each lineage pays this from its biomass → env (same for 1-cell or 72-cell)
const DISSOLVE_SINGLE_MULT       = 1.45;    // 1-cell orgs dissolve slightly faster when e<=0
const DISSOLVE_BASE              = 0.016;

// Connectivity pressure: split-born shards cannot reproduce immediately.
const SPLIT_CHILD_EXTRA_COOLDOWN = 12;

// Morphogen physics
const MORPHOGEN_DIFFUSION     = 0.2;   // 20% spreads to each neighbor per tick
const MORPHOGEN_DECAY         = 0.05;  // 5% decay per tick

// Specialization marker bump per action execution
const MARKER_BUMP = 4;
const SOCIAL_SIGNAL_CONSENSUS_RATE = 0.06; // local same-org gossip coupling per tick
const SOCIAL_REPAIR_COHESION_BONUS = 0.03; // cohesive neighborhoods repair slightly more reliably
const JAM_MIN_TICKS = 2;
const JAM_MAX_EXTRA_TICKS = 3;

// Marker slots: 0=eat, 1=digest, 2=signal, 3=move
const MARKER_EAT    = 0 as const;
const MARKER_DIGEST = 1 as const;
const MARKER_SIGNAL = 2 as const;
const MARKER_MOVE   = 3 as const;
const BUDGET_RESCALE_THRESHOLD = 0.5;

interface CellCtx {
  x: number; y: number; idx: number;
  energy: number; orgId: number;
}

interface RuleEvaluatorOptions {
  neighborMode?: NeighborMode;
  budgetMode?: BudgetMode;
  suppressionMode?: SuppressionMode;
  metabolicScale?: number;
  distressFireChanceScale?: number;
}

/**
 * **Canonical simulation (including neural / signal dynamics)** lives here on the CPU, mutating
 * `World.cellData` (see `propagateSignals`). The WebGPU path in `main.ts` only **uploads** that
 * state and runs the **fragment render** shader — it does **not** step the CA / nervous system.
 *
 * `src/gpu/shaders/ca-nervous.wgsl` is a historical / experimental compute pass: **not dispatched**
 * by the app loop. If you ever wire GPU stepping, either delete CPU duplication or make GPU write
 * back and treat one path as obsolete — do not silently maintain two divergent models.
 * @see README.md § GPU vs CPU
 */
export class RuleEvaluator {
  world: World;
  organisms: OrganismManager;
  envEnergy: Float32Array;
  private envScratch: Float32Array;
  /** Per-cell accumulated DIGEST-rule boost for this tick (applied once in `digestPhase`, order-independent). */
  private digestRuleBoost: Float32Array;
  private movedThisTick: Set<number> = new Set();
  /** Total occupied cells after `evaluate()` (rules + passive phases). Used by metabolic share. */
  private liveCellCount = 1;
  /** Total occupied cells at start of rule phase (before MOVE/REPRO etc.). Used for dominance-gated reproduction. */
  private dominanceLiveCellCount = 1;
  private simTick = 0;
  private neighborMode: NeighborMode;
  private budgetMode: BudgetMode;
  private suppressionMode: SuppressionMode;
  private metabolicScale: number;
  private distressFireChanceScale: number;
  private jamTicks: Uint8Array;
  private static readonly BROKEN_CAP_FALLBACK = 0;

  /**
   * Closed thermodynamic budget: sum(env) + Σ(cell E) + Σ(stomach) = this value.
   * Mouse inject increases it; spawn moves energy from env → cell without changing the budget.
   */
  ecosystemEnergyBudget = 0;

  constructor(world: World, organisms: OrganismManager, opts: RuleEvaluatorOptions = {}) {
    this.world = world;
    this.organisms = organisms;
    this.envEnergy = new Float32Array(TOTAL_CELLS);
    this.envScratch = new Float32Array(TOTAL_CELLS);
    this.digestRuleBoost = new Float32Array(TOTAL_CELLS);
    this.neighborMode = opts.neighborMode ?? 'four';
    this.budgetMode = opts.budgetMode ?? 'local';
    this.suppressionMode = opts.suppressionMode ?? 'on';
    this.metabolicScale = opts.metabolicScale ?? 1;
    this.distressFireChanceScale = opts.distressFireChanceScale ?? 1;
    this.jamTicks = new Uint8Array(TOTAL_CELLS);
  }

  private neighborDirs(): [number, number][] {
    return this.neighborMode === 'four' ? ORTH_DIRS : EIGHT_DIRS;
  }

  /** MOVE candidates are always 8-way so diagonal approach stays available even in 4-neighbor ecology mode. */
  private moveDirs(): [number, number][] {
    return EIGHT_DIRS;
  }

  private safeCellEnergyCapForOrg(orgId: number): number {
    if (orgId <= 0) return RuleEvaluator.BROKEN_CAP_FALLBACK;
    const org = this.organisms.get(orgId);
    if (!org) return RuleEvaluator.BROKEN_CAP_FALLBACK;
    const cap = org.tape.getMaxCellEnergy();
    if (!Number.isFinite(cap) || cap <= 0) return RuleEvaluator.BROKEN_CAP_FALLBACK;
    return cap;
  }

  private clampCellEnergyForOrg(energy: number, orgId: number): number {
    const safeE = Number.isFinite(energy) ? energy : 0;
    const cap = this.safeCellEnergyCapForOrg(orgId);
    return Math.min(cap, Math.max(0, safeE));
  }

  private setCellEnergyCappedByIdx(idx: number, energy: number, orgIdHint?: number): number {
    const orgId = orgIdHint ?? this.world.getOrganismIdByIdx(idx);
    const clamped = this.clampCellEnergyForOrg(energy, orgId);
    this.world.setCellEnergyByIdx(idx, clamped);
    return clamped;
  }

  private setCellEnergyCapped(x: number, y: number, energy: number, orgIdHint?: number): number {
    const orgId = orgIdHint ?? this.world.getOrganismId(x, y);
    const clamped = this.clampCellEnergyForOrg(energy, orgId);
    this.world.setCellEnergy(x, y, clamped);
    return clamped;
  }

  private enforcePerCellEnergyCaps() {
    for (let idx = 0; idx < TOTAL_CELLS; idx++) {
      const orgId = this.world.getOrganismIdByIdx(idx);
      if (orgId === 0) continue;
      const e = this.world.getCellEnergyByIdx(idx);
      const clamped = this.clampCellEnergyForOrg(e, orgId);
      if (clamped >= e) continue;
      this.world.setCellEnergyByIdx(idx, clamped);
      this.envEnergy[idx] += (e - clamped);
    }
  }

  setEnvEnergy(data: Float32Array) {
    this.envEnergy.set(data);
    let s = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) s += this.envEnergy[i];
    this.ecosystemEnergyBudget = s;
  }

  /** Call after initial world + organisms state is ready (sets `ecosystemEnergyBudget` from current sums). */
  snapClosedEnergyBudgetFromWorld() {
    let env = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) env += this.envEnergy[i];
    let bio = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      if (this.world.getOrganismIdByIdx(i) === 0) continue;
      bio += this.world.getCellEnergyByIdx(i) + this.world.getStomachByIdx(i);
    }
    this.ecosystemEnergyBudget = env + bio;
  }

  /** Mouse / tool inject: add this much to the closed budget (env values already increased by caller). */
  addToClosedBudget(delta: number) {
    this.ecosystemEnergyBudget += delta;
  }

  /**
   * Remove `amount` uniformly from env field (for spawn: energy becomes new cell, budget unchanged).
   * Returns false if env sum is insufficient.
   */
  withdrawEnvUniform(amount: number): boolean {
    if (amount <= 0) return true;
    let s = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) s += this.envEnergy[i];
    if (s < amount - 1e-6) return false;
    const f = (s - amount) / s;
    for (let i = 0; i < TOTAL_CELLS; i++) this.envEnergy[i] *= f;
    return true;
  }

  /** Rescale env so sum(env) = ecosystemEnergyBudget − biomass (fixes diffusion boundary drift + float). */
  enforceClosedEnergyBudget() {
    let bio = 0;
    let es = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      es += this.envEnergy[i];
      if (this.world.getOrganismIdByIdx(i) === 0) continue;
      bio += this.world.getCellEnergyByIdx(i) + this.world.getStomachByIdx(i);
    }

    let targetEnv = this.ecosystemEnergyBudget - bio;
    if (targetEnv < -1e-4) {
      const over = -targetEnv;
      this.scaleDownBiomass(over, bio);
      let bio2 = 0;
      for (let i = 0; i < TOTAL_CELLS; i++) {
        if (this.world.getOrganismIdByIdx(i) === 0) continue;
        bio2 += this.world.getCellEnergyByIdx(i) + this.world.getStomachByIdx(i);
      }
      targetEnv = this.ecosystemEnergyBudget - bio2;
    }

    if (targetEnv <= 0) {
      for (let i = 0; i < TOTAL_CELLS; i++) this.envEnergy[i] = 0;
      return;
    }

    if (es <= 1e-12) {
      const per = targetEnv / TOTAL_CELLS;
      for (let i = 0; i < TOTAL_CELLS; i++) this.envEnergy[i] = per;
      return;
    }

    const errBeforeRescale = targetEnv - es;
    if (this.budgetMode === 'local') {
      if (Math.abs(errBeforeRescale) <= BUDGET_RESCALE_THRESHOLD) {
        const anchor = this.findBudgetAnchorIndex();
        this.envEnergy[anchor] = Math.max(0, this.envEnergy[anchor] + errBeforeRescale);
        return;
      }
    }

    const sc = targetEnv / es;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      this.envEnergy[i] = Math.max(0, this.envEnergy[i] * sc);
    }

    let s2 = 0;
    for (let i = 0; i < TOTAL_CELLS; i++) s2 += this.envEnergy[i];
    const err = targetEnv - s2;
    if (Math.abs(err) > 1e-5) {
      const anchor = this.findBudgetAnchorIndex();
      this.envEnergy[anchor] = Math.max(0, this.envEnergy[anchor] + err);
    }
  }

  private findBudgetAnchorIndex(): number {
    for (let i = 0; i < TOTAL_CELLS; i++) {
      if (this.world.getOrganismIdByIdx(i) !== 0) return i;
    }
    return 0;
  }

  private scaleDownBiomass(reduceBioBy: number, knownBio?: number) {
    let bio: number;
    if (knownBio !== undefined) {
      bio = knownBio;
    } else {
      bio = 0;
      for (let i = 0; i < TOTAL_CELLS; i++) {
        if (this.world.getOrganismIdByIdx(i) === 0) continue;
        bio += this.world.getCellEnergyByIdx(i) + this.world.getStomachByIdx(i);
      }
    }
    if (bio <= 1e-12) return;
    const newBio = Math.max(0, bio - reduceBioBy);
    const f = newBio / bio;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      if (this.world.getOrganismIdByIdx(i) === 0) continue;
      const e = this.world.getCellEnergyByIdx(i);
      const st = this.world.getStomachByIdx(i);
      this.setCellEnergyCappedByIdx(i, e * f);
      this.world.setStomachByIdx(i, st * f);
    }
  }

  // ==================== NEURAL NETWORK UPDATE ====================
  updateNeuralNetworks() {
    for (const org of this.organisms.organisms.values()) {
      if (org.cells.size === 0) continue;

      let totalE = 0;
      let totalS = 0;
      let totalEnv = 0;
      let boundaryCells = 0;
      let foreignNeighbors = 0;
      let markerDominance = 0;
      let envGradient = 0;
      for (const idx of org.cells) {
        totalE += this.world.getCellEnergyByIdx(idx);
        totalS += this.world.getStomachByIdx(idx);
        totalEnv += this.envEnergy[idx];

        const x = idx % GRID_WIDTH;
        const y = (idx - x) / GRID_WIDTH;
        if (this.isOuterCell(idx, org)) boundaryCells++;

        for (const [dx, dy] of this.neighborDirs()) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
          const nOrg = this.world.getOrganismId(nx, ny);
          if (nOrg !== 0 && nOrg !== org.id) foreignNeighbors++;
        }

        markerDominance += this.dominantMarker(idx) / 255;
        const c = this.envEnergy[idx];
        let localMax = c;
        for (const [dx, dy] of this.neighborDirs()) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
          localMax = Math.max(localMax, this.envEnergy[ny * GRID_WIDTH + nx]);
        }
        envGradient += Math.max(0, localMax - c);
      }
      const n = org.cells.size;

      // 8 inputs: avg energy, avg stomach, avg env, org size, boundary ratio,
      // foreign contact ratio, marker dominance, local env gradient.
      org.nnInput[0] = Math.min(1, (totalE / n) / 255);
      org.nnInput[1] = Math.min(1, (totalS / n) / 255);
      org.nnInput[2] = Math.min(1, (totalEnv / n) / 50);
      org.nnInput[3] = Math.min(1, n / 64);
      org.nnInput[4] = Math.min(1, boundaryCells / n);
      org.nnInput[5] = Math.min(1, foreignNeighbors / (n * this.neighborDirs().length));
      org.nnInput[6] = Math.min(1, markerDominance / n);
      org.nnInput[7] = Math.min(1, (envGradient / n) / 20);

      org.nnOutput = org.nn.forward(org.nnInput);

      let best = 0;
      for (let i = 1; i < NN_OUTPUT; i++) {
        if (org.nnOutput[i] > org.nnOutput[best]) best = i;
      }
      org.nnDominant = best;
    }
  }

  // ==================== MAIN TICK ====================
  evaluate() {
    this.simTick++;
    for (let i = 0; i < TOTAL_CELLS; i++) {
      if (this.jamTicks[i] > 0) this.jamTicks[i]--;
    }
    this.stepEnvDiffusion();
    this.movedThisTick.clear();
    this.digestRuleBoost.fill(0);
    this.enforcePerCellEnergyCaps();

    // Phase 1: passive absorption, marker decay, feedback decay for ALL cells
    for (const org of this.organisms.organisms.values()) {
      this.rebuildCellList(org);
      // Decay feedback registers (exponential: *= 0.8 approximated in u8)
      for (let i = 0; i < FEEDBACK_SLOTS; i++) {
        org.feedback[i] = Math.floor(org.feedback[i] * 0.8);
      }
      for (const idx of org.cells) {
        const cell = this.cellAt(idx, org.id);
        if (!cell) continue;
        this.passiveAbsorb(cell, org.cells.size);
        this.applyForeignInterfaceMetabolism(idx, org.id);
        this.applySocialConsensusDrift(cell, org.id);
        this.world.decayMarkers(idx);
      }
    }

    let live = 0;
    for (const org of this.organisms.organisms.values()) live += org.cells.size;
    this.dominanceLiveCellCount = Math.max(1, live);

    // Phase 2: rule evaluation per organism (snapshot IDs to avoid evaluating newborns)
    const orgSnapshot = [...this.organisms.organisms.values()];
    for (const org of orgSnapshot) {
      if (org.cells.size === 0) continue;
      // Compute average cell energy for degradation scaling
      let totalE = 0;
      for (const idx of org.cells) totalE += this.world.getCellEnergyByIdx(idx);
      const avgE = org.cells.size > 0 ? totalE / org.cells.size : 0;
      org.tape.applyReadDegradation(avgE, org.age);
      this.diffuseMorphogens(org);

      const tape = org.tape;
      const cellIndices = [...org.cells];
      for (const idx of cellIndices) {
        const cell = this.cellAt(idx, org.id);
        if (!cell) continue;
        if (this.evaluateCell(cell, org, tape)) break; // MOVE fired → old indices invalid
      }
      this.propagateSignals(org, tape);
    }

    live = 0;
    for (const org of this.organisms.organisms.values()) live += org.cells.size;
    this.liveCellCount = Math.max(1, live);
  }

  // ==================== PER-CELL EVALUATION ====================
  // Returns true if the organism moved this tick (caller must re-snapshot)
  private evaluateCell(cell: CellCtx, org: Organism, tape: Tape): boolean {
    const distressChance = getDistressFireChance(this.distressFireChanceScale);
    if (shouldTriggerDistressFire(cell.energy, distressChance, randomF32())) {
      this.actionFire(cell, org);
    }

    const ruleCount = tape.getRuleCount();
    let chainPassed = false; // chain bit: previous rule's condition passed

    const startRule = (this.simTick + cell.idx + org.id) % ruleCount;
    for (let ro = 0; ro < ruleCount; ro++) {
      const r = (startRule + ro) % ruleCount;
      const rawOpcode = tape.data[CONDITIONS_OFFSET + r * RULE_SIZE + 2] & 0xff;
      this.payRuleScanTax(cell, rawOpcode);
      const rule = tape.getRule(r);
      if (rule.actionOpcode === ActionOpcode.NOP) { chainPassed = false; continue; }

      const isChain = (rule.conditionFlags & 0x40) !== 0; // bit 6

      if (chainPassed) {
        // Previous chain rule passed — this rule must ALSO pass to execute
        if (!this.checkCondition(rule, cell, org)) { chainPassed = false; continue; }
      } else {
        if (!this.checkCondition(rule, cell, org)) continue;
      }

      if (isChain) {
        // Chain rule: condition passed, but don't execute — propagate to next
        chainPassed = true;
        continue;
      }

      chainPassed = false;

      const cost = this.getActionCost(rule.actionOpcode);
      if (cell.energy < cost + 0.5) continue;

      const success = this.executeAction(rule, cell, org, tape);
      if (success) {
        recordActionExecution(rule.actionOpcode);
        if (cost > 0) this.spendEnergy(cell, cost);
      }
      tape.applyActionWear(r);

      if (rule.actionOpcode === ActionOpcode.MOVE && this.movedThisTick.has(org.id)) {
        return true;
      }
    }
    return false;
  }

  private payRuleScanTax(cell: CellCtx, rawOpcode: number) {
    let tax = SCAN_TAX_ALL;
    if (rawOpcode === ActionOpcode.NOP) {
      tax += SCAN_TAX_NOP_EXTRA;
    } else if (rawOpcode > MAX_VALID_ACTION_OPCODE) {
      tax += SCAN_TAX_INVALID_EXTRA;
    }
    const paid = Math.min(cell.energy, tax);
    if (paid <= 0) return;
    cell.energy -= paid;
    cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
    this.envEnergy[cell.idx] += paid;
  }

  // ==================== CONDITION EVALUATION ====================
  // flags: [7:res][6:chain][5:4:comparison][3:2:item][1:0:target]
  // target: 0=self, 1=org, 2=neighbor, 3=env
  // self:   0=energy, 1=moodProb, 2=morphA, 3=morphB
  // org:    0=cells, 1=totalE, 2=age, 3=avgMorphA
  // neigh:  0=same, 1=foreign, 2=empty, 3=is_outer(0/255)
  // env:    0=here, 1=gradient, 2=max_neighbor, 3=specialization(dominant marker)
  // comp:   0=GT, 1=LT, 2=EQ±5, 3=NEQ±5

  private checkCondition(rule: ConditionRule, cell: CellCtx, org: Organism): boolean {
    const target     = rule.conditionFlags & 0x03;
    const item       = (rule.conditionFlags >> 2) & 0x03;
    const comparison = (rule.conditionFlags >> 4) & 0x03;
    const value = this.getConditionValue(target, item, cell, org);
    const thr = rule.thresholdValue;
    switch (comparison) {
      case 0: return value > thr;
      case 1: return value < thr;
      case 2: return Math.abs(value - thr) <= 5;
      case 3: return Math.abs(value - thr) > 5;
    }
    return false;
  }

  private getConditionValue(target: number, item: number, cell: CellCtx, org: Organism): number {
    switch (target) {
      case 0: // self
        switch (item) {
          case 0: return Math.min(255, cell.energy);
          case 1: return Math.min(255, Math.floor(org.nnOutput[org.nnDominant] * 255));
          case 2: return Math.min(255, this.world.getMorphogenA(cell.idx) * 10);
          case 3: return Math.min(255, this.world.getMorphogenB(cell.idx) * 10);
        }
        break;
      case 1: // org
        switch (item) {
          case 0: return org.cells.size;
          case 1: return Math.min(255, this.orgTotalEnergy(org));
          case 2: return Math.min(255, org.age);
          case 3: return Math.min(255, this.orgAvgMorphA(org) * 10);
        }
        break;
      case 2: // neighbor
        switch (item) {
          case 0: return this.countNeighborsByType(cell, 'same');
          case 1: return this.countNeighborsByType(cell, 'foreign');
          case 2: return this.countNeighborsByType(cell, 'empty');
          case 3: return this.isOuterCell(cell.idx, org) ? 255 : 0;
        }
        break;
      case 3: // env / meta
        switch (item) {
          case 0: return Math.min(255, this.envEnergy[cell.idx]);
          case 1: return Math.min(255, Math.max(0, this.envGradientScaled(cell.x, cell.y)));
          case 2: return Math.min(255, this.maxNeighborEnv(cell));
          case 3: return this.dominantMarker(cell.idx);
        }
        break;
    }
    return 0;
  }

  // ==================== ACTION DISPATCH ====================

  private buildReadCtx(cell: CellCtx, org: Organism): ReadCtx {
    return {
      cellEnergy: cell.energy,
      stomach: this.world.getStomachByIdx(cell.idx),
      orgCells: org.cells.size,
      feedback: org.feedback,
    };
  }

  private writeFeedback(org: Organism, slot: number, value: number) {
    org.feedback[slot] = Math.min(255, Math.max(0, Math.round(value)));
  }

  private collectNeighborEnvSum(cell: CellCtx): number {
    let envSum = 0;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = (cell.x + dx + GRID_WIDTH) % GRID_WIDTH;
      const ny = (cell.y + dy + GRID_HEIGHT) % GRID_HEIGHT;
      envSum += this.envEnergy[ny * GRID_WIDTH + nx];
    }
    return envSum;
  }

  private executeAction(rule: ConditionRule, cell: CellCtx, org: Organism, tape: Tape): boolean {
    return dispatchAction(rule, cell, org, tape, {
      movedThisTick: this.movedThisTick,
      markers: {
        eat: MARKER_EAT,
        digest: MARKER_DIGEST,
        signal: MARKER_SIGNAL,
        bump: MARKER_BUMP,
      },
      buildReadCtx: (c, o) => this.buildReadCtx(c, o),
      writeFeedback: (o, slot, value) => this.writeFeedback(o, slot, value),
      bumpMarker: (idx, marker, amount) => this.world.bumpMarker(idx, marker, amount),
      collectNeighborEnvSum: (c) => this.collectNeighborEnvSum(c),
      actionEat: (c, strength, orgCellCount) => this.actionEat(c, strength, orgCellCount),
      actionGive: (c, intensity01) => this.actionGive(c, intensity01),
      actionTake: (c, intensity) => this.actionTake(c, intensity),
      actionDivide: (c, o, t, minEnergy) => this.actionDivide(c, o, t, minEnergy),
      actionFire: (c, o) => this.actionFire(c, o),
      actionMove: (c, o) => this.actionMove(c, o),
      actionReproduce: (c, o, t, frac) => this.actionReproduce(c, o, t, frac),
      actionAbsorb: (c, maxSteal) => this.actionAbsorb(c, maxSteal),
      actionDigest: (c, o, rate) => this.actionDigest(c, o, rate),
      actionEmit: (c, actionParam, amount) => this.actionEmit(c, actionParam, amount),
      actionRepair: (c, o, t, intensity) => this.actionRepair(c, o, t, intensity),
      actionSpill: (c, amount) => this.actionSpill(c, amount),
      actionJam: (c, intensity) => this.actionJam(c, intensity),
    });
  }

  private getActionCost(opcode: ActionOpcode): number {
    return getActionCostForOpcode(opcode);
  }

  // ==================== ACTIONS ====================

  private passiveAbsorb(cell: CellCtx, orgSize = 1) {
    if (cell.energy <= 0) return;
    const coordination = Math.min(1, orgSize / 3); // 1-cell=33%, 3+=100%
    const avail = this.envEnergy[cell.idx];
    const take = Math.min(avail * 0.05, PASSIVE_ABSORB_RATE) * coordination;
    if (take < 0.001) return;
    this.envEnergy[cell.idx] -= take;
    const stomach = this.world.getStomachByIdx(cell.idx);
    this.world.setStomachByIdx(cell.idx, stomach + take);
  }

  private actionEat(cell: CellCtx, maxGather: number, orgSize = 1) {
    const specBonus = 1 + this.world.getMarkerByIdx(cell.idx, MARKER_EAT) / 255;
    const vitality = Math.min(1, cell.energy / 5);
    const coordination = Math.min(1, orgSize / 3); // 1-cell=33%, 2-cell=67%, 3+=100%
    const maxEat = maxGather * (0.3 + 0.7 * vitality) * specBonus * coordination;
    let gathered = 0;
    const spots: [number, number][] = [
      [cell.x, cell.y],
      [cell.x - 1, cell.y], [cell.x + 1, cell.y],
      [cell.x, cell.y - 1], [cell.x, cell.y + 1],
    ];
    for (const [px, py] of spots) {
      if (px < 0 || px >= GRID_WIDTH || py < 0 || py >= GRID_HEIGHT) continue;
      const ei = py * GRID_WIDTH + px;
      const avail = this.envEnergy[ei];
      const take = Math.min(avail * 0.1, maxEat - gathered);
      if (take < 0.001) continue;
      this.envEnergy[ei] -= take;
      gathered += take;
      if (gathered >= maxEat) break;
    }
    if (gathered > 0) {
      const stomach = this.world.getStomachByIdx(cell.idx);
      this.world.setStomachByIdx(cell.idx, stomach + gathered);
    }
  }

  /**
   * DIGEST opcode: no immediate transfer — raises this cell’s digest boost for the single `digestPhase` pass
   * (own stomach only; order of rules does not change final boost — each add is clamped to the same cap).
   */
  private actionDigest(cell: CellCtx, _org: Organism, rate: number): boolean {
    const own = this.world.getStomachByIdx(cell.idx);
    if (own < 0.01) return false;
    if (this.digestRuleBoost[cell.idx] >= DIGEST_RULE_BOOST_CAP - 1e-8) return false;
    const specBonus = 1 + this.world.getMarkerByIdx(cell.idx, MARKER_DIGEST) / 255;
    const contribution = rate * specBonus;
    this.digestRuleBoost[cell.idx] = Math.min(
      DIGEST_RULE_BOOST_CAP,
      this.digestRuleBoost[cell.idx] + contribution,
    );
    return true;
  }

  private isRuleOpcodeByte(index: number): boolean {
    const rel = index - CONDITIONS_OFFSET;
    if (rel < 0 || rel >= MAX_RULES * RULE_SIZE) return false;
    return rel % RULE_SIZE === 2;
  }

  /** 0..KIN_TRUST_CAP_FOREIGN for foreign cells; same-org callers should not use this (returns 0 if same org). */
  private kinTrustForeign(selfIdx: number, nIdx: number, selfOrgId: number): number {
    const nOrg = this.world.getOrganismIdByIdx(nIdx);
    if (nOrg === 0 || nOrg === selfOrgId) return 0;

    const sl = this.world.cellData[selfIdx * U32_PER_CELL + 7] & 0xffffff;
    const nl = this.world.cellData[nIdx * U32_PER_CELL + 7] & 0xffffff;
    const lineageSim = 1 - popcount24(sl ^ nl) / 24;

    const sigS = this.world.getMarkerByIdx(selfIdx, MARKER_SIGNAL);
    const sigN = this.world.getMarkerByIdx(nIdx, MARKER_SIGNAL);
    const signalSim = 1 - Math.min(1, Math.abs(sigS - sigN) / 96);

    const mS = this.world.getMorphogenA(selfIdx);
    const mN = this.world.getMorphogenA(nIdx);
    const morphSim = 1 - Math.min(1, Math.abs(mS - mN) / 7);

    const g = Math.max(0, lineageSim) * Math.max(0, signalSim) * Math.max(0, morphSim);
    if (g <= 0) return 0;
    return Math.min(KIN_TRUST_CAP_FOREIGN, Math.cbrt(g) * KIN_TRUST_FOREIGN_SCALE);
  }

  /** Repair quorum: 1 per same-org neighbor + weighted foreign “kin” matches (deceptive pigment helps only if chemistry aligns). */
  private repairQuorumKin(cell: CellCtx): number {
    const dirs = this.neighborDirs();
    let sum = 0;
    for (const [dx, dy] of dirs) {
      const nx = cell.x + dx,
        ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const nIdx = ny * GRID_WIDTH + nx;
      if (this.world.getCellTypeByIdx(nIdx) === CellType.Empty) continue;
      const nOrg = this.world.getOrganismIdByIdx(nIdx);
      if (nOrg === 0) continue;
      if (nOrg === cell.orgId) sum += 1;
      else sum += this.kinTrustForeign(cell.idx, nIdx, cell.orgId) * REPAIR_FOREIGN_KIN_WEIGHT;
    }
    return sum;
  }

  /** Immune repair: damaged / invalid-opcode bytes; neighbor quorum includes same-org + partial foreign kin trust. */
  private actionRepair(cell: CellCtx, _org: Organism, tape: Tape, intensity: number): boolean {
    const kinSum = this.repairQuorumKin(cell);
    const bias = 1 + REPAIR_NEIGHBOR_COEFF * kinSum;
    const socialBias = 1 + SOCIAL_REPAIR_COHESION_BONUS * this.localSignalCohesion(cell);
    const inten = Math.max(0.08, Math.min(1, intensity));

    const cands: number[] = [];
    const wts: number[] = [];
    for (let i = 0; i < TAPE_SIZE; i++) {
      let w = tape.degradation[i];
      if (this.isRuleOpcodeByte(i) && tape.data[i] > MAX_VALID_ACTION_OPCODE) w += 52;
      if (w > 0) {
        cands.push(i);
        wts.push(w);
      }
    }
    if (cands.length === 0) return false;

    let sumW = 0;
    for (const w of wts) sumW += w;
    let r = randomF32() * sumW;
    let pick = 0;
    for (let i = 0; i < cands.length; i++) {
      r -= wts[i];
      if (r <= 0) {
        pick = i;
        break;
      }
      pick = i;
    }
    const idx = cands[pick];

    const pTry = Math.min(0.93, REPAIR_BASE_P * bias * inten * socialBias);
    if (randomF32() > pTry) return false;

    if (this.isRuleOpcodeByte(idx) && tape.data[idx] > MAX_VALID_ACTION_OPCODE) {
      tape.data[idx] = ActionOpcode.NOP;
    }
    tape.degradation[idx] = Math.max(0, tape.degradation[idx] - REPAIR_DEG_HEAL);
    return true;
  }

  private actionGive(cell: CellCtx, rate: number): boolean {
    let given = 0,
      heat = 0;
    const basePush = cell.energy * rate * 0.1;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx,
        ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const nOrg = this.world.getOrganismId(nx, ny);
      if (nOrg === 0) continue;
      const nIdx = ny * GRID_WIDTH + nx;
      const ne = this.world.getCellEnergy(nx, ny);

      if (nOrg === cell.orgId) {
        if (basePush < 0.2) continue;
        if (ne < cell.energy - basePush) {
          const loss = basePush * 0.1;
          this.setCellEnergyCapped(nx, ny, ne + basePush - loss, nOrg);
          given += basePush;
          heat += loss;
        }
      } else {
        const t = this.kinTrustForeign(cell.idx, nIdx, cell.orgId);
        if (t < KIN_GIVE_MIN_TRUST) continue;
        const pushPerN = basePush * t * KIN_GIVE_RATE_CAP;
        if (pushPerN < 0.12) continue;
        if (ne < cell.energy - pushPerN) {
          const loss = pushPerN * (0.1 + KIN_GIVE_EXTRA_HEAT * (1 - t));
          this.setCellEnergyCapped(nx, ny, ne + pushPerN - loss, nOrg);
          given += pushPerN;
          heat += loss;
        }
      }
    }
    if (given > 0) {
      cell.energy -= given;
      cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
      this.envEnergy[cell.idx] += heat;
      return true;
    }
    return false;
  }

  private actionTake(cell: CellCtx, maxPull: number): boolean {
    let taken = 0, heat = 0;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx, ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      if (this.world.getOrganismId(nx, ny) !== cell.orgId) continue;
      const ne = this.world.getCellEnergy(nx, ny);
      if (ne > cell.energy) {
        const pull = Math.min(ne * 0.15, maxPull);
        if (pull < 0.1) continue;
        const loss = pull * 0.1;
        this.setCellEnergyCapped(nx, ny, ne - pull, cell.orgId);
        taken += pull - loss;
        heat += loss;
      }
    }
    if (taken > 0) {
      cell.energy += taken;
      cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
      this.envEnergy[cell.idx] += heat;
      return true;
    }
    return false;
  }

  private actionEmit(cell: CellCtx, actionParam: number, amount: number) {
    const channel = actionParam % 2;
    if (channel === 0) {
      const cur = this.world.getMorphogenA(cell.idx);
      this.world.setMorphogenA(cell.idx, cur + amount);
    } else {
      const cur = this.world.getMorphogenB(cell.idx);
      this.world.setMorphogenB(cell.idx, cur + amount);
    }
  }

  private actionDivide(cell: CellCtx, org: Organism, tape: Tape, divCost: number): boolean {
    const maxCells = tape.getMaxCells();
    if (org.cells.size >= maxCells) return false;
    if (cell.energy < divCost) return false;

    let bestDir: [number, number] | null = null;
    let bestEnv = -1;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx, ny = cell.y + dy;
      if (!this.world.isEmpty(nx, ny)) continue;
      const e = this.envEnergy[ny * GRID_WIDTH + nx];
      if (e > bestEnv) { bestEnv = e; bestDir = [nx, ny]; }
    }
    if (!bestDir) return false;

    const overhead = divCost * 0.15;
    const childE = divCost - overhead;
    const childCap = this.safeCellEnergyCapForOrg(cell.orgId);
    const childStored = Math.min(childE, childCap);
    const childOverflow = Math.max(0, childE - childStored);
    const childType = this.chooseChildCellType(cell, org, bestDir[0], bestDir[1]);
    this.world.setCell(
      bestDir[0],
      bestDir[1],
      cell.orgId,
      childType,
      childStored,
      tape.getLineagePacked(),
    );
    org.cells.add(bestDir[1] * GRID_WIDTH + bestDir[0]);
    cell.energy -= divCost;
    cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
    this.envEnergy[cell.idx] += overhead + childOverflow;
    return true;
  }

  private chooseChildCellType(cell: CellCtx, org: Organism, nx: number, ny: number): CellType {
    const envAtChild = this.envEnergy[ny * GRID_WIDTH + nx];
    const mood = org.nnDominant;

    if (mood === 2) return CellType.Motor;
    if (mood === 1 && org.cells.size >= 6) return CellType.Reproductive;
    if (envAtChild > 8 || this.envGradientScaled(cell.x, cell.y) > 2) return CellType.Sensor;
    if (this.world.getMarkerByIdx(cell.idx, MARKER_DIGEST) > 120) return CellType.Reproductive;
    return CellType.Stem;
  }

  private actionFire(cell: CellCtx, _org?: Organism): boolean {
    const b = cell.idx * U32_PER_CELL;
    const packed = this.world.cellData[b + 1];
    if (((packed >> 8) & 0xFF) !== 0) return false;
    this.world.cellData[b + 1] = (packed & 0xFF) | (1 << 8);
    return true;
  }

  private actionMove(_cell: CellCtx, org: Organism): boolean {
    if (org.cells.size === 0) return false;
    let bestDir: [number, number] = [0, 0];
    let bestScore = -Infinity;
    for (const [dx, dy] of this.moveDirs()) {
      if (!this.canMoveOrg(org, dx, dy)) continue;
      let score = 0;
      for (const idx of org.cells) {
        const cx = idx % GRID_WIDTH + dx;
        const cy = ((idx - idx % GRID_WIDTH) / GRID_WIDTH) + dy;
        if (cx >= 0 && cx < GRID_WIDTH && cy >= 0 && cy < GRID_HEIGHT) {
          score += this.envEnergy[cy * GRID_WIDTH + cx];
        }
      }
      const { foreignFaces, edgeFaces } = this.countRepulsionFacesIfOrgShifted(org, dx, dy);
      score -= MOVE_REPEL_FOREIGN_FACE * foreignFaces + MOVE_REPEL_MAP_EDGE_FACE * edgeFaces;
      if (dx !== 0 && dy !== 0) {
        const nnMove = Math.max(0, Math.min(1, org.nnOutput[NN_MOVE] ?? 0));
        const diagonalBiasPerCell =
          DIAGONAL_MOVE_SCORE_BIAS_PER_CELL_BASE + DIAGONAL_MOVE_SCORE_BIAS_PER_CELL_NN * nnMove;
        score += diagonalBiasPerCell * org.cells.size;
      }
      // slight random exploration scaled by NN move urgency
      score += (randomF32() - 0.3) * org.nnOutput[NN_MOVE] * 20;
      if (score > bestScore) { bestScore = score; bestDir = [dx, dy]; }
    }
    if (bestDir[0] !== 0 || bestDir[1] !== 0) {
      this.moveOrg(org, bestDir[0], bestDir[1]);
      this.distributeMoveCost(org, bestDir[0], bestDir[1]);
      for (const idx of org.cells) {
        this.world.bumpMarker(idx, MARKER_MOVE, MARKER_BUMP);
      }
      return true;
    }
    return false;
  }

  private distributeMoveCost(org: Organism, dx: number, dy: number) {
    const moveMult = dx !== 0 && dy !== 0 ? DIAGONAL_MOVE_COST_MULT : 1;
    for (const idx of org.cells) {
      const e = this.world.getCellEnergyByIdx(idx);
      const cost = Math.min(e, MOVE_COST_PER_CELL * moveMult);
      this.setCellEnergyCappedByIdx(idx, e - cost, org.id);
      this.envEnergy[idx] += cost;
    }
  }

  private actionReproduce(cell: CellCtx, org: Organism, _tape: Tape, childFraction: number): boolean {
    recordReproductionAttempt();
    if (org.cells.size < MIN_CELLS_TO_REPRODUCE) return false;
    if (org.reproduceCooldown > 0) return false;
    const share = org.cells.size / this.dominanceLiveCellCount;
    const dominanceFailChance = getDominanceReproduceFailChance(share, this.suppressionMode === 'on');
    if (dominanceFailChance > 0) {
      if (randomF32() < dominanceFailChance) {
        recordReproduceFailDominance();
        return false;
      }
    }
    const nOrgs = this.organisms.count;
    const crowdFailChance = getCrowdingReproduceFailChance(nOrgs, this.suppressionMode === 'on');
    if (crowdFailChance > 0) {
      if (randomF32() < crowdFailChance) {
        recordReproduceFailCrowding();
        return false;
      }
    }
    const minRequired = REPRODUCE_ACTION_COST + 1;
    if (cell.energy < minRequired) return false;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx, ny = cell.y + dy;
      if (!this.world.isEmpty(nx, ny)) continue;
      const childTape = transcribeForReproduction(org.tape);
      if (!childTape) {
        const e = cell.energy;
        const pay = Math.min(e, REPRODUCE_ACTION_COST);
        cell.energy = e - pay;
        cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
        this.envEnergy[cell.idx] += pay;
        org.reproduceCooldown = REPRODUCE_COOLDOWN_TICKS;
        return false;
      }
      const childId = this.world.nextOrganismId++;
      const budget = cell.energy - REPRODUCE_ACTION_COST;
      const childE = budget * childFraction;
      this.organisms.register(childId, childTape, { parentId: org.id, birthTick: this.simTick });
      const childCap = this.safeCellEnergyCapForOrg(childId);
      const childStored = Math.min(childE, childCap);
      const childOverflow = Math.max(0, childE - childStored);
      this.world.setCell(nx, ny, childId, CellType.Stem, childStored, childTape.getLineagePacked());
      cell.energy -= childE + REPRODUCE_ACTION_COST;
      cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
      this.envEnergy[cell.idx] += REPRODUCE_ACTION_COST + childOverflow;
      const childOrg = this.organisms.get(childId);
      if (childOrg) childOrg.cells.add(ny * GRID_WIDTH + nx);
      org.reproduceCooldown = REPRODUCE_COOLDOWN_TICKS;
      recordReproductionSuccess();
      recordBirthFromReproduce();
      return true;
    }
    return false;
  }

  /** True when morph A/B at two indices are both within match tolerance (symbiotic interface). */
  private morphAbsorbCompatible(idxA: number, idxB: number): boolean {
    const dA = Math.abs(this.world.getMorphogenA(idxA) - this.world.getMorphogenA(idxB));
    const dB = Math.abs(this.world.getMorphogenB(idxA) - this.world.getMorphogenB(idxB));
    return dA <= MORPH_ABSORB_MATCH_A && dB <= MORPH_ABSORB_MATCH_B;
  }

  private actionAbsorb(cell: CellCtx, maxSteal: number): boolean {
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx, ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const nOrg = this.world.getOrganismId(nx, ny);
      if (nOrg === 0 || nOrg === cell.orgId) continue;
      const nIdx = ny * GRID_WIDTH + nx;
      const ne = this.world.getCellEnergy(nx, ny);

      if (this.morphAbsorbCompatible(cell.idx, nIdx) && !this.isEdgeJammed(cell.idx, nIdx)) {
        const eC = cell.energy;
        if (eC <= 0 && ne <= 0) continue;
        const avg = (eC + ne) / 2;
        const r = ABSORB_BIDIR_RELAX;
        const newC = Math.max(0, eC + r * (avg - eC));
        const newN = Math.max(0, ne + r * (avg - ne));
        cell.energy = this.setCellEnergyCapped(cell.x, cell.y, newC, cell.orgId);
        this.setCellEnergyCapped(nx, ny, newN, nOrg);
        const contactFlux = Math.abs(newC - eC);
        this.tryHorizontalTapeTransfer(cell, nIdx, nOrg, contactFlux, Math.max(0.2, maxSteal * XENO_TRANSFER_CONTACT_SCALE));
        return true;
      }

      const steal = Math.min(ne, maxSteal);
      if (steal <= 0) continue;
      this.setCellEnergyCapped(nx, ny, ne - steal, nOrg);
      const stomach = this.world.getStomachByIdx(cell.idx);
      this.world.setStomachByIdx(cell.idx, stomach + steal);
      this.tryHorizontalTapeTransfer(cell, nIdx, nOrg, steal, maxSteal);
      return true;
    }
    return false;
  }

  /** Spill a small amount of own stomach into local environment (self + local neighbors). */
  private actionSpill(cell: CellCtx, amount: number): boolean {
    return spillStomachToNearbyEnv(
      cell,
      amount,
      (idx) => this.world.getStomachByIdx(idx),
      (idx, value) => this.world.setStomachByIdx(idx, value),
      (idx, delta) => {
        this.envEnergy[idx] += delta;
      },
      GRID_WIDTH,
      GRID_HEIGHT,
      this.neighborMode === 'eight',
    );
  }

  /** Defensive cut: short-lived jam at foreign boundary edges around this cell. */
  private actionJam(cell: CellCtx, intensity: number): boolean {
    const ttl = computeJamTtl(intensity, JAM_MIN_TICKS, JAM_MAX_EXTRA_TICKS);
    return applyJamToForeignBoundary(
      cell,
      ttl,
      (idx) => this.world.getOrganismIdByIdx(idx),
      (idx, jamTtl) => this.markJammed(idx, jamTtl),
      GRID_WIDTH,
      GRID_HEIGHT,
      this.neighborMode === 'eight',
    );
  }

  private markJammed(idx: number, ttl: number): void {
    const next = Math.max(this.jamTicks[idx], Math.min(255, Math.max(0, ttl)));
    this.jamTicks[idx] = next;
  }

  private isEdgeJammed(aIdx: number, bIdx: number): boolean {
    return this.jamTicks[aIdx] > 0 || this.jamTicks[bIdx] > 0;
  }

  /**
   * Low-rate cross-lineage tape transfer ("infection-like" HGT) at predatory absorb interface.
   * Transfer is resisted by local REPAIR quorum and paid as a small heat fee on success.
   */
  private tryHorizontalTapeTransfer(
    cell: CellCtx,
    donorIdx: number,
    donorOrgId: number,
    steal: number,
    maxSteal: number,
  ): void {
    if (this.isEdgeJammed(cell.idx, donorIdx)) return;
    const hostOrg = this.organisms.get(cell.orgId);
    const donorOrg = this.organisms.get(donorOrgId);
    if (!hostOrg || !donorOrg) return;
    if (steal <= 0.02 || maxSteal <= 0) return;

    const trust = this.kinTrustForeign(cell.idx, donorIdx, cell.orgId);
    const repairDefense = this.repairQuorumKin(cell);
    const contactPressure = this.foreignContactPressure(cell);
    const stealFactor = Math.min(1, steal / Math.max(0.01, maxSteal));
    const gut = this.world.getStomachByIdx(cell.idx);
    const stomachFactor = gut / (gut + XENO_TAPE_TRANSFER_STOMACH_K);
    const interfaceGain = 0.2 + 0.8 * contactPressure;
    const drive = Math.max(0, Math.min(1, (0.35 + trust) * stomachFactor * stealFactor * interfaceGain));
    const defense = repairDefense / (1 + repairDefense);
    if (drive <= defense) {
      recordXenoTransferAttempt(false, drive);
      return;
    }

    const slot = Math.floor(randomF32() * MAX_RULES);
    const idx = CONDITIONS_OFFSET + slot * RULE_SIZE + 2; // opcode byte only (behavioral takeover channel)
    let b = donorOrg.tape.data[idx];
    if (randomF32() < 0.12) {
      const bitPos = Math.floor(randomF32() * 8) & 7;
      b ^= 1 << bitPos;
    }
    hostOrg.tape.data[idx] = b & 0xff;
    hostOrg.tape.degradation[idx] = Math.min(255, hostOrg.tape.degradation[idx] + 24);
    recordXenoTransferAttempt(true, drive);

    const fee = Math.min(cell.energy, XENO_TAPE_TRANSFER_HEAT);
    if (fee > 0) {
      cell.energy -= fee;
      cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
      this.envEnergy[cell.idx] += fee;
    }
  }

  /** 0..1: foreign-face share among occupied neighbors around host cell. */
  private foreignContactPressure(cell: CellCtx): number {
    return computeForeignContactPressure(
      cell,
      (x, y) => this.world.getOrganismId(x, y),
      GRID_WIDTH,
      GRID_HEIGHT,
      this.neighborMode === 'eight',
    );
  }

  // ==================== MORPHOGEN DIFFUSION ====================
  private diffuseMorphogens(org: Organism) {
    const dirs = this.neighborDirs();
    const deltaA = new Map<number, number>();
    const deltaB = new Map<number, number>();

    for (const idx of org.cells) {
      const a = this.world.getMorphogenA(idx);
      const b = this.world.getMorphogenB(idx);
      if (a < 0.01 && b < 0.01) continue;

      const x = idx % GRID_WIDTH, y = (idx - x) / GRID_WIDTH;
      let sameNeighbors = 0;
      const neighbors: number[] = [];
      for (const [dx, dy] of dirs) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
        const ni = ny * GRID_WIDTH + nx;
        if (org.cells.has(ni)) {
          sameNeighbors++;
          neighbors.push(ni);
        }
      }
      if (sameNeighbors === 0) continue;

      const spreadA = a * MORPHOGEN_DIFFUSION / sameNeighbors;
      const spreadB = b * MORPHOGEN_DIFFUSION / sameNeighbors;
      for (const ni of neighbors) {
        deltaA.set(ni, (deltaA.get(ni) ?? 0) + spreadA);
        deltaB.set(ni, (deltaB.get(ni) ?? 0) + spreadB);
      }
      deltaA.set(idx, (deltaA.get(idx) ?? 0) - a * MORPHOGEN_DIFFUSION);
      deltaB.set(idx, (deltaB.get(idx) ?? 0) - b * MORPHOGEN_DIFFUSION);
    }

    for (const idx of org.cells) {
      let a = this.world.getMorphogenA(idx) + (deltaA.get(idx) ?? 0);
      let b = this.world.getMorphogenB(idx) + (deltaB.get(idx) ?? 0);
      a *= (1 - MORPHOGEN_DECAY);
      b *= (1 - MORPHOGEN_DECAY);
      this.world.setMorphogenA(idx, a);
      this.world.setMorphogenB(idx, b);
    }
  }

  // ==================== DIGESTION (single pipeline per tick) ====================
  // Stomach → this cell’s energy + heat to env. DIGEST opcode only pre-fills digestRuleBoost (capped); see DIGEST_RULE_BOOST_CAP.
  digestPhase() {
    for (const org of this.organisms.organisms.values()) {
      for (const idx of org.cells) {
        const stomach = this.world.getStomachByIdx(idx);
        if (stomach < 0.01) continue;
        const cellE = this.world.getCellEnergyByIdx(idx);
        const enzymeEff = Math.max(0, Math.min(1, cellE / 3));
        const specBonus = 1 + this.world.getMarkerByIdx(idx, MARKER_DIGEST) / 255;
        const boostMul = 1 + this.digestRuleBoost[idx]; // already ≤ DIGEST_RULE_BOOST_CAP
        const sameRatio = this.sameOrgNeighborRatioByIdx(idx, org.id);
        const networkMul = DIGEST_NETWORK_BASE + DIGEST_NETWORK_COEFF * sameRatio;
        const digested = stomach * PASSIVE_DIGEST_RATE * enzymeEff * specBonus * boostMul * networkMul;
        const heat = digested * DIGESTION_HEAT_LOSS;
        this.world.setStomachByIdx(idx, stomach - digested);
        this.setCellEnergyCappedByIdx(idx, cellE + digested - heat, org.id);
        this.envEnergy[idx] += heat;
      }
    }
  }

  // ==================== METABOLIC COST (density-dependent) ====================
  applyMetabolicCost() {
    const liveN = this.liveCellCount;

    for (const org of this.organisms.organisms.values()) {
      const share = org.cells.size / liveN;
      const dominanceMult = getDominanceMetabolicMultiplier(share, this.suppressionMode === 'on');
      const perCell = getPerCellMetabolicBase(org.cells.size, dominanceMult, this.metabolicScale);
      for (const idx of org.cells) {
        const sameRatio = this.sameOrgNeighborRatioByIdx(idx, org.id);
        const isolationMult = 1 + ISOLATION_METABOLIC_PENALTY * (1 - sameRatio);
        const e = this.world.getCellEnergyByIdx(idx);
        const cost = Math.min(e, perCell * isolationMult);
        const newE = e - cost;
        this.setCellEnergyCappedByIdx(idx, newE, org.id);
        this.envEnergy[idx] += cost;
        if (newE < 2) {
          const leak = Math.min(newE, LOW_ENERGY_LEAK_MAX);
          if (leak > 0) {
            this.setCellEnergyCappedByIdx(idx, newE - leak, org.id);
            this.envEnergy[idx] += leak;
          }
          if (randomF32() < 0.02) org.tape.applyReadDegradation(0, org.age);
        }
      }
    }
  }

  /** Per-lineage overhead: same absolute cost for tiny vs large org → multicell amortizes; culls org spam. */
  applyOrganismOverhead() {
    for (const org of this.organisms.organisms.values()) {
      if (org.cells.size === 0) continue;
      const indices = [...org.cells];
      let totalE = 0;
      for (const idx of indices) totalE += this.world.getCellEnergyByIdx(idx);
      if (totalE < 1e-8) continue;
      const takeTotal = Math.min(totalE, ORG_OVERHEAD_PER_TICK);
      for (const idx of indices) {
        const e = this.world.getCellEnergyByIdx(idx);
        const t = takeTotal * (e / totalE);
        this.setCellEnergyCappedByIdx(idx, e - t, org.id);
        this.envEnergy[idx] += t;
      }
    }
  }

  // ==================== CLEANUP ====================
  cleanupDeadOrganisms() {
    const nOrgs = this.organisms.count;
    const crowdDiss = getCrowdingDissolveBonus(nOrgs, this.suppressionMode === 'on');
    const baseDiss = DISSOLVE_BASE + crowdDiss;

    const dead: number[] = [];
    for (const org of this.organisms.organisms.values()) {
      const toDissolve: number[] = [];
      let p = org.cells.size === 1 ? baseDiss * DISSOLVE_SINGLE_MULT : baseDiss;
      p = Math.min(0.22, p);
      for (const idx of org.cells) {
        if (this.world.getOrganismIdByIdx(idx) !== org.id) { toDissolve.push(idx); continue; }
        if (this.world.getCellEnergyByIdx(idx) <= 0 && randomF32() < p) toDissolve.push(idx);
      }
      for (const idx of toDissolve) {
        const e = this.world.getCellEnergyByIdx(idx);
        const s = this.world.getStomachByIdx(idx);
        if (e > 0) this.envEnergy[idx] += e;
        if (s > 0) this.envEnergy[idx] += s;
        const base = idx * U32_PER_CELL;
        for (let i = 0; i < U32_PER_CELL; i++) this.world.cellData[base + i] = 0;
        org.cells.delete(idx);
      }
      if (org.cells.size === 0) dead.push(org.id);
    }
    for (const id of dead) this.organisms.remove(id);
  }

  // ==================== CONNECTIVITY CHECK ====================
  splitDisconnected() {
    const toSplit: Array<{ org: Organism; components: number[][] }> = [];
    for (const org of this.organisms.organisms.values()) {
      if (org.cells.size <= 1) continue;
      const components = this.findConnectedComponents(org);
      if (components.length > 1) toSplit.push({ org, components });
    }

    for (const { org, components } of toSplit) {
      components.sort((a, b) => b.length - a.length);
      const minFragCells = getSplitMinFragmentCells(this.organisms.count, this.suppressionMode === 'on');
      const keepWithParent = [...components[0]];
      const splitFrags: number[][] = [];
      for (let i = 1; i < components.length; i++) {
        const frag = components[i];
        if (frag.length < minFragCells) {
          keepWithParent.push(...frag);
          continue;
        }
        splitFrags.push(frag);
      }

      // Largest fragment keeps original org; tiny shards stay with parent to avoid lineage spam from singletons.
      org.cells = new Set(keepWithParent);
      if (splitFrags.length === 0) continue;
      recordSplitEvent(components[0]?.length ?? 0);

      // Multi-cell fragments become new organisms.
      for (const frag of splitFrags) {
        recordBirthFromSplit(frag.length);
        const newId = this.world.nextOrganismId++;
        const childTape = org.tape.clone();
        this.organisms.register(newId, childTape, { parentId: org.id, birthTick: this.simTick });
        const newOrg = this.organisms.get(newId)!;
        const splitCrowdExtra = getSplitCrowdExtraCooldown(this.organisms.count, this.suppressionMode === 'on');
        newOrg.reproduceCooldown = Math.round(
          Math.max(
            newOrg.reproduceCooldown,
            REPRODUCE_COOLDOWN_TICKS + SPLIT_CHILD_EXTRA_COOLDOWN + splitCrowdExtra,
          ),
        );
        for (const idx of frag) {
          // Update orgId in world data
          this.world.cellData[idx * U32_PER_CELL] = newId;
          newOrg.cells.add(idx);
        }
      }
    }
  }

  private findConnectedComponents(org: Organism): number[][] {
    const visited = new Set<number>();
    const components: number[][] = [];

    for (const start of org.cells) {
      if (visited.has(start)) continue;
      const component: number[] = [];
      const queue = [start];
      visited.add(start);

      while (queue.length > 0) {
        const idx = queue.pop()!;
        component.push(idx);
        const x = idx % GRID_WIDTH, y = (idx - x) / GRID_WIDTH;
        for (const [dx, dy] of this.neighborDirs()) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
          const ni = ny * GRID_WIDTH + nx;
          if (!visited.has(ni) && org.cells.has(ni)) {
            visited.add(ni);
            queue.push(ni);
          }
        }
      }
      components.push(component);
    }
    return components;
  }

  // ==================== HELPERS ====================

  private sameOrgNeighborRatioByIdx(idx: number, orgId: number): number {
    const dirs = this.neighborDirs();
    const x = idx % GRID_WIDTH;
    const y = (idx - x) / GRID_WIDTH;
    let same = 0;
    for (const [dx, dy] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      if (this.world.getOrganismId(nx, ny) === orgId) same++;
    }
    return same / dirs.length;
  }

  private spendEnergy(cell: CellCtx, cost: number) {
    cell.energy -= cost;
    cell.energy = this.setCellEnergyCapped(cell.x, cell.y, cell.energy, cell.orgId);
    this.envEnergy[cell.idx] += cost;
  }

  /** Spatial redistribution of env energy; global sum drift from open boundaries is corrected by `enforceClosedEnergyBudget`. */
  private stepEnvDiffusion() {
    const dirs = this.neighborDirs();
    const src = this.envEnergy;
    const dst = this.envScratch;
    for (let y = 0; y < GRID_HEIGHT; y++) {
      for (let x = 0; x < GRID_WIDTH; x++) {
        const i = y * GRID_WIDTH + x;
        const c = src[i];
        let sum = 0,
          cnt = 0;
        for (const [dx, dy] of dirs) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
          sum += src[ny * GRID_WIDTH + nx];
          cnt++;
        }
        dst[i] = c + (sum / cnt - c) * ENV_DIFFUSION_RATE;
      }
    }
    this.envEnergy = dst;
    this.envScratch = src;
  }

  /**
   * **Source of truth for same-org neural propagation** after FIRE/SIG. Reads/writes packed state in
   * `world.cellData` (+1 word: cellType | neuralState<<8 | refractory<<16). Not mirrored by any
   * active GPU compute shader in the current build.
   */
  private propagateSignals(org: Organism, tape: Tape) {
    const refPeriod = tape.getRefractoryPeriod();
    for (const idx of org.cells) {
      const b = idx * U32_PER_CELL;
      const packed = this.world.cellData[b + 1];
      const ns = (packed >> 8) & 0xFF;
      const refCnt = (packed >> 16) & 0xFF;
      const ct = packed & 0xFF;
      if (ns === 1) {
        this.world.cellData[b + 1] = ct | (2 << 8) | (refPeriod << 16);
        const x = idx % GRID_WIDTH, y = (idx - x) / GRID_WIDTH;
        for (const [dx, dy] of this.neighborDirs()) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
          const ni = ny * GRID_WIDTH + nx;
          if (!org.cells.has(ni)) continue;
          const np = this.world.cellData[ni * U32_PER_CELL + 1];
          if (((np >> 8) & 0xFF) === 0) {
            this.world.cellData[ni * U32_PER_CELL + 1] = (np & 0xFF) | (1 << 8);
          }
        }
      } else if (ns === 2) {
        this.world.cellData[b + 1] = refCnt <= 1 ? ct : ct | (2 << 8) | ((refCnt - 1) << 16);
      }
    }
  }

  /**
   * Local consensus (gossip): same-org neighbors softly pull signal marker toward their local mean.
   * This creates agreement dynamics from interaction topology without adding explicit "social mode".
   */
  private applySocialConsensusDrift(cell: CellCtx, orgId: number): void {
    const dirs = this.neighborDirs();
    let same = 0;
    let sumSignal = 0;
    for (const [dx, dy] of dirs) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const nIdx = ny * GRID_WIDTH + nx;
      if (this.world.getOrganismIdByIdx(nIdx) !== orgId) continue;
      same++;
      sumSignal += this.world.getMarkerByIdx(nIdx, MARKER_SIGNAL);
    }
    if (same === 0) return;

    const [eat, digest, signal, move] = this.world.getMarkersByIdx(cell.idx);
    const meanSignal = sumSignal / same;
    const couplingNorm = Math.max(2, dirs.length / 2);
    const coupling = SOCIAL_SIGNAL_CONSENSUS_RATE * Math.min(1, same / couplingNorm);
    const nextSignal = Math.round(signal + (meanSignal - signal) * coupling);
    this.world.setMarkersByIdx(cell.idx, eat, digest, Math.max(0, Math.min(255, nextSignal)), move);

    const cohesion = 1 - Math.min(1, Math.abs(meanSignal - signal) / 255);
    recordSocialCohesion(cohesion);
  }

  /** 0..1: how close this cell's signal marker is to same-org local mean. */
  private localSignalCohesion(cell: CellCtx): number {
    const dirs = this.neighborDirs();
    let same = 0;
    let sumSignal = 0;
    for (const [dx, dy] of dirs) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const nIdx = ny * GRID_WIDTH + nx;
      if (this.world.getOrganismIdByIdx(nIdx) !== cell.orgId) continue;
      same++;
      sumSignal += this.world.getMarkerByIdx(nIdx, MARKER_SIGNAL);
    }
    if (same === 0) return 0;
    const signal = this.world.getMarkerByIdx(cell.idx, MARKER_SIGNAL);
    const meanSignal = sumSignal / same;
    return 1 - Math.min(1, Math.abs(meanSignal - signal) / 255);
  }

  private rebuildCellList(org: Organism) {
    const alive = new Set<number>();
    for (const idx of org.cells) {
      if (this.world.getOrganismIdByIdx(idx) === org.id) alive.add(idx);
    }
    org.cells = alive;
  }

  private cellAt(idx: number, orgId: number): CellCtx | null {
    if (this.world.getCellTypeByIdx(idx) === CellType.Empty) return null;
    const x = idx % GRID_WIDTH, y = (idx - x) / GRID_WIDTH;
    return { x, y, idx, energy: this.world.getCellEnergyByIdx(idx), orgId };
  }

  private isOuterCell(idx: number, org: Organism): boolean {
    const x = idx % GRID_WIDTH, y = (idx - x) / GRID_WIDTH;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) return true;
      if (!org.cells.has(ny * GRID_WIDTH + nx)) return true;
    }
    return false;
  }

  private countNeighborsByType(cell: CellCtx, kind: 'same' | 'foreign' | 'empty'): number {
    let cnt = 0;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx, ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const nOrg = this.world.getOrganismId(nx, ny);
      const nType = this.world.getCellType(nx, ny);
      if (kind === 'same'    && nOrg === cell.orgId && nType !== CellType.Empty) cnt++;
      if (kind === 'foreign' && nOrg !== 0 && nOrg !== cell.orgId) cnt++;
      if (kind === 'empty'   && nType === CellType.Empty) cnt++;
    }
    return cnt;
  }

  /** Orthogonal neighbors occupied by a different organism (current world, not hypothetical shift). */
  private countForeignOrthFacesAtIdx(idx: number, orgId: number): number {
    const x = idx % GRID_WIDTH, y = (idx - x) / GRID_WIDTH;
    let n = 0;
    for (const [odx, ody] of ORTH_DIRS) {
      const nx = x + odx, ny = y + ody;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      const no = this.world.getOrganismId(nx, ny);
      if (no !== 0 && no !== orgId) n++;
    }
    return n;
  }

  /**
   * After rigidly shifting the whole organism by (dx,dy): count (1) orthogonal faces touching another org,
   * (2) faces on the map boundary. Empty neighbors are neutral so foraging into open space is not penalized.
   */
  private countRepulsionFacesIfOrgShifted(
    org: Organism,
    dx: number,
    dy: number,
  ): { foreignFaces: number; edgeFaces: number } {
    return countRepulsionFacesForShift(
      org.cells,
      org.id,
      dx,
      dy,
      (idx) => this.world.getCellTypeByIdx(idx),
      (idx) => this.world.getOrganismIdByIdx(idx),
      GRID_WIDTH,
      GRID_HEIGHT,
    );
  }

  private applyForeignInterfaceMetabolism(idx: number, orgId: number) {
    const faces = this.countForeignOrthFacesAtIdx(idx, orgId);
    if (faces <= 0) return;
    const e = this.world.getCellEnergyByIdx(idx);
    const tax = Math.min(Math.max(0, e - 1e-6), faces * FOREIGN_INTERFACE_METABOLISM);
    if (tax <= 0) return;
    this.setCellEnergyCappedByIdx(idx, e - tax, orgId);
    this.envEnergy[idx] += tax;
  }

  private envGradientScaled(x: number, y: number): number {
    const c = this.envEnergy[y * GRID_WIDTH + x];
    let mx = 0;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      mx = Math.max(mx, this.envEnergy[ny * GRID_WIDTH + nx]);
    }
    return (mx - c) * 10;
  }

  private maxNeighborEnv(cell: CellCtx): number {
    let mx = 0;
    for (const [dx, dy] of this.neighborDirs()) {
      const nx = cell.x + dx, ny = cell.y + dy;
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
      mx = Math.max(mx, this.envEnergy[ny * GRID_WIDTH + nx]);
    }
    return mx;
  }

  private orgTotalEnergy(org: Organism): number {
    let t = 0;
    for (const idx of org.cells) t += this.world.getCellEnergyByIdx(idx);
    return t;
  }

  private orgAvgMorphA(org: Organism): number {
    if (org.cells.size === 0) return 0;
    let t = 0;
    for (const idx of org.cells) t += this.world.getMorphogenA(idx);
    return t / org.cells.size;
  }

  private dominantMarker(idx: number): number {
    const [eat, digest, signal, move] = this.world.getMarkersByIdx(idx);
    const max = Math.max(eat, digest, signal, move);
    if (max === 0) return 0;
    if (max === eat)    return 64;
    if (max === digest) return 128;
    if (max === signal) return 192;
    return 255; // move
  }

  private canMoveOrg(org: Organism, dx: number, dy: number): boolean {
    for (const idx of org.cells) {
      const x = idx % GRID_WIDTH + dx;
      const y = ((idx - idx % GRID_WIDTH) / GRID_WIDTH) + dy;
      if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT) return false;
      const ni = y * GRID_WIDTH + x;
      if (!org.cells.has(ni) && this.world.getCellTypeByIdx(ni) !== CellType.Empty) return false;
    }
    return true;
  }

  private moveOrg(org: Organism, dx: number, dy: number) {
    const snaps: Array<{ newIdx: number; data: Uint32Array }> = [];
    for (const idx of org.cells) {
      const base = idx * U32_PER_CELL;
      snaps.push({
        newIdx: (((idx - idx % GRID_WIDTH) / GRID_WIDTH) + dy) * GRID_WIDTH + (idx % GRID_WIDTH + dx),
        data: this.world.cellData.slice(base, base + U32_PER_CELL),
      });
      for (let i = 0; i < U32_PER_CELL; i++) this.world.cellData[base + i] = 0;
    }
    org.cells.clear();
    for (const s of snaps) {
      this.world.cellData.set(s.data, s.newIdx * U32_PER_CELL);
      org.cells.add(s.newIdx);
    }
  }
}
