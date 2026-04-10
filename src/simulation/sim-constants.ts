/**
 * Centralised tuning constants for the PopComplex simulation.
 *
 * All numeric parameters that were previously scattered across `rule-evaluator.ts`,
 * `behaviors/action-costs.ts`, and `behaviors/stress-signals.ts` live here so that
 * parameter searches and design-space exploration require only a single file edit.
 *
 * Constants are grouped by subsystem with section comments.
 * Structural / architectural constants (grid dimensions, enum values, tape layout) remain
 * in `constants.ts` and `tape.ts` respectively.
 */

// ==================== ACTION COSTS ====================
/** Neural-fire / SIG heat released to env on success. */
export const ACTION_COST_FIRE        = 0.4;
/** Base MOVE heat (also exported as MOVE_COST_PER_CELL). */
export const ACTION_COST_MOVE        = 0.5;
/** Overhead for REPRODUCE action invocation (paid regardless of child size). */
export const ACTION_COST_REPRODUCE   = 5.0;
export const ACTION_COST_ABSORB      = 0.3;
export const ACTION_COST_DIGEST      = 0.2;
export const ACTION_COST_TAKE        = 0.1;
export const ACTION_COST_GIVE        = 0.05;
export const ACTION_COST_EMIT        = 0.15;
export const ACTION_COST_REPAIR      = 0.28;
export const ACTION_COST_SPILL       = 0.08;
export const ACTION_COST_JAM         = 0.06;

// ==================== METABOLIC / DISTRESS ====================
export const DISTRESS_FIRE_CHANCE       = 0.2;
export const DISTRESS_ENERGY_THRESH     = 3;

export const BASE_METABOLIC             = 0.25;
export const CROWD_METABOLIC            = 0.008;
export const DOMINANCE_SHARE_SOFT       = 0.08;
export const DOMINANCE_METABOLIC_COEFF  = 5.5;
export const DOMINANCE_METABOLIC_CAP    = 2.4;

export const DOMINANCE_SHARE_REP           = 0.1;
export const DOMINANCE_REP_FAIL_COEFF      = 2.2;
export const DOMINANCE_REP_FAIL_CAP        = 0.72;
export const REPRO_CROWD_START_ORGS        = 700;
export const REPRO_CROWD_FAIL_COEFF        = 0.00014;
export const REPRO_CROWD_FAIL_CAP          = 0.58;

export const DISSOLVE_CROWD_START          = 2500;
export const DISSOLVE_CROWD_BONUS          = 0.000045;
export const DISSOLVE_CROWD_CAP            = 0.045;

export const SPLIT_CROWD_COOLDOWN_START_ORGS    = 300;
export const SPLIT_CROWD_COOLDOWN_COEFF         = 0.03;
export const SPLIT_CROWD_COOLDOWN_CAP           = 36;
export const SPLIT_CROWD_MIN_FRAGMENT_START_ORGS = 450;
export const SPLIT_CROWD_MIN_FRAGMENT_CELLS      = 4;
export const SPLIT_MIN_FRAGMENT_CELLS            = 2;

// ==================== REPAIR ====================
/** Success mult += coeff * weighted neighbor quorum (Moore). */
export const REPAIR_NEIGHBOR_COEFF              = 0.16;
/** Base success probability before bias × intensity. */
export const REPAIR_BASE_P                      = 0.36;
/** Degradation subtracted on success (primary byte). */
export const REPAIR_DEG_HEAL                    = 12;
/** Small neighbourhood "reflow" fraction per success. */
export const REPAIR_DEG_HEAL_SPREAD_FRAC        = 0.25;
/** Quorum-gated: copy a good rule into a broken slot (re-route). */
export const REPAIR_RULE_BYPASS_BASE_P          = 0.18;
/** Extra targeting weight for invalid rule opcode bytes. */
export const REPAIR_INVALID_OPCODE_TARGET_BONUS = 220;

// ==================== PROXY / FAIL-SOFT ====================
/** Energy units paid to env per proxy attempt (scaled). */
export const PROXY_EXEC_TAX_BASE  = 0.12;
/** Energy units paid to env on fail-soft idle action (scaled). */
export const FAILSOFT_TAX_BASE    = 0.18;

// ==================== KIN TRUST ====================
export const KIN_TRUST_CAP_FOREIGN      = 0.52;
export const KIN_TRUST_FOREIGN_SCALE    = 1.12;
export const KIN_GIVE_MIN_TRUST         = 0.36;
/** Max foreign GIVE rate vs same-org GIVE strength. */
export const KIN_GIVE_RATE_CAP          = 0.24;
/** Mis-altruism waste scales with imperfect trust. */
export const KIN_GIVE_EXTRA_HEAT        = 0.2;
/** Foreign neighbour adds trust×this to repair quorum (capped trust above). */
export const REPAIR_FOREIGN_KIN_WEIGHT  = 0.4;
/** Extra heat when local signal marker + morph A disagree with same-org neighbour. */
export const SAME_ORG_TRANSFER_MISMATCH_EXTRA = 0.14;

// ==================== DIGESTION ====================
export const ISOLATION_METABOLIC_PENALTY = 0.12;
export const PASSIVE_DIGEST_RATE         = 0.15;
export const DIGESTION_HEAT_LOSS         = 0.25;
export const LOW_ENERGY_LEAK_MAX         = 0.08;
/** same-neighbour ratio 0.0 → 82% of baseline digest throughput. */
export const DIGEST_NETWORK_BASE         = 0.82;
/** ratio 1.0 (8 same Moore neighbours) → +32%. */
export const DIGEST_NETWORK_COEFF        = 0.32;
export const DIGEST_RULE_BOOST_CAP       = 1.0;
export const PASSIVE_ABSORB_RATE         = 0.15;

// ==================== SCAN TAXES ====================
export const SCAN_TAX_ALL           = 0.00005;
export const SCAN_TAX_NOP_EXTRA     = 0.00035;
export const SCAN_TAX_INVALID_EXTRA = 0.00100;

// ==================== FOREIGN INTERFACE / MOVE ====================
/** Orthogonal contact tension (energy → env, closed system). */
export const FOREIGN_INTERFACE_METABOLISM           = 0.055;
/** MOVE score penalty per foreign face after shift. */
export const MOVE_REPEL_FOREIGN_FACE                = 14;
/** MOVE score penalty per world-edge face after shift. */
export const MOVE_REPEL_MAP_EDGE_FACE               = 4;
export const DIAGONAL_MOVE_COST_MULT                = Math.SQRT2;
export const DIAGONAL_MOVE_SCORE_BIAS_PER_CELL_BASE = 0;
export const DIAGONAL_MOVE_SCORE_BIAS_PER_CELL_NN   = 0;

// ==================== DIV / BRANCHING ====================
export const DIV_CHOICE_TOP_K       = 4;
export const DIV_CHOICE_BASE_W      = 0.15;
export const ENV_GRAD_FLAT          = 0.5;
export const ENV_GRAD_STEEP         = 6.0;
export const DIV_TEMP_FLAT          = 10.0;
export const DIV_TEMP_STEEP         = 3.5;
export const OUTER_EAT_MULT_FLAT    = 1.20;
export const OUTER_EAT_MULT_STEEP   = 1.60;
export const OUTER_ABSORB_MULT_FLAT = 1.35;
export const OUTER_ABSORB_MULT_STEEP = 1.15;

// ==================== ABSORB / HGT ====================
export const ABSORB_RELAX_RATE               = 0.26;
export const XENO_TAPE_TRANSFER_STOMACH_K    = 2.4;
export const XENO_TAPE_TRANSFER_HEAT         = 0.08;
export const XENO_TRANSFER_CONTACT_SCALE     = 0.50;

// ==================== DIFFERENTIATION ====================
export const DIFF_COMMIT_RISE   = 3;
export const DIFF_COMMIT_FALL   = 4;
export const DIFF_COMMIT_UPPER  = 140;
export const DIFF_COMMIT_LOWER  = 50;
export const DIFF_MIN_SCORE     = 0.70;
/** Sensor: +12% effective eat strength. */
export const SENSOR_EAT_BONUS    = 0.12;
/** Motor: −10% move cost per cell. */
export const MOTOR_MOVE_DISCOUNT = 0.10;
/** Reproductive: +15% digest-rule boost contribution. */
export const REPRO_DIGEST_BONUS  = 0.15;

// ==================== DEVELOPMENT (LIFE HISTORY) ====================
export const DEV_JUVENILE_EAT_BONUS     = 0.15;
/** GROWING (stage=1): -10% DIV cost (efficient cell division during growth phase). */
export const DEV_GROWING_DIV_DISCOUNT   = 0.10;
export const DEV_MATURE_REPRO_DISCOUNT  = 0.15;
export const DEV_SENESCENT_ROT_PER_TICK = 0.003;

// ==================== ADAPTIVE IMMUNITY (xenoTolerance) ====================
export const XENO_LEARN_RATE  = 0.04;
export const XENO_TRUST_SCALE = 0.3;
export const XENO_NN_SCALE    = 0.5;

// ==================== METABOLIC TOXIN LOAD ====================
// Steady-state formula: toxin_ss = accumulation_per_tick / TOXIN_PASSIVE_DECAY
//   Digest-only (digested=1.0): 1.0 × 0.001 / 0.010 = 0.10
//   Foreign-only (3 faces):     3   × 0.001 / 0.010 = 0.30
//   Combined (high activity):   ≈ 0.4–0.6 (with quadratic suppress at 20%)
export const TOXIN_DIGEST_RATE   = 0.001;
/**
 * Max digest-rate suppression at full toxin load (quadratic: effectiveRate = base × (1 - toxin² × SUPPRESS)).
 * Quadratic curve: toxin=0.1 → 0.2% suppression; toxin=0.5 → 5%; toxin=1.0 → 20%.
 * This keeps the effect negligible at typical steady-state loads (~0.1–0.3) while
 * providing meaningful pressure only on genuinely overloaded cells.
 */
export const TOXIN_SUPPRESS      = 0.20;
export const TOXIN_PASSIVE_DECAY = 0.010;
export const TOXIN_FOREIGN_RATE  = 0.001;
export const TOXIN_ROT_BOOST     = 0.001;
export const TOXIN_REPAIR_CLEAR  = 0.15;
export const TOXIN_STRESS_WEIGHT = 0.10;

// ==================== REPRODUCTION ====================
export const MIN_CELLS_TO_REPRODUCE    = 2;
export const REPRODUCE_COOLDOWN_TICKS  = 40;

// ==================== ORGANISM OVERHEAD / DISSOLUTION ====================
export const ORG_OVERHEAD_PER_TICK = 0.042;
export const DISSOLVE_SINGLE_MULT  = 1.45;
export const DISSOLVE_BASE         = 0.016;

// ==================== SPLIT ====================
export const SPLIT_CHILD_EXTRA_COOLDOWN = 12;

// ==================== MORPHOGENS ====================
export const MORPHOGEN_DIFFUSION = 0.2;
export const MORPHOGEN_DECAY     = 0.05;

// ==================== MARKERS / SOCIAL ====================
export const MARKER_BUMP                  = 4;
export const SOCIAL_SIGNAL_CONSENSUS_RATE = 0.06;
export const SOCIAL_REPAIR_COHESION_BONUS = 0.03;
export const JAM_MIN_TICKS                = 2;
export const JAM_MAX_EXTRA_TICKS          = 3;

/** Marker slot indices: 0=eat, 1=digest, 2=signal, 3=move. */
export const MARKER_EAT    = 0 as const;
export const MARKER_DIGEST = 1 as const;
export const MARKER_SIGNAL = 2 as const;
export const MARKER_MOVE   = 3 as const;

// ==================== ENERGY BUDGET ====================
export const BUDGET_RESCALE_THRESHOLD = 0.5;
