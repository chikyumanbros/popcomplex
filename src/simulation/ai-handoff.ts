/**
 * Builds a markdown report for pasting into an AI chat (debug / design review).
 * Written in English in the doc blocks; user-facing button label is Japanese in UI.
 */

import { GRID_WIDTH, GRID_HEIGHT, INITIAL_ENV_ENERGY_PER_CELL, TOTAL_CELLS } from './constants';
import {
  NN_TAPE_WEIGHT_CENTER,
  NN_TAPE_WEIGHT_SCALE,
  PROTO_TAPE_NN_SEED,
  TAPE_SNAPSHOT_BYTES,
  tapeSnapshotBase64,
} from './tape';
import {
  CHANNEL_SWAP_ACCEPT_CROSS_REGION_MULT,
  CHANNEL_SWAP_ACCEPT_REPL_KEY_MULT,
  CHANNEL_SWAP_ACCEPT_RULE_TABLE_MULT,
  CHANNEL_SWAP_BASE_PROB,
} from './transcription';
import { U32_PER_CELL, type World } from './world';
import { NN_MOVE, type OrganismManager, type Organism } from './organism';
import type { RuleEvaluator } from './rule-evaluator';
import type { UIState } from '../ui/controls';
import { measureEnergyBookkeeping, biomassReservoirTotal, type EnergyBookkeeping } from './energy-metrics';
import { getRandomSeed } from './rng';
import { countInvalidRuleOpcodes } from './tape-health';
import type { TelemetrySnapshot } from './telemetry';

function percentile(sorted: number[], p01: number): number {
  if (sorted.length === 0) return 0;
  const p = Math.max(0, Math.min(1, p01));
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx]!;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function pearsonCorr(xs: number[], ys: number[]): number {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i]!;
    my += ys[i]!;
  }
  mx /= n;
  my /= n;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - mx;
    const dy = ys[i]! - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const den = Math.sqrt(dx2 * dy2);
  if (den <= 1e-12) return 0;
  return num / den;
}

function tapeHexDump(data: Uint8Array): string {
  const labels: Record<number, string> = {
    0: '── data ──',
    32: '── CA ──',
    64: '── rules ──',
    128: '── NN wt ──',
  };
  const lines: string[] = [];
  for (let off = 0; off < data.length; off += 16) {
    if (labels[off]) lines.push(labels[off]);
    const parts: string[] = [];
    for (let b = 0; b < 16 && off + b < data.length; b++) {
      parts.push(data[off + b].toString(16).padStart(2, '0'));
    }
    lines.push(`${off.toString(16).padStart(3, '0')}: ${parts.join(' ')}`);
  }
  return lines.join('\n');
}

function tapeDegradationHexDump(deg: Uint8Array): string {
  const lines: string[] = ['── degradation (u8, parallel to data indices 0..255) ──'];
  for (let off = 0; off < deg.length; off += 16) {
    const parts: string[] = [];
    for (let b = 0; b < 16 && off + b < deg.length; b++) {
      parts.push(deg[off + b].toString(16).padStart(2, '0'));
    }
    lines.push(`${off.toString(16).padStart(3, '0')}: ${parts.join(' ')}`);
  }
  return lines.join('\n');
}

type MoodIdx = 0 | 1 | 2 | 3;

const MOOD_NAMES: Record<MoodIdx, string> = {
  0: 'eat',
  1: 'grow',
  2: 'move',
  3: 'conserve',
};

interface OrgProfile {
  id: number;
  /** Public kin tag (“face”): render + foreign kin trust; may diverge from `geneticKinTag` (mimicry). */
  lineage: number;
  /** Private genetic kin tag from tape bytes 33–36; not used in `kinTrustForeign`. */
  geneticKinTag: number;
  parentId: number | null;
  cells: number;
  age: number;
  energy: number;
  stomach: number;
  biomass: number;
  meanEnergy: number;
  meanStomach: number;
  morphA: number;
  morphB: number;
  meanMarkers: [number, number, number, number];
  boundaryRatio: number;
  compactness: number;
  centerX: number;
  centerY: number;
  nnOutput: Float32Array;
  nnDominant: MoodIdx;
  invalidOpcode: number;
  validRules: number;
  nopRules: number;
  liveCellRatio: number;
  deadCellRatio: number;
  rotMean: number;
  rotMax: number;
  rotMaxDead: number;
}

function softOrgStateLabel(p: Pick<OrgProfile, 'liveCellRatio' | 'deadCellRatio' | 'rotMaxDead'>): string {
  const live = Math.max(0, Math.min(1, p.liveCellRatio));
  const dead = Math.max(0, Math.min(1, p.deadCellRatio));
  const rDead = Math.max(0, Math.min(1, p.rotMaxDead));

  // Avoid calling a functioning organism "dead" because of a single necrotic cell.
  if (live >= 0.98) return dead <= 0.02 ? 'operational' : 'operational+necrosis';
  if (live >= 0.80) return dead <= 0.10 ? 'strained' : 'strained+necrosis';
  if (live >= 0.50) return 'limping';

  // Mostly non-operational tissue: describe process without discrete death language.
  if (live > 0.05) return rDead > 0.7 ? 'dissolving' : 'inactive';
  return rDead > 0.7 ? 'near-dissolve' : 'inactive';
}

function summarizeOrganism(world: World, org: Organism): OrgProfile {
  let energy = 0;
  let stomach = 0;
  let morphA = 0;
  let morphB = 0;
  let markerEat = 0;
  let markerDigest = 0;
  let markerSignal = 0;
  let markerMove = 0;
  let cx = 0;
  let cy = 0;
  let minX = GRID_WIDTH;
  let minY = GRID_HEIGHT;
  let maxX = 0;
  let maxY = 0;
  let boundaryFaces = 0;
  let liveCells = 0;
  let deadCells = 0;
  let rotSum = 0;
  let rotMax = 0;
  let rotMaxDead = 0;

  const own = org.cells;
  for (const idx of own) {
    const x = idx % GRID_WIDTH;
    const y = (idx - x) / GRID_WIDTH;
    cx += x;
    cy += y;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;

    energy += world.getCellEnergyByIdx(idx);
    stomach += world.getStomachByIdx(idx);
    morphA += world.getMorphogenA(idx);
    morphB += world.getMorphogenB(idx);
    const e = world.getCellEnergyByIdx(idx);
    const r = world.rot[idx] ?? 0;
    if (e > 0) {
      liveCells++;
    } else {
      deadCells++;
      if (r > rotMaxDead) rotMaxDead = r;
    }
    rotSum += r;
    if (r > rotMax) rotMax = r;
    const [mkEat, mkDigest, mkSignal, mkMove] = world.getMarkersByIdx(idx);
    markerEat += mkEat;
    markerDigest += mkDigest;
    markerSignal += mkSignal;
    markerMove += mkMove;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) {
          boundaryFaces++;
          continue;
        }
        const nIdx = ny * GRID_WIDTH + nx;
        if (!own.has(nIdx)) boundaryFaces++;
      }
    }
  }

  const n = Math.max(1, own.size);
  const bboxW = Math.max(1, maxX - minX + 1);
  const bboxH = Math.max(1, maxY - minY + 1);
  const nnDom = Math.max(0, Math.min(3, org.nnDominant)) as MoodIdx;
  const rules = countInvalidRuleOpcodes(org.tape.data);

  return {
    id: org.id,
    lineage: org.tape.getPublicKinTagPacked() & 0xffffff,
    geneticKinTag: org.tape.getGeneticKinTagPacked() & 0xffffff,
    parentId: org.parentId,
    cells: own.size,
    age: org.age,
    energy,
    stomach,
    biomass: energy + stomach,
    meanEnergy: energy / n,
    meanStomach: stomach / n,
    morphA,
    morphB,
    meanMarkers: [markerEat / n, markerDigest / n, markerSignal / n, markerMove / n],
    boundaryRatio: boundaryFaces / (8 * n),
    compactness: own.size / (bboxW * bboxH),
    centerX: cx / n,
    centerY: cy / n,
    nnOutput: org.nnOutput,
    nnDominant: nnDom,
    invalidOpcode: rules.invalid,
    validRules: rules.valid,
    nopRules: rules.nop,
    liveCellRatio: liveCells / n,
    deadCellRatio: deadCells / n,
    rotMean: rotSum / n,
    rotMax,
    rotMaxDead,
  };
}

export interface AIHandoffInput {
  tick: number;
  world: World;
  organisms: OrganismManager;
  ruleEval: RuleEvaluator;
  ui?: UIState;
  /** If omitted, computed via full grid scan. */
  bookkeepingSnapshot?: EnergyBookkeeping;
  /** Optional: additional counters since last telemetry reset. */
  telemetrySnapshot?: TelemetrySnapshot;
  /**
   * Optional: how many sim ticks have elapsed since the last telemetry reset for `telemetrySnapshot`.
   * In the browser build, the periodic console logger may reset telemetry frequently.
   */
  telemetryWindowTicks?: number;
}

export type AIHandoffPromptPreset = 'review' | 'ecology' | 'tape';

export function buildAIHandoffPrompt(preset: AIHandoffPromptPreset): string {
  switch (preset) {
    case 'ecology':
      return [
        'Focus on ecology and evolution.',
        'From the mood mix, lineage shares, and notable organisms, infer active niches (forager, mover, digester, signal-coordinator, etc.), explain why they are stable/unstable, and propose 3 interventions to increase long-horizon diversity without breaking energy closure.',
        'Important: This system has explicit network/topology and quorum effects (connectivity-dependent digestion, isolation penalties, REPAIR quorum + social cohesion bonus, JAM-gated cross-lineage cooperation). Interpret robustness and “limping” strategies in that light, not as bugs by default.',
      ].join('\n');
    case 'tape':
      return [
        'Focus on tape health and behavioral reliability.',
        'Use invalidOpcode counts, degradation sums, and the provided tape dumps to identify failure modes (dead rules, brittle regions, over-mutation zones), then suggest exact checks/metrics to add in code/tests.',
        'Important: Many invalid opcodes can be neutral (fail-soft to NOP) and organisms can remain viable via network-mediated robustness. Distinguish “harmless dead rows” from true failure (loss of essential pipeline, runaway costs, or loss of adaptive response).',
      ].join('\n');
    case 'review':
    default:
      return [
        'Analyze this snapshot as an artificial-life code reviewer.',
        'Prioritize: (1) likely logic bugs/regressions, (2) ecological interpretation of niches and dominance, (3) concrete parameter/code changes with expected side effects.',
        'Use specific organism IDs/lineages from the report when explaining.',
        'Important: Account for intended robustness: same-org network effects, quorum/consensus biases, and fail-soft opcode normalization (unknown opcodes → NOP). Avoid labeling “high invalidOpcode” as broken unless it correlates with clear functional loss.',
      ].join('\n');
  }
}

export function buildAIHandoffMarkdown(input: AIHandoffInput): string {
  const { tick, world, organisms, ruleEval, ui, bookkeepingSnapshot, telemetrySnapshot, telemetryWindowTicks } = input;
  const orgList = [...organisms.organisms.values()].sort((a, b) => b.cells.size - a.cells.size);

  let orgRegistryCells = 0;
  for (const org of orgList) orgRegistryCells += org.cells.size;

  const bk = bookkeepingSnapshot ?? measureEnergyBookkeeping(world, ruleEval);
  const totalEnv = bk.envU;
  const totalCellE = bk.cellEnergy;
  const totalStomach = bk.stomach;
  const occupied = bk.occupiedCells;
  const systemE = biomassReservoirTotal(bk);
  const fillRatio = occupied / TOTAL_CELLS;

  const lines: string[] = [];

  lines.push('## PopComplex — AI handoff snapshot');
  lines.push('');
  lines.push('### How to use (for the human)');
  lines.push('Paste this whole block into an AI assistant and ask e.g.: energy conservation issues, dominance / diversity, tape robustness, or design suggestions.');
  lines.push(
    `- **Full tape + wear**: logical payload is **256B data + 256B degradation**. Hex blocks below list both. **TAPE512_B64** is fixed-length base64 of data‖degradation (${TAPE_SNAPSHOT_BYTES} raw bytes). Decode in-app: decodeTapeSnapshotBase64 → tapeFromSnapshot in tape.ts.`,
  );
  lines.push('');
  lines.push('### Request (for the AI)');
  lines.push(
    'This is a snapshot from **popcomplex**: WebGPU render + CPU `RuleEvaluator`. Payload is 256B tape **data**; **degradation** (256B) tracks wear separately — include both for bit-faithful state.',
  );
  lines.push('Please: (1) flag likely bugs or inconsistencies, (2) note if totals suggest energy leaks, (3) comment on rule-table health (invalid opcodes), (4) identify ecological niches / notable morphologies, (5) suggest concrete code-level checks if needed.');
  lines.push('');

  lines.push('### Runtime');
  lines.push(`- **tick**: ${tick}`);
  lines.push(`- **seed**: ${getRandomSeed()}`);
  lines.push(`- **organisms**: ${organisms.count}`);
  lines.push(`- **occupied cells (grid)**: ${occupied} / ${TOTAL_CELLS} (${(fillRatio * 100).toFixed(2)}% of grid)`);
  if (orgRegistryCells !== occupied) {
    lines.push(
      `- **WARN**: organism registry Σcells=${orgRegistryCells} ≠ grid occupied=${occupied} (orphan / desync — totals use **grid** scan)`,
    );
  }
  lines.push(`- **sum(envEnergy)**: ${totalEnv.toFixed(2)}`);
  lines.push(`- **sum(cell energy)**: ${totalCellE.toFixed(2)}`);
  lines.push(`- **sum(stomach)**: ${totalStomach.toFixed(2)}`);
  lines.push(`- **env + cells + stomach (measured)**: ${systemE.toFixed(2)}`);
  lines.push(
    `- **ecosystemEnergyBudget (closed)**: ${ruleEval.ecosystemEnergyBudget.toFixed(2)} — should equal measured total except float noise.`,
  );
  lines.push(
    `- **drift (measured − budget)**: ${(systemE - ruleEval.ecosystemEnergyBudget).toFixed(4)}`,
  );
  lines.push(`- **sum(morphA) / morphB** (not in closed budget): ${bk.morphA.toFixed(1)} / ${bk.morphB.toFixed(1)}`);
  if (telemetrySnapshot) {
    const t = telemetrySnapshot;
    const repairExec = t.actionExec?.[0x0c] ?? 0;
    const repairSucc = (t as any).repairSuccess;
    const repairAtt = (t as any).repairAttempts;
    const clampN = (t as any).invalidOpcodeClamps;
    lines.push(
      `- **telemetry (since last reset)**: emitA=${t.morphAEmitted.toFixed(2)} emitB=${t.morphBEmitted.toFixed(2)} ` +
        `decayA=${t.morphADecayed.toFixed(2)} decayB=${t.morphBDecayed.toFixed(2)} ` +
        `gutLeak=${t.gutLeakTotal.toFixed(2)} recovered=${t.gutLeakRecovered.toFixed(2)} toEnv=${t.gutLeakToEnv.toFixed(2)} ` +
        `repairExec=${repairExec}` +
        (typeof repairAtt === 'number' ? ` repairAtt=${repairAtt}` : '') +
        (typeof repairSucc === 'number' ? ` repairSucc=${repairSucc}` : '') +
        (typeof clampN === 'number' ? ` invalidClamp=${clampN}` : ''),
    );
    if (typeof telemetryWindowTicks === 'number' && Number.isFinite(telemetryWindowTicks) && telemetryWindowTicks >= 0) {
      lines.push(`- **telemetry window**: ~${Math.round(telemetryWindowTicks)} ticks`);
    }
  }
  if (ui) {
    lines.push(`- **UI**: paused=${ui.paused} speed=${ui.speed}`);
  }
  lines.push('');

  lines.push('### Build / grid');
  lines.push(`- **grid**: ${GRID_WIDTH}×${GRID_HEIGHT}`);
  lines.push(`- **initial env per cell**: ${INITIAL_ENV_ENERGY_PER_CELL} (see \`INITIAL_ENV_ENERGY_PER_CELL\`)`);
  lines.push(`- **proto NN seed (fixed starter)**: 0x${PROTO_TAPE_NN_SEED.toString(16)}`);
  lines.push('');

  const profiles = orgList.map((org) => summarizeOrganism(world, org));
  const moodOrgCounts = [0, 0, 0, 0];
  const moodCellCounts = [0, 0, 0, 0];
  for (const p of profiles) {
    moodOrgCounts[p.nnDominant] += 1;
    moodCellCounts[p.nnDominant] += p.cells;
  }
  const moodOrgLine = ([0, 1, 2, 3] as MoodIdx[])
    .map((m) => `${MOOD_NAMES[m]}:${moodOrgCounts[m]}`)
    .join(' | ');
  const moodCellLine = ([0, 1, 2, 3] as MoodIdx[])
    .map((m) => `${MOOD_NAMES[m]}:${moodCellCounts[m]}`)
    .join(' | ');

  let micro = 0;
  let small = 0;
  let medium = 0;
  let macro = 0;
  for (const p of profiles) {
    if (p.cells <= 1) micro += 1;
    else if (p.cells <= 4) small += 1;
    else if (p.cells <= 15) medium += 1;
    else macro += 1;
  }

  /** Counts **public** kin tags on cells (apparent lineages; mimics merge here). */
  const lineageCounts = new Map<number, number>();
  for (let i = 0; i < TOTAL_CELLS; i++) {
    if (world.getOrganismIdByIdx(i) === 0) continue;
    const lineage = world.cellData[i * U32_PER_CELL + 7] & 0xffffff;
    lineageCounts.set(lineage, (lineageCounts.get(lineage) ?? 0) + 1);
  }
  const topLineages = [...lineageCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  // Tape health distribution (ecosystem-wide).
  const invalidPerOrg: number[] = [];
  let digestIntactOrgs = 0;
  let digestIntactCells = 0;
  for (const org of orgList) {
    const rules = countInvalidRuleOpcodes(org.tape.data);
    invalidPerOrg.push(rules.invalid);
    if (org.tape.isDigestModuleIntact()) {
      digestIntactOrgs++;
      digestIntactCells += org.cells.size;
    }
  }
  invalidPerOrg.sort((a, b) => a - b);

  lines.push('### Ecosystem observability snapshot');
  lines.push(`- **mood mix by organism**: ${moodOrgLine}`);
  lines.push(`- **mood mix by occupied cells**: ${moodCellLine}`);
  lines.push(`- **size classes (org count)**: micro(1)=${micro}, small(2-4)=${small}, medium(5-15)=${medium}, macro(16+)=${macro}`);
  if (invalidPerOrg.length > 0) {
    lines.push(
      `- **invalidOpcode per organism (rules table)**: ` +
        `p50=${percentile(invalidPerOrg, 0.50).toFixed(0)} ` +
        `p90=${percentile(invalidPerOrg, 0.90).toFixed(0)} ` +
        `p99=${percentile(invalidPerOrg, 0.99).toFixed(0)} ` +
        `max=${percentile(invalidPerOrg, 1.00).toFixed(0)} (out of 16)`,
    );
  }
  if (profiles.length >= 4) {
    const inv = profiles.map((p) => p.invalidOpcode);
    const age = profiles.map((p) => p.age);
    const cellsArr = profiles.map((p) => p.cells);
    const biomass = profiles.map((p) => p.biomass);
    const ePerCell = profiles.map((p) => p.meanEnergy);
    const sPerCell = profiles.map((p) => p.meanStomach);
    const reproCd = profiles.map((p) => (organisms.get(p.id)?.reproduceCooldown ?? 0));

    const invSorted = [...inv].sort((a, b) => a - b);
    const p25 = percentile(invSorted, 0.25);
    const p75 = percentile(invSorted, 0.75);
    const low = profiles.filter((p) => p.invalidOpcode <= p25);
    const high = profiles.filter((p) => p.invalidOpcode >= p75);

    const lowAge = mean(low.map((p) => p.age));
    const highAge = mean(high.map((p) => p.age));
    const lowCells = mean(low.map((p) => p.cells));
    const highCells = mean(high.map((p) => p.cells));

    lines.push(
      `- **invalidOpcode correlations (Pearson r)**: ` +
        `age=${pearsonCorr(inv, age).toFixed(3)} ` +
        `cells=${pearsonCorr(inv, cellsArr).toFixed(3)} ` +
        `biomass=${pearsonCorr(inv, biomass).toFixed(3)} ` +
        `E/cell=${pearsonCorr(inv, ePerCell).toFixed(3)} ` +
        `S/cell=${pearsonCorr(inv, sPerCell).toFixed(3)} ` +
        `reproCd=${pearsonCorr(inv, reproCd).toFixed(3)} (snapshot-only)`,
    );
    lines.push(
      `- **invalidOpcode buckets**: ` +
        `p25=${p25.toFixed(0)} p75=${p75.toFixed(0)} ` +
        `low(n=${low.length}) meanAge=${lowAge.toFixed(1)} meanCells=${lowCells.toFixed(2)} | ` +
        `high(n=${high.length}) meanAge=${highAge.toFixed(1)} meanCells=${highCells.toFixed(2)}`,
    );
    const oldestHigh = [...high].sort((a, b) => b.age - a.age).slice(0, 3);
    const youngestHigh = [...high].sort((a, b) => a.age - b.age).slice(0, 3);
    if (oldestHigh.length > 0) {
      lines.push(
        `- **invalidOpcode high: oldest**: ${oldestHigh.map((p) => `#${p.id}(inv=${p.invalidOpcode},age=${p.age},cells=${p.cells})`).join(' | ')}`,
      );
    }
    if (youngestHigh.length > 0) {
      lines.push(
        `- **invalidOpcode high: youngest**: ${youngestHigh.map((p) => `#${p.id}(inv=${p.invalidOpcode},age=${p.age},cells=${p.cells})`).join(' | ')}`,
      );
    }
  }
  lines.push(
    `- **digest module intact**: orgs=${digestIntactOrgs}/${Math.max(1, orgList.length)} ` +
      `cells=${digestIntactCells}/${Math.max(1, occupied)}`,
  );
  if (topLineages.length > 0) {
    lines.push(
      `- **top lineages by occupied cells** (public kin “face” only): ${topLineages.map(([lin, c]) => `0x${lin.toString(16).padStart(6, '0')}:${c} (${((c / Math.max(1, occupied)) * 100).toFixed(1)}%)`).join(' | ')}`,
    );
  } else {
    lines.push('- **top lineages by occupied cells**: _(none)_');
  }
  lines.push('');

  lines.push(
    '### Physics / tuning (intended behaviour; verify in `rule-evaluator.ts`; cross-lineage edge policy in `metabolic-edge.ts`)',
  );
  lines.push('- Metabolic: `BASE_METABOLIC` + `CROWD_METABOLIC * sqrt(N)` × dominance multiplier when org share > ~8% of live cells, then × isolation multiplier (`1 + ISOLATION_METABOLIC_PENALTY * (1 - sameMooreNeighbors/8)`).');
  lines.push('- **Per-organism overhead**: each lineage pays a fixed energy/tick from its biomass → env (1-cell pays same as big colony per lineage → selection vs fragment spam).');
  lines.push('- Reproduce: `MIN_CELLS_TO_REPRODUCE=2`, cooldown ~40 ticks, dominance fail + **global crowding fail** when organism count is high.');
  lines.push('- Split policy: disconnected **1-cell shards stay with parent lineage** (no new lineage spawn); multi-cell fragments split into child lineages with extra reproduce cooldown (and additional crowd-scaled cooldown when lineage count is high). At high lineage counts, minimum split-child fragment size is raised (2 → 3 cells) to reduce tiny split spam.');
  lines.push('- Dissolution of e≤0 cells speeds up when many organisms exist; slightly faster for single-cell lineages.');
  lines.push('- Env: **conservative diffusion** on `envEnergy` (`ENV_DIFFUSION_RATE`); each tick ends with **rescale** so `sum(env)+Σcells+Σstomach = ecosystemEnergyBudget` (fixed universe except inject).');
  lines.push('- **Digestion (design notes)**');
  lines.push('  - **Single pipeline**: All stomach→cell transfer runs in `digestPhase()` once per tick (`PASSIVE_DIGEST_RATE`, `enzymeEff = min(1, cellE/3)`, `networkMul = DIGEST_NETWORK_BASE + DIGEST_NETWORK_COEFF * (sameMooreNeighbors/8)`, heat `DIGESTION_HEAT_LOSS` → env). No second hidden path.');
  lines.push('  - **`DIGEST` opcode**: Does not move energy immediately; it only raises `digestRuleBoost[idx]` (capped by `DIGEST_RULE_BOOST_CAP`). That boost multiplies the same `digestPhase` formula for that cell. **Neighbor stomachs are never read** (removed the old “strip adjacent same-org stomach” behaviour).');
  lines.push('  - **Conservation**: Digestion only moves mass own-stomach → own-cell-energy + local env heat; consistent with closed `ecosystemEnergyBudget`.');
  lines.push('  - **Rule-order nuance**: The **final boost multiplier** is order-independent (running sum clamped to cap). **Which** `DIGEST` rows still pay `ACTION_COST_DIGEST` if the cap is already full depends on rule-table order (later rows may no-op with `false`). Avoid redundant duplicate `DIGEST` rules on one cell.');
  lines.push('  - **Marker spec**: `MARKER_DIGEST` uses `/255` in the digest formula (aligned with the opcode path).');
  lines.push('- Tape: `readModifier` is op-node encoded in data[0..15], literals [16..31]; chain bit in rule flags.');
  lines.push(
    `- **TAPE_SIZE=256** is a fixed layout: region sizes in \`tape.ts\` must still sum to 256 if you change \`TAPE_SIZE\` (snapshots + offsets are not auto-derived from length alone).`,
  );
  lines.push(
    `- **NN tape decode**: each parameter = \`(u16_be(hi,lo) - ${NN_TAPE_WEIGHT_CENTER}) / ${NN_TAPE_WEIGHT_SCALE}\` (see \`decodeNNWeightBytes\` / \`getNNWeights\`). NN bytes **128–255** share the same transcribe + degradation path as rules/data — splitting NN to another buffer would isolate “mood” drift from discrete rules at implementation cost.`,
  );
  lines.push(
    `- **Degradation**: XOR one bit per hit; probability × \`tapeByteDegradationSensitivity(i)\` (\`TAPE_DEGRAD_SENS_*\`) — replication key, rule opcodes, maxCells, refractory, NN band are tuned sturdier than average literals.`,
  );
  lines.push(
    `- **Child degradation**: base child tape starts with **zero wear bits** on construction, but reproduction can intentionally add initial wear / NOPs in “degraded birth” outcomes (see \`rule-evaluator.ts\`).`,
  );
  lines.push(
    `- Reproduction **\`transcribeForReproductionOutcome()\`** (**\`transcription.ts\`**): same **length-preserving** channel as \`transcribe()\` (bit-flip + rare swap + mis-fetch), then a probabilistic **proofreading** step that can revert some mutated bytes back toward the parent (strongest on replication key and rule opcodes; NN bytes are proofread lightly). Proofreading strength scales primarily with **local same-org neighborhood quorum** (organization) with a smaller capped boost from overall colony size. **Channel swap**: base prob \`${CHANNEL_SWAP_BASE_PROB}\`, then acceptance × coarse **cross-region** \`${CHANNEL_SWAP_ACCEPT_CROSS_REGION_MULT}\` if endpoints differ in {data,CA,rules,NN}; × \`${CHANNEL_SWAP_ACCEPT_REPL_KEY_MULT}\` if either byte is replication key **60–63**; × \`${CHANNEL_SWAP_ACCEPT_RULE_TABLE_MULT}\` if either is rule table **64–127** (tunable exports). **Replication key**: parent **degradation** on those bytes **amplifies** copy noise there. Instead of hard stillbirth, a **degraded birth** outcome can spawn a weaker child (reduced initial energy + extra initial wear / NOPs).`,
  );
  lines.push('- **REPAIR (0x0C)**: immune action — weighted mend of `degradation` + clamp invalid rule opcodes to NOP; success rate × `(1 + k × quorum)` where quorum = same-org neighbors (full) + foreign neighbors weighted by multi-factor kin trust (**public** kin tag + signal marker + morph A — face-only mimicry stays weak; private genetic tag is not used).');
  lines.push('- **ABSORB (0x07)**: heterospecific contact is a **single continuous interaction**: morph affinity continuously shifts between (A) symmetric relax (cell↔cell equalization) and (B) stomach inflow (neighbor cell energy → actor stomach). JAM acts like **immunity** that dampens both good+bad coupling; large connected groups amplify both immunity and breaking. Low-rate horizontal rule-opcode transfer can occur at the interface (`donor -> host`); fixation is a deterministic contest of contact-drive (foreign-contact pressure + stomach load + interface flux + kin trust) vs local REPAIR quorum defense, with a small heat fee on successful integration.');
  lines.push('- **Social consensus (signal gossip)**: each tick, same-org Moore neighbors softly pull a cell’s signal marker toward local mean (local averaging), producing agreement/dissent dynamics from topology without explicit role flags.');
  lines.push('- **Consensus-coupled maintenance**: higher local signal cohesion slightly boosts REPAIR success, linking social agreement to collective maintenance outcomes.');
  lines.push('- **GIVE (0x04)**: same-org redistribution unchanged; may also push energy to **foreign** cells when kin trust is high enough, with extra heat loss scaling with imperfect trust (parasitic “fake kin” if **public** face + signal + morph align; private genetic tag not used).');
  lines.push('- **Deterministic decay (rot)**: cells with `energy <= 0` accumulate a deterministic `rot` gauge; large connected components slow rot, harsher metabolism speeds it. Cells dissolve when `rot >= 1` (see `cleanup-phase.ts`).');
  lines.push('- **Dead tissue gut leak + recovery**: dead cells leak a fraction of stomach each tick; same-org living neighbors preferentially recover it into stomach inflow, remainder becomes local env energy (see `rule-evaluator.ts`).');
  lines.push('');

  lines.push('### Per-organism (up to 8, largest by cells)');
  const topCells = orgList.slice(0, 8);
  for (const org of topCells) {
    let e = 0;
    let s = 0;
    for (const idx of org.cells) {
      e += world.getCellEnergyByIdx(idx);
      s += world.getStomachByIdx(idx);
    }
    const degSum = org.tape.degradation.reduce((a, b) => a + b, 0);
    const rules = countInvalidRuleOpcodes(org.tape.data);
    const moods = ['eat', 'grow', 'move', 'conserve']
      .map((m, i) => `${m}:${org.nnOutput[i].toFixed(3)}`)
      .join(' ');
    const p = profiles.find((x) => x.id === org.id);
    const state = p ? softOrgStateLabel(p) : 'unknown';
    const live = p ? p.liveCellRatio : 0;
    const rotMx = p ? p.rotMax : 0;
    const rotDead = p ? p.rotMaxDead : 0;
    lines.push(
      `- **#${org.id}** cells=${org.cells.size} age=${org.age} maxCells=${org.tape.getMaxCells()} ` +
        `reproCd=${Math.max(0, Math.round(org.reproduceCooldown))} E=${e.toFixed(1)} S=${s.toFixed(1)} ` +
        `nnDom=${org.nnDominant} (${moods}) tapeDegΣ=${degSum} ` +
        `rules: valid=${rules.valid} nop=${rules.nop} invalidOpcode=${rules.invalid} ` +
        `state=${state} live=${live.toFixed(2)} rotMax=${rotMx.toFixed(2)} rotMaxDead=${rotDead.toFixed(2)}`,
    );
  }
  if (orgList.length === 0) lines.push('- _(none)_');
  lines.push('');

  lines.push('### Per-organism (up to 8, most live cells)');
  const byLiveCells = [...profiles]
    .map((p) => ({ p, liveCells: p.liveCellRatio * Math.max(1, p.cells) }))
    .sort((a, b) => b.liveCells - a.liveCells)
    .slice(0, 8);
  for (const item of byLiveCells) {
    const p = item.p;
    const org = organisms.get(p.id);
    if (!org) continue;
    const degSum = org.tape.degradation.reduce((a, b) => a + b, 0);
    const rules = countInvalidRuleOpcodes(org.tape.data);
    const moods = ['eat', 'grow', 'move', 'conserve']
      .map((m, i) => `${m}:${org.nnOutput[i].toFixed(3)}`)
      .join(' ');
    const state = softOrgStateLabel(p);
    lines.push(
      `- **#${p.id}** liveCells≈${item.liveCells.toFixed(1)} cells=${p.cells} age=${p.age} maxCells=${org.tape.getMaxCells()} ` +
        `reproCd=${Math.max(0, Math.round(org.reproduceCooldown))} E=${p.energy.toFixed(1)} S=${p.stomach.toFixed(1)} ` +
        `nnDom=${org.nnDominant} (${moods}) tapeDegΣ=${degSum} ` +
        `rules: valid=${rules.valid} nop=${rules.nop} invalidOpcode=${rules.invalid} ` +
        `state=${state} live=${p.liveCellRatio.toFixed(2)} rotMax=${p.rotMax.toFixed(2)} rotMaxDead=${p.rotMaxDead.toFixed(2)}`,
    );
  }
  if (byLiveCells.length === 0) lines.push('- _(none)_');
  lines.push('');

  const notable: Array<{ title: string; profile: OrgProfile }> = [];
  const used = new Set<number>();
  const pickNotable = (title: string, pred: (p: OrgProfile) => boolean, score: (p: OrgProfile) => number) => {
    const candidate = profiles
      .filter((p) => !used.has(p.id) && pred(p))
      .sort((a, b) => score(b) - score(a))[0];
    if (!candidate) return;
    notable.push({ title, profile: candidate });
    used.add(candidate.id);
  };
  pickNotable('Largest colony', () => true, (p) => p.cells);
  pickNotable('Highest biomass', () => true, (p) => p.biomass);
  pickNotable('Oldest lineage', () => true, (p) => p.age);
  pickNotable('Explorer candidate (move-drive)', (p) => p.cells >= 2, (p) => p.nnOutput[NN_MOVE]);
  pickNotable(
    'Digest specialist',
    (p) => p.cells >= 2,
    (p) => p.meanMarkers[1] * 0.6 + p.meanStomach * 0.4,
  );
  pickNotable('Signal-heavy coordinator', (p) => p.cells >= 2, (p) => p.meanMarkers[2]);
  pickNotable('Filament / edge-dense', (p) => p.cells >= 3, (p) => p.boundaryRatio);

  lines.push('### Notable organisms for observation');
  if (notable.length === 0) {
    lines.push('- _(none)_');
    lines.push('');
  } else {
    for (const item of notable) {
      const p = item.profile;
      const degSum = organisms.get(p.id)?.tape.degradation.reduce((a, b) => a + b, 0) ?? 0;
      const state = softOrgStateLabel(p);
      lines.push(
        `- **${item.title} → #${p.id}** face=0x${p.lineage.toString(16).padStart(6, '0')} genetic=0x${p.geneticKinTag.toString(16).padStart(6, '0')} ` +
          `cells=${p.cells} age=${p.age} mood=${MOOD_NAMES[p.nnDominant]} ` +
          `biomass=${p.biomass.toFixed(1)} E/cell=${p.meanEnergy.toFixed(2)} S/cell=${p.meanStomach.toFixed(2)} ` +
          `markers[e,d,s,m]=[${p.meanMarkers.map((v) => v.toFixed(1)).join(', ')}] ` +
          `boundary=${p.boundaryRatio.toFixed(3)} compact=${p.compactness.toFixed(3)} ` +
          `center=(${p.centerX.toFixed(1)}, ${p.centerY.toFixed(1)}) tapeDegΣ=${degSum} ` +
          `state=${state} live=${p.liveCellRatio.toFixed(2)} rotMax=${p.rotMax.toFixed(2)} rotMaxDead=${p.rotMaxDead.toFixed(2)}`,
      );
    }
    lines.push('');
  }

  lines.push('### Full tape export (largest organism)');
  if (orgList.length > 0) {
    const t0 = orgList[0].tape;
    lines.push('**data[256]**');
    lines.push('```');
    lines.push(tapeHexDump(t0.data));
    lines.push('```');
    lines.push('**degradation[256]**');
    lines.push('```');
    lines.push(tapeDegradationHexDump(t0.degradation));
    lines.push('```');
    lines.push('**TAPE512_B64**');
    lines.push('```');
    lines.push(tapeSnapshotBase64(t0));
    lines.push('```');
  } else {
    lines.push('_N/A_');
  }
  lines.push('');

  if (orgList.length > 1) {
    const smallest = orgList[orgList.length - 1];
    if (smallest.id !== orgList[0].id) {
      const ts = smallest.tape;
      lines.push(`### Full tape export (smallest organism #${smallest.id}, ${smallest.cells.size} cells)`);
      lines.push('**data[256]**');
      lines.push('```');
      lines.push(tapeHexDump(ts.data));
      lines.push('```');
      lines.push('**degradation[256]**');
      lines.push('```');
      lines.push(tapeDegradationHexDump(ts.degradation));
      lines.push('```');
      lines.push('**TAPE512_B64**');
      lines.push('```');
      lines.push(tapeSnapshotBase64(ts));
      lines.push('```');
      lines.push('');
    }
  }

  const featureExportIds = notable
    .map((n) => n.profile.id)
    .filter((id) => id !== (orgList[0]?.id ?? -1) && id !== (orgList[orgList.length - 1]?.id ?? -1))
    .slice(0, 3);
  for (const id of featureExportIds) {
    const org = organisms.get(id);
    if (!org) continue;
    lines.push(`### Full tape export (feature organism #${org.id}, ${org.cells.size} cells)`);
    lines.push('**data[256]**');
    lines.push('```');
    lines.push(tapeHexDump(org.tape.data));
    lines.push('```');
    lines.push('**degradation[256]**');
    lines.push('```');
    lines.push(tapeDegradationHexDump(org.tape.degradation));
    lines.push('```');
    lines.push('**TAPE512_B64**');
    lines.push('```');
    lines.push(tapeSnapshotBase64(org.tape));
    lines.push('```');
    lines.push('');
  }

  lines.push('### Suggested review checklist');
  lines.push('- [ ] `invalidOpcode` high ⇒ many rules never execute.');
  lines.push('- [ ] One org huge share + fill ratio ⇒ monoculture; dominance tax should bite.');
  lines.push('- [ ] **Closed budget**: `measured total` should track `ecosystemEnergyBudget`; large drift ⇒ bug or desync. Inject increases budget intentionally.');
  lines.push('- [ ] Compare behaviour to `createProtoTape()` in `tape.ts` for “healthy” baseline.');
  lines.push('- [ ] Many duplicate `DIGEST` rules: boost hits cap early — later rows fail cheaply; check tape evolution.');
  lines.push(
    '- [ ] **TAPE512_B64**: must decode to exactly 512 bytes; reconstruct with `tapeFromSnapshot` if testing import.',
  );
  lines.push('');

  return lines.join('\n');
}
