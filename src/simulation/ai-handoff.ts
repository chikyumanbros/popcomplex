/**
 * Builds a markdown report for pasting into an AI chat (debug / design review).
 * Written in English in the doc blocks; user-facing button label is Japanese in UI.
 */

import { GRID_WIDTH, GRID_HEIGHT, TOTAL_CELLS } from './constants';
import {
  ActionOpcode,
  CONDITIONS_OFFSET,
  MAX_VALID_ACTION_OPCODE,
  NN_TAPE_WEIGHT_CENTER,
  NN_TAPE_WEIGHT_SCALE,
  PROTO_TAPE_NN_SEED,
  RULE_SIZE,
  MAX_RULES,
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

function countInvalidRuleOpcodes(data: Uint8Array): { invalid: number; nop: number; valid: number } {
  let invalid = 0;
  let nop = 0;
  let valid = 0;
  for (let r = 0; r < MAX_RULES; r++) {
    const op = data[CONDITIONS_OFFSET + r * RULE_SIZE + 2];
    if (op === ActionOpcode.NOP) nop++;
    else if (op > MAX_VALID_ACTION_OPCODE) invalid++;
    else valid++;
  }
  return { invalid, nop, valid };
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
    const [mkEat, mkDigest, mkSignal, mkMove] = world.getMarkersByIdx(idx);
    markerEat += mkEat;
    markerDigest += mkDigest;
    markerSignal += mkSignal;
    markerMove += mkMove;

    if (x === 0 || !own.has(idx - 1)) boundaryFaces++;
    if (x === GRID_WIDTH - 1 || !own.has(idx + 1)) boundaryFaces++;
    if (y === 0 || !own.has(idx - GRID_WIDTH)) boundaryFaces++;
    if (y === GRID_HEIGHT - 1 || !own.has(idx + GRID_WIDTH)) boundaryFaces++;
  }

  const n = Math.max(1, own.size);
  const bboxW = Math.max(1, maxX - minX + 1);
  const bboxH = Math.max(1, maxY - minY + 1);
  const nnDom = Math.max(0, Math.min(3, org.nnDominant)) as MoodIdx;

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
    boundaryRatio: boundaryFaces / (4 * n),
    compactness: own.size / (bboxW * bboxH),
    centerX: cx / n,
    centerY: cy / n,
    nnOutput: org.nnOutput,
    nnDominant: nnDom,
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
}

export type AIHandoffPromptPreset = 'review' | 'ecology' | 'tape';

export function buildAIHandoffPrompt(preset: AIHandoffPromptPreset): string {
  switch (preset) {
    case 'ecology':
      return [
        'Focus on ecology and evolution.',
        'From the mood mix, lineage shares, and notable organisms, infer active niches (forager, mover, digester, signal-coordinator, etc.), explain why they are stable/unstable, and propose 3 interventions to increase long-horizon diversity without breaking energy closure.',
      ].join('\n');
    case 'tape':
      return [
        'Focus on tape health and behavioral reliability.',
        'Use invalidOpcode counts, degradation sums, and the provided tape dumps to identify failure modes (dead rules, brittle regions, over-mutation zones), then suggest exact checks/metrics to add in code/tests.',
      ].join('\n');
    case 'review':
    default:
      return [
        'Analyze this snapshot as an artificial-life code reviewer.',
        'Prioritize: (1) likely logic bugs/regressions, (2) ecological interpretation of niches and dominance, (3) concrete parameter/code changes with expected side effects.',
        'Use specific organism IDs/lineages from the report when explaining.',
      ].join('\n');
  }
}

export function buildAIHandoffMarkdown(input: AIHandoffInput): string {
  const { tick, world, organisms, ruleEval, ui, bookkeepingSnapshot } = input;
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
  lines.push('Paste this whole block into an AI assistant and ask e.g.: energy conservation issues, dominance / diversity, broken rule opcodes, or design suggestions.');
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
  if (ui) {
    lines.push(`- **UI**: paused=${ui.paused} speed=${ui.speed}`);
  }
  lines.push('');

  lines.push('### Build / grid');
  lines.push(`- **grid**: ${GRID_WIDTH}×${GRID_HEIGHT}`);
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

  lines.push('### Ecosystem observability snapshot');
  lines.push(`- **mood mix by organism**: ${moodOrgLine}`);
  lines.push(`- **mood mix by occupied cells**: ${moodCellLine}`);
  lines.push(`- **size classes (org count)**: micro(1)=${micro}, small(2-4)=${small}, medium(5-15)=${medium}, macro(16+)=${macro}`);
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
  lines.push('- Metabolic: `BASE_METABOLIC` + `CROWD_METABOLIC * sqrt(N)` × dominance multiplier when org share > ~8% of live cells, then × isolation multiplier (`1 + ISOLATION_METABOLIC_PENALTY * (1 - sameOrthNeighbors/4)`).');
  lines.push('- **Per-organism overhead**: each lineage pays a fixed energy/tick from its biomass → env (1-cell pays same as big colony per lineage → selection vs fragment spam).');
  lines.push('- Reproduce: `MIN_CELLS_TO_REPRODUCE=2`, cooldown ~40 ticks, dominance fail + **global crowding fail** when organism count is high.');
  lines.push('- Split policy: disconnected **1-cell shards stay with parent lineage** (no new lineage spawn); multi-cell fragments split into child lineages with extra reproduce cooldown (and additional crowd-scaled cooldown when lineage count is high). At high lineage counts, minimum split-child fragment size is raised (2 → 3 cells) to reduce tiny split spam.');
  lines.push('- Dissolution of e≤0 cells speeds up when many organisms exist; slightly faster for single-cell lineages.');
  lines.push('- Env: **conservative diffusion** on `envEnergy` (`ENV_DIFFUSION_RATE`); each tick ends with **rescale** so `sum(env)+Σcells+Σstomach = ecosystemEnergyBudget` (fixed universe except inject).');
  lines.push('- **Digestion (design notes)**');
  lines.push('  - **Single pipeline**: All stomach→cell transfer runs in `digestPhase()` once per tick (`PASSIVE_DIGEST_RATE`, `enzymeEff = min(1, cellE/3)`, `networkMul = DIGEST_NETWORK_BASE + DIGEST_NETWORK_COEFF * (sameOrthNeighbors/4)`, heat `DIGESTION_HEAT_LOSS` → env). No second hidden path.');
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
    `- **Child degradation**: always **zeros** on \`new Tape(data)\` after transcribe — parent wear biases copy noise / stillbirth only, not inherited wear bits.`,
  );
  lines.push(
    `- Reproduction **\`transcribeForReproduction()\`** (**\`transcription.ts\`**): same **length-preserving** channel as \`transcribe()\` (bit-flip + rare swap + mis-fetch). **Channel swap**: base prob \`${CHANNEL_SWAP_BASE_PROB}\`, then acceptance × coarse **cross-region** \`${CHANNEL_SWAP_ACCEPT_CROSS_REGION_MULT}\` if endpoints differ in {data,CA,rules,NN}; × \`${CHANNEL_SWAP_ACCEPT_REPL_KEY_MULT}\` if either byte is replication key **60–63**; × \`${CHANNEL_SWAP_ACCEPT_RULE_TABLE_MULT}\` if either is rule table **64–127** (tunable exports). **Replication key**: parent **degradation** on those bytes **amplifies** copy noise there; after copy, a **viability roll** can **abort** offspring (stillbirth) — cost is reproduce heat only, **no** child, cooldown applies, action returns failure (no reproduce feedback).`,
  );
  lines.push('- **REPAIR (0x0C)**: immune action — weighted mend of `degradation` + clamp invalid rule opcodes to NOP; success rate × `(1 + k × quorum)` where quorum = same-org neighbors (full) + foreign neighbors weighted by multi-factor kin trust (**public** kin tag + signal marker + morph A — face-only mimicry stays weak; private genetic tag is not used).');
  lines.push('- **ABSORB (0x07)**: heterospecific contact is a **single continuous interaction**: morph affinity continuously shifts between (A) symmetric relax (cell↔cell equalization) and (B) stomach inflow (neighbor cell energy → actor stomach). JAM acts like **immunity** that dampens both good+bad coupling; large connected groups amplify both immunity and breaking. Low-rate horizontal rule-opcode transfer can occur at the interface (`donor -> host`); fixation is a deterministic contest of contact-drive (foreign-contact pressure + stomach load + interface flux + kin trust) vs local REPAIR quorum defense, with a small heat fee on successful integration.');
  lines.push('- **Social consensus (signal gossip)**: each tick, same-org orthogonal neighbors softly pull a cell’s signal marker toward local mean (local averaging), producing agreement/dissent dynamics from topology without explicit role flags.');
  lines.push('- **Consensus-coupled maintenance**: higher local signal cohesion slightly boosts REPAIR success, linking social agreement to collective maintenance outcomes.');
  lines.push('- **GIVE (0x04)**: same-org redistribution unchanged; may also push energy to **foreign** cells when kin trust is high enough, with extra heat loss scaling with imperfect trust (parasitic “fake kin” if **public** face + signal + morph align; private genetic tag not used).');
  lines.push('');

  lines.push('### Per-organism (up to 8, largest first)');
  const top = orgList.slice(0, 8);
  for (const org of top) {
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
    lines.push(
      `- **#${org.id}** cells=${org.cells.size} age=${org.age} maxCells=${org.tape.getMaxCells()} ` +
        `reproCd=${Math.max(0, Math.round(org.reproduceCooldown))} E=${e.toFixed(1)} S=${s.toFixed(1)} ` +
        `nnDom=${org.nnDominant} (${moods}) tapeDegΣ=${degSum} ` +
        `rules: valid=${rules.valid} nop=${rules.nop} invalidOpcode=${rules.invalid}`,
    );
  }
  if (orgList.length === 0) lines.push('- _(none)_');
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
      lines.push(
        `- **${item.title} → #${p.id}** face=0x${p.lineage.toString(16).padStart(6, '0')} genetic=0x${p.geneticKinTag.toString(16).padStart(6, '0')} ` +
          `cells=${p.cells} age=${p.age} mood=${MOOD_NAMES[p.nnDominant]} ` +
          `biomass=${p.biomass.toFixed(1)} E/cell=${p.meanEnergy.toFixed(2)} S/cell=${p.meanStomach.toFixed(2)} ` +
          `markers[e,d,s,m]=[${p.meanMarkers.map((v) => v.toFixed(1)).join(', ')}] ` +
          `boundary=${p.boundaryRatio.toFixed(3)} compact=${p.compactness.toFixed(3)} ` +
          `center=(${p.centerX.toFixed(1)}, ${p.centerY.toFixed(1)}) tapeDegΣ=${degSum}`,
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
