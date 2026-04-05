import { INITIAL_ENV_ENERGY_PER_CELL, GRID_HEIGHT, GRID_WIDTH } from '../simulation/constants';
import { World } from '../simulation/world';
import { OrganismManager } from '../simulation/organism';
import { RuleEvaluator } from '../simulation/rule-evaluator';
import { createProtoTape, CONDITIONS_OFFSET, MAX_RULES, RULE_SIZE, MAX_VALID_ACTION_OPCODE } from '../simulation/tape';
import { transcribeForReproduction } from '../simulation/transcription';
import { setRandomSeed } from '../simulation/rng';
import { measurePopulationMetrics, measureEnergyBookkeeping } from '../simulation/energy-metrics';
import { snapshotAndResetTelemetry } from '../simulation/telemetry';
import { addRegionalEnvBumps, spawnTricladProtos, DEFAULT_TRICLADE_SITES } from '../simulation/initial-inoculation';

type GateStatus = 'PASS' | 'WARN' | 'FAIL';

interface GateResult {
  key: string;
  status: GateStatus;
  value: string;
  note: string;
  blocking?: boolean;
}

interface RunSummary {
  occupiedAt20: number;
  occupiedEnd: number;
  simpsonEnd: number;
  uniqueLineagesEnd: number;
  driftEndAbs: number;
}

function statusFrom(value: number, passMin: number, warnMin: number): GateStatus {
  if (value >= passMin) return 'PASS';
  if (value >= warnMin) return 'WARN';
  return 'FAIL';
}

function statusFromMax(value: number, passMax: number, warnMax: number): GateStatus {
  if (value <= passMax) return 'PASS';
  if (value <= warnMax) return 'WARN';
  return 'FAIL';
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function byteSimilarity(a: Uint8Array, b: Uint8Array): number {
  let same = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] === b[i]) same++;
  return n > 0 ? same / n : 0;
}

function mutationPresenceRate(parent: Uint8Array, children: Uint8Array[]): number {
  if (children.length === 0) return 0;
  let mutated = 0;
  for (const child of children) {
    let diff = false;
    for (let i = 0; i < parent.length; i++) {
      if (parent[i] !== child[i]) {
        diff = true;
        break;
      }
    }
    if (diff) mutated++;
  }
  return mutated / children.length;
}

function invalidOpcodeRate(children: Uint8Array[]): number {
  let invalid = 0;
  let total = 0;
  for (const child of children) {
    for (let r = 0; r < MAX_RULES; r++) {
      const op = child[CONDITIONS_OFFSET + r * RULE_SIZE + 2];
      total++;
      if (op > MAX_VALID_ACTION_OPCODE) invalid++;
    }
  }
  return total > 0 ? invalid / total : 1;
}

function runScenario(seed: number, ticks: number, metabolicScale: number): RunSummary {
  setRandomSeed(seed);
  snapshotAndResetTelemetry();

  const world = new World();
  const organisms = new OrganismManager();
  const ruleEval = new RuleEvaluator(world, organisms, {
    neighborMode: 'four',
    budgetMode: 'local',
    suppressionMode: 'on',
    metabolicScale,
    distressFireChanceScale: 0.3,
  });

  const env = new Float32Array(GRID_WIDTH * GRID_HEIGHT);
  env.fill(INITIAL_ENV_ENERGY_PER_CELL);
  addRegionalEnvBumps(env, DEFAULT_TRICLADE_SITES, 1.0, 14);
  ruleEval.setEnvEnergy(env);
  ruleEval.snapClosedEnergyBudgetFromWorld();

  const spawnEnergy = 150;
  const tape = createProtoTape();
  spawnTricladProtos(world, organisms, ruleEval, tape, spawnEnergy, DEFAULT_TRICLADE_SITES);

  let occupiedAt20 = 0;
  for (let t = 1; t <= ticks; t++) {
    organisms.syncNeuralWeightsFromTape();
    ruleEval.updateNeuralNetworks();
    ruleEval.evaluate();
    ruleEval.digestPhase();
    ruleEval.applyMetabolicCost();
    ruleEval.applyOrganismOverhead();
    ruleEval.cleanupDeadOrganisms();
    ruleEval.splitDisconnected();
    organisms.tick();
    world.syncLineageToCells(organisms);
    ruleEval.enforceClosedEnergyBudget();

    if (t === 20) {
      occupiedAt20 = measureEnergyBookkeeping(world, ruleEval).occupiedCells;
    }
  }

  const popEnd = measurePopulationMetrics(world, organisms);
  const bkEnd = measureEnergyBookkeeping(world, ruleEval);
  const measuredEnd = bkEnd.envU + bkEnd.cellEnergy + bkEnd.stomach;
  snapshotAndResetTelemetry();

  return {
    occupiedAt20,
    occupiedEnd: bkEnd.occupiedCells,
    simpsonEnd: popEnd.simpsonDiversity,
    uniqueLineagesEnd: popEnd.uniqueLineages,
    driftEndAbs: Math.abs(measuredEnd - ruleEval.ecosystemEnergyBudget),
  };
}

function evaluateDesignGate(): GateResult[] {
  const trials = 1200;
  setRandomSeed(0x20260405);
  snapshotAndResetTelemetry();

  const parent = createProtoTape();
  const viableChildren: Uint8Array[] = [];
  let aborted = 0;
  for (let i = 0; i < trials; i++) {
    const p = createProtoTape();
    const child = transcribeForReproduction(p);
    if (!child) {
      aborted++;
      continue;
    }
    viableChildren.push(child.data);
  }
  const viableRate = viableChildren.length / trials;
  const childSimilarity = mean(viableChildren.map((d) => byteSimilarity(parent.data, d)));
  const mutPresence = mutationPresenceRate(parent.data, viableChildren);
  const invOpcode = invalidOpcodeRate(viableChildren);

  const seeds = [3006, 3007, 3010];
  const baseRuns = seeds.map((s) => runScenario(s, 220, 0.6));
  const harshRuns = seeds.map((s) => runScenario(s, 220, 0.9));

  const growth = mean(
    baseRuns.map((r) => {
      const denom = Math.max(1, r.occupiedAt20);
      return r.occupiedEnd / denom;
    }),
  );
  const diversity = mean(baseRuns.map((r) => r.simpsonEnd));
  const lineageRichness = mean(baseRuns.map((r) => r.uniqueLineagesEnd));
  const occupiedEndMean = mean(baseRuns.map((r) => r.occupiedEnd));
  const selectionSensitivity =
    mean(baseRuns.map((r) => r.occupiedEnd)) - mean(harshRuns.map((r) => r.occupiedEnd));
  const driftEndAbs = mean(baseRuns.map((r) => r.driftEndAbs));
  const ecologyCollapsed = occupiedEndMean < 5;

  const results: GateResult[] = [
    {
      key: 'heritability',
      status: statusFrom(childSimilarity, 0.90, 0.82),
      value: childSimilarity.toFixed(3),
      note: 'parent-child tape byte similarity',
      blocking: true,
    },
    {
      key: 'variability',
      status: statusFrom(mutPresence, 0.25, 0.10),
      value: mutPresence.toFixed(3),
      note: 'fraction of viable offspring with >=1 byte change',
      blocking: true,
    },
    {
      key: 'robustness',
      status: statusFrom(viableRate, 0.55, 0.35),
      value: viableRate.toFixed(3),
      note: 'viable offspring rate (1 - stillbirth)',
      blocking: true,
    },
    {
      key: 'rule-integrity',
      status: statusFromMax(invOpcode, 0.35, 0.55),
      value: invOpcode.toFixed(3),
      note: 'invalid opcode fraction in viable offspring rule table',
      blocking: true,
    },
    {
      key: 'evolvability',
      status: statusFrom(growth, 1.12, 1.0),
      value: growth.toFixed(3),
      note: 'occupied growth ratio tick20→tick220 (denom scales with multi-origin inoculation)',
      blocking: true,
    },
    {
      key: 'selection-sensitivity',
      status: ecologyCollapsed ? 'WARN' : statusFrom(selectionSensitivity, 5, 1),
      value: selectionSensitivity.toFixed(3),
      note: ecologyCollapsed
        ? 'not evaluated strictly: ecology collapsed (low occupiedEnd)'
        : 'occupied(base metabolic 0.6) - occupied(harsh 0.9)',
    },
    {
      key: 'diversity',
      status: ecologyCollapsed ? 'WARN' : statusFrom(diversity, 0.08, 0.02),
      value: diversity.toFixed(3),
      note: ecologyCollapsed ? 'not evaluated strictly: ecology collapsed (low occupiedEnd)' : 'mean Simpson diversity at end',
    },
    {
      key: 'lineage-richness',
      status: ecologyCollapsed ? 'WARN' : statusFrom(lineageRichness, 2, 1.1),
      value: lineageRichness.toFixed(3),
      note: ecologyCollapsed ? 'not evaluated strictly: ecology collapsed (low occupiedEnd)' : 'mean unique lineages at end',
    },
    {
      key: 'energy-closure',
      status: statusFromMax(driftEndAbs, 0.5, 3),
      value: driftEndAbs.toFixed(6),
      note: 'mean absolute budget drift at end',
      blocking: true,
    },
  ];

  const stillbirthRate = aborted / trials;
  results.push({
    key: 'stillbirth-rate',
    status: statusFromMax(stillbirthRate, 0.45, 0.70),
    value: stillbirthRate.toFixed(3),
    note: 'observed stillbirth ratio during heredity probe',
    blocking: true,
  });

  return results;
}

function main() {
  const results = evaluateDesignGate();
  console.log('[Design gate] PASS/WARN/FAIL');
  for (const r of results) {
    console.log(`${r.status.padEnd(4)}  ${r.key.padEnd(20)} value=${r.value}  (${r.note})`);
  }
  const fail = results.filter((r) => r.status === 'FAIL').length;
  const blockingFail = results.filter((r) => r.status === 'FAIL' && r.blocking).length;
  const warn = results.filter((r) => r.status === 'WARN').length;
  console.log(
    `[Design gate] summary: fail=${fail} (blocking=${blockingFail}) warn=${warn} pass=${results.length - fail - warn}`,
  );
  if (blockingFail > 0) process.exitCode = 1;
}

main();
