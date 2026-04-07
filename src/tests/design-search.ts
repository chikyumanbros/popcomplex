import { GRID_HEIGHT, GRID_WIDTH } from '../simulation/constants';
import { World } from '../simulation/world';
import { OrganismManager } from '../simulation/organism';
import { RuleEvaluator } from '../simulation/rule-evaluator';
import { CONDITIONS_OFFSET, RULE_SIZE, createProtoTape } from '../simulation/tape';
import { setRandomSeed } from '../simulation/rng';
import { measureEnergyBookkeeping } from '../simulation/energy-metrics';
import { snapshotAndResetTelemetry } from '../simulation/telemetry';
import { initSimulation } from '../simulation/init-simulation';

interface Profile {
  moveThr: number;
  divThr: number;
  reproThr: number;
}

interface ProfileScore {
  profile: Profile;
  growth: number;
  occupiedEndMean: number;
}

function withProfileTape(profile: Profile) {
  const t = createProtoTape();
  // rule 2 = DIV, rule 4 = MOVE, rule 5 = REPRODUCE
  t.data[CONDITIONS_OFFSET + 2 * RULE_SIZE + 1] = profile.divThr & 0xff;
  t.data[CONDITIONS_OFFSET + 4 * RULE_SIZE + 1] = profile.moveThr & 0xff;
  t.data[CONDITIONS_OFFSET + 5 * RULE_SIZE + 1] = profile.reproThr & 0xff;
  return t;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function runScenario(seed: number, ticks: number, profile: Profile): { occupiedAt20: number; occupiedEnd: number } {
  setRandomSeed(seed);
  snapshotAndResetTelemetry();

  const world = new World();
  const organisms = new OrganismManager();
  const ruleEval = new RuleEvaluator(world, organisms, {
    budgetMode: 'local',
    suppressionMode: 'on',
    metabolicScale: 0.6,
    distressFireChanceScale: 0.3,
  });

  const spawnEnergy = 150;
  const tape = withProfileTape(profile);
  initSimulation(world, organisms, ruleEval, {
    env: new Float32Array(GRID_WIDTH * GRID_HEIGHT),
    spawnEnergy,
    culture: false,
    multiOrigin: true,
    protoTape: tape,
  });

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
    if (t === 20) occupiedAt20 = measureEnergyBookkeeping(world, ruleEval).occupiedCells;
  }
  const occupiedEnd = measureEnergyBookkeeping(world, ruleEval).occupiedCells;
  return { occupiedAt20, occupiedEnd };
}

function scoreProfile(profile: Profile): ProfileScore {
  const seeds = [3006, 3007, 3010];
  const rows = seeds.map((s) => runScenario(s, 220, profile));
  const growth = mean(
    rows.map((r) => {
      const denom = Math.max(1, r.occupiedAt20);
      return r.occupiedEnd / denom;
    }),
  );
  return {
    profile,
    growth,
    occupiedEndMean: mean(rows.map((r) => r.occupiedEnd)),
  };
}

function main() {
  const moves = [96, 120, 144];
  const divs = [42, 45, 48];
  const repros = [70, 75, 80];
  const scores: ProfileScore[] = [];
  for (const moveThr of moves) {
    for (const divThr of divs) {
      for (const reproThr of repros) {
        scores.push(scoreProfile({ moveThr, divThr, reproThr }));
      }
    }
  }
  scores.sort((a, b) => b.growth - a.growth);
  console.log('[Design search] top profiles by evolvability growth');
  for (const s of scores.slice(0, 8)) {
    console.log(
      `move=${s.profile.moveThr} div=${s.profile.divThr} repro=${s.profile.reproThr} growth=${s.growth.toFixed(3)} occEndMean=${s.occupiedEndMean.toFixed(3)}`,
    );
  }
}

main();
