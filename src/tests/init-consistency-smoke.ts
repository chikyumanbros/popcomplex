import assert from 'node:assert/strict';
import { GRID_WIDTH, GRID_HEIGHT, INITIAL_ENV_ENERGY_PER_CELL } from '../simulation/constants';
import { World } from '../simulation/world';
import { OrganismManager } from '../simulation/organism';
import { RuleEvaluator } from '../simulation/rule-evaluator';
import { initSimulation } from '../simulation/init-simulation';
import { setRandomSeed } from '../simulation/rng';
import { biomassReservoirTotal, measureEnergyBookkeeping } from '../simulation/energy-metrics';

function sumEnv(env: Float32Array): number {
  let s = 0;
  for (let i = 0; i < env.length; i++) s += env[i]!;
  return s;
}

function runMode(seed: number, mode: { culture: boolean; multiOrigin: boolean }, ticks: number) {
  setRandomSeed(seed);
  const world = new World();
  const organisms = new OrganismManager();
  const ruleEval = new RuleEvaluator(world, organisms, {
    budgetMode: 'local',
    suppressionMode: 'on',
    metabolicScale: 0.6,
    distressFireChanceScale: 0.3,
  });

  const env = new Float32Array(GRID_WIDTH * GRID_HEIGHT);
  initSimulation(world, organisms, ruleEval, {
    env,
    spawnEnergy: 150,
    culture: mode.culture,
    multiOrigin: mode.multiOrigin,
  });

  const envSum0 = sumEnv(env);
  const orgs0 = organisms.count;

  for (let t = 0; t < ticks; t++) {
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
  }

  const bk = measureEnergyBookkeeping(world, ruleEval);
  const measured = biomassReservoirTotal(bk);
  const drift = measured - ruleEval.ecosystemEnergyBudget;
  return { envSum0, orgs0, driftAbs: Math.abs(drift) };
}

function main() {
  const baselineEnv = GRID_WIDTH * GRID_HEIGHT * INITIAL_ENV_ENERGY_PER_CELL;
  // Conservative bumps use float rescale; allow tiny rounding error.
  const EPS = 1e-2;

  // 1) culture=true should preserve total env energy at init (conservative bumps).
  const culture = runMode(3006, { culture: true, multiOrigin: false }, 40);
  assert.ok(Math.abs(culture.envSum0 - baselineEnv) < EPS, `culture init must preserve env sum: got=${culture.envSum0} expected=${baselineEnv}`);
  assert.ok(culture.orgs0 > 1, `culture init should inoculate multi-site by default (orgs0=${culture.orgs0})`);
  assert.ok(culture.driftAbs < 1e-2, `culture run drift too large: ${culture.driftAbs}`);

  // 2) multiOrigin=true, culture=false uses non-conservative bumps (env sum increases vs baseline).
  const multi = runMode(3006, { culture: false, multiOrigin: true }, 40);
  assert.ok(multi.envSum0 > baselineEnv + 1, `multiOrigin init should increase env sum (non-conservative bumps): got=${multi.envSum0} baseline=${baselineEnv}`);
  assert.ok(multi.orgs0 > 1, `multiOrigin init should inoculate multi-site (orgs0=${multi.orgs0})`);
  assert.ok(multi.driftAbs < 1e-2, `multiOrigin run drift too large: ${multi.driftAbs}`);

  // 3) Neither flag => single-origin baseline (one proto).
  const single = runMode(3006, { culture: false, multiOrigin: false }, 40);
  assert.equal(single.orgs0, 1, `single-origin init should spawn exactly 1 proto (orgs0=${single.orgs0})`);
  assert.ok(single.driftAbs < 1e-2, `single-origin run drift too large: ${single.driftAbs}`);

  console.log('[Init consistency smoke] OK');
}

main();

