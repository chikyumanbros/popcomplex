import assert from 'node:assert/strict';
import { INITIAL_ENV_ENERGY_PER_CELL, GRID_WIDTH, GRID_HEIGHT } from '../simulation/constants';
import { World } from '../simulation/world';
import { OrganismManager } from '../simulation/organism';
import { RuleEvaluator } from '../simulation/rule-evaluator';
import { createProtoTape, REPLICATION_KEY_LEN, REPLICATION_KEY_OFFSET } from '../simulation/tape';
import { transcribeForReproduction } from '../simulation/transcription';
import { setRandomSeed } from '../simulation/rng';
import { biomassReservoirTotal, measureEnergyBookkeeping } from '../simulation/energy-metrics';

interface MiniSnapshot {
  orgCount: number;
  occupied: number;
  budget: number;
  measured: number;
  drift: number;
  envChecksum: number;
  cellChecksum: number;
}

function estimateStillbirthRate(parentBuilder: () => ReturnType<typeof createProtoTape>, trials: number, seed: number): number {
  setRandomSeed(seed);
  let aborted = 0;
  for (let i = 0; i < trials; i++) {
    const parent = parentBuilder();
    const child = transcribeForReproduction(parent);
    if (!child) aborted++;
  }
  return aborted / trials;
}

function runMiniSimulation(seed: number, ticks: number): MiniSnapshot {
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
  env.fill(INITIAL_ENV_ENERGY_PER_CELL);
  ruleEval.setEnvEnergy(env);
  ruleEval.snapClosedEnergyBudgetFromWorld();

  const spawnEnergy = 150;
  const cx = Math.floor(GRID_WIDTH / 2);
  const cy = Math.floor(GRID_HEIGHT / 2);
  const tape = createProtoTape();
  const ok = ruleEval.withdrawEnvUniform(spawnEnergy);
  assert.equal(ok, true, 'spawn withdraw must succeed in mini simulation');
  const id = world.spawnProto(cx, cy, tape.getLineagePacked(), spawnEnergy);
  organisms.register(id, tape, { parentId: null, birthTick: 0 });
  organisms.get(id)?.cells.add(cy * GRID_WIDTH + cx);

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
  let envChecksum = 0;
  for (let i = 0; i < ruleEval.envEnergy.length; i += 257) {
    envChecksum += ruleEval.envEnergy[i];
  }
  let cellChecksum = 0;
  for (let i = 0; i < world.cellData.length; i += 257) {
    cellChecksum = (cellChecksum + world.cellData[i]) >>> 0;
  }
  return {
    orgCount: organisms.count,
    occupied: bk.occupiedCells,
    budget: Number(ruleEval.ecosystemEnergyBudget.toFixed(6)),
    measured: Number(measured.toFixed(6)),
    drift: Number((measured - ruleEval.ecosystemEnergyBudget).toFixed(6)),
    envChecksum: Number(envChecksum.toFixed(6)),
    cellChecksum,
  };
}

function testStillbirthRateRespondsToKeyWear() {
  const trials = 3000;
  const cleanRate = estimateStillbirthRate(() => createProtoTape(), trials, 0x11111111);
  const wornRate = estimateStillbirthRate(() => {
    const t = createProtoTape();
    for (let k = 0; k < REPLICATION_KEY_LEN; k++) {
      t.degradation[REPLICATION_KEY_OFFSET + k] = 255;
    }
    return t;
  }, trials, 0x11111111);

  assert.ok(cleanRate > 0.005, `clean stillbirth too low: ${cleanRate}`);
  assert.ok(cleanRate < 0.2, `clean stillbirth too high: ${cleanRate}`);
  assert.ok(wornRate > cleanRate + 0.02, `key wear must increase stillbirth rate: clean=${cleanRate}, worn=${wornRate}`);
}

function testDeterministicReplayForSameSeed() {
  const a = runMiniSimulation(3006, 150);
  const b = runMiniSimulation(3006, 150);
  assert.deepEqual(b, a, 'same seed + same params should replay exactly');
}

function main() {
  testStillbirthRateRespondsToKeyWear();
  testDeterministicReplayForSameSeed();
  console.log('[Repro stats smoke] OK');
}

main();
