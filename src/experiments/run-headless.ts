import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GRID_WIDTH, GRID_HEIGHT } from '../simulation/constants';
import { World } from '../simulation/world';
import { OrganismManager } from '../simulation/organism';
import { RuleEvaluator } from '../simulation/rule-evaluator';
import { tapeSnapshotBase64 } from '../simulation/tape';
import { setRandomSeed, getRandomSeed } from '../simulation/rng';
import { measureEnergyBookkeeping, measurePopulationMetrics, biomassReservoirTotal } from '../simulation/energy-metrics';
import { snapshotAndResetTelemetry } from '../simulation/telemetry';
import { countInvalidRuleOpcodes } from '../simulation/tape-health';
import type { BudgetMode, SuppressionMode } from '../simulation/runtime-config';
import { initSimulation } from '../simulation/init-simulation';

interface CliArgs {
  seed: number;
  ticks: number;
  logEvery: number;
  budget: BudgetMode;
  suppression: SuppressionMode;
  spawnEnergy: number;
  metabolicScale: number;
  distressScale: number;
  snapshotEvery: number;
  outDir: string;
}

const DEFAULT_SEED = 3418996282;

function parseArgs(): CliArgs {
  const map = new Map<string, string>();
  for (const token of process.argv.slice(2)) {
    const [k, v] = token.split('=');
    if (k && v !== undefined) map.set(k.replace(/^--/, ''), v);
  }
  const seedNum = Number(map.get('seed') ?? DEFAULT_SEED);
  const ticksNum = Number(map.get('ticks') ?? 1000);
  const logEveryNum = Number(map.get('logEvery') ?? 10);
  const snapshotEveryNum = Number(map.get('snapshotEvery') ?? 500);
  const budget = (map.get('budget') === 'global' ? 'global' : 'local') as BudgetMode;
  const suppression = (map.get('suppression') === 'off' ? 'off' : 'on') as SuppressionMode;
  const spawnEnergyNum = Number(map.get('spawnEnergy') ?? 60);
  const metabolicScaleNum = Number(map.get('metabolicScale') ?? 1);
  const distressScaleNum = Number(map.get('distressScale') ?? 1);
  const outDir = map.get('outDir') ?? 'runs';
  return {
    seed: Number.isFinite(seedNum) ? (Math.trunc(seedNum) >>> 0) : DEFAULT_SEED,
    ticks: Number.isFinite(ticksNum) ? Math.max(1, Math.trunc(ticksNum)) : 1000,
    logEvery: Number.isFinite(logEveryNum) ? Math.max(1, Math.trunc(logEveryNum)) : 10,
    budget,
    suppression,
    spawnEnergy: Number.isFinite(spawnEnergyNum) && spawnEnergyNum > 0 ? spawnEnergyNum : 60,
    metabolicScale: Number.isFinite(metabolicScaleNum) && metabolicScaleNum > 0 ? metabolicScaleNum : 1,
    distressScale: Number.isFinite(distressScaleNum) && distressScaleNum > 0 ? distressScaleNum : 1,
    snapshotEvery: Number.isFinite(snapshotEveryNum) ? Math.max(1, Math.trunc(snapshotEveryNum)) : 500,
    outDir,
  };
}

interface LineageNode {
  id: number;
  parentId: number | null;
  birthTick: number;
  deathTick: number | null;
}

function actionEntropy(actions: number[]): number {
  let total = 0;
  for (const a of actions) total += a;
  if (total <= 0) return 0;
  let h = 0;
  for (const a of actions) {
    if (a <= 0) continue;
    const p = a / total;
    h -= p * Math.log2(p);
  }
  return h;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function percentile(sorted: number[], p01: number): number {
  if (sorted.length === 0) return 0;
  const p = Math.max(0, Math.min(1, p01));
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx]!;
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

function main() {
  const args = parseArgs();
  setRandomSeed(args.seed);

  const world = new World();
  const organisms = new OrganismManager();
  const ruleEval = new RuleEvaluator(world, organisms, {
    budgetMode: args.budget,
    suppressionMode: args.suppression,
    metabolicScale: args.metabolicScale,
    distressFireChanceScale: args.distressScale,
  });

  // Canonical init: by default, run headless in multi-origin mode (3 sites) with non-conservative bumps.
  // For strict culture-dish comparability, use the browser with `?culture=1` (conservative bumps).
  const env = new Float32Array(GRID_WIDTH * GRID_HEIGHT);
  initSimulation(world, organisms, ruleEval, {
    env,
    spawnEnergy: args.spawnEnergy,
    culture: false,
    multiOrigin: true,
  });

  const startedAt = new Date().toISOString();
  const runId = `run-${startedAt.replace(/[:.]/g, '-')}-seed-${getRandomSeed()}`;
  const dir = resolve(process.cwd(), args.outDir, runId);
  const snapshotDir = resolve(dir, 'snapshots');
  mkdirSync(snapshotDir, { recursive: true });

  const lineage = new Map<number, LineageNode>();
  let prevIds = new Set<number>(organisms.organisms.keys());
  for (const org of organisms.organisms.values()) {
    lineage.set(org.id, {
      id: org.id,
      parentId: org.parentId,
      birthTick: org.birthTick,
      deathTick: null,
    });
  }

  const rows: string[] = [
    'tick,orgs,occupied,lineages,topShare,simpson,shannonNats,pielouEven,giniLineage,meanCellsPerOrg,perimeterRatio,avgComponents,budget,measured,drift,tapeCorr,stillbirths,repTry,repOk,birthRepro,birthSplit,splitEvents,splitFrags,splitFragMean,splitSingletonRatio,splitKeepMean,xenoTry,xenoOk,xenoDriveMean,socialCohMean,actionEntropy,noveltyProxy,invOpc_p50,invOpc_p90,invOpc_p99,invOpc_mean,invOpc_corr_age,invOpc_corr_cells,invOpc_corr_energy',
  ];

  for (let tick = 1; tick <= args.ticks; tick++) {
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

    const currentIds = new Set<number>(organisms.organisms.keys());
    for (const org of organisms.organisms.values()) {
      if (!lineage.has(org.id)) {
        lineage.set(org.id, {
          id: org.id,
          parentId: org.parentId,
          birthTick: org.birthTick,
          deathTick: null,
        });
      }
    }
    for (const idDead of prevIds) {
      if (!currentIds.has(idDead)) {
        const n = lineage.get(idDead);
        if (n && n.deathTick === null) n.deathTick = tick;
      }
    }
    prevIds = currentIds;

    if (tick % args.snapshotEvery === 0 || tick === args.ticks) {
      let largestId = -1;
      let largestN = -1;
      for (const org of organisms.organisms.values()) {
        if (org.cells.size > largestN) {
          largestN = org.cells.size;
          largestId = org.id;
        }
      }
      if (largestId > 0) {
        const org = organisms.get(largestId);
        if (org) {
          const snap = {
            tick,
            orgId: largestId,
            cells: org.cells.size,
            tape512b64: tapeSnapshotBase64(org.tape),
          };
          writeFileSync(resolve(snapshotDir, `tick-${String(tick).padStart(6, '0')}-org-${largestId}.json`), `${JSON.stringify(snap, null, 2)}\n`);
        }
      }
    }

    if (tick % args.logEvery === 0 || tick === args.ticks) {
      const bk = measureEnergyBookkeeping(world, ruleEval);
      const pop = measurePopulationMetrics(world, organisms);
      const measured = biomassReservoirTotal(bk);
      const drift = measured - ruleEval.ecosystemEnergyBudget;
      const telem = snapshotAndResetTelemetry();
      const aEntropy = actionEntropy(telem.actionExec);
      const noveltyProxy = pop.simpsonDiversity * (1 + aEntropy);
      const splitFragMean = telem.splitFragments > 0 ? telem.splitFragmentCells / telem.splitFragments : 0;
      const splitSingletonRatio = telem.splitFragments > 0 ? telem.splitFragmentSingletons / telem.splitFragments : 0;
      const splitKeepMean = telem.splitEvents > 0 ? telem.splitLargestKeptCells / telem.splitEvents : 0;
      const xenoDriveMean = telem.xenoTransferAttempts > 0 ? telem.xenoTransferDriveSum / telem.xenoTransferAttempts : 0;
      const socialCohMean = telem.socialCohesionSamples > 0 ? telem.socialCohesionSum / telem.socialCohesionSamples : 0;

      // Rule-table health (invalid opcode) vs fitness proxies (snapshot).
      const inv: number[] = [];
      const ages: number[] = [];
      const cellsArr: number[] = [];
      const energies: number[] = [];
      for (const org of organisms.organisms.values()) {
        inv.push(countInvalidRuleOpcodes(org.tape.data).invalid);
        ages.push(org.age);
        cellsArr.push(org.cells.size);
        let e = 0;
        for (const idx of org.cells) e += world.getCellEnergyByIdx(idx);
        energies.push(e);
      }
      const invSorted = [...inv].sort((a, b) => a - b);
      const invP50 = percentile(invSorted, 0.5);
      const invP90 = percentile(invSorted, 0.9);
      const invP99 = percentile(invSorted, 0.99);
      const invMean = mean(inv);
      const invCorrAge = pearsonCorr(inv, ages);
      const invCorrCells = pearsonCorr(inv, cellsArr);
      const invCorrEnergy = pearsonCorr(inv, energies);

      rows.push(
        [
          tick,
          organisms.count,
          bk.occupiedCells,
          pop.uniqueLineages,
          pop.topLineageShare.toFixed(6),
          pop.simpsonDiversity.toFixed(6),
          pop.shannonLineageNats.toFixed(6),
          pop.pielouEvenness.toFixed(6),
          pop.giniLineageSizes.toFixed(6),
          pop.meanCellsPerOrganism.toFixed(6),
          pop.perimeterRatio.toFixed(6),
          pop.avgComponents.toFixed(6),
          ruleEval.ecosystemEnergyBudget.toFixed(6),
          measured.toFixed(6),
          drift.toFixed(6),
          telem.tapeCorruptions,
          telem.stillbirths,
          telem.reproductionAttempts,
          telem.reproductionSuccess,
          telem.birthsFromReproduce,
          telem.birthsFromSplit,
          telem.splitEvents,
          telem.splitFragments,
          splitFragMean.toFixed(6),
          splitSingletonRatio.toFixed(6),
          splitKeepMean.toFixed(6),
          telem.xenoTransferAttempts,
          telem.xenoTransferSuccess,
          xenoDriveMean.toFixed(6),
          socialCohMean.toFixed(6),
          aEntropy.toFixed(6),
          noveltyProxy.toFixed(6),
          invP50.toFixed(6),
          invP90.toFixed(6),
          invP99.toFixed(6),
          invMean.toFixed(6),
          invCorrAge.toFixed(6),
          invCorrCells.toFixed(6),
          invCorrEnergy.toFixed(6),
        ].join(','),
      );
    }
  }

  const config = {
    runId,
    startedAt,
    seed: getRandomSeed(),
    ticks: args.ticks,
    logEvery: args.logEvery,
    budget: args.budget,
    suppression: args.suppression,
    spawnEnergy: args.spawnEnergy,
    metabolicScale: args.metabolicScale,
    distressScale: args.distressScale,
    snapshotEvery: args.snapshotEvery,
  };
  writeFileSync(resolve(dir, 'config.json'), JSON.stringify(config, null, 2));
  writeFileSync(resolve(dir, 'metrics.csv'), `${rows.join('\n')}\n`);
  writeFileSync(resolve(dir, 'lineage-tree.json'), `${JSON.stringify([...lineage.values()], null, 2)}\n`);

  const summary = [
    `# ${runId}`,
    '',
    `- seed: ${config.seed}`,
    `- ticks: ${config.ticks}`,
    `- budget: ${config.budget}`,
    `- suppression: ${config.suppression}`,
    `- spawnEnergy: ${config.spawnEnergy}`,
    `- metabolicScale: ${config.metabolicScale}`,
    `- distressScale: ${config.distressScale}`,
    `- snapshotEvery: ${config.snapshotEvery}`,
    '',
    'Generated files:',
    '- config.json',
    '- metrics.csv',
    '- lineage-tree.json',
    '- snapshots/*.json',
  ].join('\n');
  writeFileSync(resolve(dir, 'summary.md'), `${summary}\n`);

  console.log(`[Experiment] saved:${dir}`);
}

main();
