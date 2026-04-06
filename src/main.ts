import { initWebGPU } from './gpu/context';
import { createBuffers, writeUniform } from './gpu/buffers';
import { createRenderPipeline } from './gpu/pipelines/render';
import { World } from './simulation/world';
import { OrganismManager } from './simulation/organism';
import { ActionOpcode, createProtoTape } from './simulation/tape';
import { RuleEvaluator } from './simulation/rule-evaluator';
import { buildAIHandoffMarkdown, buildAIHandoffPrompt, type AIHandoffPromptPreset } from './simulation/ai-handoff';
import { createUI, updateStats, type UIState } from './ui/controls';
import { EcologyTrendChart } from './ui/ecology-chart';
import { setupInspector } from './ui/inspector';
import { StatsTracker } from './ui/stats';
import { GRID_WIDTH, GRID_HEIGHT } from './simulation/constants';
import {
  measureEnergyBookkeeping,
  measurePopulationMetrics,
  biomassReservoirTotal,
  diffBookkeeping,
  type EnergyBookkeeping,
  type PopulationMetrics,
} from './simulation/energy-metrics';
import { setRandomSeed, getRandomSeed } from './simulation/rng';
import { readRuntimeConfigFromUrl } from './simulation/runtime-config';
import { snapshotAndResetTelemetry } from './simulation/telemetry';
import {
  addRegionalEnvBumps,
  addRegionalEnvBumpsConservative,
  spawnTricladProtos,
  DEFAULT_TRICLADE_SITES,
} from './simulation/initial-inoculation';

/** Full-grid `measurePopulationMetrics` only every N sim ticks — halves CPU vs chart+logs at speed 1. */
const CHART_SAMPLE_EVERY_SIM_TICK = 2;

/**
 * Background tabs throttle `requestAnimationFrame` heavily; drive the CPU sim with `setInterval` instead.
 * Step budget approximates visible-tab rate (~60 rAF/s × ui.speed) over each wake interval.
 */
const BACKGROUND_TICK_MS = 200;
const MAX_BACKGROUND_SIM_STEPS = 800;
const ASSUMED_VISIBLE_FPS = 60;

async function main() {
  const cfg = readRuntimeConfigFromUrl();
  setRandomSeed(cfg.seed);
  console.log(
    `[Run] seed:${getRandomSeed()} neighbor:${cfg.neighborMode} budget:${cfg.budgetMode} suppression:${cfg.suppressionMode} spawnEnergy:${cfg.spawnInitialEnergy} metabolicScale:${cfg.metabolicScale} distressScale:${cfg.distressFireChanceScale} multiOrigin:${cfg.multiOrigin} culture:${cfg.culture}`,
  );

  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  canvas.width = GRID_WIDTH;
  canvas.height = GRID_HEIGHT;

  let gpu;
  try {
    gpu = await initWebGPU(canvas);
  } catch (e) {
    document.getElementById('error')!.textContent = `WebGPU Error: ${e}`;
    return;
  }

  const buffers = createBuffers(gpu.device);
  const renderer = createRenderPipeline(gpu, buffers);

  const world = new World();
  const organisms = new OrganismManager();
  const ruleEval = new RuleEvaluator(world, organisms, {
    neighborMode: cfg.neighborMode,
    budgetMode: cfg.budgetMode,
    suppressionMode: cfg.suppressionMode,
    metabolicScale: cfg.metabolicScale,
    distressFireChanceScale: cfg.distressFireChanceScale,
  });
  if (cfg.culture) {
    // Culture dish mode: nutrient spots with fixed total env energy.
    addRegionalEnvBumpsConservative(buffers.initialEnv, DEFAULT_TRICLADE_SITES, 1.0, 14);
    gpu.device.queue.writeBuffer(buffers.envEnergy[0], 0, buffers.initialEnv.buffer);
  } else if (cfg.multiOrigin) {
    addRegionalEnvBumps(buffers.initialEnv, DEFAULT_TRICLADE_SITES, 1.0, 14);
    gpu.device.queue.writeBuffer(buffers.envEnergy[0], 0, buffers.initialEnv.buffer);
  }
  ruleEval.setEnvEnergy(buffers.initialEnv);
  const stats = new StatsTracker();
  const ui = createUI(canvas, cfg.seed);
  const chartCanvas = document.getElementById('ecology-chart') as HTMLCanvasElement | null;
  const ecologyChart = chartCanvas ? new EcologyTrendChart(chartCanvas) : null;

  const componentMaskCpu = new Uint32Array(GRID_WIDTH * GRID_HEIGHT);
  let selectedComponentSeedIdx: number | null = null;
  let selectedComponentOrgId: number | null = null;
  let lastComponentHighlight = ui.componentHighlight;

  function rebuildSelectedComponentMask() {
    componentMaskCpu.fill(0);
    if (selectedComponentSeedIdx === null || selectedComponentOrgId === null) return;
    const seed = selectedComponentSeedIdx;
    const orgId = selectedComponentOrgId;
    if (world.getOrganismIdByIdx(seed) !== orgId) return;

    const q: number[] = [seed];
    componentMaskCpu[seed] = 1;
    let qi = 0;
    const use8 = cfg.neighborMode === 'eight';
    while (qi < q.length) {
      const idx = q[qi++]!;
      const x = idx % GRID_WIDTH;
      const y = (idx - x) / GRID_WIDTH;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (!use8 && dx !== 0 && dy !== 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
          const ni = ny * GRID_WIDTH + nx;
          if (componentMaskCpu[ni] !== 0) continue;
          if (world.getOrganismIdByIdx(ni) !== orgId) continue;
          componentMaskCpu[ni] = 1;
          q.push(ni);
        }
      }
    }
  }

  setupInspector(canvas, world, organisms, ui, (idx, orgId) => {
    selectedComponentSeedIdx = idx;
    selectedComponentOrgId = orgId;
    rebuildSelectedComponentMask();
    if (ui.componentHighlight) {
      gpu.device.queue.writeBuffer(buffers.componentMask, 0, componentMaskCpu.buffer);
    }
  });

  ruleEval.snapClosedEnergyBudgetFromWorld();

  if (cfg.multiOrigin || cfg.culture) {
    spawnTricladProtos(world, organisms, ruleEval, createProtoTape(), cfg.spawnInitialEnergy, DEFAULT_TRICLADE_SITES);
  }

  let tick = 0;
  let frameCount = 0;
  let lastLoggedTick = -1;
  let bookkeepingAtLastLog: EnergyBookkeeping | null = null;
  let previousOrgIds = new Set<number>(organisms.organisms.keys());
  let birthsSinceLastLog = 0;
  let deathsSinceLastLog = 0;
  let bkNow = measureEnergyBookkeeping(world, ruleEval);
  let measured = biomassReservoirTotal(bkNow);
  let closed = ruleEval.ecosystemEnergyBudget;
  let drift = measured - closed;

  let lastChartPopForLog: PopulationMetrics | null = null;
  let lastChartSampleTick = -1;

  // Frame rate limiting (optional: set to 0 for unlimited, or e.g. 60 for 60fps cap)
  const TARGET_FPS = 0; // 0 = unlimited
  const FRAME_MIN_MS = TARGET_FPS > 0 ? 1000 / TARGET_FPS : 0;
  let lastFrameTime = performance.now();

  let driverRaf = 0;
  let driverInterval = 0;

  function simulationStepsForBackgroundWake(): number {
    return Math.min(
      MAX_BACKGROUND_SIM_STEPS,
      Math.max(1, Math.round((ui.speed * ASSUMED_VISIBLE_FPS * BACKGROUND_TICK_MS) / 1000)),
    );
  }

  function runSimStepBatch(steps: number): boolean {
    let advanced = false;
    for (let s = 0; s < steps; s++) {
      simulationTick(world, organisms, ruleEval, ui);
      trackTurnover();
      tick++;
      advanced = true;
      if (ecologyChart && tick % CHART_SAMPLE_EVERY_SIM_TICK === 0) {
        const pop = measurePopulationMetrics(world, organisms);
        lastChartPopForLog = pop;
        lastChartSampleTick = tick;
        ecologyChart.sample(
          tick,
          organisms.count,
          pop.simpsonDiversity,
          pop.uniqueLineages,
          pop.topLineageShare,
          pop.perimeterRatio,
        );
      }
    }
    return advanced;
  }

  function refreshBookkeepingAndStats(advanced: boolean) {
    if (advanced || frameCount % 10 === 0) {
      bkNow = measureEnergyBookkeeping(world, ruleEval);
      measured = biomassReservoirTotal(bkNow);
      closed = ruleEval.ecosystemEnergyBudget;
      drift = measured - closed;
    }
    updateStats(tick, organisms.count, stats.fps, closed, measured, drift, ui.viewMode);
  }

  function maybeLogWorldState() {
    if (tick > 0 && tick % 10 === 0 && tick !== lastLoggedTick) {
      lastLoggedTick = tick;
      const pop =
        lastChartSampleTick === tick && lastChartPopForLog
          ? lastChartPopForLog
          : measurePopulationMetrics(world, organisms);
      const telem = snapshotAndResetTelemetry();
      logWorldState(organisms, ruleEval, world, {
        tick,
        bookkeepingNow: bkNow,
        bookkeepingAtLastLog,
        closedBudget: closed,
        drift,
        popMetrics: pop,
        telemetry: telem,
        birthsSinceLastLog,
        deathsSinceLastLog,
        onAfterLog(next: EnergyBookkeeping) {
          bookkeepingAtLastLog = next;
          birthsSinceLastLog = 0;
          deathsSinceLastLog = 0;
        },
      });
    }
  }

  function scheduleVisibleFrame() {
    driverRaf = requestAnimationFrame(frame);
  }

  function syncDriverToVisibility() {
    if (document.hidden) {
      if (driverRaf !== 0) {
        cancelAnimationFrame(driverRaf);
        driverRaf = 0;
      }
      if (driverInterval === 0) {
        driverInterval = window.setInterval(backgroundFrame, BACKGROUND_TICK_MS);
      }
    } else {
      if (driverInterval !== 0) {
        clearInterval(driverInterval);
        driverInterval = 0;
      }
      if (driverRaf === 0) {
        scheduleVisibleFrame();
      }
    }
  }

  if (ecologyChart) {
    const pop0 = measurePopulationMetrics(world, organisms);
    lastChartPopForLog = pop0;
    lastChartSampleTick = 0;
    ecologyChart.sample(
      0,
      organisms.count,
      pop0.simpsonDiversity,
      pop0.uniqueLineages,
      pop0.topLineageShare,
      pop0.perimeterRatio,
    );
  }

  function buildHandoffMarkdown(): string {
    const bk = measureEnergyBookkeeping(world, ruleEval);
    return buildAIHandoffMarkdown({
      tick,
      world,
      organisms,
      ruleEval,
      ui,
      bookkeepingSnapshot: bk,
    });
  }

  function copyTextForAI(text: string) {
    void navigator.clipboard.writeText(text).then(
      () => {
        alert('Copied to clipboard. Paste into your AI chat.');
      },
      () => {
        console.log(text);
        alert('Could not write to clipboard. Full text was printed to the developer console.');
      },
    );
  }

  document.getElementById('btn-ai-handoff-prompt')!.addEventListener('click', () => {
    const presetEl = document.getElementById('ai-prompt-preset') as HTMLSelectElement | null;
    const preset = (presetEl?.value ?? 'review') as AIHandoffPromptPreset;
    const report = buildHandoffMarkdown();
    const prompt = buildAIHandoffPrompt(preset);
    const payload = `${report}\n\n### Prompt to append in AI chat\n\`\`\`\n${prompt}\n\`\`\``;
    copyTextForAI(payload);
  });

  function frame() {
    driverRaf = 0;

    // Frame rate limiting
    if (FRAME_MIN_MS > 0) {
      const now = performance.now();
      const elapsed = now - lastFrameTime;
      if (elapsed < FRAME_MIN_MS) {
        if (!document.hidden) scheduleVisibleFrame();
        return;
      }
      lastFrameTime = now;
    }

    stats.recordFrame();
    frameCount++;

    let advanced = false;
    if (!ui.paused || ui.stepRequested) {
      const steps = ui.stepRequested ? 1 : ui.speed;
      ui.stepRequested = false;
      advanced = runSimStepBatch(steps);
    }

    // GPU: display only — full sim (rules, metabolism, neural propagation) runs on CPU first.
    world.uploadTo(gpu!.device, buffers.cellState[0]);
    gpu!.device.queue.writeBuffer(buffers.envEnergy[0], 0, ruleEval.envEnergy.buffer);
    // Component highlight toggle: on->upload current, off->zero once.
    if (ui.componentHighlight !== lastComponentHighlight) {
      lastComponentHighlight = ui.componentHighlight;
      if (!ui.componentHighlight) {
        componentMaskCpu.fill(0);
        gpu!.device.queue.writeBuffer(buffers.componentMask, 0, componentMaskCpu.buffer);
      } else if (selectedComponentSeedIdx !== null && selectedComponentOrgId !== null) {
        rebuildSelectedComponentMask();
        gpu!.device.queue.writeBuffer(buffers.componentMask, 0, componentMaskCpu.buffer);
      }
    }

    // Selection mask can become stale if the organism moved/split/died; rebuild cheaply when selected.
    if (ui.componentHighlight && selectedComponentSeedIdx !== null && selectedComponentOrgId !== null && advanced) {
      rebuildSelectedComponentMask();
      gpu!.device.queue.writeBuffer(buffers.componentMask, 0, componentMaskCpu.buffer);
    }

    writeUniform(gpu!.device, buffers.uniform, tick, 0, ui.viewX, ui.viewY, ui.viewZoom, ui.viewMode);
    const encoder = gpu!.device.createCommandEncoder({ label: 'render' });
    renderer.draw(encoder, 0);
    gpu!.device.queue.submit([encoder.finish()]);

    refreshBookkeepingAndStats(advanced);
    maybeLogWorldState();

    if (!document.hidden) {
      scheduleVisibleFrame();
    }
  }

  function backgroundFrame() {
    frameCount++;

    let advanced = false;
    if (!ui.paused || ui.stepRequested) {
      const steps = ui.stepRequested ? 1 : simulationStepsForBackgroundWake();
      ui.stepRequested = false;
      advanced = runSimStepBatch(steps);
    }

    refreshBookkeepingAndStats(advanced);
    maybeLogWorldState();
  }

  document.addEventListener('visibilitychange', () => syncDriverToVisibility());
  syncDriverToVisibility();

  function trackTurnover() {
    const currentIds = new Set<number>(organisms.organisms.keys());
    for (const id of currentIds) {
      if (!previousOrgIds.has(id)) birthsSinceLastLog++;
    }
    for (const id of previousOrgIds) {
      if (!currentIds.has(id)) deathsSinceLastLog++;
    }
    previousOrgIds = currentIds;
  }
}

function simulationTick(
  world: World,
  organisms: OrganismManager,
  ruleEval: RuleEvaluator,
  _ui: UIState,
) {
  // Keep genotype->phenotype mapping explicit: NN is rebuilt from tape every tick.
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

interface LogWorldStateOpts {
  tick: number;
  bookkeepingNow: EnergyBookkeeping;
  bookkeepingAtLastLog: EnergyBookkeeping | null;
  closedBudget: number;
  drift: number;
  popMetrics: PopulationMetrics;
  telemetry: ReturnType<typeof snapshotAndResetTelemetry>;
  birthsSinceLastLog: number;
  deathsSinceLastLog: number;
  onAfterLog: (bk: EnergyBookkeeping) => void;
}

function logWorldState(
  organisms: OrganismManager,
  _ruleEval: RuleEvaluator,
  world: World,
  opts: LogWorldStateOpts,
) {
  const bk = opts.bookkeepingNow;
  const measured = biomassReservoirTotal(bk);
  let deltaStr = '';
  if (opts.bookkeepingAtLastLog) {
    const d = diffBookkeeping(opts.bookkeepingAtLastLog, bk);
    const dTot = d.envU + d.cellEnergy + d.stomach;
    deltaStr = ` | sinceLastLog: Δenv ${d.envU.toFixed(0)} Δcell ${d.cellEnergy.toFixed(0)} Δstomach ${d.stomach.toFixed(0)} Δsum ${dTot.toFixed(0)} (Δocc ${d.occupiedCells})`;
  }
  opts.onAfterLog(bk);

  const orgDetails: string[] = [];
  let orgListCells = 0;
  for (const org of organisms.organisms.values()) {
    let orgE = 0;
    let orgS = 0;
    for (const idx of org.cells) {
      orgE += world.getCellEnergyByIdx(idx);
      orgS += world.getStomachByIdx(idx);
    }
    orgListCells += org.cells.size;
    const metaPerCell = (0.25 + 0.008 * org.cells.size).toFixed(2);
    orgDetails.push(
      `#${org.id}(${org.cells.size}c E:${orgE.toFixed(1)} S:${orgS.toFixed(1)} m/c:${metaPerCell} age:${org.age})`,
    );
  }

  console.log(
    `[Energy] tick:${opts.tick} orgs:${organisms.count} gridOcc:${bk.occupiedCells} orgCellsΣ:${orgListCells} | env:${(bk.envU / 1e6).toFixed(4)}M cell:${bk.cellEnergy.toFixed(0)} stomach:${bk.stomach.toFixed(0)} measured:${(measured / 1e6).toFixed(4)}M budget:${(opts.closedBudget / 1e6).toFixed(4)}M drift:${opts.drift.toFixed(3)}${deltaStr}`,
  );
  console.log(
    `[Diversity] lineages:${opts.popMetrics.uniqueLineages} topShare:${(opts.popMetrics.topLineageShare * 100).toFixed(2)}% simpson:${opts.popMetrics.simpsonDiversity.toFixed(4)} shannon:${opts.popMetrics.shannonLineageNats.toFixed(3)} pielou:${opts.popMetrics.pielouEvenness.toFixed(3)} giniLin:${opts.popMetrics.giniLineageSizes.toFixed(3)} meanCells/Org:${opts.popMetrics.meanCellsPerOrganism.toFixed(2)}`,
  );
  console.log(
    `[Morphology] perimeterRatio:${opts.popMetrics.perimeterRatio.toFixed(4)} avgComponents:${opts.popMetrics.avgComponents.toFixed(3)}`,
  );
  console.log(
    `[Ecology] birthsSinceLastLog:${opts.birthsSinceLastLog} deathsSinceLastLog:${opts.deathsSinceLastLog} births(repro/split):${opts.telemetry.birthsFromReproduce}/${opts.telemetry.birthsFromSplit}`,
  );
  const splitSingletonRatio =
    opts.telemetry.splitFragments > 0
      ? opts.telemetry.splitFragmentSingletons / opts.telemetry.splitFragments
      : 0;
  const splitMeanFragCells =
    opts.telemetry.splitFragments > 0
      ? opts.telemetry.splitFragmentCells / opts.telemetry.splitFragments
      : 0;
  const splitMeanLargestKept =
    opts.telemetry.splitEvents > 0
      ? opts.telemetry.splitLargestKeptCells / opts.telemetry.splitEvents
      : 0;
  const xenoDriveMean =
    opts.telemetry.xenoTransferAttempts > 0
      ? opts.telemetry.xenoTransferDriveSum / opts.telemetry.xenoTransferAttempts
      : 0;
  const socialCohesionMean =
    opts.telemetry.socialCohesionSamples > 0
      ? opts.telemetry.socialCohesionSum / opts.telemetry.socialCohesionSamples
      : 0;
  console.log(
    `[Selection] corr:${opts.telemetry.tapeCorruptions} wrRnd:${opts.telemetry.writeRandomizations} bitflip:${opts.telemetry.channelBitflips} swap:+${opts.telemetry.channelSwapsAccepted}/-${opts.telemetry.channelSwapsRejected} stillbirth:${opts.telemetry.stillbirths} repTry:${opts.telemetry.reproductionAttempts} repOk:${opts.telemetry.reproductionSuccess} repFailDom:${opts.telemetry.reproduceFailDominance} repFailCrowd:${opts.telemetry.reproduceFailCrowding} splitEv:${opts.telemetry.splitEvents} splitFrag:${opts.telemetry.splitFragments} splitFragMean:${splitMeanFragCells.toFixed(2)} split1cRatio:${splitSingletonRatio.toFixed(2)} splitKeepMean:${splitMeanLargestKept.toFixed(2)} xenoTry:${opts.telemetry.xenoTransferAttempts} xenoOk:${opts.telemetry.xenoTransferSuccess} xenoDriveMean:${xenoDriveMean.toFixed(4)} socialCohMean:${socialCohesionMean.toFixed(4)}`,
  );
  const spillExec = opts.telemetry.actionExec[ActionOpcode.SPILL] ?? 0;
  const jamExec = opts.telemetry.actionExec[ActionOpcode.JAM] ?? 0;
  const moveExec = opts.telemetry.actionExec[ActionOpcode.MOVE] ?? 0;
  const fireExec = (opts.telemetry.actionExec[ActionOpcode.FIRE] ?? 0) + (opts.telemetry.actionExec[ActionOpcode.SIG] ?? 0);
  console.log(
    `[StressValidation] move:${moveExec} fire+sig:${fireExec} spill(vent):${spillExec} jam(exclusion):${jamExec} repFailDom:${opts.telemetry.reproduceFailDominance} repFailCrowd:${opts.telemetry.reproduceFailCrowding}`,
  );
  console.log(
    `[Top] ${orgDetails.slice(0, 5).join(' ')}${orgDetails.length > 5 ? '...' : ''}`,
  );
}

main().catch((e) => {
  console.error(e);
  const errEl = document.getElementById('error');
  if (errEl) errEl.textContent = `Error: ${e.message}`;
});
