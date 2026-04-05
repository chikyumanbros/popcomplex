import { initWebGPU } from './gpu/context';
import { createBuffers, writeUniform } from './gpu/buffers';
import { createRenderPipeline } from './gpu/pipelines/render';
import { World } from './simulation/world';
import { OrganismManager } from './simulation/organism';
import { createProtoTape } from './simulation/tape';
import { RuleEvaluator } from './simulation/rule-evaluator';
import { buildAIHandoffMarkdown, buildAIHandoffPrompt, type AIHandoffPromptPreset } from './simulation/ai-handoff';
import {
  createUI,
  updateStats,
  viewPortCenterGrid,
  pointerClientToGrid,
  type UIState,
} from './ui/controls';
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
import { readRuntimeConfigFromUrl, DEFAULT_RUNTIME_SEED } from './simulation/runtime-config';
import { snapshotAndResetTelemetry } from './simulation/telemetry';
import {
  addRegionalEnvBumps,
  addRegionalEnvBumpsConservative,
  spawnTricladProtos,
  DEFAULT_TRICLADE_SITES,
} from './simulation/initial-inoculation';

/** Full-grid `measurePopulationMetrics` only every N sim ticks — halves CPU vs chart+logs at speed 1. */
const CHART_SAMPLE_EVERY_SIM_TICK = 2;

/** Same query shape as `design-gate` mini-world + multi-origin inoculation (browser defaults differ otherwise). */
function buildExperimentPresetUrl(): string {
  const params = new URLSearchParams({
    multiOrigin: '1',
    culture: '1',
    spawnEnergy: '150',
    seed: String(DEFAULT_RUNTIME_SEED),
    metabolicScale: '0.6',
    distressScale: '0.3',
  });
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}

function mountExperimentPresetUrlPanel(): void {
  const presetUrl = buildExperimentPresetUrl();
  console.log(`[Run] experiment preset URL: ${presetUrl}`);
  const panel = document.getElementById('quick-run-panel');
  if (!panel) return;

  panel.hidden = false;
  panel.replaceChildren();

  const label = document.createElement('div');
  label.className = 'quick-run-label';
  label.textContent = 'Experiment preset (design-gate match)';

  const link = document.createElement('a');
  link.className = 'quick-run-url';
  link.href = presetUrl;
  link.textContent = presetUrl;

  const actions = document.createElement('div');
  actions.className = 'quick-run-actions';

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.textContent = 'Copy URL';
  copyBtn.addEventListener('click', () => {
    void navigator.clipboard.writeText(presetUrl).then(
      () => {
        copyBtn.textContent = 'Copied';
        window.setTimeout(() => {
          copyBtn.textContent = 'Copy URL';
        }, 1600);
      },
      () => {
        console.log('[Run] clipboard unavailable; preset URL printed above');
      },
    );
  });

  const restartBtn = document.createElement('button');
  restartBtn.type = 'button';
  restartBtn.title = 'Reload; keeps current URL query';
  restartBtn.textContent = 'Restart';
  restartBtn.addEventListener('click', () => {
    window.location.reload();
  });

  actions.append(copyBtn, restartBtn);
  panel.append(label, link, actions);
}

async function main() {
  const cfg = readRuntimeConfigFromUrl();
  mountExperimentPresetUrlPanel();
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
  const ui = createUI(canvas);
  const chartCanvas = document.getElementById('ecology-chart') as HTMLCanvasElement | null;
  const ecologyChart = chartCanvas ? new EcologyTrendChart(chartCanvas) : null;

  setupInspector(canvas, world, organisms, ui);

  ruleEval.snapClosedEnergyBudgetFromWorld();

  if (cfg.multiOrigin || cfg.culture) {
    spawnTricladProtos(world, organisms, ruleEval, createProtoTape(), cfg.spawnInitialEnergy, DEFAULT_TRICLADE_SITES);
  }

  document.getElementById('btn-spawn')!.addEventListener('click', () => {
    const { gx, gy } = viewPortCenterGrid(ui);
    spawnProtoAt(world, organisms, ruleEval, cfg.spawnInitialEnergy, gx, gy);
  });

  canvas.addEventListener('click', (e) => {
    if (e.shiftKey) return;
    const { gx, gy } = pointerClientToGrid(canvas, ui, e.clientX, e.clientY);
    spawnProtoAt(world, organisms, ruleEval, cfg.spawnInitialEnergy, gx, gy);
  });

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

  document.getElementById('btn-ai-handoff')!.addEventListener('click', () => {
    copyTextForAI(buildHandoffMarkdown());
  });

  document.getElementById('btn-ai-handoff-prompt')!.addEventListener('click', () => {
    const presetEl = document.getElementById('ai-prompt-preset') as HTMLSelectElement | null;
    const preset = (presetEl?.value ?? 'review') as AIHandoffPromptPreset;
    const report = buildHandoffMarkdown();
    const prompt = buildAIHandoffPrompt(preset);
    const payload = `${report}\n\n### Prompt to append in AI chat\n\`\`\`\n${prompt}\n\`\`\``;
    copyTextForAI(payload);
  });

  function frame() {
    requestAnimationFrame(frame);
    stats.recordFrame();
    frameCount++;

    let advanced = false;
    if (!ui.paused || ui.stepRequested) {
      const steps = ui.stepRequested ? 1 : ui.speed;
      ui.stepRequested = false;
      for (let s = 0; s < steps; s++) {
        simulationTick(world, organisms, ruleEval, ui);
        trackTurnover();
        tick++;
        advanced = true;
      }
      if (advanced && ecologyChart && tick % CHART_SAMPLE_EVERY_SIM_TICK === 0) {
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

    // GPU: display only — full sim (rules, metabolism, neural propagation) runs on CPU first.
    world.uploadTo(gpu!.device, buffers.cellState[0]);
    gpu!.device.queue.writeBuffer(buffers.envEnergy[0], 0, ruleEval.envEnergy.buffer);

    writeUniform(gpu!.device, buffers.uniform, tick, 0, ui.viewX, ui.viewY, ui.viewZoom, ui.viewMode);
    const encoder = gpu!.device.createCommandEncoder({ label: 'render' });
    renderer.draw(encoder, 0);
    gpu!.device.queue.submit([encoder.finish()]);

    // Heavy bookkeeping scan runs only when simulation advanced.
    // While paused, refresh occasionally to keep UI values from going stale.
    if (advanced || frameCount % 10 === 0) {
      bkNow = measureEnergyBookkeeping(world, ruleEval);
      measured = biomassReservoirTotal(bkNow);
      closed = ruleEval.ecosystemEnergyBudget;
      drift = measured - closed;
    }
    updateStats(tick, organisms.count, stats.fps, closed, measured, drift, ui.viewMode);

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

  requestAnimationFrame(frame);

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

function spawnProtoAt(
  world: World,
  organisms: OrganismManager,
  ruleEval: RuleEvaluator,
  spawnInitialEnergy: number,
  x: number,
  y: number,
) {
  x = Math.max(2, Math.min(GRID_WIDTH - 3, x));
  y = Math.max(2, Math.min(GRID_HEIGHT - 3, y));
  if (!world.isEmpty(x, y)) return;
  if (!ruleEval.withdrawEnvUniform(spawnInitialEnergy)) return;

  const tape = createProtoTape();
  const id = world.spawnProto(x, y, tape.getLineagePacked(), spawnInitialEnergy);
  organisms.register(id, tape, { parentId: null, birthTick: 0 });
  const org = organisms.get(id)!;
  org.cells.add(y * GRID_WIDTH + x);
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
    `[Diversity] lineages:${opts.popMetrics.uniqueLineages} topShare:${(opts.popMetrics.topLineageShare * 100).toFixed(2)}% simpson:${opts.popMetrics.simpsonDiversity.toFixed(4)}`,
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
  console.log(
    `[Top] ${orgDetails.slice(0, 5).join(' ')}${orgDetails.length > 5 ? '...' : ''}`,
  );
}

main().catch((e) => {
  console.error(e);
  const errEl = document.getElementById('error');
  if (errEl) errEl.textContent = `Error: ${e.message}`;
});
