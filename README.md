# PopComplex

Browser-based artificial life experiment: grid cells, tape-encoded rules, closed energy bookkeeping, WebGPU visualization.

## Authorship note

This program is developed with **vibe coding** (AI-assisted iterative prototyping and refinement).

For an architecture/glossary explainer (Tape module, NN structure, and network structure), see:

- `docs/tape-nn-network-explainer.md`
- `docs/signal-morph-actions.md` — FIRE/SIG/EMIT marker bumps, feedback slots, distress FIRE vs table FIRE

## Requirements

- **Browser**: Chromium-based with **WebGPU** enabled (Chrome / Edge / Arc; Safari 18+ on macOS with WebGPU; Firefox status varies).
- **Node.js**: 18+ recommended (for Vite).

## Run locally

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

## In-app controls (quick)

- **Space**: pause / resume
- **Step / Speed**: advance simulation deterministically or faster
- **V**: cycle field visualization modes
- **H**: toggle connected-component highlight
- **Shift + click**: inspect a cell (shows organism/tape stats)
- **Copy AI handoff**: copies a full snapshot report **plus** a ready-to-paste AI prompt (select preset: review/ecology/tape)

Production build:

```bash
npm run build
npm run preview   # optional: serve dist/
```

Quick tape/NN smoke checks:

```bash
npm run test:tape
npm run test:repro
npm run test:design
npm run test:init
```

## Tuning acceptance gate (anti-arbitrary)

When tuning ecology parameters, decide adoption by the same protocol every time:

1. **Mechanism-first rule**
   - Only accept changes explainable by simulation mechanics (connectivity, transport, lineage externalities, conservation).
   - Reject purely aesthetic hacks.
2. **One-knob PDCA**
   - Change one knob at a time.
   - If side effects appear, revert immediately.
3. **Blocking safety**
   - `npm run test:design` must have `blocking=0`.
   - `npm run test:tape`, `npm run test:repro`, and `npm run build` must pass.
4. **Fixed-seed A/B**
   - Compare at fixed seeds and fixed tick horizons (do not compare different horizons).
   - Recommended seeds: `3006`, `3007`, `3010`.
5. **Adoption metrics**
   - Use `runs/*/metrics.csv` columns:
     - `birthRepro`, `birthSplit`, `splitEvents`
     - `occupied`, `orgs`, `simpson`, `lineages`
     - (optional ecology) `shannonNats`, `pielouEven`, `giniLineage`, `meanCellsPerOrg`
     - (optional tape health) `invOpc_p50`, `invOpc_p90`, `invOpc_mean`, `invOpc_corr_age`, `invOpc_corr_cells`, `invOpc_corr_energy`
   - Prefer changes that reduce split dominance (`birthSplit / max(1,birthRepro)`) while preserving long-horizon survival (`occupied > 0` at target tick).

Example medium-horizon A/B command (single seed):

```bash
npm run experiment:run -- --seed=3006 --ticks=6000 --logEvery=250 --snapshotEvery=3000 --spawnEnergy=150 --metabolicScale=0.6 --distressScale=0.3 --outDir=runs/smoke
```

Use the same command for each seed above, then compare metrics side-by-side.

For repeatable multi-seed benchmarking in one command:

```bash
npm run experiment:stability
```

Optional args:

- `--seeds=3006,3007,3010,3011,3012`
- `--ticksList=2200,6000`
- `--outDir=runs/stability`

Browser query options (examples):

- `?multiOrigin=1` — spawn three initial clades.
- `?culture=1` — conservative nutrient hotspots (culture-dish mode; total env energy preserved).
- `?culture=0&multiOrigin=1` — headless-like multi-origin without conservative rescale (env total increases at init).
- `?culture=0&multiOrigin=0` — single-origin baseline (one proto at center, no inoculation bumps).

Other query params (all optional; defaults shown):

- `?seed=21719667` — unsigned 32-bit seed (reloads apply it).
- `?budget=local|global` — energy-budget bookkeeping mode (default: `local`).
- `?suppression=on|off` — suppression toggle (default: `on`).
- `?spawnEnergy=60` — initial spawn energy scalar (default: `60`).
- `?metabolicScale=1` — global metabolic multiplier (default: `1`).
- `?distressScale=1` — distress FIRE chance scale (default: `1`).

**Spatial topology**: the simulation uses **Moore (8-neighbor)** adjacency everywhere (rules, diffusion, connectivity, MOVE, EAT sampling, etc.). There is no separate 4-neighbor mode or `?neighbor=` switch.

## GPU vs CPU (what actually runs)

| Layer | Role |
|--------|------|
| **CPU** (`RuleEvaluator`, `World`, …) | **Authoritative simulation**: rules, energy, digestion, movement, reproduction, morphogens, **neural/signal propagation** (`propagateSignals` after FIRE/SIG). |
| **GPU** | **Rendering** only: `render.wgsl` reads `cellState` + `envEnergy` buffers uploaded each frame. View pan/zoom uniforms are applied in the fragment shader. |

**Not used in the live loop** (pipelines / shaders exist under `src/gpu/` but `main.ts` does not submit their compute passes):

- `ca-nervous.wgsl` — would step a same-org nervous CA; **duplicates** logic that already runs on CPU (Moore neighbors when synced). Do not treat it as ground truth unless you reconnect it and remove or sync the CPU path.
- `cell-update.wgsl`, `env-diffusion.wgsl` — experimental / future stepping (`env-diffusion` uses Moore-averaging when aligned with CPU `stepEnvDiffusion`).
- `inject-energy.wgsl` — pipeline is created for possible GPU injection; **mouse inject was removed**; CPU writes `envEnergy` directly.

**Single source of truth for neural state**: `World.cellData` packed fields, updated by `RuleEvaluator` (including `propagateSignals`). The render shader only **displays** `neuralState` for visuals.

## Project layout (short)

- `src/main.ts` — WebGPU init, frame loop, UI hooks.
- `src/simulation/` — world, tape, rules, transcription, metrics.
- `src/gpu/shaders/render.wgsl` — fullscreen pass, cell colours from uploaded state.
- `src/ui/` — controls, inspector, stats.

## Tape layout (256B data + 256B degradation)

Authoritative comments live in `src/simulation/tape.ts` at the top of the file. In short: bytes **0–31** are the evolvable data region (op-nodes, literals, maxCells at index 4, **public kin tag 28–31** for render + foreign trust); **32–63** is the CA band (private genetic kin **33–36**, energy-cap **48–59**, replication key **60–63**); **64–127** holds **16** condition→action rules; **128–255** stores NN weight bytes.

**`TAPE_SIZE = 256` is a fixed layout contract** — offsets (`CA_RULES_OFFSET`, `CONDITIONS_OFFSET`, NN block length, etc.) are chosen to fill the tape. Changing only `TAPE_SIZE` without reallocating those regions will break snapshots and semantics.

### Transcription (`transcription.ts`)

Copy uses write noise, per-byte XOR bit-flips, rare **random byte swaps**, and adjacent mis-reads. Swaps still draw any pair of indices, but **acceptance** is scaled down when the pair crosses coarse regions (data / CA / rules / NN), touches the **replication key (60–63)**, or touches the **rule table (64–127)** — tunable via `CHANNEL_SWAP_*` exports so boundary-crossing swaps stay possible but less frequent than bulk data.

### Degradation (`Tape.applyReadDegradation`, `corruptByte`)

Wear is a **single-bit XOR** plus a bump to the parallel `degradation[]` track. Hit probability is scaled by **`tapeByteDegradationSensitivity`** (`TAPE_DEGRAD_SENS_*` in `tape.ts`): e.g. maxCells, replication key, **rule opcode bytes**, refractory (byte 32), and the NN band are harder to corrupt than generic literals — easy to retune for ecology experiments.

### Child `degradation` starts clean (but degraded birth can add initial wear)

`transcribe` returns `new Tape(data)` with a **fresh zero** wear array. Parent **degradation** never directly copies to offspring: it only **biases** transcription noise (especially on the replication key).

Reproduction uses `transcribeForReproductionOutcome`: after the noisy copy, it applies a probabilistic **proofreading** step that can revert some mutated bytes back toward the parent (strongest on the replication key, then rule opcodes; NN bytes are proofread lightly). Proofreading strength scales primarily with **local same-org neighborhood quorum** (organization), with a smaller capped boost from overall colony size. Instead of hard stillbirth, reproduction can yield a **degraded birth** outcome — a child is still spawned but starts weaker (lower initial energy) and may be given **intentional initial wear / NOPs** as “repair debt” to preserve lineage continuity while penalizing low-fidelity replication.

### NN weights (bytes 128–255)

Weights live on the same `data[]` as rules and literals, so **transcription and degradation apply to the NN block too**. Alternative designs (separate NN buffer, different mutation rates) would decouple “continuous mood weights” from discrete rule evolution at the cost of more state and UI. Decoding is explicit: each weight uses two bytes as a big-endian u16, then

`float = (u16 - NN_TAPE_WEIGHT_CENTER) / NN_TAPE_WEIGHT_SCALE`

(`NN_TAPE_WEIGHT_CENTER` = 32768, `NN_TAPE_WEIGHT_SCALE` = 16384 → about **±2** full-scale). Adjust those constants to change effective magnitude / saturation without changing tensor sizes.

Invalid rule opcode bytes are normalized to **NOP** when rules are read (`getRule`), so evaluation treats them like inactive rows until REPAIR or mutation fixes the raw byte.

Recent low-intensity opcode additions keep the same compatibility rule:

- `SPILL` — spills a small amount of own stomach to local environment (self + **Moore / 8-neighbor** env tiles).
- `JAM` — applies a short-lived defensive boundary jam that cuts cross-lineage coupling routes (e.g. horizontal tape transfer, foreign ABSORB coupling path) near the acting cell.

### Seed vs initial genome

- Runtime `seed` controls stochastic events during simulation/transcription (action noise, corruption events, swaps, stillbirth, etc.).
- The canonical starter genome from `createProtoTape()` is fixed by design.
- In particular, proto NN bytes are generated from `PROTO_TAPE_NN_SEED`, so changing runtime `seed` does **not** change the initial NN genome unless proto generation logic is changed.

## Vite / build notes

See `vite.config.ts`. Production builds use **`build.sourcemap: false`** by default (smaller `dist/`, no published `.map` files). To debug a production build with maps, run:

```bash
npx vite build --sourcemap
```

or set `build.sourcemap: true` or `'hidden'` in config.
