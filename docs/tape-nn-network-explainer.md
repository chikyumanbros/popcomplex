# PopComplex Explainer (Tape / NN / Network)

## Development Style Note

This program is developed with **vibe coding** (AI-assisted iterative prototyping and refinement).  
The implementation style prioritizes quick run-observe-adjust loops over rigid up-front specification.

---

## 1) Tape Module Glossary

| Term | Meaning |
|---|---|
| `Tape` | Core per-organism data structure containing both "genome" and wear history: `data[256]` plus `degradation[256]`. |
| `TAPE_SIZE` | Total tape byte count (`256`). This is a layout and snapshot contract value. |
| `data` | Evolvable byte array holding rules, constants, and NN weights. |
| `degradation` | Per-byte wear amount. Parent wear affects copy noise, but offspring start with zero wear. |
| Data region (`0..31`) | Operation nodes (`0..15`) and literal pool (`16..31`); `4` is structural `maxCells`. |
| Public kin tag (`28..31`) | “Face”: packed 24-bit tag for **render tint** and **foreign kin trust** (`kinTrustForeign` lineageSim). |
| Private genetic kin tag (`33..36`) | True clade signal in CA padding; **not** used in kin trust (mimicry: face can match while genetics diverge). If still unset (`0x80`×4, legacy snapshots), genetic packed equals public. |
| CA band (`32..63`) | Refractory seed (`byte 32` low nibble), **33–36** genetic kin, energy-cap modules **48–59**, replication key **60–63**. |
| `REPLICATION_KEY` (`60..63`) | Key bytes affecting reproduction viability; degradation/noise raises birth failure risk. |
| `ENERGY_CAP_BANK` (`48..59`) | Encodes surviving module slots used to decode per-cell energy cap. |
| Rule table (`64..127`) | 16 x 4-byte condition->action rows executed by the evaluator. |
| NN region (`128..255`) | 2 bytes per decoded weight parameter, for 64 total NN parameters. |
| `readModifier` | Evaluates operation nodes from `actionParam` to produce action intensity/value. |
| `applyReadDegradation` | Natural read-side wear progression; low energy and old age increase break probability. |
| `applyActionWear` | Adds usage wear around actively executed rule bytes. |
| `normalizeActionOpcode` | Maps corrupt/unknown opcode bytes to `NOP` for safe evaluation. |
| `SPILL` | Low-intensity action opcode that redistributes own stomach to local environment (self + **Moore / 8-neighbor** tiles). |
| `JAM` | Defensive action opcode that temporarily cuts cross-lineage boundary coupling near the acting cell. |
| `tapeSnapshotBase64` | Snapshot format for the full 512 bytes (`data + degradation`) as base64. |

### Tape Layout Summary

- `0..31`: operation/literal/structural values
- `32..63`: CA support band (refractory, energy-cap, replication key)
- `64..127`: condition->action rules
- `128..255`: NN weights

This layout is a fixed contract. Changing only `TAPE_SIZE` breaks semantics and compatibility.

---

## 2) Neural Network (NN) Structure

### Topology

- Input: `8`
- Hidden: `4` (activation: `tanh`)
- Output: `4` (followed by `softmax`)

In short: **8 -> 4 -> 4** compact MLP.

### Parameter Breakdown (Total: 64)

- `weightsIH`: `8 * 4 = 32`
- `weightsHO`: `4 * 4 = 16`
- `inputGain`: `8`
- `biasH`: `4`
- `biasO`: `4`

Total = `32 + 16 + 8 + 4 + 4 = 64`.  
These are decoded from the `128` NN bytes on tape (2 bytes per parameter).

### Meaning of the 8 Inputs

Built per organism in `RuleEvaluator.updateNeuralNetworks()`:

1. Average cell energy
2. Average stomach buffer
3. Average local environment energy
4. Organism size
5. Boundary-cell ratio
6. Foreign-contact ratio
7. Marker dominance
8. Local environment gradient

### Meaning of the 4 Outputs ("mood")

- `0`: `NN_EAT` (feeding urgency)
- `1`: `NN_GROW` (growth/reproduction urgency)
- `2`: `NN_MOVE` (movement urgency)
- `3`: `NN_CONSERVE` (conserve/digest urgency)

Outputs form a probability distribution; argmax is stored as `nnDominant` and reused in rule conditions.

---

## 3) Network Structure (Cell Coupling and Signals)

### A. Morphological Network (Spatial Graph)

- Each organism is a set of occupied cells, interacting through a **Moore (8-neighbor)** neighborhood on the grid (single topology; no 4-neighbor mode).
- Boundary checks, same/foreign contact counts, and empty-neighbor counts are all local-topology derived from that neighborhood.
- Digestion efficiency is multiplied by same-org connectivity, so isolated cells are structurally disadvantaged.

### B. Neural-like Signal Propagation

`propagateSignals()` drives same-organism signal spread:

- `neuralState = 0`: idle
- `neuralState = 1`: firing (will propagate to **same-org Moore neighbors**)
- `neuralState = 2`: refractory

Refractory duration comes from tape (`refractoryPeriod`, low nibble of byte `32`).

### C. Social Consensus Drift (Local Synchronization)

`applySocialConsensusDrift()` softly pulls each cell's signal marker toward the local same-organism **Moore-neighborhood** mean.  
This creates agreement-like dynamics from topology and local interaction, without a centralized controller.

### D. Runtime Responsibility Split

- **CPU**: authoritative simulation (rules, metabolism, NN update, signal propagation)
- **GPU**: rendering only

In the current build, GPU compute shaders are not the source of truth for simulation state.

---

## 4) Energy / Metabolism (Authoritative Invariants)

These are **implementation-true** laws for reasoning about transfers (see `simulationTick` in `main.ts` and `RuleEvaluator`).

1. **Closed budget**: Each tick, `sum(envEnergy) + sum(cell energy) + sum(stomach)` matches `ecosystemEnergyBudget` after enforcement.
2. **Environment → organism intake**: Flow from `envEnergy` into the organism goes through **`stomachInflow`** (passive absorb + `EAT`). No direct `env → cellEnergy` shortcut.
3. **Stomach → cell gate**: Net movement from stomach buffer into cell energy runs in **`digestPhase()`** only. The **`DIGEST` opcode** only raises per-cell `digestRuleBoost` consumed there; it does not immediately move energy.
4. **Direct cell↔cell**: Same-organism `GIVE` / `TAKE` move **cell energy** between **Moore-adjacent** cells. Xenogeneic direct paths exist only where the evaluator allows them (e.g. kin-trust `GIVE`); predation steal paths use **actor stomach** (`stomachInflow`).
5. **Stomach overflow**: `stomachInflow` clamps to cap; excess returns to **local `envEnergy`** (closed budget).

Digestion module corruption: if `Tape.isDigestModuleIntact()` is false, `digestPhase` skips that organism’s cells, but EAT/passive intake can still fill stomach (converter off, buffer may fill).

**JAM vs cross-lineage cooperation**: `foreignKinCooperationEdgeOpen(!jammed)` gates **kin-trust foreign `GIVE`**, **foreign kin weight in REPAIR quorum**, and **horizontal tape transfer (HGT)** at the absorb interface. On a jammed edge, **bidirectional morph ABSORB relax** is also off; **predation steal** to stomach still runs (see `foreignAbsorbInteraction`).

---

## 4.1) Observability: invalid opcodes (“dead rules”)

Rule table opcode bytes can corrupt; invalid/unknown opcodes are normalized to `NOP` for safe evaluation.  
High invalid-opcode rates are not automatically “bad” (they can act like neutral drift / exploration), but you should check whether they correlate with fitness proxies.

Two built-in places to observe this:

- **AI handoff report** (`Copy AI handoff` button): prints `invalidOpcode` distribution plus simple **Pearson correlations** vs age/size/biomass densities (snapshot-only).
- **Headless experiment CSV** (`npm run experiment:run`): `runs/*/metrics.csv` includes `invOpc_p50/p90/p99/mean` and `invOpc_corr_*` columns for tick-by-tick tracking.

---

## 5) Rule Table: Scan Order and Chain Flag

- **Scan order**: For each cell, rules are visited in a **rotated** order: start index = `(simTick + cellIdx + orgId) % ruleCount`, then wrap through all rules once. Not always row `0` first.
- **Multiple firings**: The loop does not stop after the first successful action (except **MOVE** invalidates further evaluation for that cell in the same tick by returning early).
- **Chain bit**: Condition flag **bit 6** (`0x40`). If set and the condition passes, **no action runs**; `chainPassed` becomes true for the **next** rule, which must also pass its condition before an action can execute. **`NOP` clears** the chain. This is **logical chaining**, not spatial connectivity (contrast with morphological neighbors in §3).

---

## Reference Files

- `src/simulation/tape.ts`
- `src/simulation/tape-health.ts` — invalid opcode counting, wear level helpers
- `src/simulation/metabolic-edge.ts` — morph match thresholds, `foreignAbsorbInteraction`, `canPassiveIntakeFromEnv`, `allowsForeignKinGive`, `foreignKinCooperationEdgeOpen`
- `src/simulation/neural-network.ts`
- `src/simulation/organism.ts`
- `src/simulation/rule-evaluator.ts`
- `src/simulation/world.ts`
- `src/simulation/ai-handoff.ts` — AI snapshot report (tape dumps + rule health + correlations)
- `src/experiments/run-headless.ts` — headless `metrics.csv` producer (`invOpc_*` columns)
- `README.md`
