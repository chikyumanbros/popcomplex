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
| Lineage bytes (`28..31`) | Bytes used for lineage/kin tint and clade-style identity. |
| CA band (`32..63`) | Stores refractory seed (`byte 32` low nibble), replication key, and energy-cap modules. |
| `REPLICATION_KEY` (`60..63`) | Key bytes affecting reproduction viability; degradation/noise raises birth failure risk. |
| `ENERGY_CAP_BANK` (`48..59`) | Encodes surviving module slots used to decode per-cell energy cap. |
| Rule table (`64..127`) | 16 x 4-byte condition->action rows executed by the evaluator. |
| NN region (`128..255`) | 2 bytes per decoded weight parameter, for 64 total NN parameters. |
| `readModifier` | Evaluates operation nodes from `actionParam` to produce action intensity/value. |
| `applyReadDegradation` | Natural read-side wear progression; low energy and old age increase break probability. |
| `applyActionWear` | Adds usage wear around actively executed rule bytes. |
| `normalizeActionOpcode` | Maps corrupt/unknown opcode bytes to `NOP` for safe evaluation. |
| `SPILL` | Low-intensity action opcode that redistributes own stomach to local environment (self + orthogonal neighbors). |
| `JAM` | Defensive action opcode that temporarily cuts cross-lineage boundary coupling near the acting cell. |
| `VENT` | Low-intensity action opcode that vents own cell energy to local environment when energy pressure is high. |
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
Runtime applies a small EMA smoothing to NN outputs before argmax (mood inertia), so behavior is less jittery while keeping the same tape format and parameter count.

---

## 3) Network Structure (Cell Coupling and Signals)

### A. Morphological Network (Spatial Graph)

- Each organism is a set of occupied cells, interacting through local neighborhoods (4-neighbor or 8-neighbor mode).
- Boundary checks, same/foreign contact counts, and empty-neighbor counts are all local-topology derived.
- Digestion efficiency is multiplied by same-org connectivity, so isolated cells are structurally disadvantaged.

### B. Neural-like Signal Propagation

`propagateSignals()` drives same-organism signal spread:

- `neuralState = 0`: idle
- `neuralState = 1`: firing (will propagate to neighbors)
- `neuralState = 2`: refractory

Refractory duration comes from tape (`refractoryPeriod`, low nibble of byte `32`).

### C. Social Consensus Drift (Local Synchronization)

`applySocialConsensusDrift()` softly pulls each cell's signal marker toward the local same-organism neighborhood mean.  
This creates agreement-like dynamics from topology and local interaction, without a centralized controller.

### D. Runtime Responsibility Split

- **CPU**: authoritative simulation (rules, metabolism, NN update, signal propagation)
- **GPU**: rendering only

In the current build, GPU compute shaders are not the source of truth for simulation state.

---

## Reference Files

- `src/simulation/tape.ts`
- `src/simulation/neural-network.ts`
- `src/simulation/organism.ts`
- `src/simulation/rule-evaluator.ts`
- `src/simulation/world.ts`
- `README.md`
