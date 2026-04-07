# Signal and morphogen-related action side effects

Opcode dispatch lives in [`src/simulation/behaviors/action-dispatch.ts`](../src/simulation/behaviors/action-dispatch.ts). Neural **FIRE** (refractory gate in packed `cellData`) is implemented by `RuleEvaluator.actionFire` in [`rule-evaluator.ts`](../src/simulation/rule-evaluator.ts).

## Related implementation files (quick links)

- `src/simulation/behaviors/action-dispatch.ts` — opcode → marker/feedback wiring (table below)
- `src/simulation/rule-evaluator.ts` — `actionFire`, distress-trigger FIRE path, and `propagateSignals`
- `src/simulation/tape.ts` — opcode enum (`ActionOpcode`) and tape layout constants

## Marker bumps (`bumpMarker` via dispatch)

| Opcode | Marker slot | Notes |
|--------|-------------|--------|
| EAT | `eat` | — |
| DIGEST | `digest` | — |
| FIRE, SIG | `signal` | Same handler: `actionFire` then signal bump |
| EMIT | `signal` | Then `actionEmit` (morph A/B channel from `actionParam % 2`) |
| REPAIR | `signal` | On success only |
| ABSORB | `eat` | Predation / coupling path |
| MOVE | `move` | Bumped inside `actionMove` for **all** org cells (not in this table’s dispatch-only view) |

## Organism `writeFeedback` (tape-facing u8 slots, dispatch only)

| Opcode | Slot | Value (summary) |
|--------|------|-----------------|
| EAT | 0 | strength-scaled |
| ABSORB | 2 | `mod` |
| GIVE / TAKE | 3 | `mod` |
| DIV | 4 | `mod` |
| FIRE / SIG | 5 | Moore-neighbor env sum scaled |
| MOVE | 6 | Moore-neighbor env sum scaled (capped) |
| REPRODUCE | 7 | energy × fraction scaled (capped) |
| DIGEST | 1 | `mod` |
| REPAIR | 1 | repair hint (capped) |
| SPILL | 2 | `mod` (if spill ran) |
| JAM | 5 | `mod` (if jam applied) |
| EMIT | — | **No** `writeFeedback` in dispatch |

## FIRE without dispatch (distress)

`evaluateCell` may call `actionFire` directly when low-energy distress triggers. That path **does not** run `dispatchAction`, so it **does not** bump the signal marker or write feedback — only sets the refractory / pulse bit in `cellData` (same low-level helper as table FIRE/SIG).

## Post-rule neural propagation

After rules, `RuleEvaluator.propagateSignals` runs per organism (same-org graph signal on packed neural fields). It is **downstream** of both table FIRE/SIG and distress FIRE.
