/**
 * Reader index: **paths that end in `RuleEvaluator.stomachInflow`** (non-negative inflow; overflow → same cell `envEnergy`).
 *
 * Authoritative implementation: [`rule-evaluator.ts`](../rule-evaluator.ts) private `stomachInflow`.
 *
 * | Source | Function / site | What moves |
 * |--------|-----------------|------------|
 * | Passive intake | `passiveAbsorb` — Phase 1 in `evaluate()` | Local `envEnergy` → stomach |
 * | EAT opcode | `actionEat` | Self + **Moore (8-neighbor)** `envEnergy` tiles → actor stomach |
 * | ABSORB (predator) | `actionAbsorb` steal branch | Neighbor **cell energy** → actor stomach |
 *
 * Stomach → environment venting is **not** listed here; see [`vent-actions.ts`](./vent-actions.ts) (`actionSpill` / `spillStomachToNearbyEnv`).
 *
 * This file intentionally contains **no runtime code** — only documentation so refactors stay grep-friendly.
 */
export {};
