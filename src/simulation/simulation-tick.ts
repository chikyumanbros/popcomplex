/**
 * Canonical single-tick step for the PopComplex simulation.
 *
 * All callers (main loop, design gate, headless runner, smoke tests) must use
 * this function to ensure the phase ordering stays consistent.
 * The optional `afterEvaluate` callback lets callers (e.g. headless runner)
 * insert measurement logic between phases without re-duplicating the sequence.
 */
import type { World } from './world';
import type { OrganismManager } from './organism';
import type { RuleEvaluator } from './rule-evaluator';

export function simulationTick(
  world: World,
  organisms: OrganismManager,
  ruleEval: RuleEvaluator,
): void {
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
