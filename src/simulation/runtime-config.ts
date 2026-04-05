export type NeighborMode = 'four' | 'eight';
export type BudgetMode = 'global' | 'local';
export type SuppressionMode = 'on' | 'off';
/** Default URL `seed=` and headless default; design-gate scenarios use this among others. */
export const DEFAULT_RUNTIME_SEED = 3006;

export interface RuntimeConfig {
  seed: number;
  neighborMode: NeighborMode;
  budgetMode: BudgetMode;
  suppressionMode: SuppressionMode;
  spawnInitialEnergy: number;
  metabolicScale: number;
  distressFireChanceScale: number;
  /** Match headless / design-gate: three separated protos, same genome, distinct lineage tags. */
  multiOrigin: boolean;
  /** Culture dish mode: conservative nutrient hotspots in environment (total env energy preserved). */
  culture: boolean;
}

function parseNeighborMode(v: string | null): NeighborMode {
  if (v === '8' || v === 'eight') return 'eight';
  return 'four';
}

function parseBudgetMode(v: string | null): BudgetMode {
  if (v === 'global') return 'global';
  return 'local';
}

function parseSuppressionMode(v: string | null): SuppressionMode {
  if (v === 'off' || v === '0' || v === 'false') return 'off';
  return 'on';
}

function parsePositiveFinite(v: string | null, fallback: number): number {
  if (v === null) return fallback;
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function parseQueryFlag(v: string | null, fallback: boolean): boolean {
  if (v === null) return fallback;
  const s = v.trim().toLowerCase();
  if (s === '1' || s === 'true' || s === 'yes' || s === 'on') return true;
  if (s === '0' || s === 'false' || s === 'no' || s === 'off') return false;
  return fallback;
}

export function readRuntimeConfigFromUrl(): RuntimeConfig {
  const qs = new URLSearchParams(window.location.search);
  const seedRaw = qs.get('seed');
  const seedNum = seedRaw !== null ? Number(seedRaw) : DEFAULT_RUNTIME_SEED;
  const seed = Number.isFinite(seedNum) ? (Math.trunc(seedNum) >>> 0) : DEFAULT_RUNTIME_SEED;

  return {
    seed,
    neighborMode: parseNeighborMode(qs.get('neighbor')),
    budgetMode: parseBudgetMode(qs.get('budget')),
    suppressionMode: parseSuppressionMode(qs.get('suppression')),
    spawnInitialEnergy: parsePositiveFinite(qs.get('spawnEnergy'), 60),
    metabolicScale: parsePositiveFinite(qs.get('metabolicScale'), 1),
    distressFireChanceScale: parsePositiveFinite(qs.get('distressScale'), 1),
    multiOrigin: parseQueryFlag(qs.get('multiOrigin'), false),
    culture: parseQueryFlag(qs.get('culture'), true),
  };
}
