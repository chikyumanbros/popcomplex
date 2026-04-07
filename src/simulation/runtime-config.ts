export type BudgetMode = 'global' | 'local';
export type SuppressionMode = 'on' | 'off';
/** Default URL `seed=` and headless default; design-gate scenarios use this among others. */
export const DEFAULT_RUNTIME_SEED = 147341274;

export interface RuntimeConfig {
  seed: number;
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

/** Same normalization as URL `seed=` parsing (unsigned 32-bit). */
export function normalizeRuntimeSeed(n: number): number {
  return Number.isFinite(n) ? (Math.trunc(n) >>> 0) : DEFAULT_RUNTIME_SEED;
}

/** Update `?seed=` on the current URL and reload; other query params are kept. */
export function reloadPageWithSeed(seed: number): void {
  const s = normalizeRuntimeSeed(seed);
  const url = new URL(window.location.href);
  url.searchParams.set('seed', String(s));
  window.location.assign(url.toString());
}

/** Random 32-bit seed via `crypto`, then same as `reloadPageWithSeed`. */
export function reloadPageWithRandomSeed(): void {
  const u = new Uint32Array(1);
  crypto.getRandomValues(u);
  reloadPageWithSeed(u[0]!);
}

export function readRuntimeConfigFromUrl(): RuntimeConfig {
  const qs = new URLSearchParams(window.location.search);
  const seedRaw = qs.get('seed');
  const seedNum = seedRaw !== null ? Number(seedRaw) : DEFAULT_RUNTIME_SEED;
  const seed = normalizeRuntimeSeed(seedNum);

  return {
    seed,
    budgetMode: parseBudgetMode(qs.get('budget')),
    suppressionMode: parseSuppressionMode(qs.get('suppression')),
    spawnInitialEnergy: parsePositiveFinite(qs.get('spawnEnergy'), 60),
    metabolicScale: parsePositiveFinite(qs.get('metabolicScale'), 1),
    distressFireChanceScale: parsePositiveFinite(qs.get('distressScale'), 1),
    multiOrigin: parseQueryFlag(qs.get('multiOrigin'), false),
    culture: parseQueryFlag(qs.get('culture'), true),
  };
}
