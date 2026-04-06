import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

interface RunConfig {
  runId: string;
  seed: number;
  neighbor: string;
  budget: string;
  suppression: string;
  spawnEnergy?: number;
  metabolicScale?: number;
  distressScale?: number;
}

interface Row {
  orgs: number;
  occupied: number;
  simpson: number;
  drift: number;
  noveltyProxy: number;
}

function parseArgs() {
  const map = new Map<string, string>();
  for (const token of process.argv.slice(2)) {
    const [k, v] = token.split('=');
    if (k && v !== undefined) map.set(k.replace(/^--/, ''), v);
  }
  return {
    input: map.get('input') ?? 'runs',
    output: map.get('output') ?? 'report.md',
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}

function parseLastRow(csvText: string): Row | null {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return null;
  const header = lines[0].split(',');
  const last = lines[lines.length - 1].split(',');
  if (last.length !== header.length) return null;
  const col = (name: string) => {
    const i = header.indexOf(name);
    return i >= 0 ? Number(last[i]) : NaN;
  };
  const orgs = col('orgs');
  const occupied = col('occupied');
  const simpson = col('simpson');
  const drift = col('drift');
  const noveltyProxy = col('noveltyProxy');
  if (![orgs, occupied, simpson, drift, noveltyProxy].every((x) => Number.isFinite(x))) return null;
  return { orgs, occupied, simpson, drift, noveltyProxy };
}

function main() {
  const args = parseArgs();
  const base = resolve(process.cwd(), args.input);
  const dirs = readdirSync(base, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => resolve(base, d.name));

  const grouped = new Map<string, { cfg: RunConfig; rows: Row[] }>();
  for (const dir of dirs) {
    try {
      const cfg = JSON.parse(readFileSync(resolve(dir, 'config.json'), 'utf8')) as RunConfig;
      const row = parseLastRow(readFileSync(resolve(dir, 'metrics.csv'), 'utf8'));
      if (!row) continue;
      const spawnEnergy = cfg.spawnEnergy ?? 60;
      const metabolicScale = cfg.metabolicScale ?? 1;
      const distressScale = cfg.distressScale ?? 1;
      const key = `${cfg.neighbor}/${cfg.budget}/${cfg.suppression}/spawn${spawnEnergy}/meta${metabolicScale}/distress${distressScale}`;
      const g = grouped.get(key) ?? { cfg, rows: [] };
      g.rows.push(row);
      grouped.set(key, g);
    } catch {
      // ignore non-run directories
    }
  }

  const lines: string[] = ['# Experiment Summary', '', '| condition | runs | survivalRate | occupiedMean | simpsonMean | absDriftMean | noveltyProxyMean |', '|---|---:|---:|---:|---:|---:|---:|'];
  for (const [key, g] of grouped) {
    const runs = g.rows.length;
    const survivalRate = mean(g.rows.map((r) => (r.orgs > 0 ? 1 : 0)));
    const occupiedMean = mean(g.rows.map((r) => r.occupied));
    const simpsonMean = mean(g.rows.map((r) => r.simpson));
    const absDriftMean = mean(g.rows.map((r) => Math.abs(r.drift)));
    const noveltyProxyMean = mean(g.rows.map((r) => r.noveltyProxy));
    lines.push(
      `| ${key} | ${runs} | ${survivalRate.toFixed(3)} | ${occupiedMean.toFixed(3)} | ${simpsonMean.toFixed(3)} | ${absDriftMean.toExponential(2)} | ${noveltyProxyMean.toFixed(3)} |`,
    );
  }

  const out = resolve(process.cwd(), args.output);
  writeFileSync(out, `${lines.join('\n')}\n`);
  console.log(`[Report] saved:${out}`);
}

main();
