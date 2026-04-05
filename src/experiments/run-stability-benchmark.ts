import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

interface RunConfig {
  runId?: string;
  seed: number;
  ticks: number;
}

interface RunSummary {
  seed: number;
  ticks: number;
  runDirName: string;
  birthRepro: number;
  birthSplit: number;
  splitPerRepro: number;
  finalOrgs: number;
  finalOcc: number;
  maxOrgs: number;
  maxOcc: number;
}

function parseListArg(raw: string | undefined, fallback: number[]): number[] {
  if (!raw) return fallback;
  const out = raw
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n))
    .map((n) => Math.trunc(n));
  return out.length > 0 ? out : fallback;
}

function parseArgs() {
  const map = new Map<string, string>();
  for (const token of process.argv.slice(2)) {
    const [k, v] = token.split('=');
    if (k && v !== undefined) map.set(k.replace(/^--/, ''), v);
  }
  return {
    seeds: parseListArg(map.get('seeds'), [3006, 3007, 3010, 3011, 3012]),
    ticksList: parseListArg(map.get('ticksList'), [2200, 6000]),
    outDir: map.get('outDir') ?? 'runs/stability',
    logEvery: Number(map.get('logEvery') ?? 250),
    snapshotEvery: Number(map.get('snapshotEvery') ?? 3000),
    spawnEnergy: Number(map.get('spawnEnergy') ?? 150),
    metabolicScale: Number(map.get('metabolicScale') ?? 0.6),
    distressScale: Number(map.get('distressScale') ?? 0.3),
  };
}

function runOne(seed: number, ticks: number, args: ReturnType<typeof parseArgs>) {
  const cmdArgs = [
    'run',
    'experiment:run',
    '--',
    `--seed=${seed}`,
    `--ticks=${ticks}`,
    `--logEvery=${args.logEvery}`,
    `--snapshotEvery=${args.snapshotEvery}`,
    `--spawnEnergy=${args.spawnEnergy}`,
    `--metabolicScale=${args.metabolicScale}`,
    `--distressScale=${args.distressScale}`,
    `--outDir=${args.outDir}`,
  ];
  const res = spawnSync('npm', cmdArgs, { stdio: 'inherit' });
  if (res.status !== 0) {
    throw new Error(`experiment run failed: seed=${seed} ticks=${ticks} exit=${res.status}`);
  }
}

function loadJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, 'utf8')) as T;
}

function latestMatchingRunDir(root: string, seed: number, ticks: number): string {
  const dirs = readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && d.name.startsWith('run-'))
    .map((d) => d.name)
    .sort()
    .reverse();
  for (const name of dirs) {
    const cfgPath = resolve(root, name, 'config.json');
    try {
      const cfg = loadJson<RunConfig>(cfgPath);
      if (cfg.seed === seed && cfg.ticks === ticks) return resolve(root, name);
    } catch {
      // skip unreadable dirs
    }
  }
  throw new Error(`no run found for seed=${seed} ticks=${ticks} under ${root}`);
}

function summarizeMetrics(runDir: string, seed: number, ticks: number): RunSummary {
  const metricsPath = resolve(runDir, 'metrics.csv');
  const lines = readFileSync(metricsPath, 'utf8').trim().split('\n');
  if (lines.length < 2) throw new Error(`metrics empty: ${metricsPath}`);
  const header = lines[0].split(',');
  const idx = new Map<string, number>();
  header.forEach((h, i) => idx.set(h, i));

  const required = ['birthRepro', 'birthSplit', 'orgs', 'occupied'];
  for (const k of required) {
    if (!idx.has(k)) throw new Error(`missing column "${k}" in ${metricsPath}`);
  }

  let birthRepro = 0;
  let birthSplit = 0;
  let maxOrgs = 0;
  let maxOcc = 0;
  let finalOrgs = 0;
  let finalOcc = 0;
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    const orgs = Number(cols[idx.get('orgs')!]);
    const occ = Number(cols[idx.get('occupied')!]);
    birthRepro += Number(cols[idx.get('birthRepro')!]);
    birthSplit += Number(cols[idx.get('birthSplit')!]);
    if (orgs > maxOrgs) maxOrgs = orgs;
    if (occ > maxOcc) maxOcc = occ;
    if (i === lines.length - 1) {
      finalOrgs = orgs;
      finalOcc = occ;
    }
  }
  return {
    seed,
    ticks,
    runDirName: runDir.split('/').pop() ?? runDir,
    birthRepro,
    birthSplit,
    splitPerRepro: birthRepro > 0 ? birthSplit / birthRepro : Number.POSITIVE_INFINITY,
    finalOrgs,
    finalOcc,
    maxOrgs,
    maxOcc,
  };
}

function mean(nums: number[]): number {
  if (nums.length === 0) return 0;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function formatRatio(n: number): string {
  if (!Number.isFinite(n)) return 'inf';
  return n.toFixed(3);
}

function main() {
  const args = parseArgs();
  const root = resolve(process.cwd(), args.outDir);
  console.log(
    `[Stability benchmark] seeds=${args.seeds.join(',')} ticks=${args.ticksList.join(',')} outDir=${args.outDir}`,
  );

  for (const seed of args.seeds) {
    for (const ticks of args.ticksList) {
      runOne(seed, ticks, args);
    }
  }

  const summaries: RunSummary[] = [];
  for (const seed of args.seeds) {
    for (const ticks of args.ticksList) {
      const runDir = latestMatchingRunDir(root, seed, ticks);
      summaries.push(summarizeMetrics(runDir, seed, ticks));
    }
  }

  console.log(
    'seed,ticks,run,birthRepro,birthSplit,splitPerRepro,finalOrgs,finalOcc,maxOrgs,maxOcc',
  );
  for (const r of summaries.sort((a, b) => a.ticks - b.ticks || a.seed - b.seed)) {
    console.log(
      [
        r.seed,
        r.ticks,
        r.runDirName,
        r.birthRepro,
        r.birthSplit,
        formatRatio(r.splitPerRepro),
        r.finalOrgs,
        r.finalOcc,
        r.maxOrgs,
        r.maxOcc,
      ].join(','),
    );
  }

  for (const ticks of args.ticksList) {
    const rows = summaries.filter((r) => r.ticks === ticks);
    const survival = rows.filter((r) => r.finalOcc > 0).length;
    const splitVals = rows.filter((r) => Number.isFinite(r.splitPerRepro)).map((r) => r.splitPerRepro);
    console.log(
      `[Stability aggregate] ticks=${ticks} survival=${survival}/${rows.length} meanFinalOcc=${mean(
        rows.map((r) => r.finalOcc),
      ).toFixed(1)} meanSplitPerRepro=${mean(splitVals).toFixed(3)}`,
    );
  }
}

main();
