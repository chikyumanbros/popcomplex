const EIGHT_DIRS: readonly [number, number][] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
  [1, -1],
  [1, 1],
  [-1, 1],
  [-1, -1],
];

export interface VentCell {
  x: number;
  y: number;
  idx: number;
}

/**
 * APOPTOSE action: voluntarily accelerate the rot gauge of this cell (or the weakest same-org
 * Moore neighbor), and recycle the target's cell energy to surviving same-org neighbors.
 *
 * - `targetSelf`: true → always operate on the acting cell; false → prefer the weakest (lowest
 *   energy) same-org Moore neighbor, falling back to self if none exist.
 * - `rotBoost`: added to `rot[targetIdx]`, clamped at 1.0. Dissolution itself still happens
 *   only when the existing `cleanupDeadOrganismsPhase` sees rot ≥ 1 — no new death event.
 * - `energyDumpFrac`: fraction (0..1) of target cell energy transferred to living same-org
 *   Moore neighbors of the *target* (equal share). Falls to env if no living neighbors exist.
 *
 * Conservation: energy is conserved because every unit taken from the target goes either to
 * same-org neighbors or to env — no energy is created or destroyed.
 */
export function apoptoseCell(
  cell: VentCell,
  orgId: number,
  targetSelf: boolean,
  rotBoost: number,
  energyDumpFrac: number,
  getOrgIdByIdx: (idx: number) => number,
  getCellEnergyByIdx: (idx: number) => number,
  setCellEnergyByIdx: (idx: number, value: number) => void,
  addEnvEnergyAtIdx: (idx: number, delta: number) => void,
  rotArray: Float32Array,
  gridWidth: number,
  gridHeight: number,
): boolean {
  // --- 1. Target selection ---
  let targetIdx = cell.idx;
  if (!targetSelf) {
    let minE = Infinity;
    for (const [dx, dy] of EIGHT_DIRS) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;
      if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
      const ni = ny * gridWidth + nx;
      if (getOrgIdByIdx(ni) !== orgId) continue;
      const e = getCellEnergyByIdx(ni);
      if (e < minE) { minE = e; targetIdx = ni; }
    }
    // If no same-org neighbor found, fall back to self
  }

  let changed = false;

  // --- 2. Boost rot (溶解は cleanupPhase が担当) ---
  const prevRot = rotArray[targetIdx] ?? 0;
  if (prevRot < 1 - 1e-8) {
    rotArray[targetIdx] = Math.min(1, prevRot + rotBoost);
    changed = true;
  }

  // --- 3. Energy recycling: target's energy → living same-org Moore neighbors of target ---
  const targetEnergy = getCellEnergyByIdx(targetIdx);
  const dump = targetEnergy * energyDumpFrac;
  if (dump > 1e-8) {
    const tx = targetIdx % gridWidth;
    const ty = (targetIdx - tx) / gridWidth;
    const sinks: number[] = [];
    for (const [dx, dy] of EIGHT_DIRS) {
      const nx = tx + dx;
      const ny = ty + dy;
      if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
      const ni = ny * gridWidth + nx;
      if (ni === cell.idx && !targetSelf) continue; // skip acting cell when neighbor-targeting
      if (getOrgIdByIdx(ni) !== orgId) continue;
      if (getCellEnergyByIdx(ni) <= 0) continue; // only living cells receive
      sinks.push(ni);
    }
    setCellEnergyByIdx(targetIdx, Math.max(0, targetEnergy - dump));
    if (sinks.length > 0) {
      const per = dump / sinks.length;
      for (const si of sinks) {
        setCellEnergyByIdx(si, getCellEnergyByIdx(si) + per);
      }
    } else {
      // No living same-org neighbors: return energy to env (closed budget)
      addEnvEnergyAtIdx(targetIdx, dump);
    }
    changed = true;
  }

  return changed;
}

export function spillStomachToNearbyEnv(
  cell: VentCell,
  amount: number,
  getStomachByIdx: (idx: number) => number,
  setStomachByIdx: (idx: number, value: number) => void,
  addEnvEnergyAtIdx: (idx: number, delta: number) => void,
  gridWidth: number,
  gridHeight: number,
): boolean {
  const stomach = getStomachByIdx(cell.idx);
  const spill = Math.min(stomach, Math.max(0, amount));
  if (spill < 0.01) return false;

  setStomachByIdx(cell.idx, stomach - spill);
  const selfShare = spill * 0.5;
  const neighborShare = spill - selfShare;
  addEnvEnergyAtIdx(cell.idx, selfShare);

  const neighbors: number[] = [];
  for (const [dx, dy] of EIGHT_DIRS) {
    const nx = cell.x + dx;
    const ny = cell.y + dy;
    if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
    neighbors.push(ny * gridWidth + nx);
  }
  if (neighbors.length === 0) {
    addEnvEnergyAtIdx(cell.idx, neighborShare);
    return true;
  }

  const per = neighborShare / neighbors.length;
  for (const nIdx of neighbors) addEnvEnergyAtIdx(nIdx, per);
  return true;
}
