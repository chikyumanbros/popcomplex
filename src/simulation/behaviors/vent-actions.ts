const EIGHT_DIRS: [number, number][] = [
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
