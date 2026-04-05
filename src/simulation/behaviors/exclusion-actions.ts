import { CellType } from '../constants';

const ORTH_DIRS: [number, number][] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
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

export interface ExclusionCell {
  x: number;
  y: number;
  idx: number;
  orgId: number;
}

export function computeJamTtl(intensity: number, jamMinTicks: number, jamMaxExtraTicks: number): number {
  const extra = Math.round(Math.max(0, Math.min(1, intensity)) * jamMaxExtraTicks);
  return jamMinTicks + extra;
}

export function applyJamToForeignBoundary(
  cell: ExclusionCell,
  ttl: number,
  getOrganismIdByIdx: (idx: number) => number,
  markJammed: (idx: number, jamTtl: number) => void,
  gridWidth: number,
  gridHeight: number,
  useEightNeighbors = false,
): boolean {
  const dirs = useEightNeighbors ? EIGHT_DIRS : ORTH_DIRS;
  let applied = false;
  for (const [dx, dy] of dirs) {
    const nx = cell.x + dx;
    const ny = cell.y + dy;
    if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
    const nIdx = ny * gridWidth + nx;
    const nOrg = getOrganismIdByIdx(nIdx);
    if (nOrg === 0 || nOrg === cell.orgId) continue;
    markJammed(cell.idx, ttl);
    markJammed(nIdx, ttl);
    applied = true;
  }
  return applied;
}

export function countRepulsionFacesForShift(
  orgCells: Iterable<number>,
  orgId: number,
  dx: number,
  dy: number,
  getCellTypeByIdx: (idx: number) => CellType,
  getOrganismIdByIdx: (idx: number) => number,
  gridWidth: number,
  gridHeight: number,
): { foreignFaces: number; edgeFaces: number } {
  const shifted = new Set<number>();
  for (const idx of orgCells) {
    const x = (idx % gridWidth) + dx;
    const y = ((idx - (idx % gridWidth)) / gridWidth) + dy;
    shifted.add(y * gridWidth + x);
  }

  let foreignFaces = 0;
  let edgeFaces = 0;
  for (const idx of shifted) {
    const x = idx % gridWidth;
    const y = (idx - x) / gridWidth;
    for (const [ox, oy] of ORTH_DIRS) {
      const nx = x + ox;
      const ny = y + oy;
      if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) {
        edgeFaces++;
        continue;
      }
      const ni = ny * gridWidth + nx;
      if (shifted.has(ni)) continue;
      if (getCellTypeByIdx(ni) === CellType.Empty) continue;
      const nOrg = getOrganismIdByIdx(ni);
      if (nOrg !== 0 && nOrg !== orgId) foreignFaces++;
    }
  }
  return { foreignFaces, edgeFaces };
}
