import { GRID_WIDTH, GRID_HEIGHT } from '../constants';
import { U32_PER_CELL, type World } from '../world';
import type { Organism } from '../organism';
import type { Tape } from '../tape';

const EIGHT_DIRS: [number, number][] = [
  [0, -1], [1, 0], [0, 1], [-1, 0],
  [1, -1], [1, 1], [-1, 1], [-1, -1],
];

/**
 * Source of truth for same-org neural propagation after FIRE/SIG.
 * Reads/writes packed state in `world.cellData` (+1 word: cellType | neuralState<<8 | refractory<<16).
 * Not mirrored by any active GPU compute shader in the current build.
 */
export function runPropagateSignals(world: World, org: Organism, tape: Tape): void {
  const refPeriod = tape.getRefractoryPeriod();
  for (const idx of org.cells) {
    const b = idx * U32_PER_CELL;
    const packed = world.cellData[b + 1]!;
    const ns = (packed >> 8) & 0xFF;
    const refCnt = (packed >> 16) & 0xFF;
    const ct = packed & 0xFF;
    // Preserve bits 24-31 (commit) through all neural state writes.
    const commitBits = packed & 0xFF000000;
    if (ns === 1) {
      world.cellData[b + 1] = commitBits | ct | (2 << 8) | (refPeriod << 16);
      const x = idx % GRID_WIDTH;
      const y = (idx - x) / GRID_WIDTH;
      for (const [dx, dy] of EIGHT_DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
        const ni = ny * GRID_WIDTH + nx;
        if (!org.cells.has(ni)) continue;
        const np = world.cellData[ni * U32_PER_CELL + 1]!;
        if (((np >> 8) & 0xFF) === 0) {
          // Preserve neighbour's commit bits too.
          world.cellData[ni * U32_PER_CELL + 1] = (np & 0xFF000000) | (np & 0xFF) | (1 << 8);
        }
      }
    } else if (ns === 2) {
      world.cellData[b + 1] = commitBits | ct | (refCnt <= 1 ? ct : ct | (2 << 8) | ((refCnt - 1) << 16));
    }
  }
}
