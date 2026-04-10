import { GRID_WIDTH, GRID_HEIGHT } from '../constants';
import { ENV_DIFFUSION_RATE } from '../constants';

const EIGHT_DIRS: [number, number][] = [
  [0, -1], [1, 0], [0, 1], [-1, 0],
  [1, -1], [1, 1], [-1, 1], [-1, -1],
];

/**
 * One step of environment-energy spatial diffusion (Moore neighbourhood average).
 * Writes the diffused result into `dst`; the caller is responsible for swapping
 * src/dst buffers so the next tick reads the updated values.
 *
 * @param src  Read-only source buffer (current env energy).
 * @param dst  Write-only destination buffer (receives diffused values).
 */
export function runEnvDiffusion(
  src: Float32Array,
  dst: Float32Array,
  diffusionRate: number = ENV_DIFFUSION_RATE,
): void {
  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      const i = y * GRID_WIDTH + x;
      const c = src[i]!;
      let sum = 0;
      let cnt = 0;
      for (const [dx, dy] of EIGHT_DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;
        sum += src[ny * GRID_WIDTH + nx]!;
        cnt++;
      }
      dst[i] = c + (sum / cnt - c) * diffusionRate;
    }
  }
}
