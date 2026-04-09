/**
 * Lenia-style continuous cellular automaton engine.
 * Parameters are derived from an organism's tape + NN state and used to generate
 * a living "portrait" that reflects the organism's genetic signature.
 *
 * Field is 64×64, toroidal (wrapping), computed on CPU.
 * Kernel is precomputed once per parameter set.
 */

import type { Tape } from '../simulation/tape';
import type { World } from '../simulation/world';
import { GRID_WIDTH } from '../simulation/constants';

export const LENIA_W = 64;
export const LENIA_H = 64;

/** Lenia kernel + growth function parameters derived from an organism's tape. */
export interface LeniaParams {
  R: number;         // kernel radius in cells (3..10)
  beta: [number, number, number]; // ring peak heights (0..1)
  delta: number;     // ring width (0.05..0.4)
  mu: number;        // growth function center (0.1..0.5)
  sigma: number;     // growth function width (0.02..0.15)
  dt: number;        // time step (0.05..0.3)
}

/** Mood index → hue shift applied on top of kin color.
 * 0=EAT (+0°, accent orange), 1=GROW (+100°, green),
 * 2=MOVE (+220°, cyan-blue), 3=CONSERVE (+180°, cool)
 */
const MOOD_HUE_SHIFT = [0, 100, 220, 180];

/**
 * Extract Lenia parameters from tape bytes, constrained to the known-stable regime.
 *
 * Lenia stability requires mu ≈ the expected convolution output for the seed pattern.
 * For a normalized kernel and a seed covering ~10-20% of a 64×64 field, u_typical ≈ 0.1..0.2.
 * mu must be close to u_typical, and sigma small enough to be selective but wide enough
 * to not require exact seeding. Classical Orbium: R=13, mu=0.15, sigma=0.015, dt=0.1.
 */
export function deriveLeniaParams(tapeData: Uint8Array): LeniaParams {
  // R from CA band refractory byte: 5..13
  const R = 5 + Math.round((tapeData[32]! & 0xff) / 255 * 8); // 5..13

  // Beta: dominant first ring (always large), optional secondary rings
  const b0 = 0.5  + (tapeData[48]! & 0xff) / 255 * 0.5;  // 0.5..1.0
  const b1 = (tapeData[50]! & 0xff) / 255 * 0.6;          // 0.0..0.6
  const b2 = (tapeData[52]! & 0xff) / 255 * 0.35;         // 0.0..0.35
  const beta: [number, number, number] = [b0, b1, b2];

  // Delta: ring width — keep in a range that allows pattern formation
  const delta = 0.08 + (tapeData[60]! & 0xff) / 255 * 0.14; // 0.08..0.22

  // mu and sigma must stay in the classical stable band.
  // mu = 0.10..0.22, sigma = 0.008..0.030
  const mu    = 0.10 + (tapeData[64]! & 0xff) / 255 * 0.12; // 0.10..0.22
  const sigma = 0.008 + (tapeData[65]! & 0xff) / 255 * 0.022; // 0.008..0.030

  // dt small for stability: 0.05..0.12
  const dt = 0.05 + (tapeData[66]! & 0xff) / 255 * 0.07; // 0.05..0.12

  return { R, beta, delta, mu, sigma, dt };
}

/** Build a normalized multi-ring kernel for the given params.
 * Returns a Float32Array of length (2R+1)^2, laid out row-major. */
function buildKernel(params: LeniaParams): Float32Array {
  const { R, beta, delta } = params;
  const size = 2 * R + 1;
  const k = new Float32Array(size * size);
  const N = beta.length;
  let total = 0;

  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const r = Math.sqrt(dx * dx + dy * dy) / R;
      if (r > 1.0) continue;
      let val = 0;
      for (let i = 0; i < N; i++) {
        const peakR = (i + 0.5) / N;
        const diff = r - peakR;
        val += beta[i]! * Math.exp(-(diff * diff) / (2 * delta * delta));
      }
      const ki = (dy + R) * size + (dx + R);
      k[ki] = val;
      total += val;
    }
  }
  if (total > 0) for (let i = 0; i < k.length; i++) k[i] /= total;
  return k;
}

/** Gaussian bell: exp(-x^2 / (2 sigma^2)) */
function bell(x: number, mu: number, sigma: number): number {
  const d = x - mu;
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

export class LeniaEngine {
  readonly W = LENIA_W;
  readonly H = LENIA_H;
  field: Float32Array;
  private next: Float32Array;
  private kernel: Float32Array;
  private kernelR: number;
  params: LeniaParams;

  constructor(params: LeniaParams) {
    this.params = params;
    this.field = new Float32Array(this.W * this.H);
    this.next = new Float32Array(this.W * this.H);
    this.kernelR = params.R;
    this.kernel = buildKernel(params);
  }

  /**
   * Seed initial Lenia field from an organism.
   *
   * Strategy: build a Gaussian ring blob centered in the field — large enough to
   * guarantee that the convolution output lies in the viable range for the derived
   * mu/sigma, so the pattern survives and evolves.  The ring radius and texture are
   * modulated by the organism's NN weight bytes and cell energy to make each portrait
   * unique, while still starting from a stable configuration.
   *
   * The actual cell positions from the world grid are blended in as a secondary layer
   * (up to 40% weight), giving the shape some resemblance to the real organism outline.
   */
  seedFromOrganism(org: { tape: Tape; cells: Set<number> }, world: World): void {
    const { W, H, field } = this;
    const { R } = this.params;
    field.fill(0);

    const cx = W / 2;
    const cy = H / 2;
    // Seed ring radius: 60-80% of kernel radius, so convolution output lands near mu.
    const seedR = R * 0.7 + 2;
    const nnBytes = org.tape.data;

    // Layer 1: Gaussian ring blob centred in field, textured by NN weight bytes.
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const r = Math.sqrt(dx * dx + dy * dy);
        // Ring profile: Gaussian centred at seedR, width = seedR*0.35
        const rNorm = r / seedR;
        const ringProfile = Math.exp(-((rNorm - 1) * (rNorm - 1)) / (2 * 0.12 * 0.12));
        if (ringProfile < 0.01) continue;

        // Texture from NN weight bytes mapped onto a 8×8 tile.
        const ti = ((y & 7) << 3 | (x & 7)) & 0x3f;
        const na = (nnBytes[128 + ti * 2]! & 0xff) / 255;
        const nb = (nnBytes[129 + ti * 2]! & 0xff) / 255;
        const texVal = (na + nb) * 0.5;

        // Blend ring profile with texture; keep base value high enough to survive.
        field[y * W + x] = Math.min(1, ringProfile * (0.65 + texVal * 0.35));
      }
    }

    // Layer 2: project actual org cell layout (scaled to fit) and blend at 35% weight.
    if (org.cells.size > 0) {
      let minX = GRID_WIDTH, maxX = 0, minY = 99999, maxY = 0;
      for (const idx of org.cells) {
        const gx = idx % GRID_WIDTH;
        const gy = (idx - gx) / GRID_WIDTH;
        if (gx < minX) minX = gx;
        if (gx > maxX) maxX = gx;
        if (gy < minY) minY = gy;
        if (gy > maxY) maxY = gy;
      }
      const bw = Math.max(1, maxX - minX + 1);
      const bh = Math.max(1, maxY - minY + 1);
      // Scale so the org bounding box fills at most 70% of the field.
      const fitScale = Math.min((W * 0.7) / bw, (H * 0.7) / bh);
      const oX = Math.floor((W - bw * fitScale) / 2);
      const oY = Math.floor((H - bh * fitScale) / 2);

      // Build a small temp cell map for the projected org.
      const cellLayer = new Float32Array(W * H);
      for (const idx of org.cells) {
        const gx = idx % GRID_WIDTH;
        const gy = (idx - gx) / GRID_WIDTH;
        const px = Math.floor(oX + (gx - minX) * fitScale);
        const py = Math.floor(oY + (gy - minY) * fitScale);
        const energy = world.getCellEnergyByIdx(idx);
        const eNorm = Math.min(1, Math.max(0.3, energy / 50));
        // Small splat around each projected cell.
        for (let dy2 = -1; dy2 <= 1; dy2++) {
          for (let dx2 = -1; dx2 <= 1; dx2++) {
            const fx = ((px + dx2) % W + W) % W;
            const fy = ((py + dy2) % H + H) % H;
            const v = (dx2 === 0 && dy2 === 0) ? eNorm : eNorm * 0.5;
            if (cellLayer[fy * W + fx]! < v) cellLayer[fy * W + fx] = v;
          }
        }
      }
      // Blend cell layer into field (35% cell, 65% ring keeps viability).
      for (let i = 0; i < W * H; i++) {
        field[i] = Math.min(1, field[i]! * 0.65 + cellLayer[i]! * 0.35);
      }
    }
  }

  /** Advance one Lenia step (toroidal boundary). */
  step(): void {
    const { W, H, field, next, kernel, kernelR, params } = this;
    const size = 2 * kernelR + 1;
    const { mu, sigma, dt } = params;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let u = 0;
        for (let dy = -kernelR; dy <= kernelR; dy++) {
          const ny = ((y + dy) % H + H) % H;
          const kRow = (dy + kernelR) * size;
          const fRow = ny * W;
          for (let dx = -kernelR; dx <= kernelR; dx++) {
            const nx = ((x + dx) % W + W) % W;
            u += kernel[kRow + dx + kernelR]! * field[fRow + nx]!;
          }
        }
        // G(u) = 2 * bell(u, mu, sigma) - 1  → growth in [-1, 1]
        const g = 2 * bell(u, mu, sigma) - 1;
        next[y * W + x] = Math.max(0, Math.min(1, field[y * W + x]! + dt * g));
      }
    }
    this.field.set(next);
  }

  /** Render field to an ImageData.
   * @param kinTag 24-bit lineage color tag
   * @param nnDominant mood index 0..3
   * @param nnPrimitives hidden layer activations (for saturation/brightness modulation)
   */
  renderToImageData(
    out: ImageData,
    kinTag: number,
    nnDominant: number,
    nnPrimitives: Float32Array,
  ): void {
    const { W, H, field } = this;
    const data = out.data;

    // Derive base hue from kinTag using a simple hash spread over 360°.
    // Multiply by a large prime to spread the 24-bit tag uniformly.
    const baseHue = (((kinTag * 2654435761) >>> 0) % 360);

    // Mood-based hue shift.
    const moodShift = MOOD_HUE_SHIFT[nnDominant] ?? 0;
    const hue = (baseHue + moodShift) % 360;

    // Saturation: driven by mean absolute nnPrimitive activation.
    let primMag = 0;
    for (let i = 0; i < nnPrimitives.length; i++) primMag += Math.abs(nnPrimitives[i]!);
    primMag = Math.min(1, primMag / Math.max(1, nnPrimitives.length));
    const saturation = 0.6 + primMag * 0.35; // 0.60..0.95

    // Background colour (for v≈0): very dark version of hue.
    const [bgR, bgG, bgB] = hslToRgb(hue, 0.4, 0.04);

    for (let i = 0; i < W * H; i++) {
      const v = field[i]!;
      if (v < 0.01) {
        // Fully opaque dark background — no transparency artefacts.
        data[i * 4 + 0] = bgR;
        data[i * 4 + 1] = bgG;
        data[i * 4 + 2] = bgB;
        data[i * 4 + 3] = 255;
      } else {
        // Lightness: low v → 8% (near-black tinted), high v → 80% (bright).
        const lightness = 0.08 + v * 0.72;
        const [ri, gi, bi] = hslToRgb(hue, saturation, lightness);
        data[i * 4 + 0] = ri;
        data[i * 4 + 1] = gi;
        data[i * 4 + 2] = bi;
        data[i * 4 + 3] = 255;
      }
    }
  }
}

/** HSL → [r, g, b] each 0..255. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h = h % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60)       { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else              { r = c; g = 0; b = x; }
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}
