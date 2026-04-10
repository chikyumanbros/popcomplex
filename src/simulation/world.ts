import { GRID_WIDTH, GRID_HEIGHT, TOTAL_CELLS, CellType } from './constants';
import type { OrganismManager } from './organism';

export const U32_PER_CELL = 8;

// Shared conversion buffers (avoid per-call allocation)
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

export class World {
  cellData: Uint32Array;
  /** Per-cell local rule routing (3×u8 packed into u32). Used for same-org proxy execution / redundancy. */
  ruleRoutes: Uint32Array;
  /** Per-cell decay gauge (0..1): deterministic rot progression when cell energy is dead. */
  rot: Float32Array;
  /**
   * Per-cell metabolic toxin load (0..1, dimensionless modifier).
   * Accumulates as a byproduct of digestion and foreign-contact stress.
   * Suppresses digest efficiency; high load accelerates rot on energy-depleted cells.
   * Not itself energy — does not affect the closed energy budget.
   */
  toxin: Float32Array;
  nextOrganismId = 1;

  constructor() {
    this.cellData = new Uint32Array(TOTAL_CELLS * U32_PER_CELL);
    this.ruleRoutes = new Uint32Array(TOTAL_CELLS);
    this.rot = new Float32Array(TOTAL_CELLS);
    this.toxin = new Float32Array(TOTAL_CELLS);
  }

  // --- positional helpers ---
  private base(x: number, y: number): number {
    return (y * GRID_WIDTH + x) * U32_PER_CELL;
  }

  setCell(x: number, y: number, orgId: number, cellType: CellType, energy: number, lineagePacked = 0) {
    const b = this.base(x, y);
    const idx = y * GRID_WIDTH + x;
    this.cellData[b] = orgId;
    this.cellData[b + 1] = cellType & 0xFF;
    _f32[0] = energy;
    this.cellData[b + 2] = _u32[0];
    this.cellData[b + 3] = 0;
    this.cellData[b + 4] = 0;
    this.cellData[b + 5] = 0;
    this.cellData[b + 6] = 0;
    this.cellData[b + 7] = lineagePacked & 0xffffff;
    this.ruleRoutes[idx] = 0;
    this.rot[idx] = 0;
    this.toxin[idx] = 0;
  }

  getCellType(x: number, y: number): CellType {
    return (this.cellData[this.base(x, y) + 1] & 0xFF) as CellType;
  }

  getOrganismId(x: number, y: number): number {
    return this.cellData[this.base(x, y)];
  }

  getCellEnergy(x: number, y: number): number {
    _u32[0] = this.cellData[this.base(x, y) + 2];
    return _f32[0];
  }

  setCellEnergy(x: number, y: number, energy: number) {
    _f32[0] = Math.max(0, energy);
    this.cellData[this.base(x, y) + 2] = _u32[0];
  }

  // --- index-based accessors (skip x/y division) ---
  getOrganismIdByIdx(idx: number): number {
    return this.cellData[idx * U32_PER_CELL];
  }

  getCellTypeByIdx(idx: number): CellType {
    return (this.cellData[idx * U32_PER_CELL + 1] & 0xFF) as CellType;
  }

  getCellEnergyByIdx(idx: number): number {
    _u32[0] = this.cellData[idx * U32_PER_CELL + 2];
    return _f32[0];
  }

  setCellEnergyByIdx(idx: number, energy: number) {
    _f32[0] = Math.max(0, energy);
    this.cellData[idx * U32_PER_CELL + 2] = _u32[0];
  }

  // Stomach (digestion buffer) stored in field [4]
  getStomachByIdx(idx: number): number {
    _u32[0] = this.cellData[idx * U32_PER_CELL + 4];
    return _f32[0];
  }

  setStomachByIdx(idx: number, value: number) {
    _f32[0] = Math.max(0, value);
    this.cellData[idx * U32_PER_CELL + 4] = _u32[0];
  }

  // Specialization markers stored in field [3] — packed 4×u8
  // byte0=eat, byte1=digest, byte2=signal, byte3=move
  getMarkersByIdx(idx: number): [number, number, number, number] {
    const v = this.cellData[idx * U32_PER_CELL + 3];
    return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF];
  }

  setMarkersByIdx(idx: number, eat: number, digest: number, signal: number, move: number) {
    this.cellData[idx * U32_PER_CELL + 3] =
      (eat & 0xFF) | ((digest & 0xFF) << 8) | ((signal & 0xFF) << 16) | ((move & 0xFF) << 24);
  }

  bumpMarker(idx: number, slot: 0 | 1 | 2 | 3, amount: number) {
    const off = idx * U32_PER_CELL + 3;
    const shift = slot * 8;
    const cur = (this.cellData[off] >> shift) & 0xFF;
    const next = Math.min(255, cur + amount);
    this.cellData[off] = (this.cellData[off] & ~(0xFF << shift)) | (next << shift);
  }

  decayMarkers(idx: number) {
    const off = idx * U32_PER_CELL + 3;
    let v = this.cellData[off];
    for (let s = 0; s < 32; s += 8) {
      const cur = (v >> s) & 0xFF;
      if (cur > 0) {
        v = (v & ~(0xFF << s)) | (Math.max(0, cur - 1) << s);
      }
    }
    this.cellData[off] = v;
  }

  getMarkerByIdx(idx: number, slot: 0 | 1 | 2 | 3): number {
    return (this.cellData[idx * U32_PER_CELL + 3] >> (slot * 8)) & 0xFF;
  }

  /**
   * Differentiation commit level (0..255) for this cell, packed in bits 24-31 of cellData[b+1].
   * 0 = uncommitted (Stem), 255 = fully committed to current type.
   * This byte is preserved by all neural-state writes (actionFire, propagateSignals).
   */
  getCommitByIdx(idx: number): number {
    return (this.cellData[idx * U32_PER_CELL + 1] >>> 24) & 0xFF;
  }

  setCommitByIdx(idx: number, value: number) {
    const b = idx * U32_PER_CELL + 1;
    this.cellData[b] = (this.cellData[b] & 0x00FFFFFF) | ((value & 0xFF) << 24);
  }

  /** Set cell type (bits 0-7 of cellData[b+1]) while preserving neuralState, refractory, and commit bits. */
  setCellTypeByIdx(idx: number, type: number) {
    const b = idx * U32_PER_CELL + 1;
    this.cellData[b] = (this.cellData[b] & 0xFFFFFF00) | (type & 0xFF);
  }

  /** Local routing table for proxy rule execution: returns three donor rule indices (0..MAX_RULES-1). */
  getRuleRoutesByIdx(idx: number): [number, number, number] {
    const p = this.ruleRoutes[idx] >>> 0;
    return [p & 0xff, (p >>> 8) & 0xff, (p >>> 16) & 0xff];
  }

  setRuleRoutesByIdx(idx: number, r0: number, r1: number, r2: number) {
    this.ruleRoutes[idx] = ((r0 & 0xff) | ((r1 & 0xff) << 8) | ((r2 & 0xff) << 16)) >>> 0;
  }

  // Morphogen A stored in field [5], Morphogen B in field [6]
  getMorphogenA(idx: number): number {
    _u32[0] = this.cellData[idx * U32_PER_CELL + 5];
    return _f32[0];
  }

  setMorphogenA(idx: number, value: number) {
    _f32[0] = Math.max(0, value);
    this.cellData[idx * U32_PER_CELL + 5] = _u32[0];
  }

  getMorphogenB(idx: number): number {
    _u32[0] = this.cellData[idx * U32_PER_CELL + 6];
    return _f32[0];
  }

  setMorphogenB(idx: number, value: number) {
    _f32[0] = Math.max(0, value);
    this.cellData[idx * U32_PER_CELL + 6] = _u32[0];
  }

  // --- spawn / query ---
  spawnProto(cx: number, cy: number, lineagePacked = 0, initialEnergy = 60): number {
    const id = this.nextOrganismId++;
    this.setCell(cx, cy, id, CellType.Stem, initialEnergy, lineagePacked);
    return id;
  }

  /**
   * Push each organism's public kin tag ("face") into GPU field [7] for all its cells.
   * Bits  0-23: lineage color tag (tint + neighbor trust).
   * Bits 24-31: developmental stage (0-3) for optional GPU-side visualization.
   */
  syncLineageToCells(organisms: OrganismManager) {
    for (const org of organisms.organisms.values()) {
      const p = org.tape.getPublicKinTagPacked() & 0xffffff;
      const packed = ((org.stage & 0xFF) << 24) | p;
      for (const idx of org.cells) {
        this.cellData[idx * U32_PER_CELL + 7] = packed;
      }
    }
  }

  isInBounds(x: number, y: number): boolean {
    return x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT;
  }

  isEmpty(x: number, y: number): boolean {
    return this.isInBounds(x, y) && this.getCellType(x, y) === CellType.Empty;
  }

  uploadTo(device: GPUDevice, buffer: GPUBuffer) {
    device.queue.writeBuffer(buffer, 0, this.cellData.buffer);
  }
}
