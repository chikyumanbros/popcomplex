import { GRID_WIDTH, GRID_HEIGHT, TOTAL_CELLS, CellType } from './constants';
import type { OrganismManager } from './organism';

export const U32_PER_CELL = 8;

// Shared conversion buffers (avoid per-call allocation)
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

export class World {
  cellData: Uint32Array;
  nextOrganismId = 1;

  constructor() {
    this.cellData = new Uint32Array(TOTAL_CELLS * U32_PER_CELL);
  }

  // --- positional helpers ---
  private base(x: number, y: number): number {
    return (y * GRID_WIDTH + x) * U32_PER_CELL;
  }

  setCell(x: number, y: number, orgId: number, cellType: CellType, energy: number, lineagePacked = 0) {
    const b = this.base(x, y);
    this.cellData[b] = orgId;
    this.cellData[b + 1] = cellType & 0xFF;
    _f32[0] = energy;
    this.cellData[b + 2] = _u32[0];
    this.cellData[b + 3] = 0;
    this.cellData[b + 4] = 0;
    this.cellData[b + 5] = 0;
    this.cellData[b + 6] = 0;
    this.cellData[b + 7] = lineagePacked & 0xffffff;
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

  /** Push each organism’s **public** kin tag (“face”) into GPU field [7] for all its cells (tint + neighbor trust). */
  syncLineageToCells(organisms: OrganismManager) {
    for (const org of organisms.organisms.values()) {
      const p = org.tape.getPublicKinTagPacked() & 0xffffff;
      for (const idx of org.cells) {
        this.cellData[idx * U32_PER_CELL + 7] = p;
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
