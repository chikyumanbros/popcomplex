export const GRID_WIDTH = 256;
export const GRID_HEIGHT = 256;
export const TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT;

/** Uniform initial env fill: per-cell average before first tick (lower = scarcer closed universe). */
export const INITIAL_ENV_ENERGY_PER_CELL = 1;
export const INITIAL_TOTAL_ENERGY = TOTAL_CELLS * INITIAL_ENV_ENERGY_PER_CELL;

/** Conservative env mixing (closed system: sum(env)+Σcells+Σstomach held fixed except inject/spawn bookkeeping). */
export const ENV_DIFFUSION_RATE = 0.055;

export const CELL_BYTES = 32;

export enum CellType {
  Empty = 0,
  Stem = 1,
  Sensor = 2,
  Motor = 3,
  Reproductive = 4,
}

export enum NeuralState {
  Resting = 0,
  Excited = 1,
  Refractory = 2,
}

export const WORKGROUP_SIZE = 8;
