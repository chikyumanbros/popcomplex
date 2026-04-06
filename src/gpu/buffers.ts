import { GRID_WIDTH, GRID_HEIGHT, TOTAL_CELLS, INITIAL_TOTAL_ENERGY } from '../simulation/constants';

export interface SimBuffers {
  cellState: [GPUBuffer, GPUBuffer];
  envEnergy: [GPUBuffer, GPUBuffer];
  componentMask: GPUBuffer;
  uniform: GPUBuffer;
  staging: GPUBuffer;
  initialEnv: Float32Array;
}

export function createBuffers(device: GPUDevice): SimBuffers {
  const cellStateSize = TOTAL_CELLS * 32; // 32 bytes per cell
  const envEnergySize = TOTAL_CELLS * 4;  // f32 per cell
  const componentMaskSize = TOTAL_CELLS * 4; // u32 per cell (0/1)

  const cellState: [GPUBuffer, GPUBuffer] = [
    device.createBuffer({
      label: 'cellState0',
      size: cellStateSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    device.createBuffer({
      label: 'cellState1',
      size: cellStateSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
  ];

  const envEnergy: [GPUBuffer, GPUBuffer] = [
    device.createBuffer({
      label: 'envEnergy0',
      size: envEnergySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    device.createBuffer({
      label: 'envEnergy1',
      size: envEnergySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
  ];

  const componentMask = device.createBuffer({
    label: 'componentMask',
    size: componentMaskSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // Ensure deterministic start (uninitialized storage buffers are undefined).
  device.queue.writeBuffer(componentMask, 0, new Uint32Array(TOTAL_CELLS).buffer);

  const uniform = device.createBuffer({
    label: 'uniform',
    size: 32, // 8 fields * 4 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const staging = device.createBuffer({
    label: 'staging',
    size: Math.max(cellStateSize, envEnergySize),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const initialEnv = createInitialEnvEnergy();
  device.queue.writeBuffer(envEnergy[0], 0, initialEnv.buffer);

  return { cellState, envEnergy, componentMask, uniform, staging, initialEnv };
}

export function createInitialEnvEnergy(): Float32Array {
  const data = new Float32Array(TOTAL_CELLS);
  const perCell = INITIAL_TOTAL_ENERGY / TOTAL_CELLS;
  for (let i = 0; i < TOTAL_CELLS; i++) {
    data[i] = perCell;
  }
  return data;
}

export function writeUniform(
  device: GPUDevice,
  buffer: GPUBuffer,
  tick: number,
  pingpong: number,
  viewX = 0,
  viewY = 0,
  viewZoom = 1,
  viewMode = 0,
) {
  const data = new ArrayBuffer(32);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);
  u32[0] = GRID_WIDTH;
  u32[1] = GRID_HEIGHT;
  u32[2] = tick;
  u32[3] = pingpong;
  f32[4] = viewX;
  f32[5] = viewY;
  f32[6] = viewZoom;
  u32[7] = viewMode >>> 0;
  device.queue.writeBuffer(buffer, 0, data);
}
