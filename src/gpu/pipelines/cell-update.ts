/** Compute pipeline for `cell-update.wgsl` — not dispatched from `main.ts`; sim runs on CPU. */
import type { SimBuffers } from '../buffers';
import shaderCode from '../shaders/cell-update.wgsl?raw';
import { GRID_WIDTH, GRID_HEIGHT, WORKGROUP_SIZE } from '../../simulation/constants';

export interface CellUpdatePipeline {
  dispatch(encoder: GPUCommandEncoder, pingpong: number): void;
}

export function createCellUpdatePipeline(
  device: GPUDevice,
  buffers: SimBuffers,
): CellUpdatePipeline {
  const module = device.createShaderModule({ label: 'cell-update', code: shaderCode });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'cell-update-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: 'cell-update-pl',
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: 'cell-update',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' },
  });

  const bindGroups = [
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.cellState[0] } },
        { binding: 2, resource: { buffer: buffers.cellState[1] } },
        { binding: 3, resource: { buffer: buffers.envEnergy[0] } },
      ],
    }),
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.cellState[1] } },
        { binding: 2, resource: { buffer: buffers.cellState[0] } },
        { binding: 3, resource: { buffer: buffers.envEnergy[1] } },
      ],
    }),
  ];

  const wgX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const wgY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);

  return {
    dispatch(encoder, pingpong) {
      const pass = encoder.beginComputePass({ label: 'cell-update' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[pingpong % 2]);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    },
  };
}
