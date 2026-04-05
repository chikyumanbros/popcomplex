/**
 * Compute pipeline for `ca-nervous.wgsl`. **Not wired from `main.ts`** — the running sim uses
 * `RuleEvaluator.propagateSignals` on CPU. Safe to ignore for gameplay; useful only if re-enabling
 * GPU-side nervous stepping.
 */
import type { SimBuffers } from '../buffers';
import shaderCode from '../shaders/ca-nervous.wgsl?raw';
import { GRID_WIDTH, GRID_HEIGHT, WORKGROUP_SIZE } from '../../simulation/constants';

export interface CANervousPipeline {
  dispatch(encoder: GPUCommandEncoder, pingpong: number): void;
}

export function createCANervousPipeline(
  device: GPUDevice,
  buffers: SimBuffers,
): CANervousPipeline {
  const module = device.createShaderModule({ label: 'ca-nervous', code: shaderCode });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'ca-nervous-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: 'ca-nervous-pl',
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: 'ca-nervous',
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
      ],
    }),
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.cellState[1] } },
        { binding: 2, resource: { buffer: buffers.cellState[0] } },
      ],
    }),
  ];

  const wgX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const wgY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);

  return {
    dispatch(encoder, pingpong) {
      const pass = encoder.beginComputePass({ label: 'ca-nervous' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[pingpong % 2]);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    },
  };
}
