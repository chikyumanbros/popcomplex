/** Compute pipeline for `env-diffusion.wgsl` — not dispatched from `main.ts`; CPU does env diffusion. */
import type { SimBuffers } from '../buffers';
import shaderCode from '../shaders/env-diffusion.wgsl?raw';
import { GRID_WIDTH, GRID_HEIGHT, WORKGROUP_SIZE } from '../../simulation/constants';

export interface EnvDiffusionPipeline {
  dispatch(encoder: GPUCommandEncoder, pingpong: number): void;
}

export function createEnvDiffusionPipeline(
  device: GPUDevice,
  buffers: SimBuffers,
): EnvDiffusionPipeline {
  const module = device.createShaderModule({ label: 'env-diffusion', code: shaderCode });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'env-diffusion-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: 'env-diffusion-pl',
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: 'env-diffusion',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' },
  });

  const bindGroups = [
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.envEnergy[0] } },
        { binding: 2, resource: { buffer: buffers.envEnergy[1] } },
      ],
    }),
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.envEnergy[1] } },
        { binding: 2, resource: { buffer: buffers.envEnergy[0] } },
      ],
    }),
  ];

  const wgX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const wgY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);

  return {
    dispatch(encoder, pingpong) {
      const pass = encoder.beginComputePass({ label: 'env-diffusion' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[pingpong % 2]);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    },
  };
}
