/**
 * Compute pipeline for `env-diffusion.wgsl`.
 * Dispatched from `main.ts` when `ruleEval.useGpuDiffusion` is true;
 * CPU does env diffusion otherwise (headless/test paths).
 */
import type { SimBuffers } from '../buffers';
import shaderCode from '../shaders/env-diffusion.wgsl?raw';
import { GRID_WIDTH, GRID_HEIGHT, WORKGROUP_SIZE } from '../../simulation/constants';

export interface EnvDiffusionPipeline {
  /**
   * Record a single diffusion pass into the command encoder.
   * @param pingpong  0 → reads envEnergy[0], writes envEnergy[1]; 1 → reversed.
   */
  dispatch(encoder: GPUCommandEncoder, pingpong: number): void;
  /** Update the diffusion rate uniform (call when ENV_DIFFUSION_RATE changes). */
  setRate(device: GPUDevice, rate: number): void;
}

export function createEnvDiffusionPipeline(
  device: GPUDevice,
  buffers: SimBuffers,
  diffusionRate: number,
): EnvDiffusionPipeline {
  const module = device.createShaderModule({ label: 'env-diffusion', code: shaderCode });

  // Separate uniform buffer for the diffusion rate (4-byte float, 16-byte minimum).
  const paramsBuffer = device.createBuffer({
    label: 'env-diffusion-params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const writeRate = (rate: number) => {
    const data = new Float32Array(4);
    data[0] = rate;
    device.queue.writeBuffer(paramsBuffer, 0, data);
  };
  writeRate(diffusionRate);

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'env-diffusion-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
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
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.envEnergy[1] } },
        { binding: 2, resource: { buffer: buffers.envEnergy[0] } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    }),
  ];

  const wgX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const wgY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);

  return {
    dispatch(encoder, pingpong) {
      const pass = encoder.beginComputePass({ label: 'env-diffusion' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[pingpong % 2]!);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    },
    setRate(dev, rate) {
      writeRate(rate);
      void dev; // device captured by closure
    },
  };
}
