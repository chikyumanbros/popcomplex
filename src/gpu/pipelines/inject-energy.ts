/**
 * GPU energy injection helper — pipeline is constructed in `main.ts` but `inject()` is not called;
 * env is updated on CPU (`RuleEvaluator.envEnergy`).
 */
import type { SimBuffers } from '../buffers';
import shaderCode from '../shaders/inject-energy.wgsl?raw';
import { GRID_WIDTH, GRID_HEIGHT, WORKGROUP_SIZE } from '../../simulation/constants';

export interface InjectEnergyPipeline {
  inject(device: GPUDevice, cx: number, cy: number, radius: number, amount: number, pingpong: number): void;
}

export function createInjectEnergyPipeline(
  device: GPUDevice,
  buffers: SimBuffers,
): InjectEnergyPipeline {
  const module = device.createShaderModule({ label: 'inject-energy', code: shaderCode });

  const paramBuffer = device.createBuffer({
    label: 'inject-params',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'inject-energy-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: 'inject-energy-pl',
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: 'inject-energy',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' },
  });

  const bindGroups = [
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramBuffer } },
        { binding: 1, resource: { buffer: buffers.envEnergy[0] } },
      ],
    }),
    device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramBuffer } },
        { binding: 1, resource: { buffer: buffers.envEnergy[1] } },
      ],
    }),
  ];

  const wgX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const wgY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);

  return {
    inject(device, cx, cy, radius, amount, pingpong) {
      const data = new ArrayBuffer(32);
      const f32 = new Float32Array(data, 0, 4);
      const u32 = new Uint32Array(data, 16, 4);
      f32[0] = cx;
      f32[1] = cy;
      f32[2] = radius;
      f32[3] = amount;
      u32[0] = GRID_WIDTH;
      u32[1] = GRID_HEIGHT;
      u32[2] = 0;
      u32[3] = 0;
      device.queue.writeBuffer(paramBuffer, 0, data);

      const encoder = device.createCommandEncoder({ label: 'inject-energy' });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[pingpong % 2]);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
  };
}
