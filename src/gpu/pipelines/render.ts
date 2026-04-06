/** Fragment-only visualization; cell/env buffers are filled on CPU then uploaded each frame. */
import type { SimBuffers } from '../buffers';
import type { GPUContext } from '../context';
import shaderCode from '../shaders/render.wgsl?raw';

export interface RenderPipeline {
  draw(encoder: GPUCommandEncoder, pingpong: number): void;
}

export function createRenderPipeline(
  gpu: GPUContext,
  buffers: SimBuffers,
): RenderPipeline {
  const module = gpu.device.createShaderModule({ label: 'render', code: shaderCode });

  const bindGroupLayout = gpu.device.createBindGroupLayout({
    label: 'render-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });

  const pipelineLayout = gpu.device.createPipelineLayout({
    label: 'render-pl',
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = gpu.device.createRenderPipeline({
    label: 'render',
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vert' },
    fragment: {
      module,
      entryPoint: 'frag',
      targets: [{ format: gpu.format }],
    },
    primitive: { topology: 'triangle-strip', stripIndexFormat: 'uint32' },
  });

  const bindGroups = [
    gpu.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.cellState[0] } },
        { binding: 2, resource: { buffer: buffers.envEnergy[0] } },
        { binding: 3, resource: { buffer: buffers.componentMask } },
        { binding: 4, resource: { buffer: buffers.rot } },
      ],
    }),
    gpu.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniform } },
        { binding: 1, resource: { buffer: buffers.cellState[1] } },
        { binding: 2, resource: { buffer: buffers.envEnergy[1] } },
        { binding: 3, resource: { buffer: buffers.componentMask } },
        { binding: 4, resource: { buffer: buffers.rot } },
      ],
    }),
  ];

  return {
    draw(encoder, pingpong) {
      const textureView = gpu.context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        label: 'render',
        colorAttachments: [{
          view: textureView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear' as GPULoadOp,
          storeOp: 'store' as GPUStoreOp,
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[pingpong % 2]);
      pass.draw(4);
      pass.end();
    },
  };
}
