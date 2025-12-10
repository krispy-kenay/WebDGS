import renderWGSL from '../shaders/gaussian.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import { ForwardPass } from './forward-pass';

export interface RasterizerConfig {
  device: GPUDevice;
  forwardPass: ForwardPass;
  format: GPUTextureFormat;
}

export class Rasterizer {
  private readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;
  private readonly splatBindGroup: GPUBindGroup;
  private readonly sortedBindGroup: GPUBindGroup;
  private readonly settingsBindGroup: GPUBindGroup;
  private readonly indirectArgs: GPUBuffer;

  constructor(config: RasterizerConfig) {
    const { device, forwardPass, format } = config;
    this.device = device;
    const resources = forwardPass.getResources();
    this.indirectArgs = resources.indirectArgs;

    const module = device.createShaderModule({
      label: 'gaussian-render',
      code: `${commonWGSL}\n${renderWGSL}`,
    });

    this.pipeline = device.createRenderPipeline({
      label: 'gaussian-render-pipeline',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
            writeMask: GPUColorWrite.ALL,
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.splatBindGroup = device.createBindGroup({
      label: 'rasterizer-splats',
      layout: this.pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: resources.splatBuffer } }],
    });

    this.sortedBindGroup = device.createBindGroup({
      label: 'rasterizer-sorted',
      layout: this.pipeline.getBindGroupLayout(2),
      entries: [{ binding: 0, resource: { buffer: forwardPass.getSortedIndicesBuffer() } }],
    });

    this.settingsBindGroup = device.createBindGroup({
      label: 'rasterizer-settings',
      layout: this.pipeline.getBindGroupLayout(3),
      entries: [{ binding: 0, resource: { buffer: resources.settingsBuffer } }],
    });
  }

  encode(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    clearColor: GPUColorDict = { r: 0, g: 0, b: 0, a: 1 }
  ) {
    const pass = encoder.beginRenderPass({
      label: 'gaussian-render-pass',
      colorAttachments: [
        {
          view: targetView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: clearColor,
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(1, this.splatBindGroup);
    pass.setBindGroup(2, this.sortedBindGroup);
    pass.setBindGroup(3, this.settingsBindGroup);
    pass.drawIndirect(this.indirectArgs, 0);
    pass.end();
  }
}
