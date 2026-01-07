import forwardWGSL from '../shaders/forward.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import { get_sorter, C, SortStuff } from '../sort/sort';
import { PointCloud } from '../utils/load';

export type RenderMode = 'pointcloud' | 'gaussian';

const WORKGROUP_SIZE = 256;
const NULLING_DATA = new Uint32Array([0]);

export interface ForwardPassConfig {
  viewportWidth: number;
  viewportHeight: number;
  gaussianScale?: number;
  pointSizePx?: number;
  maxSplatRadiusPx?: number;
  renderMode?: RenderMode;
}

export interface ForwardResources {
  splatBuffer: GPUBuffer;
  settingsBuffer: GPUBuffer;
  indirectArgs: GPUBuffer;
  sorter: SortStuff;
}

export interface ForwardPassOptions {
  skipSort?: boolean;
}

const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export class ForwardPass {
  private readonly device: GPUDevice;
  private readonly pointCloud: PointCloud;
  private readonly sorter: SortStuff;
  private readonly forwardPipeline: GPUComputePipeline;
  private cameraBindGroup: GPUBindGroup;
  private readonly sceneBindGroup: GPUBindGroup;
  private readonly sortBindGroup: GPUBindGroup;
  private readonly splatBuffer: GPUBuffer;
  private readonly settingsBuffer: GPUBuffer;
  private readonly settingsData: Float32Array;
  private readonly indirectArgs: GPUBuffer;
  private readonly numWorkgroups: number;
  private renderMode: RenderMode;

  constructor(
    device: GPUDevice,
    pointCloud: PointCloud,
    cameraBuffer: GPUBuffer,
    config: ForwardPassConfig
  ) {
    this.device = device;
    this.pointCloud = pointCloud;
    this.renderMode = config.renderMode ?? 'pointcloud';

    this.sorter = get_sorter(pointCloud.num_points, device);
    this.numWorkgroups = Math.ceil(pointCloud.num_points / WORKGROUP_SIZE);

    this.settingsData = new Float32Array([
      config.gaussianScale ?? 1,
      pointCloud.sh_deg,
      config.viewportWidth,
      config.viewportHeight,
      config.pointSizePx ?? 3,
      this.renderMode === 'gaussian' ? 1 : 0,
      config.maxSplatRadiusPx ?? 128,
    ]);

    this.settingsBuffer = createBuffer(
      device,
      'render settings',
      this.settingsData.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      this.settingsData
    );

    this.indirectArgs = createBuffer(
      device,
      'indirect args',
      4 * 4,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      new Uint32Array([6, pointCloud.num_points, 0, 0])
    );

    const splatStride = 6 * 4; // 6 packed u32 entries per splat
    this.splatBuffer = createBuffer(
      device,
      'splat buffer',
      pointCloud.num_points * splatStride,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );

    this.forwardPipeline = device.createComputePipeline({
      label: 'forward',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: `${commonWGSL}\n${forwardWGSL}` }),
        entryPoint: 'forward',
        constants: {
          workgroupSize: WORKGROUP_SIZE,
          sortKeyPerThread: (C.histogram_wg_size * C.rs_histogram_block_rows) / WORKGROUP_SIZE,
        },
      },
    });

    this.cameraBindGroup = this.createCameraBindGroup(cameraBuffer);

    this.sceneBindGroup = device.createBindGroup({
      label: 'forward-scene',
      layout: this.forwardPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: pointCloud.gaussian_3d_buffer } },
        { binding: 1, resource: { buffer: pointCloud.sh_buffer } },
        { binding: 2, resource: { buffer: this.splatBuffer } },
      ],
    });

    this.sortBindGroup = device.createBindGroup({
      label: 'forward-sort-bindings',
      layout: this.forwardPipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: this.sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: this.sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: this.sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: this.sorter.sort_dispatch_indirect_buffer } },
      ],
    });
  }

  encode(encoder: GPUCommandEncoder, options?: ForwardPassOptions) {
    this.device.queue.writeBuffer(this.sorter.sort_info_buffer, 0, NULLING_DATA);
    this.device.queue.writeBuffer(this.sorter.sort_dispatch_indirect_buffer, 0, NULLING_DATA);

    const pass = encoder.beginComputePass({ label: 'forward-preprocess-pass' });
    pass.setPipeline(this.forwardPipeline);
    pass.setBindGroup(0, this.cameraBindGroup);
    pass.setBindGroup(1, this.sceneBindGroup);
    pass.setBindGroup(2, this.sortBindGroup);
    pass.dispatchWorkgroups(this.numWorkgroups);
    pass.end();

    encoder.copyBufferToBuffer(this.sorter.sort_info_buffer, 0, this.indirectArgs, 4, 4);

    if (!options?.skipSort) {
      this.sorter.sort(encoder);
    }
  }

  setCameraBuffer(buffer: GPUBuffer) {
    this.cameraBindGroup = this.createCameraBindGroup(buffer);
  }

  setGaussianScale(value: number) {
    this.settingsData[0] = value;
    this.flushSettings();
  }

  setPointSize(value: number) {
    this.settingsData[4] = value;
    this.flushSettings();
  }

  setRenderMode(mode: RenderMode) {
    this.renderMode = mode;
    this.settingsData[5] = this.renderMode === 'gaussian' ? 1 : 0;
    this.flushSettings();
  }

  setViewport(width: number, height: number) {
    this.settingsData[2] = width;
    this.settingsData[3] = height;
    this.flushSettings();
  }

  getResources(): ForwardResources {
    return {
      splatBuffer: this.splatBuffer,
      settingsBuffer: this.settingsBuffer,
      indirectArgs: this.indirectArgs,
      sorter: this.sorter,
    };
  }

  getSortedIndicesBuffer(): GPUBuffer {
    return this.sorter.ping_pong[this.sorter.final_out_index].sort_indices_buffer;
  }

  private flushSettings() {
    this.device.queue.writeBuffer(this.settingsBuffer, 0, this.settingsData);
  }

  private createCameraBindGroup(cameraBuffer: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'forward-camera',
      layout: this.forwardPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: this.settingsBuffer } },
      ],
    });
  }
}
