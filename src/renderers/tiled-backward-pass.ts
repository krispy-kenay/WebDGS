/*
 * Tiled Backward Pass Scaffolding
 * 
 * Propagates gradients from 2D screen space back to 3D Gaussians.
 */

import { PointCloud } from '../utils/load';
import commonWGSL from '../shaders/common.wgsl';
import lossWGSL from '../shaders/loss.wgsl';
import backwardRasterizeWGSL from '../shaders/tiled-backward-rasterize.wgsl';
import backwardGeometryWGSL from '../shaders/tiled-backward.wgsl';

// Constants
const GEOMETRY_WORKGROUP_SIZE = 64;

export interface TrainingConfig {
    lambda_l1: number;
    lambda_l2: number;
    lambda_dssim: number;
    c1?: number;
    c2?: number;
}

export interface TiledBackwardPassConfig {
    viewportWidth: number;
    viewportHeight: number;
    trainingConfig: TrainingConfig;
    gaussianScale?: number;
    pointSizePx?: number;
}

export interface TiledBackwardResources {
    // TODO: Define resource interfaces
}

export interface TiledBackwardResources {
    // Buffers from Forward Pass
    splatBuffer: GPUBuffer;
    tileOffsetsBuffer: GPUBuffer;
    tileIndicesBuffer: GPUBuffer;
    cameraBuffer: GPUBuffer;

    // Textures from Rasterizer
    alphaTexture: GPUTextureView;
    nContribTexture: GPUTextureView;
}

export interface TiledBackwardPassOptions {
    // Options
}

// Helper to create GPU buffer
function createBuffer(
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer | ArrayBufferView
): GPUBuffer {
    const buffer = device.createBuffer({ label, size, usage });
    if (data) {
        device.queue.writeBuffer(buffer, 0, data);
    }
    return buffer;
}

export class TiledBackwardPass {
    private readonly device: GPUDevice;
    private readonly pointCloud: PointCloud;

    // Pipelines
    private readonly preprocessPipeline: GPUComputePipeline;
    private readonly accumulatePipeline: GPUComputePipeline;
    private readonly lossPipeline: GPUComputePipeline;
    private readonly backwardRasterizePipeline: GPUComputePipeline;
    private readonly backwardGeometryPipeline: GPUComputePipeline;

    // Bind Groups
    private lossBindGroup: GPUBindGroup | null = null;
    private backwardRasterizeBindGroups: GPUBindGroup[] = [];
    private backwardGeometryBindGroups: GPUBindGroup[] = [];

    // Buffers and Textures
    private readonly gradMeans2D: GPUBuffer;
    private readonly gradConics: GPUBuffer;
    private readonly gradOpacity: GPUBuffer;
    private readonly gradColors: GPUBuffer;
    private readonly outGradientsBuffer: GPUBuffer;

    // Intermediate
    private lossGradientTexture: GPUTexture;

    // Settings data
    private readonly settingsData: Float32Array;
    private readonly settingsBuffer: GPUBuffer;

    private readonly trainingConfigData: Float32Array;
    private readonly trainingConfigBuffer: GPUBuffer;
    private trainingConfig: TrainingConfig;

    // Dimensions
    private viewportWidth: number;
    private viewportHeight: number;

    // Dispatch info
    private readonly numWorkgroups: number;
    private readonly numPoints: number;

    constructor(
        device: GPUDevice,
        pointCloud: PointCloud,
        config: TiledBackwardPassConfig
    ) {
        this.device = device;
        this.pointCloud = pointCloud;
        this.numPoints = pointCloud.num_points;

        this.numWorkgroups = Math.ceil(this.numPoints / GEOMETRY_WORKGROUP_SIZE);

        this.viewportWidth = config.viewportWidth;
        this.viewportHeight = config.viewportHeight;

        // Settings uniform buffer
        this.settingsData = new Float32Array([
            config.gaussianScale ?? 1.0,
            pointCloud.sh_deg,
            config.viewportWidth,
            config.viewportHeight,
            config.pointSizePx ?? 3.0,
            0.0
        ]);
        this.settingsBuffer = createBuffer(
            device, 'backward-settings', this.settingsData.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            this.settingsData
        );

        // Training Config Buffer
        this.trainingConfig = config.trainingConfig;
        this.trainingConfigData = new Float32Array([
            this.trainingConfig.lambda_l1,
            this.trainingConfig.lambda_l2,
            this.trainingConfig.lambda_dssim,
            this.trainingConfig.c1 ?? 0.01 * 0.01,
            this.trainingConfig.c2 ?? 0.03 * 0.03,
            0, 0, 0
        ]);
        this.trainingConfigBuffer = createBuffer(
            device, 'training-config', this.trainingConfigData.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            this.trainingConfigData
        );

        // Loss Gradient Texture
        this.lossGradientTexture = device.createTexture({
            label: 'loss-gradient-texture',
            size: { width: config.viewportWidth, height: config.viewportHeight },
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        // Loss Pipeline
        const lossModule = device.createShaderModule({
            label: 'loss-shader',
            code: lossWGSL
        });

        this.lossPipeline = device.createComputePipeline({
            label: 'loss-compute',
            layout: 'auto',
            compute: { module: lossModule, entryPoint: 'compute_loss_grad' }
        });

        // Atomic Buffers
        this.gradMeans2D = createBuffer(device, 'grad-means-2d', this.numPoints * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.gradConics = createBuffer(device, 'grad-conics', this.numPoints * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.gradOpacity = createBuffer(device, 'grad-opacity', this.numPoints * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.gradColors = createBuffer(device, 'grad-colors', this.numPoints * 3 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        // Output Buffers
        this.outGradientsBuffer = createBuffer(device, 'out-gradients', this.numPoints * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        // Backward Rasterize Pipeline
        const bwRasterizeModule = device.createShaderModule({
            label: 'bw-rasterize-shader',
            code: `${commonWGSL}\n${backwardRasterizeWGSL}`
        });
        this.backwardRasterizePipeline = device.createComputePipeline({
            label: 'bw-rasterize-compute',
            layout: 'auto',
            compute: { module: bwRasterizeModule, entryPoint: 'backward_rasterize_main' }
        });

        // Backward Geometry Pipeline
        const bwGeometryModule = device.createShaderModule({
            label: 'bw-geometry-shader',
            code: `${commonWGSL}\n${backwardGeometryWGSL}`
        });
        this.backwardGeometryPipeline = device.createComputePipeline({
            label: 'bw-geometry-compute',
            layout: 'auto',
            compute: { module: bwGeometryModule, entryPoint: 'main_geometry_backward' }
        });

        // Placeholder pipelines
        const module = device.createShaderModule({
            label: 'tiled-backward-placeholder',
            code: `
                @compute @workgroup_size(${GEOMETRY_WORKGROUP_SIZE})
                fn main() {}
            `
        });

        this.preprocessPipeline = device.createComputePipeline({
            label: 'backward-preprocess',
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });

        this.accumulatePipeline = device.createComputePipeline({
            label: 'backward-accumulate',
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });
    }

    // Runs only the loss computation step
    computeLossOnly(
        encoder: GPUCommandEncoder,
        predictedTexture: GPUTextureView,
        targetTexture: GPUTextureView
    ): void {
        this.lossBindGroup = this.device.createBindGroup({
            label: 'loss-bind-group',
            layout: this.lossPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: predictedTexture },
                { binding: 1, resource: targetTexture },
                { binding: 2, resource: this.lossGradientTexture.createView() },
                { binding: 3, resource: { buffer: this.trainingConfigBuffer } }
            ]
        });

        const pass = encoder.beginComputePass({ label: 'loss-compute-pass' });
        pass.setPipeline(this.lossPipeline);
        pass.setBindGroup(0, this.lossBindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(this.viewportWidth / 16),
            Math.ceil(this.viewportHeight / 16)
        );
        pass.end();
    }

    getLossTextureView(): GPUTextureView {
        return this.lossGradientTexture.createView();
    }

    encode(
        encoder: GPUCommandEncoder,
        predictedTexture: GPUTextureView,
        targetTexture: GPUTextureView,
        forwardResources: TiledBackwardResources,
        options?: TiledBackwardPassOptions
    ): void {

        // Compute Loss Gradients
        {
            this.lossBindGroup = this.device.createBindGroup({
                label: 'loss-bind-group',
                layout: this.lossPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: predictedTexture },
                    { binding: 1, resource: targetTexture },
                    { binding: 2, resource: this.lossGradientTexture.createView() },
                    { binding: 3, resource: { buffer: this.trainingConfigBuffer } }
                ]
            });

            const pass = encoder.beginComputePass({ label: 'loss-compute-pass' });
            pass.setPipeline(this.lossPipeline);
            pass.setBindGroup(0, this.lossBindGroup);
            pass.dispatchWorkgroups(
                Math.ceil(this.viewportWidth / 16),
                Math.ceil(this.viewportHeight / 16)
            );
            pass.end();
        }

        // Clear Atomic Buffers
        encoder.clearBuffer(this.gradMeans2D);
        encoder.clearBuffer(this.gradConics);
        encoder.clearBuffer(this.gradOpacity);
        encoder.clearBuffer(this.gradColors);

        // Backward Rasterize
        {
            if (forwardResources) {
                const bG0 = this.device.createBindGroup({
                    label: 'bw-rast-camera',
                    layout: this.backwardRasterizePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.settingsBuffer } },
                    ],
                });

                const bG1 = this.device.createBindGroup({
                    label: 'bw-rast-scene',
                    layout: this.backwardRasterizePipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: forwardResources.tileOffsetsBuffer } },
                        { binding: 1, resource: { buffer: forwardResources.tileIndicesBuffer } },
                        { binding: 2, resource: { buffer: forwardResources.splatBuffer } },
                        { binding: 4, resource: forwardResources.alphaTexture },
                        { binding: 5, resource: forwardResources.nContribTexture }
                    ]
                });

                const bG2 = this.device.createBindGroup({
                    label: 'bw-rast-loss',
                    layout: this.backwardRasterizePipeline.getBindGroupLayout(2),
                    entries: [
                        { binding: 0, resource: this.lossGradientTexture.createView() }
                    ]
                });

                const bG3 = this.device.createBindGroup({
                    label: 'bw-rast-atomic',
                    layout: this.backwardRasterizePipeline.getBindGroupLayout(3),
                    entries: [
                        { binding: 0, resource: { buffer: this.gradMeans2D } },
                        { binding: 1, resource: { buffer: this.gradConics } },
                        { binding: 2, resource: { buffer: this.gradOpacity } },
                        { binding: 3, resource: { buffer: this.gradColors } }
                    ]
                });

                const pass = encoder.beginComputePass({ label: 'backward-rasterize-pass' });
                pass.setPipeline(this.backwardRasterizePipeline);
                pass.setBindGroup(0, bG0);
                pass.setBindGroup(1, bG1);
                pass.setBindGroup(2, bG2);
                pass.setBindGroup(3, bG3);
                pass.dispatchWorkgroups(
                    Math.ceil(this.viewportWidth / 16),
                    Math.ceil(this.viewportHeight / 16)
                );
                pass.end();
            }
        }

        // Backward Geometry
        {
            if (forwardResources) {
                const bG0 = this.device.createBindGroup({
                    label: 'bw-geom-camera',
                    layout: this.backwardGeometryPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: forwardResources.cameraBuffer } },
                        { binding: 1, resource: { buffer: this.settingsBuffer } },
                    ],
                });

                const bG1 = this.device.createBindGroup({
                    label: 'bw-geom-input',
                    layout: this.backwardGeometryPipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: this.pointCloud.gaussian_3d_buffer } }, // Gaussians 3D (source)
                        { binding: 1, resource: { buffer: this.gradMeans2D } },
                        { binding: 2, resource: { buffer: this.gradConics } },
                        { binding: 3, resource: { buffer: this.gradOpacity } },
                        { binding: 4, resource: { buffer: this.gradColors } }
                    ]
                });

                const bG2 = this.device.createBindGroup({
                    label: 'bw-geom-output',
                    layout: this.backwardGeometryPipeline.getBindGroupLayout(2),
                    entries: [
                        { binding: 0, resource: { buffer: this.outGradientsBuffer } }
                    ]
                });

                const pass = encoder.beginComputePass({ label: 'backward-geometry-pass' });
                pass.setPipeline(this.backwardGeometryPipeline);
                pass.setBindGroup(0, bG0);
                pass.setBindGroup(1, bG1);
                pass.setBindGroup(2, bG2);
                pass.dispatchWorkgroups(this.numWorkgroups);
                pass.end();
            }
        }

        {
            const pass = encoder.beginComputePass({ label: 'backward-preprocess-pass' });
            pass.setPipeline(this.preprocessPipeline);
            pass.dispatchWorkgroups(this.numWorkgroups);
            pass.end();
        }

        {
            const pass = encoder.beginComputePass({ label: 'backward-accumulate-pass' });
            pass.setPipeline(this.accumulatePipeline);
            pass.dispatchWorkgroups(this.numWorkgroups);
            pass.end();
        }
    }

    setViewport(width: number, height: number): void {
        this.viewportWidth = width;
        this.viewportHeight = height;
        this.settingsData[2] = width;
        this.settingsData[3] = height;
        this.flushSettings();

        // Resize loss gradient texture
        this.lossGradientTexture.destroy();
        this.lossGradientTexture = this.device.createTexture({
            label: 'loss-gradient-texture',
            size: { width, height },
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
    }

    setTrainingConfig(config: Partial<TrainingConfig>): void {
        if (config.lambda_l1 !== undefined) this.trainingConfig.lambda_l1 = config.lambda_l1;
        if (config.lambda_l2 !== undefined) this.trainingConfig.lambda_l2 = config.lambda_l2;
        if (config.lambda_dssim !== undefined) this.trainingConfig.lambda_dssim = config.lambda_dssim;
        if (config.c1 !== undefined) this.trainingConfig.c1 = config.c1;
        if (config.c2 !== undefined) this.trainingConfig.c2 = config.c2;

        this.trainingConfigData[0] = this.trainingConfig.lambda_l1;
        this.trainingConfigData[1] = this.trainingConfig.lambda_l2;
        this.trainingConfigData[2] = this.trainingConfig.lambda_dssim;
        this.trainingConfigData[3] = this.trainingConfig.c1 ?? 0.01 * 0.01;
        this.trainingConfigData[4] = this.trainingConfig.c2 ?? 0.03 * 0.03;

        this.device.queue.writeBuffer(this.trainingConfigBuffer, 0, this.trainingConfigData);
    }

    private flushSettings(): void {
        this.device.queue.writeBuffer(this.settingsBuffer, 0, this.settingsData);
    }

    getGradientsBuffer(): GPUBuffer {
        return this.outGradientsBuffer;
    }
}
