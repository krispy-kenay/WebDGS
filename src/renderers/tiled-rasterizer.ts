/*
 * Tiled Rasterizer Pass
 * 
 * Uses compute shaders to rasterize forward pass results into an arbitrary output texture.
 */

import tileRangesWGSL from '../shaders/tile-ranges.wgsl';
import tiledRasterizerWGSL from '../shaders/tiled-rasterizer.wgsl';
import blitWGSL from '../shaders/blit.wgsl';
import { TiledForwardPass, TiledForwardResources } from './tiled-forward-pass';

const TILE_WIDTH = 16;
const TILE_HEIGHT = 16;
const RANGES_WORKGROUP_SIZE = 256;

export interface TiledRasterizerConfig {
    device: GPUDevice;
    forwardPass: TiledForwardPass;
    format: GPUTextureFormat;
}

export class TiledRasterizer {
    private readonly device: GPUDevice;
    private readonly forwardPass: TiledForwardPass;
    private readonly format: GPUTextureFormat;
    private readonly blitSampler: GPUSampler;

    // Pipelines
    private readonly initOffsetsPipeline: GPUComputePipeline;
    private readonly buildOffsetsPipeline: GPUComputePipeline;
    private readonly rasterizePipeline: GPUComputePipeline;
    private readonly blitPipeline: GPURenderPipeline;

    // Buffers
    private readonly tileOffsetsBuffer: GPUBuffer;
    private readonly resources: TiledForwardResources;

    // Bind groups
    private offsetsBindGroup: GPUBindGroup | null = null;
    private rasterizeConfigBindGroup: GPUBindGroup;
    private rasterizeDataBindGroup: GPUBindGroup | null = null;
    private rasterizeOutputBindGroup: GPUBindGroup | null = null;
    private blitBindGroup: GPUBindGroup | null = null;

    // Output texture
    private outputTexture: GPUTexture | null = null;
    private outputTextureView: GPUTextureView | null = null;
    private outputAlphaTexture: GPUTexture | null = null;
    private outputAlphaTextureView: GPUTextureView | null = null;
    private outputNContribTexture: GPUTexture | null = null;
    private outputNContribTextureView: GPUTextureView | null = null;
    private viewportWidth: number;
    private viewportHeight: number;

    constructor(config: TiledRasterizerConfig) {
        const { device, forwardPass, format } = config;
        this.device = device;
        this.forwardPass = forwardPass;
        this.format = format;
        this.resources = forwardPass.getResources();

        this.viewportWidth = 0;
        this.viewportHeight = 0;

        // Create tile offsets buffer
        this.tileOffsetsBuffer = device.createBuffer({
            label: 'tile offsets buffer',
            size: (this.resources.totalTiles + 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Create tile offsets pipelines
        const offsetsModule = device.createShaderModule({
            label: 'tile-offsets',
            code: tileRangesWGSL,
        });

        const offsetsBindGroupLayout = device.createBindGroupLayout({
            label: 'tile-offsets-layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });

        const offsetsPipelineLayout = device.createPipelineLayout({
            label: 'tile-offsets-pipeline-layout',
            bindGroupLayouts: [offsetsBindGroupLayout],
        });

        this.initOffsetsPipeline = device.createComputePipeline({
            label: 'init-tile-offsets',
            layout: offsetsPipelineLayout,
            compute: {
                module: offsetsModule,
                entryPoint: 'init_tile_offsets',
                constants: { workgroupSize: RANGES_WORKGROUP_SIZE },
            },
        });

        this.buildOffsetsPipeline = device.createComputePipeline({
            label: 'build-tile-offsets',
            layout: offsetsPipelineLayout,
            compute: {
                module: offsetsModule,
                entryPoint: 'build_tile_offsets',
                constants: { workgroupSize: RANGES_WORKGROUP_SIZE },
            },
        });

        // Create rasterizer pipeline
        const rasterModule = device.createShaderModule({
            label: 'tiled-rasterizer',
            code: tiledRasterizerWGSL,
        });

        this.rasterizePipeline = device.createComputePipeline({
            label: 'tiled-rasterize',
            layout: 'auto',
            compute: {
                module: rasterModule,
                entryPoint: 'tiled_rasterize',
                constants: {
                    tileWidth: TILE_WIDTH,
                    tileHeight: TILE_HEIGHT,
                },
            },
        });

        // Create blit pipeline
        const blitModule = device.createShaderModule({
            label: 'tiled-blit',
            code: blitWGSL,
        });
        this.blitPipeline = device.createRenderPipeline({
            label: 'tiled-blit-pipeline',
            layout: 'auto',
            vertex: {
                module: blitModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: blitModule,
                entryPoint: 'fs_main',
                targets: [
                    {
                        format: this.format,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
        this.blitSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        this.blitBindGroup = null;

        // Create config and data bind groups
        this.rasterizeConfigBindGroup = device.createBindGroup({
            label: 'rasterize-config',
            layout: this.rasterizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.resources.tileInfoBuffer } },
                { binding: 1, resource: { buffer: this.resources.settingsBuffer } },
            ],
        });


    }

    encode(
        encoder: GPUCommandEncoder,
        width: number,
        height: number
    ) {
        // Ensure output texture matches target size
        this.ensureOutputTexture(width, height);

        // Recreate bind groups that depend on ping-pong buffers
        this.offsetsBindGroup = this.device.createBindGroup({
            label: 'tile-offsets-bind-group',
            layout: this.initOffsetsPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.resources.tileInfoBuffer } },
                { binding: 1, resource: { buffer: this.resources.sorter.sort_info_buffer } },
                { binding: 2, resource: { buffer: this.forwardPass.getSortedKeysBuffer() } },
                { binding: 3, resource: { buffer: this.tileOffsetsBuffer } },
                { binding: 4, resource: { buffer: this.forwardPass.getStatsBuffer() } },
            ],
        });

        this.rasterizeDataBindGroup = this.device.createBindGroup({
            label: 'rasterize-data',
            layout: this.rasterizePipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.resources.splatBuffer } },
                { binding: 1, resource: { buffer: this.tileOffsetsBuffer } },
                { binding: 2, resource: { buffer: this.forwardPass.getSortedIndicesBuffer() } },
                { binding: 3, resource: { buffer: this.forwardPass.getSortedKeysBuffer() } },
                { binding: 4, resource: { buffer: this.forwardPass.getStatsBuffer() } },
            ],
        });

        // Initialize offsets
        {
            const pass = encoder.beginComputePass({ label: 'init-offsets' });
            pass.setPipeline(this.initOffsetsPipeline);
            pass.setBindGroup(0, this.offsetsBindGroup!);
            pass.dispatchWorkgroups(Math.ceil((this.resources.totalTiles + 1) / RANGES_WORKGROUP_SIZE));
            pass.end();
        }

        // Build offsets
        {
            const pass = encoder.beginComputePass({ label: 'build-offsets' });
            pass.setPipeline(this.buildOffsetsPipeline);
            pass.setBindGroup(0, this.offsetsBindGroup!);
            const dispatchSize = Math.ceil(this.resources.maxTileEntries / RANGES_WORKGROUP_SIZE);
            pass.dispatchWorkgroups(dispatchSize);
            pass.end();
        }

        // Rasterize
        {
            const pass = encoder.beginComputePass({ label: 'tiled-rasterize' });
            pass.setPipeline(this.rasterizePipeline);
            pass.setBindGroup(0, this.rasterizeConfigBindGroup);
            pass.setBindGroup(1, this.rasterizeDataBindGroup!);
            pass.setBindGroup(2, this.rasterizeOutputBindGroup!);
            pass.dispatchWorkgroups(this.resources.numTilesX, this.resources.numTilesY);
            pass.end();
        }
    }

    private ensureOutputTexture(width: number, height: number) {
        if (this.viewportWidth === width && this.viewportHeight === height && this.outputTexture) {
            return;
        }

        // Destroy old texture
        this.outputTexture?.destroy();
        this.outputAlphaTexture?.destroy();
        this.outputNContribTexture?.destroy();

        // Create new output texture with storage usage
        this.outputTexture = this.device.createTexture({
            label: 'tiled-rasterizer-output',
            size: { width, height },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        this.outputTextureView = this.outputTexture.createView();

        // Create alpha output texture
        this.outputAlphaTexture = this.device.createTexture({
            label: 'tiled-rasterizer-alpha',
            size: { width, height },
            format: 'r32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.outputAlphaTextureView = this.outputAlphaTexture.createView();

        // Create n_contrib output texture
        this.outputNContribTexture = this.device.createTexture({
            label: 'tiled-rasterizer-n-contrib',
            size: { width, height },
            format: 'r32uint',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.outputNContribTextureView = this.outputNContribTexture.createView();

        // Recreate output bind group
        this.rasterizeOutputBindGroup = this.device.createBindGroup({
            label: 'rasterize-output',
            layout: this.rasterizePipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: this.outputTextureView },
                { binding: 1, resource: this.outputAlphaTextureView! },
                { binding: 2, resource: this.outputNContribTextureView! },
            ],
        });

        // Bind group for blit pass
        this.blitBindGroup = this.device.createBindGroup({
            label: 'tiled-blit-bind-group',
            layout: this.blitPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.outputTextureView },
                { binding: 1, resource: this.blitSampler },
            ],
        });

        this.viewportWidth = width;
        this.viewportHeight = height;
    }

    // Method for training pipeline to get output texture view
    getOutputTextureView(): GPUTextureView {
        if (!this.outputTextureView) {
            throw new Error('TiledRasterizer: output texture has not been created yet. Call encode() first.');
        }
        return this.outputTextureView;
    }
    getTileOffsetsBuffer(): GPUBuffer {
        return this.tileOffsetsBuffer;
    }

    getAlphaTextureView(): GPUTextureView {
        if (!this.outputAlphaTextureView) {
            throw new Error('TiledRasterizer: no alpha texture. Call encode() first.');
        }
        return this.outputAlphaTextureView;
    }

    getNContribTextureView(): GPUTextureView {
        if (!this.outputNContribTextureView) {
            throw new Error('TiledRasterizer: no n_contrib texture. Call encode() first.');
        }
        return this.outputNContribTextureView;
    }

    // Method for viewer to blit output texture to target texture
    blitToTexture(
        encoder: GPUCommandEncoder,
        targetView: GPUTextureView,
        clearColor: GPUColorDict = { r: 0, g: 0, b: 0, a: 1 },
    ) {
        if (!this.outputTexture) {
            throw new Error('TiledRasterizer: no output texture to blit from. Call encode() first.');
        }

        const pass = encoder.beginRenderPass({
            label: 'tiled-blit-pass',
            colorAttachments: [
                {
                    view: targetView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: clearColor,
                },
            ],
        });
        pass.setPipeline(this.blitPipeline);
        pass.setBindGroup(0, this.blitBindGroup);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }
}
