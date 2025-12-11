/*
 * Tiled Forward Pass
 * 
 * Projects 3D Gaussians to 2D screen space and assigns them to tiles for efficient per-tile rasterization.
 */

import tiledForwardWGSL from '../shaders/tiled-forward.wgsl';
import updateStatsWGSL from '../shaders/update-stats.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import { get_dynamic_sorter, DynamicSortStuff, C } from '../sort/sort_dynamic';
import { get_prefix_scanner, PrefixScanner, PREFIX_CONSTANTS } from '../prefix/prefix';
import { PointCloud } from '../utils/load';

export type RenderMode = 'pointcloud' | 'gaussian';

// Constants
const WORKGROUP_SIZE = 256;
const TILE_WIDTH = 16;
const TILE_HEIGHT = 16;

// Maximum tiles per Gaussian
const MAX_TILES_PER_GAUSSIAN = 256;

export interface TiledForwardPassConfig {
    viewportWidth: number;
    viewportHeight: number;
    gaussianScale?: number;
    pointSizePx?: number;
    renderMode?: RenderMode;
}

export interface TiledForwardResources {
    splatBuffer: GPUBuffer;
    settingsBuffer: GPUBuffer;
    tileInfoBuffer: GPUBuffer;
    tileKeysBuffer: GPUBuffer;
    tileIndicesBuffer: GPUBuffer;
    sorter: DynamicSortStuff;
    numTilesX: number;
    numTilesY: number;
    totalTiles: number;
    maxTileEntries: number;
}

export interface TiledForwardPassOptions {
    skipSort?: boolean;
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

export class TiledForwardPass {
    private readonly device: GPUDevice;
    private readonly pointCloud: PointCloud;

    // Pipelines
    private readonly countPipeline: GPUComputePipeline;
    private readonly emitPipeline: GPUComputePipeline;
    private readonly updateStatsPipeline: GPUComputePipeline;

    // GPU Utilities
    private readonly sorter: DynamicSortStuff;
    private readonly prefixScanner: PrefixScanner;

    // Bind Groups
    private countCameraBindGroup: GPUBindGroup;
    private readonly countSceneBindGroup: GPUBindGroup;
    private emitCameraBindGroup: GPUBindGroup;
    private readonly emitSceneBindGroup: GPUBindGroup;
    private readonly emitOffsetBindGroup: GPUBindGroup;
    private readonly emitSortBindGroup: GPUBindGroup;
    private readonly updateStatsBindGroup: GPUBindGroup;

    // Buffers
    private readonly splatBuffer: GPUBuffer;
    private readonly depthsBuffer: GPUBuffer;
    private readonly tileCountsBuffer: GPUBuffer;
    private readonly tileOffsetsBuffer: GPUBuffer;
    private readonly tileKeysBuffer: GPUBuffer;
    private readonly tileIndicesBuffer: GPUBuffer;
    private readonly settingsBuffer: GPUBuffer;
    private readonly tileInfoBuffer: GPUBuffer;
    private readonly pipelineStatsBuffer: GPUBuffer;
    private readonly numGaussiansBuffer: GPUBuffer;

    // Settings data (CPU-side for updates)
    private readonly settingsData: Float32Array;
    private readonly tileInfoData: Uint32Array;

    // Dispatch info
    private readonly numWorkgroups: number;
    private readonly numPoints: number;

    // Tile info
    private numTilesX: number;
    private numTilesY: number;
    private totalTiles: number;
    private maxTileEntries: number;

    // Render mode
    private renderMode: RenderMode;

    constructor(
        device: GPUDevice,
        pointCloud: PointCloud,
        cameraBuffer: GPUBuffer,
        config: TiledForwardPassConfig
    ) {
        this.device = device;
        this.pointCloud = pointCloud;
        this.numPoints = pointCloud.num_points;
        this.renderMode = config.renderMode ?? 'gaussian';

        // tile grid
        this.numTilesX = Math.ceil(config.viewportWidth / TILE_WIDTH);
        this.numTilesY = Math.ceil(config.viewportHeight / TILE_HEIGHT);
        this.totalTiles = this.numTilesX * this.numTilesY;

        // Compute maxTileEntries constraints
        const avgTilesPerGaussian = 30;
        const keys_per_workgroup = C.histogram_wg_size * C.rs_histogram_block_rows;

        // Workload estimate
        const baseEstimate = Math.min(
            this.numPoints * avgTilesPerGaussian,
            this.numPoints * MAX_TILES_PER_GAUSSIAN
        );

        // Buffer size constraints
        const MAX_BUFFER_SIZE = 128 * 1024 * 1024;  // 128MB
        const BYTES_PER_KEY = 4;
        const maxFromBufferLimit = Math.floor(MAX_BUFFER_SIZE / BYTES_PER_KEY);

        // Prefix sum limit
        const maxFromPrefixSum = PREFIX_CONSTANTS.MAX_ELEMENTS;
        const constrainedMax = Math.min(baseEstimate, maxFromBufferLimit, maxFromPrefixSum);
        this.maxTileEntries = Math.ceil(constrainedMax / keys_per_workgroup) * keys_per_workgroup;

        console.log(`maxTileEntries: ${this.maxTileEntries.toLocaleString()} ` +
            `(estimate: ${baseEstimate.toLocaleString()}, buffer limit: ${maxFromBufferLimit.toLocaleString()}, ` +
            `prefix limit: ${maxFromPrefixSum.toLocaleString()})`);

        this.numWorkgroups = Math.ceil(this.numPoints / WORKGROUP_SIZE);

        // Pipeline stats buffer
        this.pipelineStatsBuffer = createBuffer(
            device, 'pipeline-stats', 16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            new Uint32Array([0, 0, 0, 0])
        );

        // GPU utilities
        this.sorter = get_dynamic_sorter(this.maxTileEntries, device, this.pipelineStatsBuffer);
        this.prefixScanner = get_prefix_scanner(this.numPoints, device);

        // Settings uniform buffer
        this.settingsData = new Float32Array([
            config.gaussianScale ?? 1.0,
            pointCloud.sh_deg,
            config.viewportWidth,
            config.viewportHeight,
            config.pointSizePx ?? 3.0,
            this.renderMode === 'gaussian' ? 1.0 : 0.0
        ]);
        this.settingsBuffer = createBuffer(
            device, 'settings', this.settingsData.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            this.settingsData
        );

        // Tile info uniform buffer
        this.tileInfoData = new Uint32Array([
            this.numTilesX,
            this.numTilesY,
            this.totalTiles,
            this.maxTileEntries,
        ]);
        this.tileInfoBuffer = createBuffer(
            device, 'tile info', this.tileInfoData.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            this.tileInfoData
        );

        // Splat buffer
        const splatStride = 6 * 4;
        this.splatBuffer = createBuffer(
            device, 'splat buffer', this.numPoints * splatStride,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        );

        // Depths buffer
        this.depthsBuffer = createBuffer(
            device, 'depths buffer', this.numPoints * 4,
            GPUBufferUsage.STORAGE
        );

        // Prefix scanner buffers
        this.tileCountsBuffer = this.prefixScanner.input_buffer;
        this.tileOffsetsBuffer = this.prefixScanner.output_buffer;

        // Sorter buffers
        this.tileKeysBuffer = this.sorter.ping_pong[0].sort_depths_buffer;
        this.tileIndicesBuffer = this.sorter.ping_pong[0].sort_indices_buffer;

        // Num gaussians buffer
        this.numGaussiansBuffer = createBuffer(
            device, 'num-gaussians', 4,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            new Uint32Array([this.numPoints])
        );

        // Shaders
        const shaderModule = device.createShaderModule({
            label: 'tiled-forward',
            code: `${commonWGSL}\n${tiledForwardWGSL}`,
        });

        // Pipelines
        this.countPipeline = device.createComputePipeline({
            label: 'tiled-forward-count',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'count_main',
                constants: {
                    workgroupSize: WORKGROUP_SIZE,
                    tileWidth: TILE_WIDTH,
                    tileHeight: TILE_HEIGHT,
                },
            },
        });

        this.emitPipeline = device.createComputePipeline({
            label: 'tiled-forward-emit',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'emit_main',
                constants: {
                    workgroupSize: WORKGROUP_SIZE,
                    tileWidth: TILE_WIDTH,
                    tileHeight: TILE_HEIGHT,
                },
            },
        });

        // Bind groups
        this.countCameraBindGroup = this.createCountCameraBindGroup(cameraBuffer);

        this.countSceneBindGroup = device.createBindGroup({
            label: 'count-scene',
            layout: this.countPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: pointCloud.gaussian_3d_buffer } },
                { binding: 1, resource: { buffer: pointCloud.sh_buffer } },
                { binding: 2, resource: { buffer: this.splatBuffer } },
                { binding: 3, resource: { buffer: this.depthsBuffer } },
                { binding: 4, resource: { buffer: this.tileCountsBuffer } },
                { binding: 5, resource: { buffer: this.pipelineStatsBuffer } },
            ],
        });

        this.emitCameraBindGroup = this.createEmitCameraBindGroup(cameraBuffer);

        this.emitSceneBindGroup = device.createBindGroup({
            label: 'emit-scene',
            layout: this.emitPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: pointCloud.gaussian_3d_buffer } },
                { binding: 2, resource: { buffer: this.splatBuffer } },
                { binding: 3, resource: { buffer: this.depthsBuffer } },
                { binding: 4, resource: { buffer: this.tileCountsBuffer } },
            ],
        });

        this.emitOffsetBindGroup = device.createBindGroup({
            label: 'emit-offsets',
            layout: this.emitPipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: this.tileOffsetsBuffer } },
            ],
        });

        this.emitSortBindGroup = device.createBindGroup({
            label: 'emit-sort',
            layout: this.emitPipeline.getBindGroupLayout(3),
            entries: [
                { binding: 0, resource: { buffer: this.tileKeysBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
            ],
        });

        const updateStatsModule = device.createShaderModule({
            label: 'update-stats',
            code: updateStatsWGSL,
        });

        this.updateStatsPipeline = device.createComputePipeline({
            label: 'update-stats',
            layout: 'auto',
            compute: {
                module: updateStatsModule,
                entryPoint: 'update_stats',
            },
        });

        this.updateStatsBindGroup = device.createBindGroup({
            label: 'update-stats',
            layout: this.updateStatsPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.tileOffsetsBuffer } },
                { binding: 1, resource: { buffer: this.tileCountsBuffer } },
                { binding: 2, resource: { buffer: this.pipelineStatsBuffer } },
                { binding: 3, resource: { buffer: this.numGaussiansBuffer } },
            ],
        });

        this.prefixScanner.set_count(this.numPoints);
        this.updateSorterDispatch();

    }

    encode(encoder: GPUCommandEncoder, options?: TiledForwardPassOptions): void {


        // Clear buffers (redundant but safe)
        encoder.clearBuffer(this.pipelineStatsBuffer);
        encoder.clearBuffer(this.tileKeysBuffer);
        encoder.clearBuffer(this.tileIndicesBuffer);

        // Count tiles per Gaussian
        {
            const pass = encoder.beginComputePass({ label: 'count-pass' });
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, this.countCameraBindGroup);
            pass.setBindGroup(1, this.countSceneBindGroup);
            pass.dispatchWorkgroups(this.numWorkgroups);
            pass.end();
        }

        // GPU Prefix Sum
        this.prefixScanner.scan(encoder);

        // Update stats
        {
            const pass = encoder.beginComputePass({ label: 'update-stats-pass' });
            pass.setPipeline(this.updateStatsPipeline);
            pass.setBindGroup(0, this.updateStatsBindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
        }

        // Emit key-value pairs
        {
            const pass = encoder.beginComputePass({ label: 'emit-pass' });
            pass.setPipeline(this.emitPipeline);
            pass.setBindGroup(0, this.emitCameraBindGroup);
            pass.setBindGroup(1, this.emitSceneBindGroup);
            pass.setBindGroup(2, this.emitOffsetBindGroup);
            pass.setBindGroup(3, this.emitSortBindGroup);
            pass.dispatchWorkgroups(this.numWorkgroups);
            pass.end();
        }

        // Radix sort
        if (!options?.skipSort) {
            this.sorter.sort(encoder);
        }
    }

    setCameraBuffer(buffer: GPUBuffer): void {
        this.countCameraBindGroup = this.createCountCameraBindGroup(buffer);
    }

    setGaussianScale(value: number): void {
        this.settingsData[0] = value;
        this.flushSettings();
    }

    setPointSize(value: number): void {
        this.settingsData[4] = value;
        this.flushSettings();
    }

    setRenderMode(mode: RenderMode): void {
        this.renderMode = mode;
        this.settingsData[5] = mode === 'gaussian' ? 1.0 : 0.0;
        this.flushSettings();
    }

    setViewport(width: number, height: number): void {
        // Update settings
        this.settingsData[2] = width;
        this.settingsData[3] = height;
        this.flushSettings();

        // Update tile info
        this.numTilesX = Math.ceil(width / TILE_WIDTH);
        this.numTilesY = Math.ceil(height / TILE_HEIGHT);
        this.totalTiles = this.numTilesX * this.numTilesY;

        this.tileInfoData[0] = this.numTilesX;
        this.tileInfoData[1] = this.numTilesY;
        this.tileInfoData[2] = this.totalTiles;

        this.device.queue.writeBuffer(this.tileInfoBuffer, 0, this.tileInfoData);
    }

    // Getters
    getResources(): TiledForwardResources {
        return {
            splatBuffer: this.splatBuffer,
            settingsBuffer: this.settingsBuffer,
            tileInfoBuffer: this.tileInfoBuffer,
            tileKeysBuffer: this.tileKeysBuffer,
            tileIndicesBuffer: this.tileIndicesBuffer,
            sorter: this.sorter,
            numTilesX: this.numTilesX,
            numTilesY: this.numTilesY,
            totalTiles: this.totalTiles,
            maxTileEntries: this.maxTileEntries,
        };
    }

    getSortedIndicesBuffer(): GPUBuffer {
        return this.sorter.ping_pong[this.sorter.final_out_index].sort_indices_buffer;
    }

    getSortedKeysBuffer(): GPUBuffer {
        return this.sorter.ping_pong[this.sorter.final_out_index].sort_depths_buffer;
    }

    getTileOffsetsBuffer(): GPUBuffer {
        return this.tileOffsetsBuffer;
    }

    getStatsBuffer(): GPUBuffer {
        return this.pipelineStatsBuffer;
    }

    // (re)create bind groups
    private createCountCameraBindGroup(cameraBuffer: GPUBuffer): GPUBindGroup {
        return this.device.createBindGroup({
            label: 'count-camera',
            layout: this.countPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cameraBuffer } },
                { binding: 1, resource: { buffer: this.settingsBuffer } },
                { binding: 2, resource: { buffer: this.tileInfoBuffer } },
            ],
        });
    }

    private createEmitCameraBindGroup(cameraBuffer: GPUBuffer): GPUBindGroup {
        return this.device.createBindGroup({
            label: 'emit-camera',
            layout: this.emitPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 1, resource: { buffer: this.settingsBuffer } },
                { binding: 2, resource: { buffer: this.tileInfoBuffer } },
            ],
        });
    }

    // flush settings
    private flushSettings(): void {
        this.device.queue.writeBuffer(this.settingsBuffer, 0, this.settingsData);
    }

    // update sorter dispatch
    private updateSorterDispatch(actualCount?: number): void {
        // Use actual count if provided, otherwise use maxTileEntries
        const keysize = actualCount ?? this.maxTileEntries;
        const num_pass = 4; // From sort.ts

        // Scatter/histogram size calculation
        const scatter_block_kvs = C.histogram_wg_size * C.rs_scatter_block_rows;
        const scatter_blocks_ru = Math.floor((keysize + scatter_block_kvs - 1) / scatter_block_kvs);
        const count_ru_scatter = scatter_blocks_ru * scatter_block_kvs;

        const histo_block_kvs = C.histogram_wg_size * C.rs_histogram_block_rows;
        const histo_blocks_ru = Math.floor((count_ru_scatter + histo_block_kvs - 1) / histo_block_kvs);
        const count_ru_histo = histo_blocks_ru * histo_block_kvs;

        // Update full sort_info_buffer structure
        this.device.queue.writeBuffer(
            this.sorter.sort_info_buffer,
            0,
            new Uint32Array([keysize, count_ru_histo, num_pass, 0, 0])
        );
        this.device.queue.writeBuffer(
            this.sorter.sort_dispatch_indirect_buffer,
            0,
            new Uint32Array([scatter_blocks_ru, 1, 1])
        );
    }
}
