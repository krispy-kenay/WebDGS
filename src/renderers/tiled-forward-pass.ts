/*
 * Tiled Forward Pass
 * 
 * Projects 3D Gaussians to 2D screen space and assigns them to tiles for efficient per-tile rasterization.
 */

import tiledForwardWGSL from '../shaders/tiled-forward.wgsl';
import updateStatsWGSL from '../shaders/update-stats.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import { get_dynamic_sorter, DynamicSortStuff, C } from '../sort/sort_dynamic';
import { get_prefix_scanner, PrefixScanner } from '../prefix/prefix';
import { PointCloud } from '../utils/load';

export type RenderMode = 'pointcloud' | 'gaussian';

// Constants
const WORKGROUP_SIZE = 256;
const TILE_WIDTH = 16;
const TILE_HEIGHT = 16;
// Sort block size, tile-entry capacities are multiples of it
const KEYS_PER_SORT_BLOCK = C.histogram_wg_size * C.rs_histogram_block_rows;

// Largest screen radius a splat keeps (box plus margin then spans at most 44 x 44 tiles)
export const MAX_SPLAT_RADIUS_PX = 336;

// Initial sort capacity per Gaussian, grown on demand (see syncCapacity)
const INITIAL_TILES_PER_GAUSSIAN = 10;

export interface TiledForwardPassConfig {
    viewportWidth: number;
    viewportHeight: number;
    gaussianScale?: number;
    pointSizePx?: number;
    maxSplatRadiusPx?: number;
    renderMode?: RenderMode;
    // Initial capacity for tile entries
    tileEntriesHint?: number;
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
    tileOffsetsBuffer: GPUBuffer;
    tileCountsBuffer: GPUBuffer;
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
    private sorter: DynamicSortStuff;
    private readonly prefixScanner: PrefixScanner;

    // Bind Groups
    private countCameraBindGroup: GPUBindGroup;
    private readonly countSceneBindGroup: GPUBindGroup;
    private emitCameraBindGroup: GPUBindGroup;
    private readonly emitSceneBindGroup: GPUBindGroup;
    private readonly emitOffsetBindGroup: GPUBindGroup;
    private emitSortBindGroup: GPUBindGroup;
    private readonly updateStatsBindGroup: GPUBindGroup;

    // Buffers
    private readonly splatBuffer: GPUBuffer;
    private readonly depthsBuffer: GPUBuffer;
    private readonly tileCountsBuffer: GPUBuffer;
    private readonly tileOffsetsBuffer: GPUBuffer;
    private tileKeysBuffer: GPUBuffer;
    private tileIndicesBuffer: GPUBuffer;
    private readonly settingsBuffer: GPUBuffer;
    private readonly tileInfoBuffer: GPUBuffer;
    private readonly pipelineStatsBuffer: GPUBuffer;
    private readonly numGaussiansBuffer: GPUBuffer;
    // Copy of the pipeline stats of the last encoded frame
    private readonly statsReadbackBuffer: GPUBuffer;
    private statsReadbackState: 'idle' | 'copied' | 'mapping' = 'idle';

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
    private readonly maxTileEntriesLimit: number;
    private lastRequestedTileEntries = 0;

    // Render mode
    private renderMode: RenderMode;

    private destroyed = false;

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

        // Every sort buffer (4 bytes per entry, payload one block larger) must fit in one buffer / storage binding.
        const bytesLimit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
        this.maxTileEntriesLimit = (Math.floor(bytesLimit / 4 / KEYS_PER_SORT_BLOCK) - 1) * KEYS_PER_SORT_BLOCK;
        const estimate = config.tileEntriesHint ?? this.numPoints * INITIAL_TILES_PER_GAUSSIAN;
        this.maxTileEntries = this.tileEntryCapacity(estimate);

        console.log(`maxTileEntries: ${this.maxTileEntries.toLocaleString()} ` +
            `(estimate: ${Math.round(estimate).toLocaleString()}, limit: ${this.maxTileEntriesLimit.toLocaleString()})`);

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
            this.renderMode === 'gaussian' ? 1.0 : 0.0,
            config.maxSplatRadiusPx ?? MAX_SPLAT_RADIUS_PX,
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

        // Num gaussians + tile-entry capacity (update-stats)
        this.numGaussiansBuffer = createBuffer(
            device, 'num-gaussians', 16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            new Uint32Array([this.numPoints, this.maxTileEntries, 0, 0])
        );

        this.statsReadbackBuffer = createBuffer(
            device, 'pipeline-stats-readback', 16,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
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

        this.emitSortBindGroup = this.createEmitSortBindGroup();

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

        // Keep this frame's tile-entry count for syncCapacity()
        if (this.statsReadbackState === 'idle') {
            encoder.copyBufferToBuffer(this.pipelineStatsBuffer, 0, this.statsReadbackBuffer, 0, 16);
            this.statsReadbackState = 'copied';
        }
    }

    // Reads back how many tile entries the last encoded frame needed (call after that frame was submitted) and grows the sort buffers if they did not fit.
    async syncCapacity(): Promise<void> {
        if (this.statsReadbackState !== 'copied' || this.destroyed) return;
        this.statsReadbackState = 'mapping';
        try {
            await this.statsReadbackBuffer.mapAsync(GPUMapMode.READ);
        } catch {
            this.statsReadbackState = 'idle';
            return;
        }
        const requested = new Uint32Array(this.statsReadbackBuffer.getMappedRange())[2];
        this.statsReadbackBuffer.unmap();
        this.statsReadbackState = 'idle';
        this.lastRequestedTileEntries = requested;
        if (requested > this.maxTileEntries && !this.destroyed) {
            this.growTileEntries(requested);
        }
    }

    // Tile entries needed by the last frame read back by syncCapacity(), and the current capacity / limit
    getTileEntryStats(): { requested: number; capacity: number; limit: number } {
        return { requested: this.lastRequestedTileEntries, capacity: this.maxTileEntries, limit: this.maxTileEntriesLimit };
    }

    private tileEntryCapacity(entries: number): number {
        const blocks = Math.ceil(Math.max(1, entries) / KEYS_PER_SORT_BLOCK);
        return Math.min(blocks * KEYS_PER_SORT_BLOCK, this.maxTileEntriesLimit);
    }

    private growTileEntries(requested: number): void {
        const capacity = this.tileEntryCapacity(requested * 1.5);
        if (capacity <= this.maxTileEntries) {
            return;
        }
        console.log(`maxTileEntries: ${this.maxTileEntries.toLocaleString()} -> ${capacity.toLocaleString()} (needed ${requested.toLocaleString()})`);

        this.sorter.destroy();
        this.maxTileEntries = capacity;
        this.sorter = get_dynamic_sorter(capacity, this.device, this.pipelineStatsBuffer);
        this.tileKeysBuffer = this.sorter.ping_pong[0].sort_depths_buffer;
        this.tileIndicesBuffer = this.sorter.ping_pong[0].sort_indices_buffer;
        this.emitSortBindGroup = this.createEmitSortBindGroup();

        this.tileInfoData[3] = capacity;
        this.device.queue.writeBuffer(this.tileInfoBuffer, 0, this.tileInfoData);
        this.device.queue.writeBuffer(this.numGaussiansBuffer, 0, new Uint32Array([this.numPoints, capacity, 0, 0]));
        this.updateSorterDispatch();
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
            tileOffsetsBuffer: this.tileOffsetsBuffer,
            tileCountsBuffer: this.tileCountsBuffer,
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

    private createEmitSortBindGroup(): GPUBindGroup {
        return this.device.createBindGroup({
            label: 'emit-sort',
            layout: this.emitPipeline.getBindGroupLayout(3),
            entries: [
                { binding: 0, resource: { buffer: this.tileKeysBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
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

    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;

        this.splatBuffer.destroy();
        this.depthsBuffer.destroy();
        this.tileCountsBuffer.destroy();
        this.tileOffsetsBuffer.destroy();
        this.settingsBuffer.destroy();
        this.tileInfoBuffer.destroy();
        this.pipelineStatsBuffer.destroy();
        this.numGaussiansBuffer.destroy();
        this.statsReadbackBuffer.destroy();

        this.prefixScanner.destroy();
        this.sorter.destroy();
    }
}
