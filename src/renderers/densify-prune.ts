import { PointCloud } from '../utils/load';
import decideWGSL from '../shaders/densify-prune-decide.wgsl';
import totalWGSL from '../shaders/densify-prune-total.wgsl';
import capWGSL from '../shaders/densify-prune-cap.wgsl';
import scatterGaussiansWGSL from '../shaders/densify-prune-scatter-gaussians.wgsl';
import scatterOptPosWGSL from '../shaders/densify-prune-scatter-opt-pos.wgsl';
import scatterOptVec4WGSL from '../shaders/densify-prune-scatter-opt-vec4.wgsl';
import scatterOptFloatWGSL from '../shaders/densify-prune-scatter-opt-float.wgsl';
import scatterOptShWGSL from '../shaders/densify-prune-scatter-opt-sh.wgsl';
import scatterOptScaleWGSL from '../shaders/densify-prune-scatter-opt-scale.wgsl';
import { get_prefix_scanner, PrefixScanner } from '../prefix/prefix';
import { OptimizerStateBuffers } from './optimizer';

export type DensifyPruneStrategy = 'cpu_rebuild' | 'gpu_rebuild';

export type DensifyPruneAction = 0 | 1 | 2 | 3;

export interface DensifyPruneConfig {
    strategy: DensifyPruneStrategy;
    numViews?: number;
    cloneThreshold?: number;
    splitThreshold?: number;
    pruneThreshold?: number;
    maxNewPointsPerStep?: number;
    maxBufferBytes?: number;
}

export interface DensifyPruneInputs {
    pointCloud: PointCloud;
    metricCountsBuffer?: GPUBuffer;
    metricMapTexture?: GPUTextureView;
    pruningScoreBuffer?: GPUBuffer;
    maxRadiiBuffer?: GPUBuffer;
    gradAccumBuffer?: GPUBuffer;
}

export interface DensifyPruneOutputs {
    actionBuffer: GPUBuffer;
    outCountBuffer: GPUBuffer;
}

export interface DensifyPrunePrepared {
    actionBuffer: GPUBuffer;
    outCountBuffer: GPUBuffer;
    outOffsetBuffer: GPUBuffer;
    outTotalBuffer: GPUBuffer;
    maxOutPoints: number;
}

export interface DensifyPruneScatterTargets {
    outPointCloud: PointCloud;
    outOptimizerState?: OptimizerStateBuffers;
}

export interface DensifyPruneScatterInputs {
    pointCloud: PointCloud;
    optimizerState?: OptimizerStateBuffers;
    outOffsetBuffer: GPUBuffer;
    outNumPoints: number;
    resetNewOptimizerState?: boolean;
}

function createBuffer(
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer | ArrayBufferView
): GPUBuffer {
    const buffer = device.createBuffer({ label, size, usage });
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
}

export class DensifyPrunePass {
    private readonly device: GPUDevice;
    private config: DensifyPruneConfig;

    private numPoints = 0;

    private readonly decidePipeline: GPUComputePipeline;
    private readonly decideConfigBuffer: GPUBuffer;

    private readonly capPipeline: GPUComputePipeline;
    private readonly capInfoBuffer: GPUBuffer;

    private readonly totalPipeline: GPUComputePipeline;
    private readonly totalInfoBuffer: GPUBuffer;
    private readonly outTotalBuffer: GPUBuffer;

    private readonly scatterGaussiansPipeline: GPUComputePipeline;
    private readonly scatterOptPosPipeline: GPUComputePipeline;
    private readonly scatterOptVec4Pipeline: GPUComputePipeline;
    private readonly scatterOptFloatPipeline: GPUComputePipeline;
    private readonly scatterOptShPipeline: GPUComputePipeline;
    private readonly scatterOptScalePipeline: GPUComputePipeline;
    private readonly scatterInfoBuffer: GPUBuffer;

    private actionBuffer: GPUBuffer;
    private outCountBuffer: GPUBuffer;

    private readonly dummyU32Buffer: GPUBuffer;
    private readonly dummyF32Buffer: GPUBuffer;

    private prefixScanner?: PrefixScanner;
    private prefixScannerCapacity = 0;

    constructor(device: GPUDevice, config?: Partial<DensifyPruneConfig>) {
        this.device = device;
        this.config = {
            strategy: 'cpu_rebuild',
            numViews: 1,
            cloneThreshold: 0,
            splitThreshold: 0,
            pruneThreshold: 0,
            maxNewPointsPerStep: 0,
            maxBufferBytes: 128 * 1024 * 1024,
            ...config,
        };

        const decideModule = device.createShaderModule({
            label: 'densify-prune decide',
            code: decideWGSL,
        });
        this.decidePipeline = device.createComputePipeline({
            label: 'densify-prune decide',
            layout: 'auto',
            compute: { module: decideModule, entryPoint: 'decide_main' },
        });

        const capModule = device.createShaderModule({
            label: 'densify-prune cap',
            code: capWGSL,
        });
        this.capPipeline = device.createComputePipeline({
            label: 'densify-prune cap',
            layout: 'auto',
            compute: { module: capModule, entryPoint: 'cap_main' },
        });

        const totalModule = device.createShaderModule({
            label: 'densify-prune total',
            code: totalWGSL,
        });
        this.totalPipeline = device.createComputePipeline({
            label: 'densify-prune total',
            layout: 'auto',
            compute: { module: totalModule, entryPoint: 'total_main' },
        });

        this.decideConfigBuffer = createBuffer(
            device,
            'densify-prune decide config',
            32,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );

        this.capInfoBuffer = createBuffer(
            device,
            'densify-prune cap info',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );

        this.scatterInfoBuffer = createBuffer(
            device,
            'densify-prune scatter info',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );

        this.totalInfoBuffer = createBuffer(
            device,
            'densify-prune total info',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );

        const scatterGaussModule = device.createShaderModule({
            label: 'densify-prune scatter gaussians',
            code: scatterGaussiansWGSL,
        });
        this.scatterGaussiansPipeline = device.createComputePipeline({
            label: 'densify-prune scatter gaussians',
            layout: 'auto',
            compute: { module: scatterGaussModule, entryPoint: 'scatter_gaussians_main' },
        });

        const scatterOptPosModule = device.createShaderModule({
            label: 'densify-prune scatter opt pos',
            code: scatterOptPosWGSL,
        });
        this.scatterOptPosPipeline = device.createComputePipeline({
            label: 'densify-prune scatter opt pos',
            layout: 'auto',
            compute: { module: scatterOptPosModule, entryPoint: 'scatter_opt_pos_main' },
        });

        const scatterOptVec4Module = device.createShaderModule({
            label: 'densify-prune scatter opt vec4',
            code: scatterOptVec4WGSL,
        });
        this.scatterOptVec4Pipeline = device.createComputePipeline({
            label: 'densify-prune scatter opt vec4',
            layout: 'auto',
            compute: { module: scatterOptVec4Module, entryPoint: 'scatter_opt_vec4_main' },
        });

        const scatterOptFloatModule = device.createShaderModule({
            label: 'densify-prune scatter opt float',
            code: scatterOptFloatWGSL,
        });
        this.scatterOptFloatPipeline = device.createComputePipeline({
            label: 'densify-prune scatter opt float',
            layout: 'auto',
            compute: { module: scatterOptFloatModule, entryPoint: 'scatter_opt_float_main' },
        });

        const scatterOptShModule = device.createShaderModule({
            label: 'densify-prune scatter opt sh',
            code: scatterOptShWGSL,
        });
        this.scatterOptShPipeline = device.createComputePipeline({
            label: 'densify-prune scatter opt sh',
            layout: 'auto',
            compute: { module: scatterOptShModule, entryPoint: 'scatter_opt_sh_main' },
        });

        const scatterOptScaleModule = device.createShaderModule({
            label: 'densify-prune scatter opt scale',
            code: scatterOptScaleWGSL,
        });
        this.scatterOptScalePipeline = device.createComputePipeline({
            label: 'densify-prune scatter opt scale',
            layout: 'auto',
            compute: { module: scatterOptScaleModule, entryPoint: 'scatter_opt_scale_main' },
        });

        this.actionBuffer = createBuffer(
            device,
            'densify-prune-actions',
            4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            new Uint32Array([0])
        );

        this.outCountBuffer = createBuffer(
            device,
            'densify-prune-out-counts',
            4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            new Uint32Array([1])
        );

        this.outTotalBuffer = createBuffer(
            device,
            'densify-prune total out',
            4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            new Uint32Array([0])
        );

        this.dummyU32Buffer = createBuffer(
            device,
            'densify-prune dummy u32',
            4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            new Uint32Array([0])
        );
        this.dummyF32Buffer = createBuffer(
            device,
            'densify-prune dummy f32',
            4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            new Float32Array([0])
        );
    }

    setConfig(next: Partial<DensifyPruneConfig>): void {
        this.config = { ...this.config, ...next };
    }

    getConfig(): Readonly<DensifyPruneConfig> {
        return this.config;
    }

    ensureSize(numPoints: number): void {
        if (numPoints <= 0) numPoints = 1;
        if (this.numPoints === numPoints) return;
        this.numPoints = numPoints;
        this.actionBuffer.destroy();
        this.outCountBuffer.destroy();
        this.actionBuffer = createBuffer(
            this.device,
            'densify-prune-actions',
            this.numPoints * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            new Uint32Array(this.numPoints)
        );
        this.outCountBuffer = createBuffer(
            this.device,
            'densify-prune-out-counts',
            this.numPoints * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            new Uint32Array(this.numPoints)
        );

        if (this.prefixScanner && this.prefixScannerCapacity < this.numPoints) {
            this.prefixScanner = undefined;
            this.prefixScannerCapacity = 0;
        }
    }

    getActionBuffer(): GPUBuffer {
        return this.actionBuffer;
    }

    getOutCountBuffer(): GPUBuffer {
        return this.outCountBuffer;
    }

    getOutTotalBuffer(): GPUBuffer {
        return this.outTotalBuffer;
    }

    /** Exclusive scan over `outCountBuffer` into an internal prefix output buffer. */
    encodePrefixSum(encoder: GPUCommandEncoder): GPUBuffer {
        if (!this.prefixScanner || this.prefixScannerCapacity < this.numPoints) {
            this.prefixScannerCapacity = Math.max(1, this.numPoints);
            this.prefixScanner = get_prefix_scanner(this.prefixScannerCapacity, this.device);
        }
        const scanner = this.prefixScanner;
        scanner.set_count(this.numPoints);
        encoder.copyBufferToBuffer(this.outCountBuffer, 0, scanner.input_buffer, 0, this.numPoints * 4);
        scanner.scan(encoder);
        return scanner.output_buffer;
    }

    encodeTotalOut(encoder: GPUCommandEncoder, outOffsetBuffer: GPUBuffer): GPUBuffer {
        const uni = new Uint32Array([this.numPoints, 0, 0, 0]);
        this.device.queue.writeBuffer(this.totalInfoBuffer, 0, uni);

        const bindGroup = this.device.createBindGroup({
            label: 'densify-prune total',
            layout: this.totalPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.totalInfoBuffer } },
                { binding: 1, resource: { buffer: outOffsetBuffer } },
                { binding: 2, resource: { buffer: this.outCountBuffer } },
                { binding: 3, resource: { buffer: this.outTotalBuffer } },
            ],
        });

        const pass = encoder.beginComputePass({ label: 'densify-prune total' });
        pass.setPipeline(this.totalPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        return this.outTotalBuffer;
    }

    encodeCapToMax(
        encoder: GPUCommandEncoder,
        outOffsetBuffer: GPUBuffer,
        maxOutPoints: number
    ): void {
        const maxOut = Math.max(0, Math.floor(maxOutPoints));
        const uni = new Uint32Array([this.numPoints, maxOut, 0, 0]);
        this.device.queue.writeBuffer(this.capInfoBuffer, 0, uni);

        const bindGroup = this.device.createBindGroup({
            label: 'densify-prune cap',
            layout: this.capPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.capInfoBuffer } },
                { binding: 1, resource: { buffer: outOffsetBuffer } },
                { binding: 2, resource: { buffer: this.outCountBuffer } },
                { binding: 3, resource: { buffer: this.actionBuffer } },
            ],
        });

        const pass = encoder.beginComputePass({ label: 'densify-prune cap' });
        pass.setPipeline(this.capPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numPoints / 256));
        pass.end();
    }

    private computeMaxOutPoints(pointCloud: PointCloud): number {
        const maxBytes = this.config.maxBufferBytes ?? 128 * 1024 * 1024;
        const safeBytes = Math.max(0, Math.floor(maxBytes));
        const n = Math.max(1, pointCloud.num_points);

        const bytesPerGaussian = Math.max(1, Math.floor(pointCloud.gaussian_3d_buffer.size / n));
        let maxFromGaussian = Math.floor(safeBytes / bytesPerGaussian);

        if (pointCloud.sh_buffer) {
            const bytesPerSh = Math.max(1, Math.floor(pointCloud.sh_buffer.size / n));
            const maxFromSh = Math.floor(safeBytes / bytesPerSh);
            maxFromGaussian = Math.min(maxFromGaussian, maxFromSh);
        }

        const maxNew = Math.max(0, Math.floor(this.config.maxNewPointsPerStep ?? 0));
        if (maxNew > 0) {
            maxFromGaussian = Math.min(maxFromGaussian, n + maxNew);
        }

        return Math.max(0, maxFromGaussian);
    }

    encodeDecision(encoder: GPUCommandEncoder, inputs: DensifyPruneInputs): DensifyPruneOutputs {
        this.ensureSize(inputs.pointCloud.num_points);

        const numViews = Math.max(1, Math.floor(this.config.numViews ?? 1));
        const cloneThreshold = Math.max(0, Math.floor(this.config.cloneThreshold ?? 0));
        const pruneOpacity = this.config.pruneThreshold ?? 0;
        const splitScaleThreshold = this.config.splitThreshold ?? 1e9;

        const uni = new ArrayBuffer(32);
        const u32 = new Uint32Array(uni);
        const f32 = new Float32Array(uni);
        u32[0] = this.numPoints;
        u32[1] = numViews;
        u32[2] = cloneThreshold;
        u32[3] = 0;
        f32[4] = pruneOpacity;
        f32[5] = splitScaleThreshold;
        f32[6] = 0;
        f32[7] = 0;
        this.device.queue.writeBuffer(this.decideConfigBuffer, 0, uni);

        const bindGroup = this.device.createBindGroup({
            label: 'densify-prune decide',
            layout: this.decidePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.decideConfigBuffer } },
                { binding: 1, resource: { buffer: inputs.pointCloud.gaussian_3d_buffer } },
                { binding: 2, resource: { buffer: inputs.metricCountsBuffer ?? this.dummyU32Buffer } },
                { binding: 3, resource: { buffer: inputs.pruningScoreBuffer ?? this.dummyF32Buffer } },
                { binding: 4, resource: { buffer: inputs.maxRadiiBuffer ?? this.dummyF32Buffer } },
                { binding: 5, resource: { buffer: inputs.gradAccumBuffer ?? this.dummyF32Buffer } },
                { binding: 6, resource: { buffer: this.outCountBuffer } },
                { binding: 7, resource: { buffer: this.actionBuffer } },
            ],
        });

        const pass = encoder.beginComputePass({ label: 'densify-prune decide' });
        pass.setPipeline(this.decidePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numPoints / 256));
        pass.end();

        return { actionBuffer: this.actionBuffer, outCountBuffer: this.outCountBuffer };
    }

    /** Convenience milestone-1 encoder: decide + prefix sum + total_out. */
    encodePrepare(encoder: GPUCommandEncoder, inputs: DensifyPruneInputs): DensifyPrunePrepared {
        const maxOutPoints = this.computeMaxOutPoints(inputs.pointCloud);
        const outputs = this.encodeDecision(encoder, inputs);

        // First scan to compute offsets; clamp counts/actions to capacity; rescan to get final offsets.
        const outOffsetBufferPre = this.encodePrefixSum(encoder);
        this.encodeCapToMax(encoder, outOffsetBufferPre, maxOutPoints);
        const outOffsetBuffer = this.encodePrefixSum(encoder);
        const outTotalBuffer = this.encodeTotalOut(encoder, outOffsetBuffer);
        return { ...outputs, outOffsetBuffer, outTotalBuffer, maxOutPoints };
    }

    encodeScatter(
        encoder: GPUCommandEncoder,
        inputs: DensifyPruneScatterInputs,
        targets: DensifyPruneScatterTargets
    ): void {
        const inPoints = Math.max(1, inputs.pointCloud.num_points);
        const outPoints = Math.max(1, Math.floor(inputs.outNumPoints));

        if (targets.outPointCloud.num_points !== outPoints) {
            throw new Error(`encodeScatter: outPointCloud.num_points (${targets.outPointCloud.num_points}) != outNumPoints (${outPoints})`);
        }

        const resetNew = inputs.resetNewOptimizerState ?? true;
        const uni = new Uint32Array([inPoints, outPoints, resetNew ? 1 : 0, 0]);
        this.device.queue.writeBuffer(this.scatterInfoBuffer, 0, uni);

        // Packed gaussians + SH
        {
            const bg = this.device.createBindGroup({
                label: 'densify-prune scatter gaussians',
                layout: this.scatterGaussiansPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.scatterInfoBuffer } },
                    { binding: 1, resource: { buffer: inputs.pointCloud.gaussian_3d_buffer } },
                    { binding: 2, resource: { buffer: inputs.pointCloud.sh_buffer ?? this.dummyU32Buffer } },
                    { binding: 3, resource: { buffer: inputs.outOffsetBuffer } },
                    { binding: 4, resource: { buffer: this.outCountBuffer } },
                    { binding: 5, resource: { buffer: this.actionBuffer } },
                    { binding: 6, resource: { buffer: targets.outPointCloud.gaussian_3d_buffer } },
                    { binding: 7, resource: { buffer: targets.outPointCloud.sh_buffer ?? this.dummyU32Buffer } },
                ],
            });
            const pass = encoder.beginComputePass({ label: 'densify-prune scatter gaussians' });
            pass.setPipeline(this.scatterGaussiansPipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(inPoints / 256));
            pass.end();
        }

        // Optimizer state (optional)
        if (inputs.optimizerState && targets.outOptimizerState) {
            const dispatchCount = Math.ceil(inPoints / 256);

            // Position: needs access to rot/scale so clone/split perturbations persist through update-gaussians.
            {
                const bg0 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt pos bg0',
                    layout: this.scatterOptPosPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.scatterInfoBuffer } },
                        { binding: 1, resource: { buffer: inputs.outOffsetBuffer } },
                        { binding: 2, resource: { buffer: this.outCountBuffer } },
                        { binding: 3, resource: { buffer: this.actionBuffer } },
                    ],
                });
                const bg1 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt pos bg1',
                    layout: this.scatterOptPosPipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: inputs.optimizerState.optPosBuffer } },
                        { binding: 1, resource: { buffer: inputs.optimizerState.optRotBuffer } },
                        { binding: 2, resource: { buffer: inputs.optimizerState.optScaleBuffer } },
                    ],
                });
                const bg2 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt pos bg2',
                    layout: this.scatterOptPosPipeline.getBindGroupLayout(2),
                    entries: [{ binding: 0, resource: { buffer: targets.outOptimizerState.optPosBuffer } }],
                });
                const pass = encoder.beginComputePass({ label: 'densify-prune scatter opt pos' });
                pass.setPipeline(this.scatterOptPosPipeline);
                pass.setBindGroup(0, bg0);
                pass.setBindGroup(1, bg1);
                pass.setBindGroup(2, bg2);
                pass.dispatchWorkgroups(dispatchCount);
                pass.end();
            }

            // Rotation (generic vec4 scatter)
            {
                const bg0 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt rot bg0',
                    layout: this.scatterOptVec4Pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.scatterInfoBuffer } },
                        { binding: 1, resource: { buffer: inputs.outOffsetBuffer } },
                        { binding: 2, resource: { buffer: this.outCountBuffer } },
                        { binding: 3, resource: { buffer: this.actionBuffer } },
                    ],
                });
                const bg1 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt rot bg1',
                    layout: this.scatterOptVec4Pipeline.getBindGroupLayout(1),
                    entries: [{ binding: 0, resource: { buffer: inputs.optimizerState.optRotBuffer } }],
                });
                const bg2 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt rot bg2',
                    layout: this.scatterOptVec4Pipeline.getBindGroupLayout(2),
                    entries: [{ binding: 0, resource: { buffer: targets.outOptimizerState.optRotBuffer } }],
                });
                const pass = encoder.beginComputePass({ label: 'densify-prune scatter opt rot' });
                pass.setPipeline(this.scatterOptVec4Pipeline);
                pass.setBindGroup(0, bg0);
                pass.setBindGroup(1, bg1);
                pass.setBindGroup(2, bg2);
                pass.dispatchWorkgroups(dispatchCount);
                pass.end();
            }

            // Scale: shrink for split + action-aware reset.
            {
                const bg0 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt scale bg0',
                    layout: this.scatterOptScalePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.scatterInfoBuffer } },
                        { binding: 1, resource: { buffer: inputs.outOffsetBuffer } },
                        { binding: 2, resource: { buffer: this.outCountBuffer } },
                        { binding: 3, resource: { buffer: this.actionBuffer } },
                    ],
                });
                const bg1 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt scale bg1',
                    layout: this.scatterOptScalePipeline.getBindGroupLayout(1),
                    entries: [{ binding: 0, resource: { buffer: inputs.optimizerState.optScaleBuffer } }],
                });
                const bg2 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt scale bg2',
                    layout: this.scatterOptScalePipeline.getBindGroupLayout(2),
                    entries: [{ binding: 0, resource: { buffer: targets.outOptimizerState.optScaleBuffer } }],
                });
                const pass = encoder.beginComputePass({ label: 'densify-prune scatter opt scale' });
                pass.setPipeline(this.scatterOptScalePipeline);
                pass.setBindGroup(0, bg0);
                pass.setBindGroup(1, bg1);
                pass.setBindGroup(2, bg2);
                pass.dispatchWorkgroups(dispatchCount);
                pass.end();
            }

            {
                const bg0 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt float bg0',
                    layout: this.scatterOptFloatPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.scatterInfoBuffer } },
                        { binding: 1, resource: { buffer: inputs.outOffsetBuffer } },
                        { binding: 2, resource: { buffer: this.outCountBuffer } },
                        { binding: 3, resource: { buffer: this.actionBuffer } },
                    ],
                });
                const bg1 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt float bg1',
                    layout: this.scatterOptFloatPipeline.getBindGroupLayout(1),
                    entries: [{ binding: 0, resource: { buffer: inputs.optimizerState.optOpacityBuffer } }],
                });
                const bg2 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt float bg2',
                    layout: this.scatterOptFloatPipeline.getBindGroupLayout(2),
                    entries: [{ binding: 0, resource: { buffer: targets.outOptimizerState.optOpacityBuffer } }],
                });

                const pass = encoder.beginComputePass({ label: 'densify-prune scatter opt opacity' });
                pass.setPipeline(this.scatterOptFloatPipeline);
                pass.setBindGroup(0, bg0);
                pass.setBindGroup(1, bg1);
                pass.setBindGroup(2, bg2);
                pass.dispatchWorkgroups(dispatchCount);
                pass.end();
            }

            {
                const bg0 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt sh bg0',
                    layout: this.scatterOptShPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.scatterInfoBuffer } },
                        { binding: 1, resource: { buffer: inputs.outOffsetBuffer } },
                        { binding: 2, resource: { buffer: this.outCountBuffer } },
                        { binding: 3, resource: { buffer: this.actionBuffer } },
                    ],
                });
                const bg1 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt sh bg1',
                    layout: this.scatterOptShPipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: inputs.optimizerState.paramSH } },
                        { binding: 1, resource: { buffer: inputs.optimizerState.stateSH } },
                    ],
                });
                const bg2 = this.device.createBindGroup({
                    label: 'densify-prune scatter opt sh bg2',
                    layout: this.scatterOptShPipeline.getBindGroupLayout(2),
                    entries: [
                        { binding: 0, resource: { buffer: targets.outOptimizerState.paramSH } },
                        { binding: 1, resource: { buffer: targets.outOptimizerState.stateSH } },
                    ],
                });

                const pass = encoder.beginComputePass({ label: 'densify-prune scatter opt sh' });
                pass.setPipeline(this.scatterOptShPipeline);
                pass.setBindGroup(0, bg0);
                pass.setBindGroup(1, bg1);
                pass.setBindGroup(2, bg2);
                pass.dispatchWorkgroups(dispatchCount);
                pass.end();
            }
        }
    }

    async applyActions(inputs: DensifyPruneInputs): Promise<PointCloud> {
        this.ensureSize(inputs.pointCloud.num_points);
        if (this.config.strategy === 'cpu_rebuild') {
            throw new Error('DensifyPrunePass.applyActions(cpu_rebuild) not implemented');
        }
        throw new Error('DensifyPrunePass.applyActions(gpu_rebuild) not implemented');
    }
}
