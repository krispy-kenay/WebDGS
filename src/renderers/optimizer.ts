import { PointCloud } from '../utils/load';
import adamWGSL from '../shaders/adam.wgsl';
import updateGaussiansWGSL from '../shaders/update-gaussians.wgsl';
import { AdamHyperparameters, DEFAULT_ADAM_HYPERPARAMETERS } from './adam-config';
import { TiledBackwardResources } from './tiled-backward-pass';

export const OPTIMIZER_LAYOUT = {
    OPT_VEC4_BYTES: 48,
    OPT_FLOAT_BYTES: 12,
    SH_FLOATS_PER_POINT: 48,
} as const;

export interface OptimizerStateBuffers {
    optPosBuffer: GPUBuffer;
    optRotBuffer: GPUBuffer;
    optScaleBuffer: GPUBuffer;
    optOpacityBuffer: GPUBuffer;
    paramSH: GPUBuffer;
    stateSH: GPUBuffer;
}

export interface OptimizerInitialState {
    iteration?: number;
    buffers: OptimizerStateBuffers;
}

export function allocateOptimizerStateBuffers(device: GPUDevice, numPoints: number): OptimizerStateBuffers {
    const N = Math.max(1, Math.floor(numPoints));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    return {
        optPosBuffer: device.createBuffer({ label: 'opt-pos', size: N * OPTIMIZER_LAYOUT.OPT_VEC4_BYTES, usage }),
        optRotBuffer: device.createBuffer({ label: 'opt-rot', size: N * OPTIMIZER_LAYOUT.OPT_VEC4_BYTES, usage }),
        optScaleBuffer: device.createBuffer({ label: 'opt-scale', size: N * OPTIMIZER_LAYOUT.OPT_VEC4_BYTES, usage }),
        optOpacityBuffer: device.createBuffer({ label: 'opt-opacity', size: N * OPTIMIZER_LAYOUT.OPT_FLOAT_BYTES, usage }),
        paramSH: device.createBuffer({ label: 'param-sh', size: N * OPTIMIZER_LAYOUT.SH_FLOATS_PER_POINT * 4, usage }),
        stateSH: device.createBuffer({ label: 'state-sh', size: N * OPTIMIZER_LAYOUT.SH_FLOATS_PER_POINT * 2 * 4, usage }),
    };
}

export class Optimizer {
    private readonly device: GPUDevice;
    private readonly numPoints: number;

    // Config
    private params: AdamHyperparameters;
    private configBuffer: GPUBuffer;
    private iteration = 0;

    // Pipelines
    private adamPipeline: GPUComputePipeline;
    private updatePipeline: GPUComputePipeline;

    // Combined Optimizer Buffers 
    private optPosBuffer: GPUBuffer;
    private optRotBuffer: GPUBuffer;
    private optScaleBuffer: GPUBuffer;
    private optOpacityBuffer: GPUBuffer;

    // SH remains separate due to size
    private paramSH: GPUBuffer;
    private stateSH: GPUBuffer;

    // Bind Groups
    private adamConfigBindGroup: GPUBindGroup | null = null;
    private adamBuffersBindGroup: GPUBindGroup;

    private updateInputsBindGroup: GPUBindGroup;

    private destroyed = false;

    constructor(device: GPUDevice, pointCloud: PointCloud, params?: Partial<AdamHyperparameters>, initialState?: OptimizerInitialState) {
        this.device = device;
        this.numPoints = pointCloud.num_points;

        // Defaults
        this.params = {
            ...DEFAULT_ADAM_HYPERPARAMETERS,
            ...params
        };

        if (initialState?.buffers) {
            this.iteration = initialState.iteration ?? 0;
            this.optPosBuffer = initialState.buffers.optPosBuffer;
            this.optRotBuffer = initialState.buffers.optRotBuffer;
            this.optScaleBuffer = initialState.buffers.optScaleBuffer;
            this.optOpacityBuffer = initialState.buffers.optOpacityBuffer;
            this.paramSH = initialState.buffers.paramSH;
            this.stateSH = initialState.buffers.stateSH;
        } else {
            // Initialize f32 buffers from point cloud data
            this.initBuffers(pointCloud);
        }

        // Create Shaders/Pipelines
        const adamModule = device.createShaderModule({
            label: 'adam-shader',
            code: adamWGSL
        });
        this.adamPipeline = device.createComputePipeline({
            label: 'adam-pipeline',
            layout: 'auto',
            compute: { module: adamModule, entryPoint: 'main' }
        });

        const updateModule = device.createShaderModule({
            label: 'update-gaussians-shader',
            code: updateGaussiansWGSL
        });
        this.updatePipeline = device.createComputePipeline({
            label: 'update-gaussians-pipeline',
            layout: 'auto',
            compute: { module: updateModule, entryPoint: 'main' }
        });

        // Create Fixed Bind Groups
        this.configBuffer = device.createBuffer({
            label: 'adam-config',
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.updateConfigBuffer();

        // Adam Buffers BG 
        this.adamBuffersBindGroup = device.createBindGroup({
            label: 'adam-buffers-bg',
            layout: this.adamPipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: this.optPosBuffer } },
                { binding: 1, resource: { buffer: this.optRotBuffer } },
                { binding: 2, resource: { buffer: this.optScaleBuffer } },
                { binding: 3, resource: { buffer: this.optOpacityBuffer } },
                { binding: 4, resource: { buffer: this.paramSH } },
                { binding: 5, resource: { buffer: this.stateSH } },
            ]
        });

        // Update Inputs BG
        this.updateInputsBindGroup = device.createBindGroup({
            label: 'update-inputs-bg',
            layout: this.updatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.optPosBuffer } },
                { binding: 1, resource: { buffer: this.optRotBuffer } },
                { binding: 2, resource: { buffer: this.optScaleBuffer } },
                { binding: 3, resource: { buffer: this.optOpacityBuffer } },
                { binding: 4, resource: { buffer: this.paramSH } },
            ]
        });
    }

    private initBuffers(pc: PointCloud) {
        const N = this.numPoints;
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

        // Calculate sizes for interleaved buffers
        this.optPosBuffer = this.device.createBuffer({ size: N * OPTIMIZER_LAYOUT.OPT_VEC4_BYTES, usage });
        this.optRotBuffer = this.device.createBuffer({ size: N * OPTIMIZER_LAYOUT.OPT_VEC4_BYTES, usage });
        this.optScaleBuffer = this.device.createBuffer({ size: N * OPTIMIZER_LAYOUT.OPT_VEC4_BYTES, usage });
        this.optOpacityBuffer = this.device.createBuffer({ size: N * OPTIMIZER_LAYOUT.OPT_FLOAT_BYTES, usage });

        // SH separate
        this.paramSH = this.device.createBuffer({ size: N * OPTIMIZER_LAYOUT.SH_FLOATS_PER_POINT * 4, usage }); // f32
        this.stateSH = this.device.createBuffer({ size: N * OPTIMIZER_LAYOUT.SH_FLOATS_PER_POINT * 2 * 4, usage }); // 2x f32

        // Dispatch Unpacker
        const unpackCode = `
            struct OptVec4 {
                param: vec4<f32>,
                m: vec4<f32>,
                v: vec4<f32>,
            };
            struct OptFloat {
                param: f32,
                m: f32,
                v: f32,
            };

            struct Gaussian {
                pos_opacity: array<u32, 2>,
                rot:         array<u32, 2>,
                scale:       array<u32, 2>,
            };
            @group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
            @group(0) @binding(1) var<storage, read> sh_buffer: array<u32>;
            
            @group(1) @binding(0) var<storage, read_write> opt_pos: array<OptVec4>;
            @group(1) @binding(1) var<storage, read_write> opt_rot: array<OptVec4>;
            @group(1) @binding(2) var<storage, read_write> opt_scale: array<OptVec4>;
            @group(1) @binding(3) var<storage, read_write> opt_opacity: array<OptFloat>;
            @group(1) @binding(4) var<storage, read_write> param_sh: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if (idx >= arrayLength(&opt_pos)) { return; }
                
                let g = gaussians[idx];
                
                // Pos + Opacity
                let po0 = unpack2x16float(g.pos_opacity[0]);
                let po1 = unpack2x16float(g.pos_opacity[1]);
                opt_pos[idx].param = vec4<f32>(po0.x, po0.y, po1.x, 1.0);
                opt_opacity[idx].param = po1.y;

                // Rot
                let r0 = unpack2x16float(g.rot[0]);
                let r1 = unpack2x16float(g.rot[1]);
                opt_rot[idx].param = vec4<f32>(r0.x, r0.y, r1.x, r1.y);

                // Scale
                let s0 = unpack2x16float(g.scale[0]);
                let s1 = unpack2x16float(g.scale[1]);
                opt_scale[idx].param = vec4<f32>(s0.x, s0.y, s1.x, 0.0);

                // SH
                let base = idx * 24u;
                for (var i = 0u; i < 24u; i++) {
                    let pair = unpack2x16float(sh_buffer[base + i]);
                    param_sh[idx * 48u + i * 2u] = pair.x;
                    param_sh[idx * 48u + i * 2u + 1u] = pair.y;
                }
            }
        `;

        const unpackModule = this.device.createShaderModule({ code: unpackCode });
        const unpackPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: unpackModule, entryPoint: 'main' }
        });

        const encoder = this.device.createCommandEncoder({ label: 'init-optimizer' });
        const pass = encoder.beginComputePass();
        pass.setPipeline(unpackPipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
            layout: unpackPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
                { binding: 1, resource: { buffer: pc.sh_buffer } }
            ]
        }));
        pass.setBindGroup(1, this.device.createBindGroup({
            layout: unpackPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.optPosBuffer } },
                { binding: 1, resource: { buffer: this.optRotBuffer } },
                { binding: 2, resource: { buffer: this.optScaleBuffer } },
                { binding: 3, resource: { buffer: this.optOpacityBuffer } },
                { binding: 4, resource: { buffer: this.paramSH } },
            ]
        }));
        pass.dispatchWorkgroups(Math.ceil(N / 256));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    getIteration(): number {
        return this.iteration;
    }

    getHyperparameters(): Readonly<AdamHyperparameters> {
        return this.params;
    }

    setHyperparameters(next: Partial<AdamHyperparameters>): void {
        this.params = { ...this.params, ...next };
        this.updateConfigBuffer();
    }

    getStateBuffers(): OptimizerStateBuffers {
        return {
            optPosBuffer: this.optPosBuffer,
            optRotBuffer: this.optRotBuffer,
            optScaleBuffer: this.optScaleBuffer,
            optOpacityBuffer: this.optOpacityBuffer,
            paramSH: this.paramSH,
            stateSH: this.stateSH,
        };
    }

    private updateConfigBuffer() {
        const data = new Float32Array([
            this.params.lr_pos,
            this.params.lr_color,
            this.params.lr_opacity,
            this.params.lr_scale,
            this.params.lr_rot,
            this.params.beta1,
            this.params.beta2,
            this.params.epsilon
        ]);
        this.device.queue.writeBuffer(this.configBuffer, 0, data);
        this.device.queue.writeBuffer(this.configBuffer, 32, new Uint32Array([this.iteration]));
    }

    step(
        encoder: GPUCommandEncoder,
        coefficients: PointCloud,
        gradientsBuffer: GPUBuffer,
        tileCountsBuffer: GPUBuffer
    ) {
        this.iteration++;
        this.updateConfigBuffer();

        // Adam Step
        this.adamConfigBindGroup = this.device.createBindGroup({
            label: 'adam-config-bg',
            layout: this.adamPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.configBuffer } },
                { binding: 1, resource: { buffer: tileCountsBuffer } }
            ]
        });

        const adamGradsBindGroup = this.device.createBindGroup({
            label: 'adam-grads-bg',
            layout: this.adamPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: gradientsBuffer } }
            ]
        });

        {
            const pass = encoder.beginComputePass({ label: 'adam-step' });
            pass.setPipeline(this.adamPipeline);
            pass.setBindGroup(0, this.adamConfigBindGroup!);
            pass.setBindGroup(1, adamGradsBindGroup);
            pass.setBindGroup(2, this.adamBuffersBindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numPoints / 256));
            pass.end();
        }

        // Update Gaussians
        {
            const outputBindGroup = this.device.createBindGroup({
                label: 'update-output-bg',
                layout: this.updatePipeline.getBindGroupLayout(1),
                entries: [
                    { binding: 0, resource: { buffer: coefficients.gaussian_3d_buffer } },
                    { binding: 1, resource: { buffer: coefficients.sh_buffer } }
                ]
            });

            const pass = encoder.beginComputePass({ label: 'update-gaussians' });
            pass.setPipeline(this.updatePipeline);
            pass.setBindGroup(0, this.updateInputsBindGroup);
            pass.setBindGroup(1, outputBindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numPoints / 256));
            pass.end();
        }
    }

    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.configBuffer.destroy();
        this.optPosBuffer.destroy();
        this.optRotBuffer.destroy();
        this.optScaleBuffer.destroy();
        this.optOpacityBuffer.destroy();
        this.paramSH.destroy();
        this.stateSH.destroy();
    }
}
