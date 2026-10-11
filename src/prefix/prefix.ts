/*
    GPU Parallel Prefix Sum (Exclusive Scan) 

    Usage follows the same pattern as sort.ts:

    1. Create scanner: const scanner = get_prefix_scanner(max_elements, device)
    2. Set count: scanner.set_count(count)
    3. Write input: device.queue.writeBuffer(scanner.input_buffer, 0, inputData)
    4. Scan: scanner.scan(encoder)
    5. Read output: scanner.output_buffer contains exclusive prefix sums

    The scanner also exposes the total sum in output_buffer[count-1] + input[count-1] if needed.
    There is no size limit: block totals are scanned recursively, one level per factor of 512.
*/

import prefix_sum_wgsl from './prefix_sum.wgsl';

const WORKGROUP_SIZE = 256;
const ELEMENTS_PER_BLOCK = WORKGROUP_SIZE * 2;  // 512 elements per workgroup

export interface PrefixScannerConfig {
    maxElements: number;
}

export interface PrefixScanner {
    // Encode the prefix sum passes to a command encoder
    scan: (encoder: GPUCommandEncoder) => void;
    // Uniform buffer containing count and num_workgroups
    info_buffer: GPUBuffer;
    // Input buffer - write your u32 values here before scanning
    input_buffer: GPUBuffer;
    // Output buffer - contains exclusive prefix sums after scanning
    output_buffer: GPUBuffer;
    // Block sums buffer of the first level
    block_sums_buffer: GPUBuffer;
    // Maximum number of elements this scanner can handle
    max_elements: number;
    // Update the element count and dispatch the scan (helper)
    set_count: (count: number) => { num_workgroups: number };
    // Destroy owned GPU buffers
    destroy: () => void;
}

// Configuration constants
export const PREFIX_CONSTANTS = {
    WORKGROUP_SIZE,
    ELEMENTS_PER_BLOCK,
    // Kept for compatibility
    MAX_ELEMENTS: 8192 * ELEMENTS_PER_BLOCK,
};

function create_pipelines(device: GPUDevice) {
    const module = device.createShaderModule({
        label: 'prefix sum',
        code: prefix_sum_wgsl,
    });

    const bind_group_layout = device.createBindGroupLayout({
        label: 'prefix sum',
        entries: [
            // info uniform
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            },
            // input
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            },
            // output
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            // block_sums
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });

    const pipeline_layout = device.createPipelineLayout({
        label: 'prefix sum',
        bindGroupLayouts: [bind_group_layout],
    });

    return {
        bind_group_layout,
        local_scan: device.createComputePipeline({
            label: 'prefix sum - local scan',
            layout: pipeline_layout,
            compute: {
                module,
                entryPoint: 'local_scan',
                constants: {
                    WORKGROUP_SIZE,
                },
            },
        }),
        add_offsets: device.createComputePipeline({
            label: 'prefix sum - add offsets',
            layout: pipeline_layout,
            compute: {
                module,
                entryPoint: 'add_offsets',
                constants: {
                    WORKGROUP_SIZE,
                },
            },
        }),
    };
}

/**
 * Create a GPU prefix sum scanner.
 *
 * @param maxElements - Maximum number of elements to scan
 * @param device - WebGPU device
 * @returns PrefixScanner object with scan method and buffers
 */
export function get_prefix_scanner(maxElements: number, device: GPUDevice): PrefixScanner {
    const pipelines = create_pipelines(device);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;

    const input_buffer = device.createBuffer({
        label: 'prefix sum input',
        size: Math.max(maxElements, 1) * 4,
        usage,
    });

    const output_buffer = device.createBuffer({
        label: 'prefix sum output',
        size: Math.max(maxElements, 1) * 4,
        usage,
    });

    // One level per factor of 512. level L scans input, writes output and its block totals to block_sums,
    // which are the input of level L + 1. The last level has a single block.
    interface Level {
        info: GPUBuffer;
        input: GPUBuffer;
        output: GPUBuffer;
        block_sums: GPUBuffer;
        scan_bind_group?: GPUBindGroup;
        add_bind_group?: GPUBindGroup;
    }
    const levels: Level[] = [];
    let capacity = Math.max(maxElements, 1);
    let levelInput = input_buffer;
    let levelOutput = output_buffer;
    for (;;) {
        const blocks = Math.ceil(capacity / ELEMENTS_PER_BLOCK);
        const level: Level = {
            info: device.createBuffer({
                label: `prefix sum info L${levels.length}`,
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),
            input: levelInput,
            output: levelOutput,
            block_sums: device.createBuffer({
                label: `prefix sum block sums L${levels.length}`,
                size: Math.max(blocks * 4, 16),
                usage,
            }),
        };
        levels.push(level);
        if (blocks <= 1) break;
        capacity = blocks;
        levelInput = level.block_sums;
        levelOutput = device.createBuffer({ label: `prefix sum output L${levels.length}`, size: blocks * 4, usage });
    }

    for (let l = 0; l < levels.length; l++) {
        const level = levels[l];
        level.scan_bind_group = device.createBindGroup({
            label: `prefix sum scan L${l}`,
            layout: pipelines.bind_group_layout,
            entries: [
                { binding: 0, resource: { buffer: level.info } },
                { binding: 1, resource: { buffer: level.input } },
                { binding: 2, resource: { buffer: level.output } },
                { binding: 3, resource: { buffer: level.block_sums } },
            ],
        });
        if (l + 1 < levels.length) {
            // The next level's output is the exclusive scan of this level's block totals
            level.add_bind_group = device.createBindGroup({
                label: `prefix sum add L${l}`,
                layout: pipelines.bind_group_layout,
                entries: [
                    { binding: 0, resource: { buffer: level.info } },
                    { binding: 1, resource: { buffer: level.input } },
                    { binding: 2, resource: { buffer: level.output } },
                    { binding: 3, resource: { buffer: levels[l + 1].output } },
                ],
            });
        }
    }

    // Blocks per level for the current count
    let activeBlocks: number[] = [1];

    function set_count(count: number): { num_workgroups: number } {
        activeBlocks = [];
        let levelCount = Math.max(0, Math.min(count, maxElements));
        for (const level of levels) {
            const blocks = Math.max(1, Math.ceil(levelCount / ELEMENTS_PER_BLOCK));
            device.queue.writeBuffer(level.info, 0, new Uint32Array([levelCount, blocks, 0, 0]));
            activeBlocks.push(blocks);
            if (blocks === 1) break;
            levelCount = blocks;
        }
        return { num_workgroups: activeBlocks[0] };
    }

    // Spread blocks workgroups over a 2D dispatch to stay within maxComputeWorkgroupsPerDimension
    function dispatch(pass: GPUComputePassEncoder, blocks: number) {
        const x = Math.min(blocks, maxPerDimension);
        pass.dispatchWorkgroups(x, Math.ceil(blocks / x));
    }

    function scan(encoder: GPUCommandEncoder) {
        // Up: local scan of every level
        for (let l = 0; l < activeBlocks.length; l++) {
            const pass = encoder.beginComputePass({ label: `prefix sum - local scan L${l}` });
            pass.setPipeline(pipelines.local_scan);
            pass.setBindGroup(0, levels[l].scan_bind_group!);
            dispatch(pass, activeBlocks[l]);
            pass.end();
        }

        // Down: add the scanned block offsets, from the second-highest level to the first
        for (let l = activeBlocks.length - 2; l >= 0; l--) {
            const pass = encoder.beginComputePass({ label: `prefix sum - add offsets L${l}` });
            pass.setPipeline(pipelines.add_offsets);
            pass.setBindGroup(0, levels[l].add_bind_group!);
            dispatch(pass, activeBlocks[l]);
            pass.end();
        }
    }

    return {
        scan,
        info_buffer: levels[0].info,
        input_buffer,
        output_buffer,
        block_sums_buffer: levels[0].block_sums,
        max_elements: maxElements,
        set_count,
        destroy: () => {
            input_buffer.destroy();
            output_buffer.destroy();
            for (const level of levels) {
                level.info.destroy();
                level.block_sums.destroy();
                if (level.output !== output_buffer) level.output.destroy();
            }
        },
    };
}
