/*
    GPU Parallel Prefix Sum (Exclusive Scan) - TypeScript API
    
    Usage follows the same pattern as sort.ts:
    
    1. Create scanner: const scanner = get_prefix_scanner(max_elements, device)
    2. Set count: device.queue.writeBuffer(scanner.info_buffer, 0, new Uint32Array([count, numWorkgroups, 0, 0]))
    3. Write input: device.queue.writeBuffer(scanner.input_buffer, 0, inputData)
    4. Scan: scanner.scan(encoder)
    5. Read output: scanner.output_buffer contains exclusive prefix sums
    
    The scanner also exposes the total sum in output_buffer[count-1] + input[count-1] if needed.
*/

import prefix_sum_wgsl from './prefix_sum.wgsl';

const WORKGROUP_SIZE = 256;
const ELEMENTS_PER_BLOCK = WORKGROUP_SIZE * 2;  // 512 elements per workgroup
const BLOCKS_PER_THREAD = 32;  // Phase 2 processes this many blocks per thread
const MAX_BLOCKS = WORKGROUP_SIZE * BLOCKS_PER_THREAD;  // 4096 blocks max

export interface PrefixScannerConfig {
    maxElements: number;
}

export interface PrefixScanner {
    /** Encode the prefix sum passes to a command encoder */
    scan: (encoder: GPUCommandEncoder) => void;
    /** Uniform buffer containing count and num_workgroups */
    info_buffer: GPUBuffer;
    /** Input buffer - write your u32 values here before scanning */
    input_buffer: GPUBuffer;
    /** Output buffer - contains exclusive prefix sums after scanning */
    output_buffer: GPUBuffer;
    /** Block sums buffer (internal use, but exposed for debugging) */
    block_sums_buffer: GPUBuffer;
    /** Maximum number of elements this scanner can handle */
    max_elements: number;
    /** Update the element count and dispatch the scan (helper) */
    set_count: (count: number) => { num_workgroups: number };
    /** Destroy owned GPU buffers */
    destroy: () => void;
}

/** Configuration constants */
export const PREFIX_CONSTANTS = {
    WORKGROUP_SIZE,
    ELEMENTS_PER_BLOCK,
    BLOCKS_PER_THREAD,
    MAX_BLOCKS,
    MAX_ELEMENTS: MAX_BLOCKS * ELEMENTS_PER_BLOCK,
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
        scan_blocks: device.createComputePipeline({
            label: 'prefix sum - scan blocks',
            layout: pipeline_layout,
            compute: {
                module,
                entryPoint: 'scan_blocks',
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
    // Calculate buffer sizes
    const maxWorkgroups = Math.ceil(maxElements / ELEMENTS_PER_BLOCK);

    // Validate size limits (our implementation handles up to 4096 workgroups = 2M elements)
    if (maxWorkgroups > MAX_BLOCKS) {
        console.warn(
            `Prefix sum for ${maxElements} elements requires ${maxWorkgroups} workgroups. ` +
            `Maximum supported is ${MAX_BLOCKS} workgroups (${MAX_BLOCKS * ELEMENTS_PER_BLOCK} elements).`
        );
    }

    // Create pipelines
    const pipelines = create_pipelines(device);

    // Create buffers
    const info_buffer = device.createBuffer({
        label: 'prefix sum info',
        size: 16, // 4 u32s: count, num_workgroups, pad, pad
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const input_buffer = device.createBuffer({
        label: 'prefix sum input',
        size: maxElements * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const output_buffer = device.createBuffer({
        label: 'prefix sum output',
        size: maxElements * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const block_sums_buffer = device.createBuffer({
        label: 'prefix sum block sums',
        size: Math.max(maxWorkgroups * 4, 16), // At least 16 bytes
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Create bind group
    const bind_group = device.createBindGroup({
        label: 'prefix sum',
        layout: pipelines.bind_group_layout,
        entries: [
            { binding: 0, resource: { buffer: info_buffer } },
            { binding: 1, resource: { buffer: input_buffer } },
            { binding: 2, resource: { buffer: output_buffer } },
            { binding: 3, resource: { buffer: block_sums_buffer } },
        ],
    });

    // Track current workgroup count (set by set_count)
    let currentNumWorkgroups = 1;

    /**
     * Set the element count and calculate dispatch size.
     * Must be called before scan() with the actual count.
     */
    function set_count(count: number): { num_workgroups: number } {
        const num_workgroups = Math.ceil(count / ELEMENTS_PER_BLOCK);
        currentNumWorkgroups = num_workgroups;

        // Write info buffer
        device.queue.writeBuffer(info_buffer, 0, new Uint32Array([count, num_workgroups, 0, 0]));

        return { num_workgroups };
    }

    /**
     * Record prefix sum passes to command encoder.
     * Call set_count() first to set the element count.
     */
    function scan(encoder: GPUCommandEncoder) {
        // Phase 1: Local scan within each workgroup
        {
            const pass = encoder.beginComputePass({ label: 'prefix sum - local scan' });
            pass.setPipeline(pipelines.local_scan);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroups(currentNumWorkgroups);
            pass.end();
        }

        // Phase 2: Scan block sums (single workgroup)
        if (currentNumWorkgroups > 1) {
            const pass = encoder.beginComputePass({ label: 'prefix sum - scan blocks' });
            pass.setPipeline(pipelines.scan_blocks);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroups(1);
            pass.end();
        }

        // Phase 3: Add block offsets to each element
        if (currentNumWorkgroups > 1) {
            const pass = encoder.beginComputePass({ label: 'prefix sum - add offsets' });
            pass.setPipeline(pipelines.add_offsets);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroups(currentNumWorkgroups);
            pass.end();
        }
    }

    return {
        scan,
        info_buffer,
        input_buffer,
        output_buffer,
        block_sums_buffer,
        max_elements: maxElements,
        set_count,
        destroy: () => {
            info_buffer.destroy();
            input_buffer.destroy();
            output_buffer.destroy();
            block_sums_buffer.destroy();
        },
    };
}
