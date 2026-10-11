/*
 * Radix sort for dynamic dispatch
 */
import radix_sort_wgsl from './radix_sort_dynamic.wgsl';
import update_dispatch_wgsl from './update-sort-dispatch.wgsl';
import { get_prefix_scanner } from '../prefix/prefix';

export interface DynamicSortStuff {
    sort: (encoder: GPUCommandEncoder) => void,
    sort_info_buffer: GPUBuffer,
    sort_dispatch_indirect_buffer: GPUBuffer,
    histogram_buffer: GPUBuffer,

    // ping-pong
    ping_pong: {
        sort_indices_buffer: GPUBuffer,
        sort_depths_buffer: GPUBuffer,
    }[]

    final_out_index: number,

    destroy: () => void,
}


function create_ping_pong_buffer(adjusted_count: number, keysize: number, device: GPUDevice) {
    return {
        // payload
        sort_indices_buffer: device.createBuffer({
            label: 'ping pong sort indices',
            size: adjusted_count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }),
        // key
        sort_depths_buffer: device.createBuffer({
            label: 'ping pong sort depths',
            size: keysize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }),
    };
}

export const c_histogram_block_rows = 15;

export const C = {
    histogram_wg_size: 256,
    rs_histogram_block_rows: c_histogram_block_rows,
    rs_scatter_block_rows: c_histogram_block_rows,
};

const RADIX_BITS = 4;
const RADIX = 1 << RADIX_BITS;
const NUM_PASSES = 32 / RADIX_BITS;
const KEYS_PER_BLOCK = C.histogram_wg_size * C.rs_scatter_block_rows;

export function get_dynamic_sorter(max_capacity: number, device: GPUDevice, stats_buffer: GPUBuffer): DynamicSortStuff {
    const keysize = Math.max(1, max_capacity); // We allocate for max, but sort logic will be dynamic
    const keys_count_adjusted = (Math.floor((keysize + KEYS_PER_BLOCK - 1) / KEYS_PER_BLOCK) + 1) * KEYS_PER_BLOCK;
    const num_blocks = Math.ceil(keysize / KEYS_PER_BLOCK);
    const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;

    console.log(`[Dynamic Sorter] Allocating for max capacity: ${keysize}`);

    const sort_info_buffer = device.createBuffer({
        label: 'sort info',
        size: 5 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const sort_dispatch_indirect_buffer = device.createBuffer({
        label: 'sort dispatch indirect',
        size: 3 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });

    // Pipeline for updating the key count from the pipeline stats
    const update_dispatch_pipeline = device.createComputePipeline({
        label: 'update sort dispatch',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: update_dispatch_wgsl
            }),
            entryPoint: 'update_dispatch',
        }
    });

    const update_dispatch_bind_group = device.createBindGroup({
        label: 'update sort dispatch bind group',
        layout: update_dispatch_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sort_info_buffer } },
            { binding: 1, resource: { buffer: sort_dispatch_indirect_buffer } },
            { binding: 2, resource: { buffer: stats_buffer } },
        ]
    });

    const ping_pong = [
        create_ping_pong_buffer(keys_count_adjusted, keysize, device),
        create_ping_pong_buffer(keys_count_adjusted, keysize, device),
    ];

    // Digit counts per (digit, block), digit-major, and their exclusive scan
    const scanner = get_prefix_scanner(RADIX * num_blocks, device);
    scanner.set_count(RADIX * num_blocks);

    // Safe default until the first update from the stats buffer
    device.queue.writeBuffer(sort_info_buffer, 0, new Uint32Array([keysize, keys_count_adjusted, NUM_PASSES, 0, 0]));
    device.queue.writeBuffer(sort_dispatch_indirect_buffer, 0, new Uint32Array([num_blocks, 1, 1]));

    const module = device.createShaderModule({ label: 'radix sort', code: radix_sort_wgsl });
    const histogram_pipeline = device.createComputePipeline({
        label: 'radix sort block histogram',
        layout: 'auto',
        compute: { module, entryPoint: 'block_histogram' },
    });
    const scatter_pipeline = device.createComputePipeline({
        label: 'radix sort scatter',
        layout: 'auto',
        compute: { module, entryPoint: 'scatter' },
    });

    // One digit position per pass; pass p reads ping_pong[p % 2] and writes ping_pong[(p + 1) % 2]
    const pass_info_buffers: GPUBuffer[] = [];
    const histogram_bind_groups: GPUBindGroup[] = [];
    const scatter_bind_groups: GPUBindGroup[] = [];
    for (let p = 0; p < NUM_PASSES; p++) {
        const pass_info = device.createBuffer({
            label: `radix sort pass ${p}`,
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pass_info, 0, new Uint32Array([p * RADIX_BITS, num_blocks, 0, 0]));
        pass_info_buffers.push(pass_info);

        const src = ping_pong[p % 2], dst = ping_pong[(p + 1) % 2];
        histogram_bind_groups.push(device.createBindGroup({
            label: `radix sort histogram ${p}`,
            layout: histogram_pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: sort_info_buffer } },
                { binding: 1, resource: { buffer: pass_info } },
                { binding: 2, resource: { buffer: src.sort_depths_buffer } },
                { binding: 6, resource: { buffer: scanner.input_buffer } },
            ],
        }));
        scatter_bind_groups.push(device.createBindGroup({
            label: `radix sort scatter ${p}`,
            layout: scatter_pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: sort_info_buffer } },
                { binding: 1, resource: { buffer: pass_info } },
                { binding: 2, resource: { buffer: src.sort_depths_buffer } },
                { binding: 3, resource: { buffer: src.sort_indices_buffer } },
                { binding: 4, resource: { buffer: dst.sort_depths_buffer } },
                { binding: 5, resource: { buffer: dst.sort_indices_buffer } },
                { binding: 7, resource: { buffer: scanner.output_buffer } },
            ],
        }));
    }

    // Spread the blocks over a 2D dispatch to stay within maxComputeWorkgroupsPerDimension
    function dispatch_blocks(pass: GPUComputePassEncoder) {
        const x = Math.min(num_blocks, maxPerDimension);
        pass.dispatchWorkgroups(x, Math.ceil(num_blocks / x));
    }

    function sort(encoder: GPUCommandEncoder) {
        // 1. Take the current key count from the pipeline stats
        const update_pass = encoder.beginComputePass({
            label: 'update sort dispatch params'
        });
        update_pass.setPipeline(update_dispatch_pipeline);
        update_pass.setBindGroup(0, update_dispatch_bind_group);
        update_pass.dispatchWorkgroups(1);
        update_pass.end();

        // 2. Count, scan, scatter in one pass per digit: 
        for (let p = 0; p < NUM_PASSES; p++) {
            {
                const pass = encoder.beginComputePass({ label: `radix sort histogram ${p}` });
                pass.setPipeline(histogram_pipeline);
                pass.setBindGroup(0, histogram_bind_groups[p]);
                dispatch_blocks(pass);
                pass.end();
            }
            scanner.scan(encoder);
            {
                const pass = encoder.beginComputePass({ label: `radix sort scatter ${p}` });
                pass.setPipeline(scatter_pipeline);
                pass.setBindGroup(0, scatter_bind_groups[p]);
                dispatch_blocks(pass);
                pass.end();
            }
        }
    };

    const final_out_index = (NUM_PASSES % 2 === 0) ? 0 : 1;

    return {
        sort_info_buffer,
        sort_dispatch_indirect_buffer,
        histogram_buffer: scanner.input_buffer,
        ping_pong,
        sort,
        final_out_index,
        destroy: () => {
            sort_info_buffer.destroy();
            sort_dispatch_indirect_buffer.destroy();
            scanner.destroy();
            for (const buffer of pass_info_buffers) buffer.destroy();
            for (const pp of ping_pong) {
                pp.sort_indices_buffer.destroy();
                pp.sort_depths_buffer.destroy();
            }
        },
    };
}
