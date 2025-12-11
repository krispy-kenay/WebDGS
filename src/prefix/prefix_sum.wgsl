/*
    GPU Parallel Prefix Sum (Exclusive Scan)
    
    Three-phase Blelloch-style implementation:
    - Phase 1 (local_scan): Each workgroup computes local prefix sum and writes its total to block_sums
    - Phase 2 (scan_blocks): Single workgroup computes prefix sum over block totals 
    - Phase 3 (add_offsets): Each workgroup adds the scanned block offset to its local results
    
    This implementation is designed for arrays up to ~16M elements (256 * 256 * 256 workgroups).
*/

// Workgroup size - 256 is efficient for most GPUs
override WORKGROUP_SIZE: u32 = 256u;

struct PrefixSumInfo {
    count: u32,             // Number of elements to scan
    num_workgroups: u32,    // Number of workgroups in Phase 1/3
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> info: PrefixSumInfo;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

// Shared memory for local prefix sum
var<workgroup> shared_data: array<u32, 512>;  // 2 * WORKGROUP_SIZE for Blelloch

// Blelloch scan - up-sweep and down-sweep within shared memory
fn blelloch_scan(lid: u32, n: u32) {
    let ai = lid;
    let bi = lid + n;
    
    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = n; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if lid < d {
            let ai_idx = offset * (2u * lid + 1u) - 1u;
            let bi_idx = offset * (2u * lid + 2u) - 1u;
            if bi_idx < 2u * n {
                shared_data[bi_idx] += shared_data[ai_idx];
            }
        }
        offset <<= 1u;
    }
    
    // Clear last element for exclusive scan
    if lid == 0u {
        shared_data[2u * n - 1u] = 0u;
    }
    
    // Down-sweep phase
    for (var d = 1u; d < 2u * n; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if lid < d {
            let ai_idx = offset * (2u * lid + 1u) - 1u;
            let bi_idx = offset * (2u * lid + 2u) - 1u;
            if bi_idx < 2u * n {
                let t = shared_data[ai_idx];
                shared_data[ai_idx] = shared_data[bi_idx];
                shared_data[bi_idx] += t;
            }
        }
    }
    workgroupBarrier();
}

// Simple linear scan for small arrays (used in Phase 2)
fn linear_exclusive_scan(lid: u32, count: u32) {
    // Only thread 0 does the work - simple but sufficient for small arrays
    if lid == 0u {
        var sum = 0u;
        for (var i = 0u; i < count; i++) {
            let val = shared_data[i];
            shared_data[i] = sum;
            sum += val;
        }
    }
    workgroupBarrier();
}

// Phase 1: Local prefix sum within each workgroup
// Each thread loads 2 elements, we scan, then write results
// Also outputs block sum for Phase 2
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn local_scan(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let block_id = wg_id.x;
    let local_id = lid.x;
    let elements_per_block = WORKGROUP_SIZE * 2u;
    let block_start = block_id * elements_per_block;
    
    // Load 2 elements per thread into shared memory
    let ai = local_id;
    let bi = local_id + WORKGROUP_SIZE;
    let global_ai = block_start + ai;
    let global_bi = block_start + bi;
    
    // Load with bounds check
    if global_ai < info.count {
        shared_data[ai] = input[global_ai];
    } else {
        shared_data[ai] = 0u;
    }
    
    if global_bi < info.count {
        shared_data[bi] = input[global_bi];
    } else {
        shared_data[bi] = 0u;
    }
    
    workgroupBarrier();
    
    // Store the total before scanning (for block_sums)
    // Sum of all elements in this block
    var block_total = 0u;
    if local_id == 0u {
        for (var i = 0u; i < elements_per_block; i++) {
            if block_start + i < info.count {
                block_total += shared_data[i];
            }
        }
    }
    
    // Perform Blelloch exclusive scan
    blelloch_scan(local_id, WORKGROUP_SIZE);
    
    // Write scanned results back to output
    if global_ai < info.count {
        output[global_ai] = shared_data[ai];
    }
    if global_bi < info.count {
        output[global_bi] = shared_data[bi];
    }
    
    // Write block sum for Phase 2
    if local_id == 0u {
        block_sums[block_id] = block_total;
    }
}

// Phase 2: Scan the block sums
// Each thread processes BLOCKS_PER_THREAD block sums to support large arrays
// With 256 threads × 16 blocks/thread = 4096 blocks = 2M elements max
const BLOCKS_PER_THREAD: u32 = 32u;
const MAX_BLOCKS_PHASE2: u32 = 4096u;  // WORKGROUP_SIZE * BLOCKS_PER_THREAD

// Separate shared memory for phase 2 (larger to hold all block sums)
var<workgroup> block_shared: array<u32, 4096>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scan_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_id = lid.x;
    let num_blocks = info.num_workgroups;
    
    // Each thread loads BLOCKS_PER_THREAD block sums
    for (var i = 0u; i < BLOCKS_PER_THREAD; i++) {
        let block_idx = local_id * BLOCKS_PER_THREAD + i;
        if block_idx < num_blocks {
            block_shared[block_idx] = block_sums[block_idx];
        } else {
            block_shared[block_idx] = 0u;
        }
    }
    workgroupBarrier();
    
    // Use simple linear exclusive scan (single thread, but handles all blocks)
    // This is efficient enough for Phase 2 since it's only called once per frame
    if local_id == 0u {
        var sum = 0u;
        for (var i = 0u; i < num_blocks; i++) {
            let val = block_shared[i];
            block_shared[i] = sum;
            sum += val;
        }
    }
    workgroupBarrier();
    
    // Each thread writes BLOCKS_PER_THREAD block sums back
    for (var i = 0u; i < BLOCKS_PER_THREAD; i++) {
        let block_idx = local_id * BLOCKS_PER_THREAD + i;
        if block_idx < num_blocks {
            block_sums[block_idx] = block_shared[block_idx];
        }
    }
}

// Phase 3: Add block offset to each element
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn add_offsets(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let block_id = wg_id.x;
    let local_id = lid.x;
    let elements_per_block = WORKGROUP_SIZE * 2u;
    let block_start = block_id * elements_per_block;
    
    // Load block offset (same for all threads in this workgroup)
    let block_offset = block_sums[block_id];
    
    // Add offset to both elements this thread is responsible for
    let global_ai = block_start + local_id;
    let global_bi = block_start + local_id + WORKGROUP_SIZE;
    
    if global_ai < info.count {
        output[global_ai] = output[global_ai] + block_offset;
    }
    if global_bi < info.count {
        output[global_bi] = output[global_bi] + block_offset;
    }
}
