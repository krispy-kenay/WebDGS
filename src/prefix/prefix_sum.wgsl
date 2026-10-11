/*
    GPU Parallel Prefix Sum (Exclusive Scan)

    Multi-level reduce-then-scan for arrays of any size:
    - local_scan: each workgroup scans one block of 512 elements in shared memory (Blelloch) and writes the
      block total to block_sums
    - the block totals are scanned with the same kernel one level up, repeated until a single block remains
    - add_offsets: walking back down, each block adds its scanned block offset
    No workgroup ever waits on another one. Blocks are laid out over a 2D dispatch so that very large arrays stay
    within maxComputeWorkgroupsPerDimension.
*/

// Workgroup size - 256 is efficient for most GPUs
override WORKGROUP_SIZE: u32 = 256u;

struct PrefixSumInfo {
    count: u32,             // Number of elements to scan at this level
    num_workgroups: u32,    // Number of blocks at this level
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

// Block handled by this workgroup (blocks are spread over a 2D dispatch)
fn block_index(wg_id: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wg_id.y * num_wg.x + wg_id.x;
}

// Local prefix sum of one block; writes the block total for the next level
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn local_scan(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let block_id = block_index(wg_id, num_wg);
    if block_id >= info.num_workgroups {
        return;
    }
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

    // Exclusive scan result of the last element + its value = block total
    let last_value = shared_data[elements_per_block - 1u];

    // Perform Blelloch exclusive scan
    blelloch_scan(local_id, WORKGROUP_SIZE);

    // Write scanned results back to output
    if global_ai < info.count {
        output[global_ai] = shared_data[ai];
    }
    if global_bi < info.count {
        output[global_bi] = shared_data[bi];
    }

    // Write block total for the next level
    if local_id == 0u {
        block_sums[block_id] = shared_data[elements_per_block - 1u] + last_value;
    }
}

// Add the scanned block offset (block_sums = exclusive scan of this level's block totals) to each element
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn add_offsets(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let block_id = block_index(wg_id, num_wg);
    if block_id >= info.num_workgroups {
        return;
    }
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
