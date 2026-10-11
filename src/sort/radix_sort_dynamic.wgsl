/*
 * Stable LSD radix sort of u32 keys with a u32 payload (used by sort_dynamic.ts).
 *
 * 4-bit digits, 8 passes over 32-bit keys. Every pass is reduce-then-scan:
 *   block_histogram: each block of KEYS_PER_BLOCK keys counts its digits
 *                    -> digit_counts[digit * num_blocks + block]
 *   prefix scan:     exclusive scan of digit_counts (src/prefix), giving every (digit, block) its first output slot
 *   scatter:         each key goes to its slot + its stable rank among same-digit keys of the block
 * No workgroup waits on another one and nothing relies on subgroups running in lockstep, so the sort is
 * deterministic on any WebGPU implementation. Keys beyond the buffer capacity are ignored.
 */

const RADIX_BITS: u32 = 4u;
const RADIX: u32 = 16u;
const WG_SIZE: u32 = 256u;
const KEYS_PER_THREAD: u32 = 15u;
const KEYS_PER_BLOCK: u32 = WG_SIZE * KEYS_PER_THREAD;

struct SortInfo {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

struct PassInfo {
    shift: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> info: SortInfo;
@group(0) @binding(1) var<uniform> pass_info: PassInfo;
@group(0) @binding(2) var<storage, read> keys_in: array<u32>;
@group(0) @binding(3) var<storage, read> payload_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> payload_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> digit_counts: array<u32>;
@group(0) @binding(7) var<storage, read> digit_offsets: array<u32>;

// Keys to sort
fn key_count() -> u32 {
    return min(info.keys_size, arrayLength(&keys_in));
}

// Blocks are spread over a 2D dispatch to stay within maxComputeWorkgroupsPerDimension
fn block_index(wg_id: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wg_id.y * num_wg.x + wg_id.x;
}

fn digit_of(key: u32) -> u32 {
    return extractBits(key, pass_info.shift, RADIX_BITS);
}

// Per-block digit counts
var<workgroup> block_counts: array<atomic<u32>, RADIX>;

@compute @workgroup_size(WG_SIZE)
fn block_histogram(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32
) {
    let block = block_index(wg_id, num_wg);
    if (block >= pass_info.num_blocks) {
        return;
    }
    if (lid < RADIX) {
        atomicStore(&block_counts[lid], 0u);
    }
    workgroupBarrier();

    let n = key_count();
    let base = block * KEYS_PER_BLOCK;
    for (var i = 0u; i < KEYS_PER_THREAD; i++) {
        let idx = base + i * WG_SIZE + lid;
        if (idx < n) {
            atomicAdd(&block_counts[digit_of(keys_in[idx])], 1u);
        }
    }
    workgroupBarrier();

    // Digit-major, so one exclusive scan gives each (digit, block) its first output position
    if (lid < RADIX) {
        digit_counts[lid * pass_info.num_blocks + block] = atomicLoad(&block_counts[lid]);
    }
}

// Scatter
var<workgroup> scan_shared: array<vec4<u32>, WG_SIZE>;

// Exclusive scan of one vec4 per thread across the workgroup
fn workgroup_exclusive_scan(lid: u32, value: vec4<u32>) -> vec4<u32> {
    workgroupBarrier();
    scan_shared[lid] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < WG_SIZE; offset <<= 1u) {
        var add = vec4<u32>(0u);
        if (lid >= offset) {
            add = scan_shared[lid - offset];
        }
        workgroupBarrier();
        scan_shared[lid] += add;
        workgroupBarrier();
    }
    return scan_shared[lid] - value;
}

@compute @workgroup_size(WG_SIZE)
fn scatter(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32
) {
    let block = block_index(wg_id, num_wg);
    if (block >= pass_info.num_blocks) {
        return;
    }
    let n = key_count();
    // Each thread owns KEYS_PER_THREAD consecutive keys, so (thread, i) order is the input order
    let base = block * KEYS_PER_BLOCK + lid * KEYS_PER_THREAD;

    var keys: array<u32, KEYS_PER_THREAD>;
    var payload: array<u32, KEYS_PER_THREAD>;
    var counts: array<u32, RADIX>;
    for (var i = 0u; i < KEYS_PER_THREAD; i++) {
        let idx = base + i;
        if (idx < n) {
            keys[i] = keys_in[idx];
            payload[i] = payload_in[idx];
            counts[digit_of(keys[i])] += 1u;
        }
    }

    // Same-digit keys in earlier threads of this block, four digits per scan
    var slot: array<u32, RADIX>;
    for (var g = 0u; g < RADIX; g += 4u) {
        let before = workgroup_exclusive_scan(lid, vec4<u32>(counts[g], counts[g + 1u], counts[g + 2u], counts[g + 3u]));
        slot[g] = before.x;
        slot[g + 1u] = before.y;
        slot[g + 2u] = before.z;
        slot[g + 3u] = before.w;
    }

    // Plus where this block's keys of each digit start in the output
    for (var d = 0u; d < RADIX; d++) {
        slot[d] += digit_offsets[d * pass_info.num_blocks + block];
    }

    for (var i = 0u; i < KEYS_PER_THREAD; i++) {
        if (base + i < n) {
            let d = digit_of(keys[i]);
            let pos = slot[d];
            slot[d] = pos + 1u;
            keys_out[pos] = keys[i];
            payload_out[pos] = payload[i];
        }
    }
}
