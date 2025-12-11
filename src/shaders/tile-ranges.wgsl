/*
 * Tile Offsets Computation Shader
 * 
 * After sorting tile_keys, this shader finds the start offset for each tile
 * using atomicMin to avoid race conditions.
 */
override workgroupSize: u32 = 256u;

struct TileInfo {
    num_tiles_x: u32,
    num_tiles_y: u32,
    total_tiles: u32,
    max_tile_entries: u32,
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct TilePipelineStats {
    total_tile_entries: u32,
    visible_gaussians: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> tile_info: TileInfo;
@group(0) @binding(1) var<storage, read> sort_infos: SortInfos;
@group(0) @binding(2) var<storage, read> sorted_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> pipeline_stats: TilePipelineStats;

fn get_tile_id(key: u32) -> u32 {
    let encoded_tile = key >> 16u;
    if (encoded_tile == 0u) {
        return 0xFFFFFFFFu;
    }
    return encoded_tile - 1u;
}

// Initialize all tile offsets to MAX_UINT
@compute @workgroup_size(workgroupSize, 1, 1)
fn init_tile_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx > tile_info.total_tiles) {
        return;
    }
    atomicStore(&tile_offsets[idx], 0xFFFFFFFFu);
}

// Build tile offsets for each tile
@compute @workgroup_size(workgroupSize, 1, 1)
fn build_tile_offsets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    // Compute linear index from 2D dispatch
    let idx = gid.y * num_wg.x * workgroupSize + gid.x;
    
    // Constants matching sort.ts/swgl configuration
    let keys_per_workgroup = 256u * 15u;
    
    // Define the range of sorted keys to process
    let total_entries = pipeline_stats.total_tile_entries;
    let padded_count = total_entries;
    let base_offset = 0u;
    
    // First thread sets the sentinel for tile ranges end marker
    if (idx == 0u) {
        atomicStore(&tile_offsets[tile_info.total_tiles], padded_count);
    }
    
    // Only process the valid sorted range
    let sorted_idx = base_offset + idx;
    if (sorted_idx >= padded_count) {
        return;
    }
    
    let key = sorted_keys[sorted_idx];
    
    // Skip zero keys
    if (key == 0u) {
        return;
    }
    
    // Decode shifted tile_id
    let encoded_tile = key >> 16u;
    if (encoded_tile == 0u) {
        return;
    }
    let tile_id = encoded_tile - 1u;
    
    // Bounds check
    if (tile_id >= tile_info.total_tiles) {
        return;
    }
    
    // atomicMin ensures the smallest index wins
    atomicMin(&tile_offsets[tile_id], sorted_idx);
}
