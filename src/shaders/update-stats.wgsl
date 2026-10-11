/*
 * Update-Stats Shader
 * 
 * Single-thread compute pass that computes total_tile_entries after prefix sum.
 */
struct TilePipelineStats {
    total_tile_entries: u32,
    visible_gaussians: u32,
    requested_tile_entries: u32,
    _pad1: u32,
};

struct UpdateStatsInfo {
    num_gaussians: u32,
    max_tile_entries: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> pipeline_stats: TilePipelineStats;
@group(0) @binding(3) var<uniform> info: UpdateStatsInfo;

// Single thread calculates total_tile_entries = tile_offsets[N-1] + tile_counts[N-1]
@compute @workgroup_size(1, 1, 1)
fn update_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    if (info.num_gaussians == 0u) {
        pipeline_stats.total_tile_entries = 0u;
        pipeline_stats.requested_tile_entries = 0u;
        return;
    }

    let last_idx = info.num_gaussians - 1u;
    let last_offset = tile_offsets[last_idx];
    let last_count = tile_counts[last_idx];
    let requested = last_offset + last_count;

    pipeline_stats.requested_tile_entries = requested;
    pipeline_stats.total_tile_entries = min(requested, info.max_tile_entries);
}
