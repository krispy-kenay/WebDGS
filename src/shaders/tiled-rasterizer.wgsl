/*
 * Tiled Rasterizer Compute Shader
 * 
 * Rasterizes preprocessed gaussians to a texture
 */

override tileWidth: u32 = 16u;
override tileHeight: u32 = 16u;

struct TileInfo {
    num_tiles_x: u32,
    num_tiles_y: u32,
    total_tiles: u32,
    max_tile_entries: u32,
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    viewport_x: f32,
    viewport_y: f32,
    point_size_px: f32,
    gaussian_mode: f32,
};

struct Splat {
    pos: u32,
    radius: u32,
    conic_xy: u32,
    conic_z: u32,
    color_rg: u32,
    color_ba: u32,
};

struct TilePipelineStats {
    total_tile_entries: u32,
    visible_gaussians: u32,
    _pad0: u32,
    _pad1: u32,
};


@group(0) @binding(0) var<uniform> tile_info: TileInfo;
@group(0) @binding(1) var<uniform> settings: RenderSettings;

@group(1) @binding(0) var<storage, read> splats: array<Splat>;
@group(1) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(1) @binding(2) var<storage, read> sorted_indices: array<u32>;
@group(1) @binding(3) var<storage, read> sorted_keys: array<u32>;
@group(1) @binding(4) var<storage, read> pipeline_stats: TilePipelineStats;

@group(2) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;


const BACKGROUND_COLOR = vec4<f32>(0.0, 0.0, 0.0, 1.0);
const BATCH_SIZE = 256u;  
const MAX_BATCHES = 32u;

struct SharedSplat {
    center_px: vec2<f32>,
    conic: vec3<f32>,
    color: vec3<f32>,
    opacity: f32,
};

var<workgroup> shared_splats: array<SharedSplat, BATCH_SIZE>;
var<workgroup> shared_valid: array<u32, BATCH_SIZE>;
var<workgroup> batch_has_any: atomic<u32>;

fn get_tile_id(key: u32) -> u32 {
    let encoded_tile = key >> 16u;
    if (encoded_tile == 0u) {
        return 0xFFFFFFFFu;
    }
    return encoded_tile - 1u;
}

@compute @workgroup_size(tileWidth, tileHeight, 1)
fn tiled_rasterize(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    let tile_id = wg_id.y * tile_info.num_tiles_x + wg_id.x;
    
    let pixel_x = wg_id.x * tileWidth + local_id.x;
    let pixel_y = wg_id.y * tileHeight + local_id.y;
    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    
    let tile_valid = tile_id < tile_info.total_tiles;
    let in_bounds = pixel_x < u32(viewport.x) && pixel_y < u32(viewport.y);
    let should_process = tile_valid && in_bounds;

    let pixel = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);
    
    // Constants matching sort.ts/swgl configuration
    let keys_per_workgroup = 256u * 15u;
    
    // Define the range of sorted keys to process
    let total_entries = pipeline_stats.total_tile_entries;
    let padded_count = total_entries;
    let base_offset = 0u;
    
    var start = 0u;
    if tile_valid {
        start = tile_offsets[tile_id];
    }
    
    // Valid data: start must be in valid range and not MAX_UINT
    let tile_has_data = tile_valid && start >= base_offset && start < padded_count && start < 0xFFFFFFFFu;
    
    var accum_color = vec3<f32>(0.0);
    var accum_alpha = 0.0;
    var entries_processed = 0u;
    var still_processing = tile_has_data;
    
    // Fixed iteration count
    for (var batch = 0u; batch < MAX_BATCHES; batch++) {
        if (local_idx == 0u) {
            atomicStore(&batch_has_any, 0u);
        }
        workgroupBarrier();

        let current_entry = start + batch * BATCH_SIZE;
        
        // Load batch into shared memory
        if local_idx < BATCH_SIZE {
            var is_valid = false;

            if still_processing {
                let entry_idx = current_entry + local_idx;
                
                if entry_idx >= base_offset && entry_idx < padded_count {
                    let key = sorted_keys[entry_idx];
                    
                    // Skip zero keys
                    if key != 0u {
                        let entry_tile_id = get_tile_id(key);
                        
                        if entry_tile_id == tile_id {
                            let gaussian_idx = sorted_indices[entry_idx];
                            
                            if gaussian_idx < arrayLength(&splats) {
                                let splat = splats[gaussian_idx];
                                
                                let pos_ndc = unpack2x16float(splat.pos);
                                let conic_xy = unpack2x16float(splat.conic_xy);
                                let conic_z = unpack2x16float(splat.conic_z);
                                let color_rg = unpack2x16float(splat.color_rg);
                                let color_ba = unpack2x16float(splat.color_ba);
                                
                                let center_px = (pos_ndc * vec2<f32>(0.5, -0.5) + 0.5) * viewport;

                                shared_splats[local_idx].center_px = center_px;
                                shared_splats[local_idx].conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z.x);
                                shared_splats[local_idx].color = vec3<f32>(color_rg.x, color_rg.y, color_ba.x);
                                shared_splats[local_idx].opacity = color_ba.y;

                                is_valid = true;
                                
                                // Mark that this batch has at least one valid entry
                                atomicStore(&batch_has_any, 1u);
                            }
                        }
                    }
                }
            }

            // Write per-slot validity flag (0 or 1) so we don't read stale splats
            shared_valid[local_idx] = select(0u, 1u, is_valid);
        }
        
        workgroupBarrier();
        
        // Read whether anyone in the workgroup saw a valid entry this batch
        let has_any = atomicLoad(&batch_has_any);
        
        // If no valid entries, all remaining batches will also be empty
        if has_any == 0u {
            still_processing = false;
        }
        
        // Process valid entries
        if should_process && has_any > 0u {
            for (var i = 0u; i < BATCH_SIZE; i++) {
                if shared_valid[i] == 0u {
                    continue;
                }

                if accum_alpha > 0.99 {
                    continue;
                }

                let shared_splat = shared_splats[i];
                let delta = pixel - shared_splat.center_px;
                let conic = shared_splat.conic;
                
                if settings.gaussian_mode < 0.5 {
                    let dist_sq = dot(delta, delta);
                    let limit = settings.point_size_px;
                    if dist_sq <= limit * limit {
                        accum_color = vec3<f32>(1.0, 1.0, 0.0);
                        accum_alpha = 1.0;
                    }
                    continue;
                }
                
                let exp_q = (conic.x * delta.x * delta.x) + 
                           (2.0 * conic.y * delta.x * delta.y) + 
                           (conic.z * delta.y * delta.y);

                let gaussian_weight = exp(-0.5 * exp_q);
                let alpha = clamp(gaussian_weight * shared_splat.opacity, 0.0, 0.99);
                
                let vis = 1.0 - accum_alpha;
                accum_color += shared_splat.color * alpha * vis;
                accum_alpha += alpha * vis;
            }
        }
        
        entries_processed += BATCH_SIZE;
        
        workgroupBarrier();
    }
    
    // Write final color
    if should_process {
        let vis = 1.0 - accum_alpha;
        var final_color = accum_color + BACKGROUND_COLOR.rgb * vis;
        
        textureStore(
            output_texture,
            vec2<i32>(i32(pixel_x), i32(pixel_y)),
            vec4<f32>(final_color, 1.0)
        );
    }
}