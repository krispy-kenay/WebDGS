/*
 * Tiled Forward Pass Shader
 * 
 * Projects 3D Gaussians to 2D and assigns them to screen-space tiles
 */

const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

// Tile configuration - override from TypeScript
override tileWidth: u32 = 16u;
override tileHeight: u32 = 16u;

// Helper to convert f32 to f16 precision using a round trip (so that culling conditions align)
fn to_f16_precision(v: vec2<f32>) -> vec2<f32> {
    return unpack2x16float(pack2x16float(v));
}

fn clamp_finite(v: vec2<f32>, limit: f32) -> vec2<f32> {
    // Avoid infinities/NaNs turning into undefined f16 packing behavior.
    return clamp(v, vec2<f32>(-limit), vec2<f32>(limit));
}

struct TileInfo {
    num_tiles_x: u32,
    num_tiles_y: u32,
    total_tiles: u32,
    max_tile_entries: u32,
};

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

const SH_WORDS_PER_POINT : u32 = 24u;
fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    return vec3<f32>(
        read_f16_coeff(base_word, eR),
        read_f16_coeff(base_word, eG),
        read_f16_coeff(base_word, eB)
    );
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn float_to_ordered_uint(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
  return bits ^ mask;
}

// Quantize depth to 16 bits for packing with tile ID
fn quantize_depth_16(depth_ordered: u32) -> u32 {
    return depth_ordered >> 16u;
}

// Build a combined tile key
fn make_tile_key(tile_id: u32, depth_ordered: u32) -> u32 {
    let depth_16 = quantize_depth_16(depth_ordered);
    return ((tile_id + 1u) << 16u) | depth_16;
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;
@group(0) @binding(2) var<uniform> tile_info: TileInfo;

@group(1) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) var<storage, read> sh_buffer : array<u32>;
@group(1) @binding(2) var<storage, read_write> splats : array<Splat>;
@group(1) @binding(3) var<storage, read_write> depths : array<u32>;
@group(1) @binding(4) var<storage, read_write> tile_counts : array<u32>;
@group(1) @binding(5) var<storage, read_write> pipeline_stats : TilePipelineStatsAtomic;

struct TilePipelineStatsAtomic {
    total_tile_entries: u32,
    visible_gaussians: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
};

@group(2) @binding(0) var<storage, read> tile_offsets : array<u32>;
@group(3) @binding(0) var<storage, read_write> tile_keys : array<u32>;
@group(3) @binding(1) var<storage, read_write> tile_indices : array<u32>;

// Counts and projects the splats to the correct tile
@compute @workgroup_size(workgroupSize, 1, 1)
fn count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // Default to 0 counts
    tile_counts[idx] = 0u;

    // Unpacking 3D Gaussian data
    let gaussian = gaussians[idx];
    let rot_0 = unpack2x16float(gaussian.rot[0]);
    let rot_1 = unpack2x16float(gaussian.rot[1]);
    let quaternion = vec4<f32>(rot_0.x, rot_0.y, rot_1.x, rot_1.y);

    let scale_0 = unpack2x16float(gaussian.scale[0]);
    let scale_1 = unpack2x16float(gaussian.scale[1]);
    let gaussian_scale = exp(vec3<f32>(scale_0.x, scale_0.y, scale_1.x));

    let pos_0 = unpack2x16float(gaussian.pos_opacity[0]);
    let pos_1 = unpack2x16float(gaussian.pos_opacity[1]);
    let gaussian_position = vec3<f32>(pos_0.x, pos_0.y, pos_1.x);
    let gaussian_opacity = pos_1.y;
    let opacity_sigmoid = 1.0 / (1.0 + exp(-gaussian_opacity));

    // Transform to clip space / NDC
    let position_world = vec4<f32>(gaussian_position, 1.0);
    let world_to_view = camera.view * position_world;
    let view_to_clip = camera.proj * world_to_view;
    
    if (view_to_clip.w == 0.0) {
        return;
    }
    let gaussian_ndc = view_to_clip.xyz / view_to_clip.w;

    // Culling
    if (gaussian_ndc.x < -1.2 || gaussian_ndc.x > 1.2 ||
        gaussian_ndc.y < -1.2 || gaussian_ndc.y > 1.2 ||
        gaussian_ndc.z < 0.0 || gaussian_ndc.z > 1.0) {
        return;
    }

    // Covariance computation
    let C3D = covariance3D(quaternion, gaussian_scale);
    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    let C2D = covariance2D(C3D, world_to_view, camera.focal, viewport, camera.view);
    let det = (C2D.x * C2D.z) - (C2D.y * C2D.y);
    if (det <= 0.0) {
        return;
    }
    let det_inv = 1.0 / det;
    // conic = inverse of 2D covariance
    let conic = vec3<f32>(C2D.z * det_inv, -C2D.y * det_inv, C2D.x * det_inv);

    // Discriminant for ellipse validity
    let disc = conic.y * conic.y - conic.x * conic.z;
    if (conic.x <= 0.0 || conic.z <= 0.0 || disc >= 0.0) {
        return;
    }
    
    // Filter out low opacity gaussians
    let opacity_threshold = 128.0;
    let t = 2.0 * log(opacity_sigmoid * opacity_threshold);
    if (t <= 0.0) {
        return;
    }
    
    // Compute bounding box with tight bounds using SnugBox extents
    let x_extent = sqrt(t * conic.z / (-disc));
    let y_extent = sqrt(t * conic.x / (-disc));
    let cap = select(1e9, settings.max_splat_radius_px, settings.max_splat_radius_px > 0.0);
    let x_extent_cap = min(x_extent, cap);
    let y_extent_cap = min(y_extent, cap);
    // Store unclamped NDC center to avoid the snapping to border issue
    let ndc_store = to_f16_precision(clamp_finite(gaussian_ndc.xy, 60000.0));
    let pixel_center = (ndc_store * vec2<f32>(0.5, -0.5) + 0.5) * viewport;
    let tile_margin = 2.0; 
    let extents_f16 = to_f16_precision(vec2<f32>(x_extent_cap, y_extent_cap));
    let bbox_min_raw = pixel_center - extents_f16 - tile_margin;
    let bbox_max_raw = pixel_center + extents_f16 + tile_margin;

    // reject if bbox does not intersect the viewport (before clamping)
    if (bbox_max_raw.x < 0.0 || bbox_max_raw.y < 0.0 ||
        bbox_min_raw.x >= viewport.x || bbox_min_raw.y >= viewport.y) {
        return;
    }

    // Clamp only for tile coordinate conversion
    let bbox_min = max(bbox_min_raw, vec2<f32>(0.0));
    let bbox_max = min(bbox_max_raw, viewport - vec2<f32>(1.0));
    
    // Skip if bounding box is invalid
    if (bbox_max.x < bbox_min.x || bbox_max.y < bbox_min.y) {
        return;
    }

    // Color from SH
    let camera_world_pos = camera.view_inv[3].xyz;
    let color_direction = normalize(gaussian_position - camera_world_pos);
    let color = computeColorFromSH(color_direction, idx, u32(settings.sh_deg));

    // Convert to tile coordinates
    let tile_min_x = u32(bbox_min.x) / tileWidth;
    let tile_min_y = u32(bbox_min.y) / tileHeight;
    let tile_max_x = min(u32(bbox_max.x) / tileWidth, tile_info.num_tiles_x - 1u);
    let tile_max_y = min(u32(bbox_max.y) / tileHeight, tile_info.num_tiles_y - 1u);

    // Count tiles
    let tiles_x = tile_max_x - tile_min_x + 1u;
    let tiles_y = tile_max_y - tile_min_y + 1u;
    var num_tiles = tiles_x * tiles_y;
    
    // Safety cap
    if (num_tiles > 2048u) {
        return;
    }

    // Write data
    splats[idx].pos = pack2x16float(ndc_store);
    splats[idx].radius = pack2x16float(vec2<f32>(x_extent_cap, y_extent_cap));
    splats[idx].conic_xy = pack2x16float(conic.xy);
    splats[idx].conic_z = pack2x16float(vec2<f32>(conic.z, 0.0));
    splats[idx].color_rg = pack2x16float(clamp(color.rg, vec2<f32>(0.0), vec2<f32>(1.0)));
    splats[idx].color_ba = pack2x16float(vec2<f32>(clamp(color.b, 0.0, 1.0), clamp(opacity_sigmoid, 0.0, 1.0)));

    let depth_ordered = float_to_ordered_uint(world_to_view.z);
    depths[idx] = depth_ordered;
    tile_counts[idx] = num_tiles;

    // Atomically increment visible gaussians count
    atomicAdd(&pipeline_stats.visible_gaussians, 1u);

}

// Emit splats to tiles
@compute @workgroup_size(workgroupSize, 1, 1)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let num_tiles = tile_counts[idx];
    if (num_tiles == 0u) {
        return;
    }

    let start_offset = tile_offsets[idx];
    
    let pos_packed = unpack2x16float(splats[idx].pos);
    let extents_raw = unpack2x16float(splats[idx].radius);
    let cap = select(1e9, settings.max_splat_radius_px, settings.max_splat_radius_px > 0.0);
    let extents = min(extents_raw, vec2<f32>(cap));
    
    let ndc_pos = pos_packed;

    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    let pixel_center = (ndc_pos * vec2<f32>(0.5, -0.5) + 0.5) * viewport;

    // Use both extents for tight bounding box
    let tile_margin = 2.0;
    let bbox_min_raw = pixel_center - extents - tile_margin;
    let bbox_max_raw = pixel_center + extents + tile_margin;

    if (bbox_max_raw.x < 0.0 || bbox_max_raw.y < 0.0 ||
        bbox_min_raw.x >= viewport.x || bbox_min_raw.y >= viewport.y) {
        return;
    }

    let bbox_min = max(bbox_min_raw, vec2<f32>(0.0));
    let bbox_max = min(bbox_max_raw, viewport - vec2<f32>(1.0));

    let tile_min_x = u32(bbox_min.x) / tileWidth;
    let tile_min_y = u32(bbox_min.y) / tileHeight;
    let tile_max_x = min(u32(bbox_max.x) / tileWidth, tile_info.num_tiles_x - 1u);
    let tile_max_y = min(u32(bbox_max.y) / tileHeight, tile_info.num_tiles_y - 1u);


    let depth_ordered = depths[idx];

    var offset = 0u;
    for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
        for (var tx = tile_min_x; tx <= tile_max_x; tx++) {
            let tile_id = ty * tile_info.num_tiles_x + tx;
            
            let key_idx = start_offset + offset;
            tile_keys[key_idx] = make_tile_key(tile_id, depth_ordered);
            tile_indices[key_idx] = idx;
            
            offset++;
        }
    }
}
