/*
 * Tiled Backward Rasterization
 *
 * Propagates gradients from pixels to Gaussians (2D means, conics, opacity, colors).
 */

struct Gaussian3D {
    _pad: u32,
};

// Bindings
@group(0) @binding(0) var<uniform> settings: RenderSettings;
@group(1) @binding(0) var<storage, read> tile_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> tile_instances: array<u32>;
@group(1) @binding(2) var<storage, read> splats: array<Splat>;
// Transmittance, Contributions and Loss stored as textures
@group(1) @binding(4) var final_Ts_tex: texture_2d<f32>; 
@group(1) @binding(5) var n_contrib_tex: texture_2d<u32>;
@group(2) @binding(0) var loss_gradient: texture_2d<f32>;
// Gradients
@group(3) @binding(0) var<storage, read_write> grad_means_2d: array<atomic<i32>>; 
@group(3) @binding(1) var<storage, read_write> grad_conics: array<atomic<i32>>; // x, y, z
@group(3) @binding(2) var<storage, read_write> grad_opacity: array<atomic<i32>>;
@group(3) @binding(3) var<storage, read_write> grad_colors: array<atomic<i32>>; // RGB

// Constants
const BLOCK_SIZE: u32 = 16;


fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(16, 16)
fn backward_rasterize_main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let pix_id = global_id.xy;
    let width = u32(settings.viewport_x);
    let height = u32(settings.viewport_y);

    if (pix_id.x >= width || pix_id.y >= height) {
        return;
    }

    // Tile Info
    let tile_x = group_id.x;
    let tile_y = group_id.y;
    let num_tiles_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let tile_idx = tile_y * num_tiles_x + tile_x;

    // Load Tile Range
    let range_start = tile_offsets[tile_idx];
    let range_end = tile_offsets[tile_idx + 1];
    
    // Pixel Data
    let n_contrib_val = textureLoad(n_contrib_tex, vec2<i32>(pix_id), 0).r;
    if (n_contrib_val == 0u) {
        return;
    }
    
    // Clamp to the actual number of instances in the tile
    let tile_entries = select(0u, range_end - range_start, range_end > range_start);
    let pix_n_contrib = min(n_contrib_val, tile_entries);
    
    var T = textureLoad(final_Ts_tex, vec2<i32>(pix_id), 0).r;
    
    // Gradient from Loss
    let dL_dpixel = textureLoad(loss_gradient, vec2<i32>(pix_id), 0); 
    
    // Pre-calculations
    let pixf = vec2<f32>(pix_id) + 0.5; // Pixel center
    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    
    var accum_rec = vec3<f32>(0.0);
    var last_color = vec3<f32>(0.0);
    var last_alpha = 0.0f;
    
    // Iterate backward over the contributions
    for (var i = pix_n_contrib; i > 0u; i--) {
        let idx_in_tile = i - 1u;
        let global_gaussian_idx = tile_instances[range_start + idx_in_tile];
        
        let splat = splats[global_gaussian_idx];
        
        // Unpack Splat
        let pos_ndc = unpack2x16float(splat.pos);
        let conic_xy = unpack2x16float(splat.conic_xy);
        let conic_z = unpack2x16float(splat.conic_z);
        let color_rg = unpack2x16float(splat.color_rg);
        let color_ba = unpack2x16float(splat.color_ba);
        
        let center_px = (pos_ndc * vec2<f32>(0.5, -0.5) + 0.5) * viewport;
        let conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z.x);
        let color = vec3<f32>(color_rg.x, color_rg.y, color_ba.x);
        let opacity = color_ba.y;

        // Forward Eval
        let delta = pixf - center_px; 
        
        // Power calculation
        let power = (conic.x * delta.x * delta.x) + 
                    (2.0 * conic.y * delta.x * delta.y) + 
                    (conic.z * delta.y * delta.y);
        
        // Recompute G and Alpha
        let G = exp(-0.5 * power);
        let alpha = min(0.99f, opacity * G);
        
        // Skip invalid alpha
         if (alpha < 1.0/255.0) {
             continue; 
         }
         
        // Recover T before this gaussian
        T = T / (1.0 - alpha);
        
        // Gradient Computation
        var dL_dalpha = 0.0f;
        
        // RGB Gradients
        accum_rec = last_alpha * last_color + (1.0 - last_alpha) * accum_rec;
        
        for (var ch = 0u; ch < 3u; ch++) {
            let grad_pix = dL_dpixel[ch];
            
            // Color gradient
            let dchannel_dcolor = alpha * T;
            let dL_dc = dchannel_dcolor * grad_pix;
            
            atomicAddFloat(&grad_colors[global_gaussian_idx * 3u + ch], dL_dc);
            dL_dalpha += (color[ch] - accum_rec[ch]) * grad_pix;
        }
        
        dL_dalpha *= T;

        // Prepare for next iteration
        last_alpha = alpha;
        last_color = color;
        
        // Geometry Gradients
        let dL_dG = opacity * dL_dalpha;
        
        let dL_dopacity = G * dL_dalpha;
        atomicAddFloat(&grad_opacity[global_gaussian_idx], dL_dopacity);

        let dpow_dx = 2.0 * conic.x * delta.x + 2.0 * conic.y * delta.y;
        let dpow_dy = 2.0 * conic.z * delta.y + 2.0 * conic.y * delta.x;
        
        let dG_ddelta_x = -0.5 * G * dpow_dx;
        let dG_ddelta_y = -0.5 * G * dpow_dy;
        
        let dL_dmean_x = dL_dG * (-dG_ddelta_x);
        let dL_dmean_y = dL_dG * (-dG_ddelta_y);
        
        atomicAddFloat(&grad_means_2d[global_gaussian_idx * 2u + 0u], dL_dmean_x);
        atomicAddFloat(&grad_means_2d[global_gaussian_idx * 2u + 1u], dL_dmean_y);
        
        let dL_dconic_x = dL_dG * (-0.5 * G * delta.x * delta.x);
        let dL_dconic_y = dL_dG * (-0.5 * G * 2.0 * delta.x * delta.y);
        let dL_dconic_z = dL_dG * (-0.5 * G * delta.y * delta.y);
        
        atomicAddFloat(&grad_conics[global_gaussian_idx * 4u + 0u], dL_dconic_x);
        atomicAddFloat(&grad_conics[global_gaussian_idx * 4u + 1u], dL_dconic_y);
        atomicAddFloat(&grad_conics[global_gaussian_idx * 4u + 3u], dL_dconic_z);
    }
}
