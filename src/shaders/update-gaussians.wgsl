/*
 * Update Gaussians Shader
 *
 * Packs f32 training parameters back into the packed f16 layout used by the renderer.
 */

struct Gaussian {
  pos_opacity: array<u32, 2>,
  rot:         array<u32, 2>,
  scale:       array<u32, 2>,
};

struct OptVec4 {
    param: vec4<f32>,
    m: vec4<f32>,
    v: vec4<f32>,
};

struct OptFloat {
    param: f32,
    m: f32,
    v: f32,
};

// Bindings
@group(0) @binding(0) var<storage, read> opt_pos: array<OptVec4>;
@group(0) @binding(1) var<storage, read> opt_rot: array<OptVec4>;
@group(0) @binding(2) var<storage, read> opt_scale: array<OptVec4>;
@group(0) @binding(3) var<storage, read> opt_opacity: array<OptFloat>;
@group(0) @binding(4) var<storage, read> param_sh: array<f32>;

@group(1) @binding(0) var<storage, read_write> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> sh_buffer: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&opt_pos)) {
        return;
    }

    // Pack Position + Opacity
    let p = opt_pos[idx].param;
    let op = opt_opacity[idx].param;
    
    gaussians[idx].pos_opacity[0] = pack2x16float(vec2<f32>(p.x, p.y));
    gaussians[idx].pos_opacity[1] = pack2x16float(vec2<f32>(p.z, op));

    // Pack Rotation
    let r = opt_rot[idx].param;
    gaussians[idx].rot[0] = pack2x16float(vec2<f32>(r.x, r.y));
    gaussians[idx].rot[1] = pack2x16float(vec2<f32>(r.z, r.w));

    // Pack Scale
    let s = opt_scale[idx].param;
    gaussians[idx].scale[0] = pack2x16float(vec2<f32>(s.x, s.y));
    gaussians[idx].scale[1] = pack2x16float(vec2<f32>(s.z, 0.0));

    // Pack SH (DC only)
    const SH_WORDS_PER_POINT: u32 = 24u;
    let base_word = idx * SH_WORDS_PER_POINT;
    
    {
        let c0 = param_sh[idx * 48u + 0u];
        let c1 = param_sh[idx * 48u + 1u];
        sh_buffer[base_word + 0u] = pack2x16float(vec2<f32>(c0, c1));
    }
    {
        let c2 = param_sh[idx * 48u + 2u];
        let old_word_1 = sh_buffer[base_word + 1u];
        let unpacked_1 = unpack2x16float(old_word_1);
        let c3 = unpacked_1.y;
        
        sh_buffer[base_word + 1u] = pack2x16float(vec2<f32>(c2, c3));
    }
    
}
