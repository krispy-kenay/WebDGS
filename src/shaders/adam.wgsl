/*
 * Adam Optimizer Shader
 *
 * Updates Gaussian parameters using gradients from the backward pass.
 */

struct AdamConfig {
    lr_pos: f32,
    lr_color: f32,
    lr_opacity: f32,
    lr_scale: f32,
    lr_rot: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    iteration: u32,
};

struct GaussianGradient {
    pos_opacity: array<u32, 2>,
    rot:         array<u32, 2>,
    scale:       array<u32, 2>,
    color:       array<u32, 2>,
};

// Bindings
@group(0) @binding(0) var<uniform> config: AdamConfig;
@group(0) @binding(1) var<storage, read> tile_counts: array<u32>;

@group(1) @binding(0) var<storage, read> gradients: array<GaussianGradient>;

// Optimization Structs
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

@group(2) @binding(0) var<storage, read_write> opt_pos: array<OptVec4>;
@group(2) @binding(1) var<storage, read_write> opt_rot: array<OptVec4>;
@group(2) @binding(2) var<storage, read_write> opt_scale: array<OptVec4>;
@group(2) @binding(3) var<storage, read_write> opt_opacity: array<OptFloat>;
@group(2) @binding(4) var<storage, read_write> param_sh: array<f32>;
@group(2) @binding(5) var<storage, read_write> state_sh: array<vec2<f32>>;

// Helpers
fn adam_step(param: f32, grad: f32, m: f32, v: f32, lr: f32) -> vec3<f32> {
    let beta1 = config.beta1;
    let beta2 = config.beta2;
    let eps = config.epsilon;

    let m_new = beta1 * m + (1.0 - beta1) * grad;
    let v_new = beta2 * v + (1.0 - beta2) * grad * grad;
    
    let step = -lr * m_new / (sqrt(v_new) + eps);
    let param_new = param + step;

    return vec3<f32>(param_new, m_new, v_new);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&opt_pos)) {
        return;
    }

    if (tile_counts[idx] == 0u) {
        return;
    }

    // Unpack Gradients
    let grad_packed = gradients[idx];
    
    let g_pos_xy = unpack2x16float(grad_packed.pos_opacity[0]);
    let g_pos_z_op = unpack2x16float(grad_packed.pos_opacity[1]);
    let grad_pos = vec3<f32>(g_pos_xy.x, g_pos_xy.y, g_pos_z_op.x);
    let grad_opac = g_pos_z_op.y;

    let g_rot_xy = unpack2x16float(grad_packed.rot[0]);
    let g_rot_zw = unpack2x16float(grad_packed.rot[1]);
    let grad_rot = vec4<f32>(g_rot_xy.x, g_rot_xy.y, g_rot_zw.x, g_rot_zw.y);

    let g_scale_xy = unpack2x16float(grad_packed.scale[0]);
    let g_scale_z_ = unpack2x16float(grad_packed.scale[1]);
    let grad_scale = vec3<f32>(g_scale_xy.x, g_scale_xy.y, g_scale_z_.x);

    let g_col_rg = unpack2x16float(grad_packed.color[0]);
    let g_col_b_ = unpack2x16float(grad_packed.color[1]);
    let grad_color = vec3<f32>(g_col_rg.x, g_col_rg.y, g_col_b_.x);

    // Update Position
    {
        let p = opt_pos[idx].param;
        let m_vec = opt_pos[idx].m;
        let v_vec = opt_pos[idx].v;
        
        let res_x = adam_step(p.x, grad_pos.x, m_vec.x, v_vec.x, config.lr_pos);
        let res_y = adam_step(p.y, grad_pos.y, m_vec.y, v_vec.y, config.lr_pos);
        let res_z = adam_step(p.z, grad_pos.z, m_vec.z, v_vec.z, config.lr_pos);
        
        opt_pos[idx].param = vec4<f32>(res_x.x, res_y.x, res_z.x, 1.0);
        opt_pos[idx].m = vec4<f32>(res_x.y, res_y.y, res_z.y, 0.0);
        opt_pos[idx].v = vec4<f32>(res_x.z, res_y.z, res_z.z, 0.0);
    }

    // Update Rotation
    {
        let r = opt_rot[idx].param;
        let m_vec = opt_rot[idx].m;
        let v_vec = opt_rot[idx].v;

        let res_x = adam_step(r.x, grad_rot.x, m_vec.x, v_vec.x, config.lr_rot);
        let res_y = adam_step(r.y, grad_rot.y, m_vec.y, v_vec.y, config.lr_rot);
        let res_z = adam_step(r.z, grad_rot.z, m_vec.z, v_vec.z, config.lr_rot);
        let res_w = adam_step(r.w, grad_rot.w, m_vec.w, v_vec.w, config.lr_rot);

        var new_rot = vec4<f32>(res_x.x, res_y.x, res_z.x, res_w.x);
        new_rot = normalize(new_rot);

        opt_rot[idx].param = new_rot;
        opt_rot[idx].m = vec4<f32>(res_x.y, res_y.y, res_z.y, res_w.y);
        opt_rot[idx].v = vec4<f32>(res_x.z, res_y.z, res_z.z, res_w.z);
    }

    // Update Scale
    {
        let s = opt_scale[idx].param;
        let m_vec = opt_scale[idx].m;
        let v_vec = opt_scale[idx].v;

        let res_x = adam_step(s.x, grad_scale.x, m_vec.x, v_vec.x, config.lr_scale);
        let res_y = adam_step(s.y, grad_scale.y, m_vec.y, v_vec.y, config.lr_scale);
        let res_z = adam_step(s.z, grad_scale.z, m_vec.z, v_vec.z, config.lr_scale);

        opt_scale[idx].param = vec4<f32>(res_x.x, res_y.x, res_z.x, 0.0);
        opt_scale[idx].m = vec4<f32>(res_x.y, res_y.y, res_z.y, 0.0);
        opt_scale[idx].v = vec4<f32>(res_x.z, res_y.z, res_z.z, 0.0);
    }

    // Update Opacity
    {
        let op = opt_opacity[idx].param;
        let m_val = opt_opacity[idx].m;
        let v_val = opt_opacity[idx].v;
        
        let res = adam_step(op, grad_opac, m_val, v_val, config.lr_opacity);
        
        opt_opacity[idx].param = res.x;
        opt_opacity[idx].m = res.y;
        opt_opacity[idx].v = res.z;
    }

    // Update Color (DC)
    {
        for (var c = 0u; c < 3u; c++) {
             let sh_idx = idx * 48u + c; 
             
             let p = param_sh[sh_idx];
             let m_vec = state_sh[sh_idx];
             let g = grad_color[c];
             
             let res = adam_step(p, g, m_vec.x, m_vec.y, config.lr_color);
             
             param_sh[sh_idx] = res.x;
             state_sh[sh_idx] = vec2<f32>(res.y, res.z);
        }
    }
}
