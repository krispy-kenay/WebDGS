/*
 * Tiled Backward Geometry
 *
 * Propagates gradients from 2D Gaussian properties to 3D properties (Means, Scales, Rotations, SH/Color).
 */

// Bindings
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;

@group(1) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> grad_means_2d: array<atomic<i32>>; 
@group(1) @binding(2) var<storage, read_write> grad_conics: array<atomic<i32>>;
@group(1) @binding(3) var<storage, read_write> grad_opacity: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grad_colors: array<atomic<i32>>; 

// Packed Gradient Struct
struct GaussianGradient {
    pos_opacity: array<u32, 2>,
    rot:         array<u32, 2>,
    scale:       array<u32, 2>,
    color:       array<u32, 2>,
};

@group(2) @binding(0) var<storage, read_write> gradients: array<GaussianGradient>;

// Constants
const WORKGROUP_SIZE: u32 = 256;
const SCALE_MODIFIER = 1.0;

// Helper to unpack f16
fn unpack_half2(val: u32) -> vec2<f32> {
    return unpack2x16float(val);
}

// Transform point
fn transformPoint4x3(pos: vec3<f32>, view: mat4x4<f32>) -> vec3<f32> {
    return (view * vec4<f32>(pos, 1.0)).xyz;
}

@compute @workgroup_size(64)
fn main_geometry_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // 1. Load Gradients
    var dL_dmean2D_px = vec2<f32>(
        atomicLoadFloat(&grad_means_2d[idx * 2u + 0u]),
        atomicLoadFloat(&grad_means_2d[idx * 2u + 1u])
    );
    
    // grad_conics
    let dL_dconic = vec3<f32>(
        atomicLoadFloat(&grad_conics[idx * 4u + 0u]),
        atomicLoadFloat(&grad_conics[idx * 4u + 1u]),
        atomicLoadFloat(&grad_conics[idx * 4u + 3u])
    );
    
    let dL_dopac = atomicLoadFloat(&grad_opacity[idx]);
    
    // Load Gaussian State
    let g = gaussians[idx];
    let p_xy = unpack2x16float(g.pos_opacity[0]);
    let p_z_op = unpack2x16float(g.pos_opacity[1]);
    let mean3D = vec3<f32>(p_xy.x, p_xy.y, p_z_op.x);
    // Opacity handled via gradient directly
    let opacity_raw = p_z_op.y;
    let opacity_sigmoid = 1.0 / (1.0 + exp(-opacity_raw));
    
    let r_xy = unpack2x16float(g.rot[0]);
    let r_zw = unpack2x16float(g.rot[1]);
    let rot = vec4<f32>(r_xy.x, r_xy.y, r_zw.x, r_zw.y);
    
    let s_xy = unpack2x16float(g.scale[0]);
    let s_z_ = unpack2x16float(g.scale[1]);

    let log_scale = vec3<f32>(s_xy.x, s_xy.y, s_z_.x);
    let scale = exp(log_scale);

    // Recompute Forward State
    let cov3D_flat = covariance3D(rot, scale);
    
    let view = camera.view;
    // let proj = camera.proj; 
    
    let t = (view * vec4<f32>(mean3D, 1.0)).xyz;
    
    // Compute Projection Gradient
    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    let dL_dmean2D_ndc = dL_dmean2D_px * 0.5 * viewport; 
    
    let view_proj = camera.proj * camera.view; 
    let p_hom = view_proj * vec4<f32>(mean3D, 1.0);
    let rw = 1.0 / (p_hom.w + 0.0000001);
    let rw2 = rw * rw;
    
    let dL_dphom = vec4<f32>(
        dL_dmean2D_ndc.x * rw,
        dL_dmean2D_ndc.y * rw,
        0.0,
        -(dL_dmean2D_ndc.x * p_hom.x + dL_dmean2D_ndc.y * p_hom.y) * rw2
    );
    
    let dL_dmean3D_proj = (transpose(view_proj) * dL_dphom).xyz;
    
    // Compute Cov2D -> Cov3D + Mean3D Gradients
    let focal_x = camera.focal.x;
    let focal_y = camera.focal.y;
    // Limits
    let limx = 1.3 * viewport.x * 0.5 / focal_x; 
    let limy = 1.3 * viewport.y * 0.5 / focal_y;
    
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    
    let t_clamped_x = min(limx, max(-limx, txtz)) * t.z;
    let t_clamped_y = min(limy, max(-limy, tytz)) * t.z;
    
    // Gradients for clamping
    var x_grad_mul = 0.0;
    if (txtz >= -limx && txtz <= limx) { x_grad_mul = 1.0; }
    var y_grad_mul = 0.0;
    if (tytz >= -limy && tytz <= limy) { y_grad_mul = 1.0; }
    
    let J = mat3x3<f32>(
        vec3<f32>(focal_x / t.z, 0.0, -(focal_x * t_clamped_x) / (t.z * t.z)),
        vec3<f32>(0.0, focal_y / t.z, -(focal_y * t_clamped_y) / (t.z * t.z)),
        vec3<f32>(0.0, 0.0, 0.0)
    );
    
    // view matrix 3x3
    let W = mat3x3<f32>(
        view[0].xyz,
        view[1].xyz,
        view[2].xyz
    );
    
    let T_mat = W * J;
    
    let Vrk = mat3x3<f32>(
        vec3<f32>(cov3D_flat[0], cov3D_flat[1], cov3D_flat[2]),
        vec3<f32>(cov3D_flat[1], cov3D_flat[3], cov3D_flat[4]),
        vec3<f32>(cov3D_flat[2], cov3D_flat[4], cov3D_flat[5])
    );
    
    // Forward Cov2D
    let cov2D = transpose(T_mat) * Vrk * T_mat;
    let a = cov2D[0][0] + 0.3;
    let b = cov2D[0][1];
    let c = cov2D[1][1] + 0.3;
    
    // Gradient Prop
    let denom = a * c - b * b;
    let denom2inv = 1.0 / ((denom * denom) + 0.0000001);
    
    var dL_da = 0.0;
    var dL_db = 0.0;
    var dL_dc = 0.0;

    if (denom2inv != 0.0) {
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2.0 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2.0 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2.0 * (b * c * dL_dconic.x - (denom + 2.0 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
    }
    
    // dL_dcov3D
    var dL_dcov3D_flat: array<f32, 6>;
    
    dL_dcov3D_flat[0] = (T_mat[0][0] * T_mat[0][0] * dL_da + T_mat[0][0] * T_mat[1][0] * dL_db + T_mat[1][0] * T_mat[1][0] * dL_dc);
    dL_dcov3D_flat[3] = (T_mat[0][1] * T_mat[0][1] * dL_da + T_mat[0][1] * T_mat[1][1] * dL_db + T_mat[1][1] * T_mat[1][1] * dL_dc);
    dL_dcov3D_flat[5] = (T_mat[0][2] * T_mat[0][2] * dL_da + T_mat[0][2] * T_mat[1][2] * dL_db + T_mat[1][2] * T_mat[1][2] * dL_dc);
    
    dL_dcov3D_flat[1] = 2.0 * T_mat[0][0] * T_mat[0][1] * dL_da + (T_mat[0][0] * T_mat[1][1] + T_mat[0][1] * T_mat[1][0]) * dL_db + 2.0 * T_mat[1][0] * T_mat[1][1] * dL_dc;
    dL_dcov3D_flat[2] = 2.0 * T_mat[0][0] * T_mat[0][2] * dL_da + (T_mat[0][0] * T_mat[1][2] + T_mat[0][2] * T_mat[1][0]) * dL_db + 2.0 * T_mat[1][0] * T_mat[1][2] * dL_dc;
    dL_dcov3D_flat[4] = 2.0 * T_mat[0][2] * T_mat[0][1] * dL_da + (T_mat[0][1] * T_mat[1][2] + T_mat[0][2] * T_mat[1][1]) * dL_db + 2.0 * T_mat[1][1] * T_mat[1][2] * dL_dc;
    
    // Gradients T
    let dL_dT00 = 2.0 * (T_mat[0][0] * Vrk[0][0] + T_mat[0][1] * Vrk[0][1] + T_mat[0][2] * Vrk[0][2]) * dL_da +
                  (T_mat[1][0] * Vrk[0][0] + T_mat[1][1] * Vrk[0][1] + T_mat[1][2] * Vrk[0][2]) * dL_db;
    let dL_dT01 = 2.0 * (T_mat[0][0] * Vrk[1][0] + T_mat[0][1] * Vrk[1][1] + T_mat[0][2] * Vrk[1][2]) * dL_da +
                  (T_mat[1][0] * Vrk[1][0] + T_mat[1][1] * Vrk[1][1] + T_mat[1][2] * Vrk[1][2]) * dL_db;
    let dL_dT02 = 2.0 * (T_mat[0][0] * Vrk[2][0] + T_mat[0][1] * Vrk[2][1] + T_mat[0][2] * Vrk[2][2]) * dL_da +
                  (T_mat[1][0] * Vrk[2][0] + T_mat[1][1] * Vrk[2][1] + T_mat[1][2] * Vrk[2][2]) * dL_db;
    let dL_dT10 = 2.0 * (T_mat[1][0] * Vrk[0][0] + T_mat[1][1] * Vrk[0][1] + T_mat[1][2] * Vrk[0][2]) * dL_dc +
                  (T_mat[0][0] * Vrk[0][0] + T_mat[0][1] * Vrk[0][1] + T_mat[0][2] * Vrk[0][2]) * dL_db;
    let dL_dT11 = 2.0 * (T_mat[1][0] * Vrk[1][0] + T_mat[1][1] * Vrk[1][1] + T_mat[1][2] * Vrk[1][2]) * dL_dc +
                  (T_mat[0][0] * Vrk[1][0] + T_mat[0][1] * Vrk[1][1] + T_mat[0][2] * Vrk[1][2]) * dL_db;
    let dL_dT12 = 2.0 * (T_mat[1][0] * Vrk[2][0] + T_mat[1][1] * Vrk[2][1] + T_mat[1][2] * Vrk[2][2]) * dL_dc +
                  (T_mat[0][0] * Vrk[2][0] + T_mat[0][1] * Vrk[2][1] + T_mat[0][2] * Vrk[2][2]) * dL_db;

    // Gradients J
    let dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    let dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    let dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    let dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;
    
    // Gradients t
    let tz = 1.0 / t.z;
    let tz2 = tz * tz;
    let tz3 = tz2 * tz;
    
    let dL_dtx = x_grad_mul * -focal_x * tz2 * dL_dJ02;
    let dL_dty = y_grad_mul * -focal_y * tz2 * dL_dJ12;
    let dL_dtz = -focal_x * tz2 * dL_dJ00 - focal_y * tz2 * dL_dJ11 + (2.0 * focal_x * t_clamped_x) * tz3 * dL_dJ02 + (2.0 * focal_y * t_clamped_y) * tz3 * dL_dJ12;
    
    let dL_dmean3D_cov = (transpose(view) * vec4<f32>(dL_dtx, dL_dty, dL_dtz, 0.0)).xyz;
    
    // Compute Scale/Rot Gradients
    let x = rot.y; let y = rot.z; let z = rot.w; let r = rot.x;
    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)),
        vec3<f32>(2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)),
        vec3<f32>(2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y))
    );
    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z)
    );
    let M = S * R; 
    
    let dL_dSigma = mat3x3<f32>(
        vec3<f32>(dL_dcov3D_flat[0], 0.5 * dL_dcov3D_flat[1], 0.5 * dL_dcov3D_flat[2]),
        vec3<f32>(0.5 * dL_dcov3D_flat[1], dL_dcov3D_flat[3], 0.5 * dL_dcov3D_flat[4]),
        vec3<f32>(0.5 * dL_dcov3D_flat[2], 0.5 * dL_dcov3D_flat[4], dL_dcov3D_flat[5])
    );
    
    let dL_dM = 2.0 * M * dL_dSigma;
    
    let dL_dMt = transpose(dL_dM);
    let Rt = transpose(R);
    
    var dL_dscale = vec3<f32>(
        dot(Rt[0], dL_dMt[0]),
        dot(Rt[1], dL_dMt[1]),
        dot(Rt[2], dL_dMt[2])
    );
    
    var dL_dMt_scaled = dL_dMt;
    dL_dMt_scaled[0] = dL_dMt[0] * scale.x; 
    dL_dMt_scaled[1] = dL_dMt[1] * scale.y;
    dL_dMt_scaled[2] = dL_dMt[2] * scale.z;
    
    // Quat gradients
    let dL_drot_x = 2.0*z*(dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) + 2.0*y*(dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) + 2.0*x*(dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]);
    let dL_drot_y = 2.0*y*(dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) + 2.0*z*(dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) + 2.0*r*(dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]) - 4.0*x*(dL_dMt_scaled[2][2] + dL_dMt_scaled[1][1]);
    let dL_drot_z = 2.0*x*(dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) + 2.0*r*(dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) + 2.0*z*(dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) - 4.0*y*(dL_dMt_scaled[2][2] + dL_dMt_scaled[0][0]);
    let dL_drot_w = 2.0*r*(dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) + 2.0*x*(dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) + 2.0*y*(dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) - 4.0*z*(dL_dMt_scaled[1][1] + dL_dMt_scaled[0][0]);
    
    // Final Means Gradient sum
    let final_dL_dmean3D = dL_dmean3D_proj + dL_dmean3D_cov;

    // Convert opacity gradient from sigmoid-space to raw-space for the optimizer.
    let dL_dopacity_raw = dL_dopac * opacity_sigmoid * (1.0 - opacity_sigmoid);

    // Convert scale gradient from scale-space to log-scale-space for the optimizer.
    var dL_dlog_scale = dL_dscale * scale;

    // Screen-space radius cap
    {
        let cap_px = settings.max_splat_radius_px;
        if (cap_px > 0.0) {
            let denom_cap = a * c - b * b;
            if (denom_cap > 0.0) {
                let conic_x = c / denom_cap;
                let conic_y = -b / denom_cap;
                let conic_z = a / denom_cap;
                let disc = conic_y * conic_y - conic_x * conic_z;

                let opacity_threshold = 128.0;
                let t_cap = 2.0 * log(opacity_sigmoid * opacity_threshold);
                if (t_cap > 0.0 && disc < 0.0) {
                    let x_extent = sqrt(t_cap * conic_z / (-disc));
                    let y_extent = sqrt(t_cap * conic_x / (-disc));
                    if (max(x_extent, y_extent) >= cap_px) {
                        dL_dlog_scale = max(dL_dlog_scale, vec3<f32>(0.0));
                    }
                }
            }
        }
    }
    
    // Store outputs (Packed)
    gradients[idx].pos_opacity[0] = pack2x16float(vec2<f32>(final_dL_dmean3D.x, final_dL_dmean3D.y));
    gradients[idx].pos_opacity[1] = pack2x16float(vec2<f32>(final_dL_dmean3D.z, dL_dopacity_raw));
    gradients[idx].rot[0] = pack2x16float(vec2<f32>(dL_drot_x, dL_drot_y));
    gradients[idx].rot[1] = pack2x16float(vec2<f32>(dL_drot_z, dL_drot_w));
    gradients[idx].scale[0] = pack2x16float(vec2<f32>(dL_dlog_scale.x, dL_dlog_scale.y));
    gradients[idx].scale[1] = pack2x16float(vec2<f32>(dL_dlog_scale.z, 0.0));
    
    let dL_dcr = atomicLoadFloat(&grad_colors[idx * 3u + 0u]);
    let dL_dcg = atomicLoadFloat(&grad_colors[idx * 3u + 1u]);
    let dL_dcb = atomicLoadFloat(&grad_colors[idx * 3u + 2u]);
    gradients[idx].color[0] = pack2x16float(vec2<f32>(dL_dcr, dL_dcg));
    gradients[idx].color[1] = pack2x16float(vec2<f32>(dL_dcb, 0.0));
}

// Helper for unpacking opacity
fn unquantize_opacity(po: array<u32, 2>) -> f32 {
    let p_z_op = unpack2x16float(po[1]);
    return 1.0 / (1.0 + exp(-p_z_op.y));
}
