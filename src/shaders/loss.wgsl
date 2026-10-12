/*
 * Loss Calculation Shader
 *
 * Gradient of lambda_l1 * L1 + lambda_l2 * L2 + lambda_dssim * (1 - SSIM), summed over pixels and channels.
 */

struct TrainingConfig {
    lambda_l1: f32,
    lambda_l2: f32,
    lambda_dssim: f32,
    c1: f32,
    c2: f32,
};

@group(0) @binding(0) var pred_texture: texture_2d<f32>;
@group(0) @binding(1) var target_texture: texture_2d<f32>;
@group(0) @binding(2) var out_gradient: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> config: TrainingConfig;
@group(0) @binding(4) var<storage, read_write> ssim_partials: array<f32>;

const WINDOW_RADIUS: i32 = 5;
const WINDOW = array<f32, 11>(0.001028380, 0.007598758, 0.036000772, 0.109360690, 0.213005538, 0.266011725,
                              0.213005538, 0.109360690, 0.036000772, 0.007598758, 0.001028380);

fn inside(p: vec2<i32>, dims: vec2<u32>) -> bool {
    return p.x >= 0 && p.y >= 0 && p.x < i32(dims.x) && p.y < i32(dims.y);
}

@compute @workgroup_size(16, 16)
fn ssim_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(pred_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    var mx = vec3<f32>(0.0);
    var my = vec3<f32>(0.0);
    var xx = vec3<f32>(0.0);
    var yy = vec3<f32>(0.0);
    var xy = vec3<f32>(0.0);
    for (var dy = -WINDOW_RADIUS; dy <= WINDOW_RADIUS; dy++) {
        for (var dx = -WINDOW_RADIUS; dx <= WINDOW_RADIUS; dx++) {
            let p = vec2<i32>(global_id.xy) + vec2<i32>(dx, dy);
            if (!inside(p, dims)) {
                continue;
            }
            let w = WINDOW[dx + WINDOW_RADIUS] * WINDOW[dy + WINDOW_RADIUS];
            let x = textureLoad(pred_texture, p, 0).rgb;
            let y = textureLoad(target_texture, p, 0).rgb;
            mx += w * x;
            my += w * y;
            xx += w * x * x;
            yy += w * y * y;
            xy += w * x * y;
        }
    }

    let var_x = xx - mx * mx;
    let var_y = yy - my * my;
    let cov_xy = xy - mx * my;
    let a1 = 2.0 * mx * my + config.c1;
    let a2 = 2.0 * cov_xy + config.c2;
    let b1 = mx * mx + my * my + config.c1;
    let b2 = var_x + var_y + config.c2;
    let ssim = a1 * a2 / (b1 * b2);

    let d_cov = 2.0 * a1 / (b1 * b2);
    let d_var = -ssim / b2;
    // mu_x also enters var_x and cov_xy, which are written in terms of E[x^2] and E[xy]
    let d_mean = 2.0 * my * a2 / (b1 * b2) - 2.0 * mx * ssim / b1 - 2.0 * mx * d_var - my * d_cov;

    let i = (global_id.y * dims.x + global_id.x) * 9u;
    ssim_partials[i] = d_mean.r;
    ssim_partials[i + 1u] = d_mean.g;
    ssim_partials[i + 2u] = d_mean.b;
    ssim_partials[i + 3u] = d_var.r;
    ssim_partials[i + 4u] = d_var.g;
    ssim_partials[i + 5u] = d_var.b;
    ssim_partials[i + 6u] = d_cov.r;
    ssim_partials[i + 7u] = d_cov.g;
    ssim_partials[i + 8u] = d_cov.b;
}

@compute @workgroup_size(16, 16)
fn compute_loss_grad(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(pred_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(global_id.xy);
    let pred = textureLoad(pred_texture, coord, 0).rgb;
    let targ = textureLoad(target_texture, coord, 0).rgb;
    let diff = pred - targ;

    var total_grad = config.lambda_l1 * sign(diff) + config.lambda_l2 * diff;

    if (config.lambda_dssim > 0.0) {
        var d_mean = vec3<f32>(0.0);
        var d_var = vec3<f32>(0.0);
        var d_cov = vec3<f32>(0.0);
        for (var dy = -WINDOW_RADIUS; dy <= WINDOW_RADIUS; dy++) {
            for (var dx = -WINDOW_RADIUS; dx <= WINDOW_RADIUS; dx++) {
                let q = coord + vec2<i32>(dx, dy);
                if (!inside(q, dims)) {
                    continue;
                }
                let w = WINDOW[dx + WINDOW_RADIUS] * WINDOW[dy + WINDOW_RADIUS];
                let i = (u32(q.y) * dims.x + u32(q.x)) * 9u;
                d_mean += w * vec3<f32>(ssim_partials[i], ssim_partials[i + 1u], ssim_partials[i + 2u]);
                d_var += w * vec3<f32>(ssim_partials[i + 3u], ssim_partials[i + 4u], ssim_partials[i + 5u]);
                d_cov += w * vec3<f32>(ssim_partials[i + 6u], ssim_partials[i + 7u], ssim_partials[i + 8u]);
            }
        }
        total_grad -= config.lambda_dssim * (d_mean + 2.0 * pred * d_var + targ * d_cov);
    }

    textureStore(out_gradient, coord, vec4<f32>(total_grad, 1.0));
}
