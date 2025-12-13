/*
 * Loss Calculation Shader
 * 
 * Computes gradients for L1, L2, and DSSIM losses.
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

fn samplePred(coord: vec2<i32>, dims: vec2<u32>) -> vec3<f32> {
    let c = clamp(coord, vec2<i32>(0, 0), vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1));
    return textureLoad(pred_texture, c, 0).rgb;
}

fn sampleTarg(coord: vec2<i32>, dims: vec2<u32>) -> vec3<f32> {
    let c = clamp(coord, vec2<i32>(0, 0), vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1));
    return textureLoad(target_texture, c, 0).rgb;
}

fn computeSSIMGrad(center: vec2<i32>, dims: vec2<u32>) -> vec3<f32> {
    var mu_x = vec3<f32>(0.0);
    var mu_y = vec3<f32>(0.0);
    let window_size = 5;
    let half_window = 2;
    let n = f32(window_size * window_size);
    
    for (var dy = -half_window; dy <= half_window; dy = dy + 1) {
        for (var dx = -half_window; dx <= half_window; dx = dx + 1) {
            let coord = center + vec2<i32>(dx, dy);
            mu_x += samplePred(coord, dims);
            mu_y += sampleTarg(coord, dims);
        }
    }
    mu_x /= n;
    mu_y /= n;
    
    var sigma_x2 = vec3<f32>(0.0);
    var sigma_y2 = vec3<f32>(0.0);
    var sigma_xy = vec3<f32>(0.0);
    
    for (var dy = -half_window; dy <= half_window; dy = dy + 1) {
        for (var dx = -half_window; dx <= half_window; dx = dx + 1) {
            let coord = center + vec2<i32>(dx, dy);
            let x = samplePred(coord, dims);
            let y = sampleTarg(coord, dims);
            let dx_val = x - mu_x;
            let dy_val = y - mu_y;
            sigma_x2 += dx_val * dx_val;
            sigma_y2 += dy_val * dy_val;
            sigma_xy += dx_val * dy_val;
        }
    }
    sigma_x2 /= n;
    sigma_y2 /= n;
    sigma_xy /= n;
    
    let num1 = 2.0 * mu_x * mu_y + config.c1;
    let num2 = 2.0 * sigma_xy + config.c2;
    let den1 = mu_x * mu_x + mu_y * mu_y + config.c1;
    let den2 = sigma_x2 + sigma_y2 + config.c2;
    
    let ssim = (num1 * num2) / (den1 * den2);
    
    let pred = samplePred(center, dims);
    let targ = sampleTarg(center, dims);
    let dssim = (vec3<f32>(1.0) - ssim) * 0.5;
    
    // Simplification for gradient direction
    let grad_ssim = dssim * (pred - targ);
    
    return grad_ssim;
}


@compute @workgroup_size(16, 16)
fn compute_loss_grad(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(pred_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
    
    let pred = textureLoad(pred_texture, coord, 0).rgb;
    let targ = textureLoad(target_texture, coord, 0).rgb;
    
    // L1 Gradient
    let diff = pred - targ;
    let grad_l1 = sign(diff);
    
    // L2 Gradient
    let grad_l2 = diff;
    
    // DSSIM Gradient
    var grad_dssim = vec3<f32>(0.0);
    if (config.lambda_dssim > 0.0) {
        grad_dssim = computeSSIMGrad(coord, dims);
    }
    
    // Combine
    let total_grad = config.lambda_l1 * grad_l1 + 
                     config.lambda_l2 * grad_l2 + 
                     config.lambda_dssim * grad_dssim;
                     
    textureStore(out_gradient, coord, vec4<f32>(total_grad, 1.0));
}
