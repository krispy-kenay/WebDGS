/*
 * Opacity reset: caps every opacity at a maximum and restarts its Adam moments.
 * Floaters that are not needed then stay transparent and get pruned.
 */

struct OptFloat {
    param: f32,
    m: f32,
    v: f32,
};

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot: array<u32, 2>,
    scale: array<u32, 2>,
};

struct ResetInfo {
    num_points: u32,
    max_opacity_raw: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> info: ResetInfo;
@group(0) @binding(1) var<storage, read_write> opt_opacity: array<OptFloat>;
@group(0) @binding(2) var<storage, read_write> gaussians: array<Gaussian>;

@compute @workgroup_size(256)
fn opacity_reset_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= info.num_points) {
        return;
    }
    let raw = min(opt_opacity[idx].param, info.max_opacity_raw);
    opt_opacity[idx] = OptFloat(raw, 0.0, 0.0);

    // The renderer reads the packed copy
    let p1 = unpack2x16float(gaussians[idx].pos_opacity[1]);
    gaussians[idx].pos_opacity[1] = pack2x16float(vec2<f32>(p1.x, raw));
}
