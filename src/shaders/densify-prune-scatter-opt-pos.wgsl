struct ScatterInfo {
  in_points: u32,
  out_points: u32,
  reset_new_state: u32,
  _pad0: u32,
};

struct OptVec4 {
  param: vec4<f32>,
  m: vec4<f32>,
  v: vec4<f32>,
};

@group(0) @binding(0) var<uniform> info: ScatterInfo;
@group(0) @binding(1) var<storage, read> out_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> out_counts: array<u32>;
@group(0) @binding(3) var<storage, read> out_actions: array<u32>;

@group(1) @binding(0) var<storage, read> in_pos: array<OptVec4>;
@group(1) @binding(1) var<storage, read> in_rot: array<OptVec4>;
@group(1) @binding(2) var<storage, read> in_scale: array<OptVec4>;

@group(2) @binding(0) var<storage, read_write> out_pos: array<OptVec4>;

fn hash_u32(x: u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u);
  v = v * 0x7feb352du;
  v = v ^ (v >> 15u);
  v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash_u32(seed)) * (1.0 / 4294967296.0);
}

fn clamp_log_scale(ls: vec3<f32>) -> vec3<f32> {
  return clamp(ls, vec3<f32>(-10.0), vec3<f32>(10.0));
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
  let len2 = max(1e-12, dot(q, q));
  return q * inverseSqrt(len2);
}

fn quat_rotate(q_in: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let q = quat_normalize(q_in);
  let u = vec3<f32>(q.y, q.z, q.w);
  let s = q.x;
  return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn randn_approx(seed: u32) -> f32 {
  var s: f32 = 0.0;
  s = s + rand01(seed ^ 0xA2C79u);
  s = s + rand01(seed ^ 0x5E2D9u);
  s = s + rand01(seed ^ 0x1B873u);
  s = s + rand01(seed ^ 0xC0FFEu);
  s = s + rand01(seed ^ 0xBADC0u);
  s = s + rand01(seed ^ 0xDEADBu);
  return (s - 3.0) * 1.41421356237;
}

fn write_slot(dst_idx: u32, src_idx: u32, variant: u32, action: u32) {
  let src = in_pos[src_idx];

  let is_new = (variant == 1u) || (action == 2u);
  let reset = (info.reset_new_state != 0u) && is_new;

  var p = src.param;

  // Derive sigma + quaternion from optimizer buffers, not from packed f16.
  let q = in_rot[src_idx].param;
  let log_sigma = clamp_log_scale(in_scale[src_idx].param.xyz);
  let sigma = exp(log_sigma);

  if (action == 1u && variant == 1u) {
    // Clone jitter
    let seed = src_idx * 1664525u + dst_idx * 1013904223u;
    let r = vec3<f32>(
      rand01(seed ^ 0xA2C79u),
      rand01(seed ^ 0x5E2D9u),
      rand01(seed ^ 0x1B873u)
    ) * 2.0 - 1.0;
    let jitter_local = 0.25 * sigma * r;
    p = vec4<f32>(p.xyz + quat_rotate(q, jitter_local), p.w);
  } else if (action == 2u) {
    // Split=replace
    let seed = src_idx * 747796405u + 2891336453u;
    let d = vec3<f32>(
      randn_approx(seed ^ 0x9E3779B9u),
      randn_approx(seed ^ 0x243F6A88u),
      randn_approx(seed ^ 0xB7E15162u)
    );
    let offset_local = 0.5 * sigma * d;
    let sgn = select(1.0, -1.0, variant == 1u);
    p = vec4<f32>(p.xyz + sgn * quat_rotate(q, offset_local), p.w);
  }

  out_pos[dst_idx].param = p;
  if (reset) {
    out_pos[dst_idx].m = vec4<f32>(0.0);
    out_pos[dst_idx].v = vec4<f32>(0.0);
  } else {
    out_pos[dst_idx].m = src.m;
    out_pos[dst_idx].v = src.v;
  }
}

@compute @workgroup_size(256)
fn scatter_opt_pos_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= info.in_points) {
    return;
  }

  let action = select(0u, out_actions[idx], idx < arrayLength(&out_actions));

  let c = out_counts[idx];
  if (c == 0u) {
    return;
  }
  let off = out_offsets[idx];
  if (off >= info.out_points) {
    return;
  }

  write_slot(off, idx, 0u, action);
  if (c == 2u) {
    let off1 = off + 1u;
    if (off1 < info.out_points) {
      write_slot(off1, idx, 1u, action);
    }
  }
}
