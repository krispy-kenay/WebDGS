struct ScatterInfo {
  in_points: u32,
  out_points: u32,
  _pad0: u32,
  _pad1: u32,
};

struct Gaussian {
  pos_opacity: array<u32, 2>,
  rot:         array<u32, 2>,
  scale:       array<u32, 2>,
};

@group(0) @binding(0) var<uniform> info: ScatterInfo;
@group(0) @binding(1) var<storage, read> in_gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> in_sh: array<u32>;

@group(0) @binding(3) var<storage, read> out_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> out_counts: array<u32>;
@group(0) @binding(5) var<storage, read> out_actions: array<u32>;

@group(0) @binding(6) var<storage, read_write> out_gaussians: array<Gaussian>;
@group(0) @binding(7) var<storage, read_write> out_sh: array<u32>;

const SH_WORDS_PER_POINT: u32 = 24u;
const LN_1P6: f32 = 0.4700036292457356;
const OPACITY_MAX: f32 = 0.8;
const OPACITY_MAX_RAW: f32 = 1.38629436112;

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
  // [0, 1)
  return f32(hash_u32(seed)) * (1.0 / 4294967296.0);
}

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn clamp_log_scale(ls: vec3<f32>) -> vec3<f32> {
  // Prevent exp overflow
  return clamp(ls, vec3<f32>(-10.0), vec3<f32>(10.0));
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
  let len2 = max(1e-12, dot(q, q));
  return q * inverseSqrt(len2);
}

fn quat_rotate(q_in: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  // Quaternion is stored as (w, x, y, z) = (q.x, q.y, q.z, q.w)
  let q = quat_normalize(q_in);
  let u = vec3<f32>(q.y, q.z, q.w);
  let s = q.x;
  return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn randn_approx(seed: u32) -> f32 {
  // Approximate N(0,1) via sum-of-uniforms (CLT)
  var s: f32 = 0.0;
  s = s + rand01(seed ^ 0xA2C79u);
  s = s + rand01(seed ^ 0x5E2D9u);
  s = s + rand01(seed ^ 0x1B873u);
  s = s + rand01(seed ^ 0xC0FFEu);
  s = s + rand01(seed ^ 0xBADC0u);
  s = s + rand01(seed ^ 0xDEADBu);
  return (s - 3.0) * 1.41421356237;
}

fn copy_point(dst_idx: u32, src_idx: u32, variant: u32, action: u32) {
  var g = in_gaussians[src_idx];

  let p0 = unpack2x16float(g.pos_opacity[0]);
  let p1 = unpack2x16float(g.pos_opacity[1]);
  let op = sigmoid(p1.y);
  let opacity_clamped = op > OPACITY_MAX;
  let opacity_raw = select(p1.y, OPACITY_MAX_RAW, opacity_clamped);

  let r0 = unpack2x16float(g.rot[0]);
  let r1 = unpack2x16float(g.rot[1]);
  let q = vec4<f32>(r0.x, r0.y, r1.x, r1.y);

  let s0 = unpack2x16float(g.scale[0]);
  let s1 = unpack2x16float(g.scale[1]);
  let log_sigma = clamp_log_scale(vec3<f32>(s0.x, s0.y, s1.x));
  let sigma = exp(log_sigma);

  var pos = vec3<f32>(p0.x, p0.y, p1.x);

  let needs_transform = (action == 2u) || (action == 1u && variant == 1u);
  if (!needs_transform && !opacity_clamped) {
    // verbatim copy
    out_gaussians[dst_idx] = g;
    let src_base = src_idx * SH_WORDS_PER_POINT;
    let dst_base = dst_idx * SH_WORDS_PER_POINT;
    for (var w = 0u; w < SH_WORDS_PER_POINT; w = w + 1u) {
      out_sh[dst_base + w] = in_sh[src_base + w];
    }
    return;
  }

  // Clone jitter: add a small random position offset so duplicates can diverge.
  if (action == 1u && variant == 1u) {
    let seed = src_idx * 1664525u + dst_idx * 1013904223u;
    let r = vec3<f32>(
      rand01(seed ^ 0xA2C79u),
      rand01(seed ^ 0x5E2D9u),
      rand01(seed ^ 0x1B873u)
    ) * 2.0 - 1.0;
    let jitter_local = 0.25 * sigma * r;
    pos = pos + quat_rotate(q, jitter_local);
  }

  // Split = replace: emit two children (slot0/slot1) and shrink scale.
  if (action == 2u) {
    let seed = src_idx * 747796405u + 2891336453u;
    let d = vec3<f32>(
      randn_approx(seed ^ 0x9E3779B9u),
      randn_approx(seed ^ 0x243F6A88u),
      randn_approx(seed ^ 0xB7E15162u)
    );
    let offset_local = 0.5 * sigma * d;
    let sgn = select(1.0, -1.0, variant == 1u);
    pos = pos + sgn * quat_rotate(q, offset_local);

    let log_child = log_sigma - vec3<f32>(LN_1P6);
    g.scale[0] = pack2x16float(log_child.xy);
    g.scale[1] = pack2x16float(vec2<f32>(log_child.z, 0.0));
  }

  g.pos_opacity[0] = pack2x16float(pos.xy);
  g.pos_opacity[1] = pack2x16float(vec2<f32>(pos.z, opacity_raw));

  out_gaussians[dst_idx] = g;

  let src_base = src_idx * SH_WORDS_PER_POINT;
  let dst_base = dst_idx * SH_WORDS_PER_POINT;
  for (var w = 0u; w < SH_WORDS_PER_POINT; w = w + 1u) {
    out_sh[dst_base + w] = in_sh[src_base + w];
  }
}

@compute @workgroup_size(256)
fn scatter_gaussians_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= info.in_points) {
    return;
  }

  // Keep layout stable even if actions not used yet.
  let _act = select(0u, out_actions[idx], idx < arrayLength(&out_actions));
  if (_act == 123456u) { }

  let c = out_counts[idx];
  if (c == 0u) {
    return;
  }
  let off = out_offsets[idx];
  if (off >= info.out_points) {
    return;
  }

  // slot 0
  copy_point(off, idx, 0u, _act);

  // slot 1
  if (c == 2u) {
    let off1 = off + 1u;
    if (off1 < info.out_points) {
      copy_point(off1, idx, 1u, _act);
    }
  }
}
