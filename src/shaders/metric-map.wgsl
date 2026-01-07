/*
 * Metric Map Shader
 *
 * compute per-pixel raw L1 error, reduce global min/max over that buffer, and normalize error to [0,1] using min/max and threshold to a binary mask
 *
 */

@group(0) @binding(0) var pred_texture: texture_2d<f32>;
@group(0) @binding(1) var target_texture: texture_2d<f32>;

struct MetricErrorConfig {
  err_scale: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(1) @binding(0) var<uniform> error_config: MetricErrorConfig;
@group(1) @binding(1) var<storage, read_write> error_pairs: array<vec2<u32>>;

fn clamp_to_u32_scaled(v: f32, scale: f32) -> u32 {
  let scaled = v * scale;
  let clamped = clamp(scaled, 0.0, 4294967295.0);
  return u32(clamped);
}

@compute @workgroup_size(16, 16)
fn metric_error_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(pred_texture);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let pred = textureLoad(pred_texture, coord, 0).rgb;
  let targ = textureLoad(target_texture, coord, 0).rgb;

  let diff = abs(pred - targ);
  let l1 = (diff.r + diff.g + diff.b) / 3.0;

  let v = clamp_to_u32_scaled(l1, error_config.err_scale);
  let idx = gid.y * dims.x + gid.x;
  error_pairs[idx] = vec2<u32>(v, v);
}

@group(2) @binding(0) var<storage, read> reduce_in: array<vec2<u32>>;
@group(2) @binding(1) var<storage, read_write> reduce_out: array<vec2<u32>>;

const REDUCE_WG_SIZE: u32 = 256u;
var<workgroup> shared_pairs: array<vec2<u32>, REDUCE_WG_SIZE>;

@compute @workgroup_size(256)
fn metric_reduce_minmax_main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let local_idx = lid.x;
  let global_idx = wg_id.x * REDUCE_WG_SIZE + local_idx;

  let count = arrayLength(&reduce_in);
  var v = vec2<u32>(0xFFFFFFFFu, 0u);
  if (global_idx < count) {
    v = reduce_in[global_idx];
  }
  shared_pairs[local_idx] = v;
  workgroupBarrier();

  var stride = REDUCE_WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (local_idx < stride) {
      let other = shared_pairs[local_idx + stride];
      let cur = shared_pairs[local_idx];
      shared_pairs[local_idx] = vec2<u32>(min(cur.x, other.x), max(cur.y, other.y));
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (local_idx == 0u) {
    reduce_out[wg_id.x] = shared_pairs[0];
  }
}

struct MetricThresholdConfig {
  threshold: f32,
  err_scale: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(3) @binding(0) var<uniform> threshold_config: MetricThresholdConfig;
@group(3) @binding(1) var<storage, read> minmax_pair: array<vec2<u32>>;
@group(3) @binding(2) var output_metric_map: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(16, 16)
fn metric_threshold_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(output_metric_map);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let idx = gid.y * dims.x + gid.x;
  let err_u = error_pairs[idx].x;
  let mm = minmax_pair[0];

  let min_u = mm.x;
  let max_u = mm.y;

  var norm = 0.0;
  if (max_u > min_u) {
    norm = (f32(err_u - min_u)) / (f32(max_u - min_u));
  }

  let flag = select(0u, 1u, norm > threshold_config.threshold);
  textureStore(output_metric_map, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(flag, 0u, 0u, 0u));
}
