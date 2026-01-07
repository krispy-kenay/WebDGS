/*
 * Metric Counts Normalization Shader
 *
 * Converts accumulated per-view metric counts into a per-view average by integer division.
 * Operates in-place on an `array<atomic<u32>>` produced by `metric-count.wgsl`.
 */

struct NormalizeInfo {
  num_points: u32,
  divisor: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> info: NormalizeInfo;
@group(0) @binding(1) var<storage, read_write> metric_counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn metric_normalize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= info.num_points) {
    return;
  }

  let d = max(1u, info.divisor);
  let v = atomicLoad(&metric_counts[idx]);
  atomicStore(&metric_counts[idx], v / d);
}

