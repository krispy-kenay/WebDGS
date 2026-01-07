/*
 * Densify/Prune Capacity Clamp
 *
 * Ensures generated output indices stay within `max_out_points` by clamping
 * `out_counts` (and adjusting `out_actions`) using the *current* exclusive
 * prefix offsets. This is a safety mechanism to avoid exceeding configured
 * buffer limits.
 */

struct CapInfo {
  num_points: u32,
  max_out_points: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> info: CapInfo;
@group(0) @binding(1) var<storage, read> out_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_actions: array<u32>;

@compute @workgroup_size(256)
fn cap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= info.num_points) {
    return;
  }
  let max_out = info.max_out_points;
  let off = out_offsets[idx];
  let c = out_counts[idx];

  if (max_out == 0u) {
    out_counts[idx] = 0u;
    out_actions[idx] = 3u;
    return;
  }

  if (off >= max_out) {
    out_counts[idx] = 0u;
    out_actions[idx] = 3u;
    return;
  }

  // If only one slot remains but we wanted to emit 2, degrade to keep-only.
  if (c == 2u && off == max_out - 1u) {
    out_counts[idx] = 1u;
    out_actions[idx] = 0u;
  }
}

