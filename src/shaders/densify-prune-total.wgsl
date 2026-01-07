/*
 * Densify/Prune Total Output Count
 *
 * Computes total number of output Gaussians after applying `out_counts`:
 *   total = prefix[N-1] + out_counts[N-1]
 * where `prefix` is an exclusive scan of `out_counts`.
 */

struct Info {
  num_points: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> prefix: array<u32>;
@group(0) @binding(2) var<storage, read> out_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_total: array<u32>;

@compute @workgroup_size(1)
fn total_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) {
    return;
  }
  let n = info.num_points;
  if (n == 0u) {
    out_total[0] = 0u;
    return;
  }
  let last = n - 1u;
  // prefix is exclusive: prefix[last] == sum(out_counts[0..last-1])
  out_total[0] = prefix[last] + out_counts[last];
}
