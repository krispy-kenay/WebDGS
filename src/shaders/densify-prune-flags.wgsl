/*
 * Densify/Prune Flags
 *
 * From the decided `out_counts`, writes a per-Gaussian "wants to densify" flag (the input of a prefix scan that
 * ranks the densify candidates) and counts densify requests and prunes for the capacity clamp.
 */

struct Info {
  num_points: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

struct Stats {
  densify_requested: atomic<u32>,
  pruned: atomic<u32>,
  max_out_points: u32,
  densify_budget: u32,
};

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> out_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> densify_flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: Stats;

var<workgroup> wg_densify: atomic<u32>;
var<workgroup> wg_pruned: atomic<u32>;

@compute @workgroup_size(256)
fn flags_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32
) {
  if (lid == 0u) {
    atomicStore(&wg_densify, 0u);
    atomicStore(&wg_pruned, 0u);
  }
  workgroupBarrier();

  let idx = gid.x;
  if (idx < info.num_points) {
    let c = out_counts[idx];
    densify_flags[idx] = select(0u, 1u, c == 2u);
    if (c == 2u) {
      atomicAdd(&wg_densify, 1u);
    }
    if (c == 0u) {
      atomicAdd(&wg_pruned, 1u);
    }
  }
  workgroupBarrier();

  // One global atomic per workgroup instead of one per Gaussian
  if (lid == 0u) {
    atomicAdd(&stats.densify_requested, atomicLoad(&wg_densify));
    atomicAdd(&stats.pruned, atomicLoad(&wg_pruned));
  }
}
