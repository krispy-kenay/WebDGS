/*
 * Densify/Prune Capacity Clamp
 *
 * Keeps the output within `max_out_points` by limiting growth only: prunes and keeps always go through, and densify
 * requests beyond the remaining budget are downgraded to keep. Every clone or split adds exactly one Gaussian, so
 * the budget is `max_out_points - survivors`. Candidates are taken in rank order (exclusive scan of the densify
 * flags) starting at a random rank, so the budget rotates over the scene from one densify step to the next instead
 * of always favouring the start of the buffer.
 */

struct CapInfo {
  num_points: u32,
  max_out_points: u32,
  start_rank: u32,
  _pad0: u32,
};

struct Stats {
  densify_requested: u32,
  pruned: u32,
  max_out_points: u32,
  densify_budget: u32,
};

@group(0) @binding(0) var<uniform> info: CapInfo;
@group(0) @binding(1) var<storage, read> densify_ranks: array<u32>;
@group(0) @binding(2) var<storage, read_write> stats: Stats;
@group(0) @binding(3) var<storage, read_write> out_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_actions: array<u32>;

@compute @workgroup_size(256)
fn cap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let requested = stats.densify_requested;
  let survivors = info.num_points - stats.pruned;
  let budget = select(0u, info.max_out_points - survivors, info.max_out_points > survivors);

  if (idx == 0u) {
    stats.max_out_points = info.max_out_points;
    stats.densify_budget = min(budget, requested);
  }
  if (idx >= info.num_points || budget >= requested) {
    return;
  }
  if (out_counts[idx] != 2u) {
    return;
  }

  // Rank relative to this step's random start, wrapping around
  let rank = (densify_ranks[idx] + requested - info.start_rank % requested) % requested;
  if (rank >= budget) {
    out_counts[idx] = 1u;
    out_actions[idx] = 0u;
  }
}
