/*
 * Densify/Prune Decision Shader
 *
 * Produces, per input Gaussian:
 * - `out_counts[i]`: how many output Gaussians this input contributes (0 prune, 1 keep, 2 clone/split)
 * - `out_actions[i]`: 0 keep, 1 clone, 2 split, 3 prune
 */

struct Gaussian {
  pos_opacity: array<u32, 2>,
  rot:         array<u32, 2>,
  scale:       array<u32, 2>,
};

struct DecideConfig {
  num_points: u32,
  num_views: u32,
  clone_threshold_count: u32,
  _pad0: u32,

  prune_opacity: f32,
  split_scale_threshold: f32,
  _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> cfg: DecideConfig;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> metric_counts: array<u32>;

// Future / optional inputs
@group(0) @binding(3) var<storage, read> pruning_score: array<f32>;
@group(0) @binding(4) var<storage, read> max_radii: array<f32>;
@group(0) @binding(5) var<storage, read> grad_accum: array<f32>;

@group(0) @binding(6) var<storage, read_write> out_counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_actions: array<u32>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn decide_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= cfg.num_points) {
    return;
  }

  // Keep future buffers "alive" in the bind group layout.
  var _dummy: f32 = 0.0;
  if (idx < arrayLength(&pruning_score)) { _dummy = _dummy + pruning_score[idx]; }
  if (idx < arrayLength(&max_radii)) { _dummy = _dummy + max_radii[idx]; }
  if (idx < arrayLength(&grad_accum)) { _dummy = _dummy + grad_accum[idx]; }
  if (_dummy < -1e30) {
    // Unreachable; prevents the compiler from proving _dummy unused.
    out_actions[idx] = 123456u;
  }

  let g = gaussians[idx];
  let pos_1 = unpack2x16float(g.pos_opacity[1]);
  let opacity_raw = pos_1.y;
  let opacity = sigmoid(opacity_raw);

  // Densification trigger: high importance
  var count: u32 = 0u;
  if (idx < arrayLength(&metric_counts)) {
    count = metric_counts[idx];
  }

  var action: u32 = 0u;
  var out_count: u32 = 1u;

  if (opacity < cfg.prune_opacity) {
    action = 3u;
    out_count = 0u;
  } else if (count >= cfg.clone_threshold_count) {
    // Decide clone vs split by 3D scale magnitude.
    let s0 = unpack2x16float(g.scale[0]);
    let s1 = unpack2x16float(g.scale[1]);
    let scale3 = exp(vec3<f32>(s0.x, s0.y, s1.x));
    let max_scale = max(scale3.x, max(scale3.y, scale3.z));

    action = select(1u, 2u, max_scale >= cfg.split_scale_threshold);
    out_count = 2u;
  }

  out_counts[idx] = out_count;
  out_actions[idx] = action;
}
