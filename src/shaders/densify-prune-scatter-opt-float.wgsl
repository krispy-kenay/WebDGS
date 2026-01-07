struct ScatterInfo {
  in_points: u32,
  out_points: u32,
  reset_new_state: u32,
  _pad0: u32,
};

struct OptFloat {
  param: f32,
  m: f32,
  v: f32,
};

@group(0) @binding(0) var<uniform> info: ScatterInfo;
@group(0) @binding(1) var<storage, read> out_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> out_counts: array<u32>;
@group(0) @binding(3) var<storage, read> out_actions: array<u32>;

@group(1) @binding(0) var<storage, read> in_opt: array<OptFloat>;
@group(2) @binding(0) var<storage, read_write> out_opt: array<OptFloat>;

const OPACITY_MAX: f32 = 0.8;
const OPACITY_MAX_RAW: f32 = 1.38629436112;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn write_slot(dst_idx: u32, src_idx: u32, is_new: bool) {
  // clamp opacity in sigmoid-space and reset Adam state
  let raw = in_opt[src_idx].param;
  let op = sigmoid(raw);
  out_opt[dst_idx].param = select(raw, OPACITY_MAX_RAW, op > OPACITY_MAX);
  out_opt[dst_idx].m = 0.0;
  out_opt[dst_idx].v = 0.0;
}

@compute @workgroup_size(256)
fn scatter_opt_float_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

  write_slot(off, idx, action == 2u);
  if (c == 2u) {
    let off1 = off + 1u;
    if (off1 < info.out_points) {
      write_slot(off1, idx, true);
    }
  }
}
