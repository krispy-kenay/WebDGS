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

@group(1) @binding(0) var<storage, read> in_opt: array<OptVec4>;
@group(2) @binding(0) var<storage, read_write> out_opt: array<OptVec4>;

fn write_slot(dst_idx: u32, src_idx: u32, is_new: bool) {
  let reset = (info.reset_new_state != 0u) && is_new;
  out_opt[dst_idx].param = in_opt[src_idx].param;
  if (reset) {
    out_opt[dst_idx].m = vec4<f32>(0.0);
    out_opt[dst_idx].v = vec4<f32>(0.0);
  } else {
    out_opt[dst_idx].m = in_opt[src_idx].m;
    out_opt[dst_idx].v = in_opt[src_idx].v;
  }
}

@compute @workgroup_size(256)
fn scatter_opt_vec4_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

  // For split=replace, both outputs are considered "new".
  let slot0_new = (action == 2u);
  write_slot(off, idx, slot0_new);
  if (c == 2u) {
    let off1 = off + 1u;
    if (off1 < info.out_points) {
      write_slot(off1, idx, true);
    }
  }
}
