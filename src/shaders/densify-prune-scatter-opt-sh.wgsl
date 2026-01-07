struct ScatterInfo {
  in_points: u32,
  out_points: u32,
  reset_new_state: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> info: ScatterInfo;
@group(0) @binding(1) var<storage, read> out_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> out_counts: array<u32>;
@group(0) @binding(3) var<storage, read> out_actions: array<u32>;

@group(1) @binding(0) var<storage, read> in_param_sh: array<f32>;
@group(1) @binding(1) var<storage, read> in_state_sh: array<vec2<f32>>;

@group(2) @binding(0) var<storage, read_write> out_param_sh: array<f32>;
@group(2) @binding(1) var<storage, read_write> out_state_sh: array<vec2<f32>>;

const SH_FLOATS_PER_POINT: u32 = 48u;

fn write_slot(dst_idx: u32, src_idx: u32, is_new: bool) {
  let reset = (info.reset_new_state != 0u) && is_new;

  let src_base = src_idx * SH_FLOATS_PER_POINT;
  let dst_base = dst_idx * SH_FLOATS_PER_POINT;
  for (var i = 0u; i < SH_FLOATS_PER_POINT; i = i + 1u) {
    out_param_sh[dst_base + i] = in_param_sh[src_base + i];
    out_state_sh[dst_base + i] = select(in_state_sh[src_base + i], vec2<f32>(0.0), reset);
  }
}

@compute @workgroup_size(256)
fn scatter_opt_sh_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
