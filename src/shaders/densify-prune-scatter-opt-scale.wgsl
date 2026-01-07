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

@group(1) @binding(0) var<storage, read> in_scale: array<OptVec4>;
@group(2) @binding(0) var<storage, read_write> out_scale: array<OptVec4>;

const LN_1P6: f32 = 0.4700036292457356;

fn write_slot(dst_idx: u32, src_idx: u32, variant: u32, action: u32) {
  let src = in_scale[src_idx];
  let is_new = (variant == 1u) || (action == 2u);
  let reset = (info.reset_new_state != 0u) && is_new;

  var p = src.param;
  if (action == 2u) {
    p = vec4<f32>(p.xyz - vec3<f32>(LN_1P6), p.w);
  }

  out_scale[dst_idx].param = p;
  if (reset) {
    out_scale[dst_idx].m = vec4<f32>(0.0);
    out_scale[dst_idx].v = vec4<f32>(0.0);
  } else {
    out_scale[dst_idx].m = src.m;
    out_scale[dst_idx].v = src.v;
  }
}

@compute @workgroup_size(256)
fn scatter_opt_scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

  write_slot(off, idx, 0u, action);
  if (c == 2u) {
    let off1 = off + 1u;
    if (off1 < info.out_points) {
      write_slot(off1, idx, 1u, action);
    }
  }
}
