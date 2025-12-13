/*
 * Metric Count Shader
 *
 * For each pixel flagged in metric_map, traverse that pixel's prefix of tile instances and atomically increment per-Gaussian counts for contributors.
 *
 */

@group(0) @binding(0) var<uniform> settings: RenderSettings;

@group(1) @binding(0) var<storage, read> tile_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> tile_instances: array<u32>;
@group(1) @binding(2) var<storage, read> splats: array<Splat>;
@group(1) @binding(3) var metric_map: texture_2d<u32>;
@group(1) @binding(4) var n_contrib_tex: texture_2d<u32>;

@group(2) @binding(0) var<storage, read_write> metric_counts: array<atomic<u32>>;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16)
fn metric_count_main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>
) {
  let pix = global_id.xy;
  let width = u32(settings.viewport_x);
  let height = u32(settings.viewport_y);
  if (pix.x >= width || pix.y >= height) {
    return;
  }

  let flagged = textureLoad(metric_map, vec2<i32>(pix), 0).x;
  if (flagged == 0u) {
    return;
  }

  let n_contrib = textureLoad(n_contrib_tex, vec2<i32>(pix), 0).x;
  if (n_contrib == 0u) {
    return;
  }

  let tile_x = group_id.x;
  let tile_y = group_id.y;
  let num_tiles_x = (width + TILE_SIZE - 1u) / TILE_SIZE;
  let tile_idx = tile_y * num_tiles_x + tile_x;

  let start = tile_offsets[tile_idx];
  if (start == 0xFFFFFFFFu) {
    return;
  }

  let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
  let pixf = vec2<f32>(pix) + 0.5;

  for (var i = 0u; i < n_contrib; i++) {
    let entry = start + i;
    if (entry >= arrayLength(&tile_instances)) {
      break;
    }

    let gidx = tile_instances[entry];
    if (gidx >= arrayLength(&splats) || gidx >= arrayLength(&metric_counts)) {
      continue;
    }

    let splat = splats[gidx];
    let pos_ndc = unpack2x16float(splat.pos);
    let conic_xy = unpack2x16float(splat.conic_xy);
    let conic_z = unpack2x16float(splat.conic_z);
    let color_ba = unpack2x16float(splat.color_ba);

    let center_px = (pos_ndc * vec2<f32>(0.5, -0.5) + 0.5) * viewport;
    let conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z.x);
    let opacity = color_ba.y;

    let delta = pixf - center_px;
    let power = (conic.x * delta.x * delta.x) +
                (2.0 * conic.y * delta.x * delta.y) +
                (conic.z * delta.y * delta.y);

    let G = exp(-0.5 * power);
    let alpha = min(0.99, opacity * G);
    if (alpha < (1.0 / 255.0)) {
      continue;
    }

    atomicAdd(&metric_counts[gidx], 1u);
  }
}

