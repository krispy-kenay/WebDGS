@group(3) @binding(0) var<uniform> settings: RenderSettings;
const POINT_COLOR : vec3<f32> = vec3<f32>(1.0, 1.0, 0.0);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) center_px  : vec2<f32>,
    @location(1) color      : vec3<f32>,
    @location(2) conic      : vec3<f32>,
    @location(3) opacity    : f32,
};

@group(1) @binding(0) var<storage, read> splats : array<Splat>;
@group(2) @binding(0) var<storage, read> sorted_vis_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(instance_index) iid: u32,
    @builtin(vertex_index) vid: u32
) -> VertexOutput {
    var out: VertexOutput;

    let splat = splats[sorted_vis_indices[iid]];

    let pos = unpack2x16float(splat.pos);
    let color_rg = unpack2x16float(splat.color_rg);
    let color_ba = unpack2x16float(splat.color_ba);
    let color = vec3<f32>(color_rg.x, color_rg.y, color_ba.x);
    let opacity = color_ba.y;
    let conic_xy = unpack2x16float(splat.conic_xy);
    let conic_z = unpack2x16float(splat.conic_z);
    let conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z.x);
    let radius_px = unpack2x16float(splat.radius);

    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    let radius_ndc = radius_px * vec2<f32>(2.0 / viewport.x, 2.0 / viewport.y);

    let triangle_vertex = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, 1.0), vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let current_triangle_vertex = triangle_vertex[vid % 6u];
    let clip_xy = pos + radius_ndc * current_triangle_vertex;

    out.position = vec4<f32>(clip_xy, 0.0, 1.0);
    
    out.center_px = (pos * vec2<f32>(0.5, -0.5) + 0.5) * viewport;
    out.color = color;
    out.opacity = opacity;
    out.conic = conic;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color;
    let conic = in.conic;
    let opacity = in.opacity;
    let delta = in.position.xy - in.center_px;

    let exp_q = (conic.x * delta.x * delta.x) + (2.0 * conic.y * delta.x * delta.y) + (conic.z * delta.y * delta.y);
    let gaussian_alpha = clamp(exp(-0.5 * exp_q) * opacity, 0.0, 0.99);

    if (settings.gaussian_mode < 0.5) {
        let limit = settings.point_size_px;
        if (dot(delta, delta) > limit * limit) {
            discard;
        }
        return vec4<f32>(POINT_COLOR, 1.0);
    }
    return vec4<f32>(color, gaussian_alpha);
}
