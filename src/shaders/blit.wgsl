/*
 * Simple blit shader
 * 
 * Blits a texture to the screen using a fullscreen triangle.
 */

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VSOut {
    var out : VSOut;

    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);

    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let color = textureSample(srcTex, srcSampler, in.uv);
    return color;
}

@fragment
fn fs_abs(in: VSOut) -> @location(0) vec4<f32> {
    let color = textureSample(srcTex, srcSampler, in.uv);
    return vec4<f32>(abs(color.rgb), 1.0);
}
