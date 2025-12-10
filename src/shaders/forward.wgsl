const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;
@group(1) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) var<storage, read> sh_buffer : array<u32>;
@group(1) @binding(2) var<storage, read_write> splats : array<Splat>;

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;


const SH_WORDS_PER_POINT : u32 = 24u;
fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    return vec3<f32>(
        read_f16_coeff(base_word, eR),
        read_f16_coeff(base_word, eG),
        read_f16_coeff(base_word, eB)
    );
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn float_to_ordered_uint(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
  return bits ^ mask;
}

@compute @workgroup_size(workgroupSize,1,1)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // Unpacking
    let gaussian = gaussians[idx];
    let rot_0 = unpack2x16float(gaussian.rot[0]);
    let rot_1 = unpack2x16float(gaussian.rot[1]);
    let quaternion = vec4<f32>(rot_0.x, rot_0.y, rot_1.x, rot_1.y);

    let scale_0 = unpack2x16float(gaussian.scale[0]);
    let scale_1 = unpack2x16float(gaussian.scale[1]);
    let gaussian_scale = exp(vec3<f32>(scale_0.x, scale_0.y, scale_1.x));

    let pos_0 = unpack2x16float(gaussian.pos_opacity[0]);
    let pos_1 = unpack2x16float(gaussian.pos_opacity[1]);
    let gaussian_position = vec3<f32>(pos_0.x, pos_0.y, pos_1.x);
    let gaussian_opacity = pos_1.y;
    let opacity_sigmoid = 1.0 / (1.0 + exp(-gaussian_opacity));

    // Transform to clip space / NDC
    let position_world = vec4<f32>(gaussian_position, 1.0);
    let world_to_view = camera.view * position_world;
    let view_to_clip = camera.proj * world_to_view;
    let near_plane = 0.1;
    let far_plane = 100.0;
    if (view_to_clip.w == 0.0) {
        return;
    }
    let gaussian_ndc = view_to_clip.xyz / view_to_clip.w;

    // Culling
    if (gaussian_ndc.x < -1.2 || gaussian_ndc.x > 1.2 ||
        gaussian_ndc.y < -1.2 || gaussian_ndc.y > 1.2 ||
        gaussian_ndc.z < 0.0 || gaussian_ndc.z > 1.0) {
        return;
    }

    // Covariance computation
    let C3D = covariance3D(quaternion, gaussian_scale);
    let viewport = vec2<f32>(settings.viewport_x, settings.viewport_y);
    let C2D = covariance2D(C3D, world_to_view, camera.focal, viewport, camera.view);
    let det = (C2D.x * C2D.z) - (C2D.y * C2D.y);
    if (det <= 0.0) {
        return;
    }
    let det_inv = 1.0 / det;
    let conic_inv = vec3<f32>(C2D.z * det_inv, -C2D.y * det_inv, C2D.x * det_inv);
    let mid = 0.5 * (C2D.x + C2D.z);
    let lambda1 = mid + sqrt(max(mid * mid - det, 0.01));
    let lambda2 = mid - sqrt(max(mid * mid - det, 0.01));
    let gaussian_radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Color
    let camera_world_pos = camera.view_inv[3].xyz;
    let color_direction = normalize(gaussian_position - camera_world_pos);
    let color = computeColorFromSH(color_direction, idx, u32(settings.sh_deg));

    // Sort depth
    let sort_key_idx = atomicAdd(&sort_infos.keys_size, 1u);
    sort_depths[sort_key_idx] = float_to_ordered_uint(-world_to_view.z);
    sort_indices[sort_key_idx] = sort_key_idx;
    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if ((sort_key_idx % keys_per_dispatch) == 0u) {
        _ = atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    // Write to splats
    splats[sort_key_idx].pos = pack2x16float(clamp(gaussian_ndc.xy, vec2<f32>(-1.0), vec2<f32>(1.0)));
    splats[sort_key_idx].radius = pack2x16float(vec2<f32>(gaussian_radius));
    splats[sort_key_idx].conic_xy = pack2x16float(conic_inv.xy);
    splats[sort_key_idx].conic_z = pack2x16float(vec2<f32>(conic_inv.z, 0.0));
    splats[sort_key_idx].color_rg = pack2x16float(clamp(color.rg, vec2<f32>(0.0), vec2<f32>(1.0)));
    splats[sort_key_idx].color_ba = pack2x16float(vec2<f32>(clamp(color.b, 0.0, 1.0), clamp(opacity_sigmoid, 0.0, 1.0)));
}
