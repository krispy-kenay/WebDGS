struct CameraUniforms {
  view: mat4x4<f32>,
  view_inv: mat4x4<f32>,
  proj: mat4x4<f32>,
  proj_inv: mat4x4<f32>,
  viewport: vec2<f32>,
  focal: vec2<f32>,
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    viewport_x: f32,
    viewport_y: f32,
    point_size_px: f32,
    gaussian_mode: f32,
};

struct Gaussian {
  pos_opacity: array<u32, 2>,
  rot:         array<u32, 2>,
  scale:       array<u32, 2>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos : u32,
    radius : u32,
    conic_xy : u32,
    conic_z : u32,
    color_rg : u32,
    color_ba : u32,
};

// code adapted from graphdeco-inria/diff-gaussian-rasterization repo
fn covariance3D(quaternion: vec4<f32>, scale: vec3<f32>) -> array<f32, 6> {
    let x = quaternion.y;
    let y = quaternion.z;
    let z = quaternion.w;
    let r = quaternion.x;

    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)),
        vec3<f32>(2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)),
        vec3<f32>(2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)));

    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z)
    );

    // let cov_mat = R * S * transpose(S) * transpose(R);
    let M = S * R;
    let cov_mat = transpose(M) * M;

    let flat_mat = array<f32, 6>(cov_mat[0][0], cov_mat[0][1], cov_mat[0][2], cov_mat[1][1], cov_mat[1][2], cov_mat[2][2]);
    return flat_mat;

}

// code adapted from graphdeco-inria/diff-gaussian-rasterization repo
fn covariance2D(cov_3D: array<f32, 6>, mean_view: vec4<f32>, focal: vec2<f32>, viewport: vec2<f32>, viewmatrix: mat4x4<f32>) -> vec3<f32> {
    var t = mean_view.xyz;
    let focal_x = focal.x;
    let focal_y = focal.y;

    let fovx = viewport.x * 0.5 / focal_x;
    let fovy = viewport.y * 0.5 / focal_y;
    let limx = 1.3 * fovx;
    let limy = 1.3 * fovy;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	let J = mat3x3<f32>(
		vec3<f32>(focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z)),
		vec3<f32>(0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z)),
		vec3<f32>(0.0, 0.0, 0.0));

	let W = mat3x3<f32>(
		viewmatrix[0].x, viewmatrix[1].x, viewmatrix[2].x,
		viewmatrix[0].y, viewmatrix[1].y, viewmatrix[2].y,
		viewmatrix[0].z, viewmatrix[1].z, viewmatrix[2].z);

	let T = W * J;

	let Vrk = mat3x3<f32>(
		cov_3D[0], cov_3D[1], cov_3D[2],
		cov_3D[1], cov_3D[3], cov_3D[4],
		cov_3D[2], cov_3D[4], cov_3D[5]);

	var cov = transpose(T) * transpose(Vrk) * T;

    // numerical stability
	cov[0][0] = cov[0][0] + 0.3;
	cov[1][1] = cov[1][1] + 0.3;
	return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);

}