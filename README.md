# WebDGS

Differentiable Gaussian Splatting training and viewing, entirely in the browser with WebGPU. This is a full pipeline implemented as a static site: load data, train, densify and prune, and render the evolving point cloud live, all on the GPU.

Live demo: [https://krispy-kenay.github.io/WebDGS/](https://krispy-kenay.github.io/WebDGS/)

<details>
<summary>Quick facts</summary>

| What | Details |
| --- | --- |
| Stack | TypeScript + WebGPU + WGSL (Vite) |
| Runs | Local dev server or static hosting |
| Data | COLMAP outputs (images.bin + cameras.bin or JSON) and PLY splats |
| Modes | Train new splats or view existing ones |
| Focus | Real-time, GPU-first differentiable rendering in the browser |

</details>

## What this is
WebDGS is a browser-native implementation of a Gaussian Splatting training loop. It includes tiled forward rendering, rasterization, backward gradients, Adam updates, and a densify/prune schedule, all running on the GPU via WebGPU compute passes. The goal is to make 3DGS training understandable and interactive: you can load a dataset, tweak training parameters, and see how the scene evolves without leaving the browser.

This repo is built as a static site. There is no backend service and no server-side ML framework. Everything happens client-side, which makes it easy to host and share, but also means the pipeline is constrained by browser and GPU limits. If you are reviewing this as engineering work, the interesting parts are the GPU pipeline structure, buffer management for dynamic point counts, and the UI-driven training loop that stays interactive while doing real compute.

## Why it matters
Most 3DGS pipelines live in native code and require specialized environments. WebDGS demonstrates that the full training loop can be moved to the browser, opening the door to interactive education, lightweight demos, and portable research tooling. It also shows how to structure a non-trivial GPU workload in TypeScript and WGSL without losing performance or developer ergonomics.

## Running locally
Install dependencies and start the dev server:

```bash
npm install
npm run dev
```

Build and preview a production bundle:

```bash
npm run build
npm run preview
```

The app is a static site, so the build output can be hosted anywhere that serves files (GitHub Pages, Vercel, Netlify, or a custom server).

## Browser requirements
You’ll need a WebGPU-enabled browser (a recent Chromium-based build is the safest bet). A discrete GPU is strongly recommended for larger scenes.

## Data and workflows
WebDGS supports two primary workflows. You can load a prebuilt PLY splat for instant viewing, or you can load COLMAP outputs and train directly in the browser. For COLMAP data, provide the images folder and the camera metadata. The loader accepts the standard `images.bin` and `cameras.bin` files, or a JSON camera file if you have custom preprocessing.

Once data is loaded, you can start training from the UI. The training controls expose the iteration budget, densify and prune schedule, and the key loss weights. Training progress, iteration rate, and the current Gaussian count update live while the scene renders.

The file-type detection is intentionally simple: it looks at the filename and the first bytes, then falls back to a best-effort loader.

```ts
// src/utils/load.ts
const isPly = headerView[0] === 0x70 && headerView[1] === 0x6c && headerView[2] === 0x79; // 'ply'

if (isPly || filename.endsWith('.ply')) {
  return loadPointCloud(file, device);
}

if (filename.endsWith('cameras.bin') || filename.endsWith('images.bin') || filename.endsWith('.json')) {
  return loadCamera(file);
}
```

## User experience
The UI is designed to feel more like a live lab bench than a static viewer. Training and viewing are both real-time. You can toggle Gaussian vs point-cloud rendering, adjust Gaussian scale and point size, and track the densify schedule in the control panel. The intent is to make the training loop legible: you can literally see how the representation changes as it optimizes.

## Technical outline
At a high level, training repeatedly renders the current Gaussians from one or more camera views, computes a loss against the reference images, backpropagates gradients into Gaussian parameters, and applies an optimizer update. On a schedule, the system densifies (clone/split) and prunes (drop low-value Gaussians) to keep capacity where it matters.

The implementation is GPU-first. Forward and backward passes are tiled for efficiency, and densify/prune is executed with multiple compute stages, including a GPU-side scatter that compacts and expands the Gaussian buffers.

### Shader integration
WGSL is imported directly as source strings via a Vite plugin, so pipelines can be built without custom bundling steps.

```ts
// vite.config.ts
rawPlugin({
  fileRegex: /\.wgsl$/,
})
```

In code, that means renderers can import WGSL like any other module:

```ts
// src/renderers/densify-prune.ts
import decideWGSL from '../shaders/densify-prune-decide.wgsl';
import scatterGaussiansWGSL from '../shaders/densify-prune-scatter-gaussians.wgsl';
```

### Densify and prune (GPU rebuild)
The trainer configures a densify/prune schedule and related thresholds. These defaults are meant to be safe on typical GPUs while still showing meaningful behavior in the UI.

```ts
// src/trainer.ts
this.densifyPruneConfig = {
  schedule: { enabled: true, warmupIterations: 500, interval: 100, stopIterations: 15_000 },
  metricViews: 10,
  metricDownscale: 2,
  metricThreshold: 0.5,
  maxNewPointsPerStep: 5000,
  pruneOpacity: 0.01,
  cloneThresholdCount: 500,
  splitScaleThreshold: 1.0,
};
```

On the shader side, the scatter stage is responsible for producing a new packed Gaussian buffer (and associated SH data) from the prior buffer and an action/count stream produced earlier in the densify/prune pipeline.

```wgsl
// src/shaders/densify-prune-scatter-gaussians.wgsl
@group(0) @binding(1) var<storage, read> in_gaussians: array<Gaussian>;
@group(0) @binding(6) var<storage, read_write> out_gaussians: array<Gaussian>;

@compute @workgroup_size(256)
fn scatter_gaussians_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= info.in_points) { return; }
  // ...
  copy_point(off, idx, 0u, _act);
  if (c == 2u) { copy_point(off + 1u, idx, 1u, _act); }
}
```

### Loss weighting
Training exposes the loss weights directly in the UI (L1, L2, and a DSSIM-style term). The defaults are set in the trainer:

```ts
// src/trainer.ts
this.trainingConfig = trainingConfig || {
  lambda_l1: 0.8,
  lambda_l2: 0.0,
  lambda_dssim: 0.2
};
```

## Architecture notes
The codebase is separated into a small number of focused modules:

`src/trainer.ts` orchestrates the training loop, scheduling, and swaps between point-cloud buffers. `src/renderers` contains the forward pass, rasterizer, backward pass, optimizer, and densify/prune stages. `src/shaders` holds the WGSL kernels that implement the heavy lifting, while `src/viewer.ts` provides a lighter-weight rendering path for inspection and navigation. The entry point, `src/main.ts`, wires the UI, handles WebGPU initialization, and runs the render loop.

If you are looking for a “tour route” through the code, start with `src/main.ts` (UI + initialization), then read `src/trainer.ts` (the training loop), and then dive into `src/renderers` and `src/shaders`. The densify/prune implementation is one of the more distinctive parts of the repo: it includes a decision pass, a total/cap pass, and a scatter pass that compacts and expands the Gaussian buffers entirely on the GPU. The training loop also keeps optimizer state in GPU buffers and applies Adam updates without CPU involvement.

## Performance and constraints
Because everything runs in the browser, performance depends heavily on GPU support and driver maturity. Discrete GPUs are recommended for larger scenes, and very large datasets may hit browser memory limits. The pipeline tries to stay within WebGPU storage buffer constraints and keeps intermediate buffers bounded when densifying.

## Notes on scope
WebDGS is intentionally end-to-end rather than a minimal viewer. The “interesting work” is spread across GPU compute (forward/backward, scatter, sorting), GPU memory layout and resizing for dynamic point counts, and the UI/runtime structure needed to keep training interactive in a browser app.

## Contributing and collaboration
If you want to extend this project, contributions are welcome. Areas that are particularly interesting include performance tuning, additional loss terms, improved camera and dataset loaders, and UI additions for analysis or debugging. Open a discussion or PR if you want to collaborate.

## Acknowledgments
This project is inspired by the 3D Gaussian Splatting literature and open-source differentiable rasterization work. Several shader routines follow established approaches from prior art in the community.
