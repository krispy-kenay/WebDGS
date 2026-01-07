
import { TiledForwardPass } from './renderers/tiled-forward-pass';
import { TiledRasterizer } from './renderers/tiled-rasterizer';
import { TiledBackwardPass, TrainingConfig } from './renderers/tiled-backward-pass';
import { Camera } from './camera/camera';
import { PointCloud, CameraData } from './utils/load';
import { LoadedImage } from './utils/load-images';
import { Optimizer, OptimizerInitialState } from './renderers/optimizer';
import { allocateOptimizerStateBuffers } from './renderers/optimizer';
import { allocatePointCloudLike } from './utils/allocate-pointcloud';
import { DensifyPrunePass } from './renderers/densify-prune';
import { QueueGate } from './utils/queue-gate';
import { AdamHyperparameters, DEFAULT_ADAM_HYPERPARAMETERS } from './renderers/adam-config';

// Shaders
import blitWGSL from './shaders/blit.wgsl';

export interface PointCloudSwapRequest {
    pointCloud: PointCloud;
    optimizerInitialState?: OptimizerInitialState;
}

export interface DensifyPruneScheduleConfig {
    enabled: boolean;
    warmupIterations: number;
    interval: number;
    stopIterations: number;
}

export interface DensifyPruneTrainingConfig {
    schedule: DensifyPruneScheduleConfig;
    metricViews: number;
    metricDownscale: number;
    metricThreshold: number;
    maxBufferBytes: number;
    maxNewPointsPerStep: number;
    pruneOpacity: number;
    cloneThresholdCount: number;
    splitScaleThreshold: number;
}

export class Trainer {
    public readonly camera: Camera;
    private readonly metricsCamera: Camera;

    private readonly device: GPUDevice;
    private readonly queueGate: QueueGate | null;

    // Components
    private forwardPass: TiledForwardPass | null = null;
    private rasterizer: TiledRasterizer | null = null;
    private backwardPass: TiledBackwardPass | null = null;
    private optimizer: Optimizer | null = null;
    private pointCloud: PointCloud | null = null;
    private readonly densifyPrune: DensifyPrunePass;
    private metricsForwardPass: TiledForwardPass | null = null;
    private metricsRasterizer: TiledRasterizer | null = null;
    private metricsPass: TiledBackwardPass | null = null;
    private metricsViewportWidth = 0;
    private metricsViewportHeight = 0;
    private metricsTargetTexture: GPUTexture | null = null;
    private metricsTargetTextureView: GPUTextureView | null = null;
    private metricsTargetWidth = 0;
    private metricsTargetHeight = 0;
    private densifyPruneConfig: DensifyPruneTrainingConfig;

    private trainingConfig: TrainingConfig;
    private optimizerHyperparameters: AdamHyperparameters;

    // Training state
    private isTraining = false;
    private iteration = 0;
    private maxIterations = 10_000;
    private stepItersPerSec = 0;
    private stepMs = 0;
    private lastDensifyPruneIteration: number | null = null;
    private lastViewportWidth = 1;
    private lastViewportHeight = 1;

    private pendingPointCloudSwap: PointCloudSwapRequest | null = null;

    // Debug
    private visualizeLossPipeline: GPURenderPipeline;
    private debugSampler: GPUSampler;
    private readonly metricsDownsamplePipeline: GPURenderPipeline;

    // Dataset
    private trainCameras: CameraData[] = [];
    private images: LoadedImage[] = [];

    constructor(device: GPUDevice, trainingConfig?: TrainingConfig, queueGate?: QueueGate) {
        this.device = device;
        this.queueGate = queueGate ?? null;
        // Dummy canvas for training camera
        const dummyCanvas = document.createElement('canvas');
        this.camera = new Camera(dummyCanvas, device);
        const metricsCanvas = document.createElement('canvas');
        this.metricsCamera = new Camera(metricsCanvas, device);

        this.trainingConfig = trainingConfig || {
            lambda_l1: 0.8,
            lambda_l2: 0.0,
            lambda_dssim: 0.2
        };
        this.optimizerHyperparameters = { ...DEFAULT_ADAM_HYPERPARAMETERS };

        // Initialize Loss Visualization Pipeline
        const visModule = device.createShaderModule({
            label: 'visualize-loss-shader',
            code: blitWGSL
        });

        // Pipeline with fs_abs entry point
        this.visualizeLossPipeline = device.createRenderPipeline({
            label: 'visualize-loss-pipeline',
            layout: 'auto',
            vertex: {
                module: visModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: visModule,
                entryPoint: 'fs_abs',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }] // Main canvas format
            },
            primitive: { topology: 'triangle-list' }
        });

        this.debugSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        const downsampleModule = device.createShaderModule({
            label: 'metrics-downsample-gt-shader',
            code: blitWGSL
        });
        this.metricsDownsamplePipeline = device.createRenderPipeline({
            label: 'metrics-downsample-gt-pipeline',
            layout: 'auto',
            vertex: { module: downsampleModule, entryPoint: 'vs_main' },
            fragment: {
                module: downsampleModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'rgba8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });

        const maxStorage = this.device.limits.maxStorageBufferBindingSize;
        const defaultMaxBytes = Math.min(128 * 1024 * 1024, maxStorage);
        this.densifyPruneConfig = {
            schedule: {
                enabled: true,
                warmupIterations: 500,
                interval: 100,
                stopIterations: 15_000,
            },
            metricViews: 10,
            metricDownscale: 2,
            metricThreshold: 0.5,
            maxBufferBytes: defaultMaxBytes,
            maxNewPointsPerStep: 5000,
            pruneOpacity: 0.01,
            cloneThresholdCount: 500,
            splitScaleThreshold: 1.0,
        };

        this.densifyPrune = new DensifyPrunePass(this.device, {
            strategy: 'gpu_rebuild',
            numViews: this.densifyPruneConfig.metricViews,
            maxBufferBytes: this.densifyPruneConfig.maxBufferBytes,
            maxNewPointsPerStep: this.densifyPruneConfig.maxNewPointsPerStep,
            pruneThreshold: this.densifyPruneConfig.pruneOpacity,
            cloneThreshold: this.densifyPruneConfig.cloneThresholdCount,
            splitThreshold: this.densifyPruneConfig.splitScaleThreshold,
        });
    }

    setPointCloud(pointCloud: PointCloud) {
        this.applyPointCloudSwap({ pointCloud });
    }

    requestPointCloudSwap(pointCloud: PointCloud, optimizerInitialState?: OptimizerInitialState): void {
        this.pendingPointCloudSwap = { pointCloud, optimizerInitialState };
    }

    consumePointCloudSwapRequest(): PointCloudSwapRequest | null {
        const req = this.pendingPointCloudSwap;
        this.pendingPointCloudSwap = null;
        return req;
    }

    /**
     * Debug helper: allocate a new point cloud with the same per-point layout and request a swap.
     */
    requestResizeTo(numPoints: number): void {
        if (!this.pointCloud) return;
        const resized = allocatePointCloudLike(this.device, this.pointCloud, { numPoints });
        this.requestPointCloudSwap(resized);
    }

    /** Apply a point cloud swap immediately (call only at a safe submission boundary). */
    applyPointCloudSwap(request: PointCloudSwapRequest): void {
        const oldPointCloud = this.pointCloud;
        const oldOptimizerParams = this.optimizer?.getHyperparameters();

        // Tear down old graph
        this.forwardPass?.destroy();
        this.rasterizer?.destroy();
        this.backwardPass?.destroy();
        this.metricsForwardPass?.destroy();
        this.metricsRasterizer?.destroy();
        this.metricsPass?.destroy();
        this.optimizer?.destroy();

        this.forwardPass = null;
        this.rasterizer = null;
        this.backwardPass = null;
        this.metricsForwardPass = null;
        this.metricsRasterizer = null;
        this.metricsPass = null;

        this.pointCloud = request.pointCloud;
        this.optimizer = new Optimizer(
            this.device,
            request.pointCloud,
            oldOptimizerParams ?? this.optimizerHyperparameters,
            request.optimizerInitialState
        );
        this.optimizerHyperparameters = { ...this.optimizer.getHyperparameters() };

        if (oldPointCloud && oldPointCloud !== request.pointCloud) {
            oldPointCloud.gaussian_3d_buffer.destroy();
            oldPointCloud.sh_buffer?.destroy();
        }

        // rebuild render graph for current size
        this.ensurePipelines(this.lastViewportWidth, this.lastViewportHeight);
    }

    setDataset(cameras: CameraData[], images: LoadedImage[]) {
        this.trainCameras = cameras;
        this.images = images;
    }

    getTrainingConfig(): TrainingConfig {
        return { ...this.trainingConfig };
    }

    setTrainingConfig(next: Partial<TrainingConfig>): void {
        if (next.lambda_l1 !== undefined) this.trainingConfig.lambda_l1 = next.lambda_l1;
        if (next.lambda_l2 !== undefined) this.trainingConfig.lambda_l2 = next.lambda_l2;
        if (next.lambda_dssim !== undefined) this.trainingConfig.lambda_dssim = next.lambda_dssim;
        if (next.c1 !== undefined) this.trainingConfig.c1 = next.c1;
        if (next.c2 !== undefined) this.trainingConfig.c2 = next.c2;

        this.backwardPass?.setTrainingConfig(next);
        this.metricsPass?.setTrainingConfig(next);
    }

    getOptimizerHyperparameters(): AdamHyperparameters {
        return { ...(this.optimizer?.getHyperparameters() ?? this.optimizerHyperparameters) };
    }

    setOptimizerHyperparameters(next: Partial<AdamHyperparameters>): void {
        this.optimizerHyperparameters = { ...this.optimizerHyperparameters, ...next };
        this.optimizer?.setHyperparameters(next);
    }

    setDensifyPruneConfig(next: Partial<DensifyPruneTrainingConfig>): void {
        this.densifyPruneConfig = {
            ...this.densifyPruneConfig,
            ...next,
            schedule: { ...this.densifyPruneConfig.schedule, ...(next.schedule ?? {}) },
        };

        this.densifyPrune.setConfig({
            numViews: this.densifyPruneConfig.metricViews,
            maxBufferBytes: this.densifyPruneConfig.maxBufferBytes,
            maxNewPointsPerStep: this.densifyPruneConfig.maxNewPointsPerStep,
            pruneThreshold: this.densifyPruneConfig.pruneOpacity,
            cloneThreshold: this.densifyPruneConfig.cloneThresholdCount,
            splitThreshold: this.densifyPruneConfig.splitScaleThreshold,
        });
    }

    private ensureMetricsTargetTexture(width: number, height: number): GPUTextureView {
        if (this.metricsTargetTexture && this.metricsTargetTextureView && this.metricsTargetWidth === width && this.metricsTargetHeight === height) {
            return this.metricsTargetTextureView;
        }

        this.metricsTargetTexture?.destroy();
        this.metricsTargetTexture = this.device.createTexture({
            label: 'metrics-gt-downsampled',
            size: { width, height },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.metricsTargetTextureView = this.metricsTargetTexture.createView();
        this.metricsTargetWidth = width;
        this.metricsTargetHeight = height;
        return this.metricsTargetTextureView;
    }

    private encodeDownsampleToMetrics(encoder: GPUCommandEncoder, srcView: GPUTextureView, dstView: GPUTextureView): void {
        const bg = this.device.createBindGroup({
            label: 'metrics-downsample-gt-bg0',
            layout: this.metricsDownsamplePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: srcView },
                { binding: 1, resource: this.debugSampler },
            ],
        });

        const pass = encoder.beginRenderPass({
            label: 'metrics-downsample-gt-pass',
            colorAttachments: [
                {
                    view: dstView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                },
            ],
        });
        pass.setPipeline(this.metricsDownsamplePipeline);
        pass.setBindGroup(0, bg);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    private ensureMetricsPipelines(baseWidth: number, baseHeight: number): { width: number; height: number } {
        const downscale = Math.max(1, Math.floor(this.densifyPruneConfig.metricDownscale));
        const width = Math.max(1, Math.floor(baseWidth / downscale));
        const height = Math.max(1, Math.floor(baseHeight / downscale));

        if (this.metricsForwardPass && this.metricsRasterizer && this.metricsPass && this.metricsViewportWidth === width && this.metricsViewportHeight === height) {
            return { width, height };
        }

        this.metricsForwardPass?.destroy();
        this.metricsRasterizer?.destroy();
        this.metricsPass?.destroy();
        this.metricsForwardPass = null;
        this.metricsRasterizer = null;
        this.metricsPass = null;

        if (!this.pointCloud) {
            return { width, height };
        }

        this.metricsViewportWidth = width;
        this.metricsViewportHeight = height;

        this.metricsForwardPass = new TiledForwardPass(this.device, this.pointCloud, this.metricsCamera.uniform_buffer, {
            viewportWidth: width,
            viewportHeight: height,
            renderMode: 'gaussian',
        });
        this.metricsRasterizer = new TiledRasterizer({
            device: this.device,
            forwardPass: this.metricsForwardPass,
            format: 'rgba8unorm',
        });
        this.metricsPass = new TiledBackwardPass(this.device, this.pointCloud, {
            viewportWidth: width,
            viewportHeight: height,
            trainingConfig: this.trainingConfig,
        });

        this.ensureMetricsTargetTexture(width, height);
        return { width, height };
    }

    private async runDensifyPruneMultiView(): Promise<void> {
        if (!this.pointCloud || !this.optimizer) return;
        if (this.trainCameras.length === 0 || this.images.length === 0) return;

        const baseWidth = this.lastViewportWidth;
        const baseHeight = this.lastViewportHeight;
        const { width: metricsW, height: metricsH } = this.ensureMetricsPipelines(baseWidth, baseHeight);
        if (!this.metricsForwardPass || !this.metricsRasterizer || !this.metricsPass) return;

        const viewsTarget = Math.max(1, Math.floor(this.densifyPruneConfig.metricViews));
        const threshold = this.densifyPruneConfig.metricThreshold;

        const encoder = this.device.createCommandEncoder({ label: 'densify-prune multiview metrics' });
        encoder.clearBuffer(this.metricsPass.getMetricCountsBuffer());

        let usedViews = 0;
        const maxAttempts = viewsTarget * 4;

        for (let attempt = 0; attempt < maxAttempts && usedViews < viewsTarget; attempt++) {
            const idx = Math.floor(Math.random() * this.trainCameras.length);
            const camData = this.trainCameras[idx];
            const image = this.images[idx];
            if (!camData || !image) continue;
            if (image.width !== baseWidth || image.height !== baseHeight) continue;

            // Update metrics camera (separate uniform buffer from training).
            this.metricsCamera.canvas.width = metricsW;
            this.metricsCamera.canvas.height = metricsH;
            this.metricsCamera.set_preset(camData);

            // Low-res forward+rasterize.
            this.metricsForwardPass.encode(encoder);
            this.metricsRasterizer.encode(encoder, metricsW, metricsH);

            // Downsample GT to the metrics resolution.
            const gtMetricsView = this.ensureMetricsTargetTexture(metricsW, metricsH);
            this.encodeDownsampleToMetrics(encoder, image.texture.createView(), gtMetricsView);

            // Metric map + per-gaussian counts (accumulated).
            const predView = this.metricsRasterizer.getOutputTextureView();
            this.metricsPass.computeMetricMap(encoder, predView, gtMetricsView, { threshold });
            this.metricsPass.computeMetricCounts(
                encoder,
                {
                    splatBuffer: this.metricsForwardPass.getResources().splatBuffer,
                    tileOffsetsBuffer: this.metricsRasterizer.getTileOffsetsBuffer(),
                    tileIndicesBuffer: this.metricsForwardPass.getSortedIndicesBuffer(),
                    nContribTexture: this.metricsRasterizer.getNContribTextureView(),
                },
                { clear: false }
            );

            usedViews++;
        }

        if (usedViews === 0) {
            return;
        }

        this.metricsPass.normalizeMetricCounts(encoder, { divisor: usedViews });

        this.densifyPrune.ensureSize(this.pointCloud.num_points);
        const densifyPrepared = this.densifyPrune.encodePrepare(encoder, {
            pointCloud: this.pointCloud,
            metricCountsBuffer: this.metricsPass.getMetricCountsBuffer(),
        });

        const densifyTotalReadback = this.device.createBuffer({
            label: 'densify-prune outTotal readback',
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        encoder.copyBufferToBuffer(densifyPrepared.outTotalBuffer, 0, densifyTotalReadback, 0, 4);

        const cmd = encoder.finish();
        if (this.queueGate) {
            await this.queueGate.submit([cmd]);
        } else {
            this.device.queue.submit([cmd]);
            await this.device.queue.onSubmittedWorkDone();
        }

        await densifyTotalReadback.mapAsync(GPUMapMode.READ);
        const outTotal = new Uint32Array(densifyTotalReadback.getMappedRange())[0];
        densifyTotalReadback.unmap();
        densifyTotalReadback.destroy();

        const inN = this.pointCloud.num_points;
        const outN = Math.min(outTotal, densifyPrepared.maxOutPoints);
        if (outN === 0 || outN === inN) {
            return;
        }

        const outPointCloud = allocatePointCloudLike(this.device, this.pointCloud, { numPoints: outN });
        const outOptimizerState = allocateOptimizerStateBuffers(this.device, outN);

        const scatterEncoder = this.device.createCommandEncoder({ label: 'densify-prune scatter' });
        this.densifyPrune.encodeScatter(
            scatterEncoder,
            {
                pointCloud: this.pointCloud,
                optimizerState: this.optimizer.getStateBuffers(),
                outOffsetBuffer: densifyPrepared.outOffsetBuffer,
                outNumPoints: outN,
                resetNewOptimizerState: true,
            },
            {
                outPointCloud,
                outOptimizerState,
            }
        );
        const scatterCmd = scatterEncoder.finish();
        if (this.queueGate) {
            await this.queueGate.submit([scatterCmd]);
        } else {
            this.device.queue.submit([scatterCmd]);
            await this.device.queue.onSubmittedWorkDone();
        }

        this.requestPointCloudSwap(outPointCloud, {
            iteration: this.optimizer.getIteration(),
            buffers: outOptimizerState,
        });
        this.lastDensifyPruneIteration = this.iteration;
    }

    start() {
        if (!this.pointCloud || this.trainCameras.length === 0) {
            console.error("Cannot start training: Missing point cloud or dataset.");
            return;
        }
        this.isTraining = true;
        this.iteration = 0;
        this.stepItersPerSec = 0;
        this.stepMs = 0;
        this.lastDensifyPruneIteration = null;
        console.log("Training started");
    }

    stop() {
        this.isTraining = false;
        console.log("Training stopped");
    }

    getIsTraining() {
        return this.isTraining;
    }

    setMaxIterations(maxIterations: number): void {
        const n = Math.max(1, Math.floor(maxIterations));
        this.maxIterations = n;
    }

    getMaxIterations(): number {
        return this.maxIterations;
    }

    getIteration(): number {
        return this.iteration;
    }

    getPointCount(): number {
        return this.pointCloud?.num_points ?? 0;
    }

    getLastStepMs(): number {
        return this.stepMs;
    }

    getItersPerSec(): number {
        return this.stepItersPerSec;
    }

    getLastDensifyPruneIteration(): number | null {
        return this.lastDensifyPruneIteration;
    }

    getNextDensifyPruneIteration(): number | null {
        const schedule = this.densifyPruneConfig.schedule;
        if (!schedule.enabled) return null;

        const warmup = schedule.warmupIterations;
        const interval = Math.max(1, schedule.interval);
        const stop = schedule.stopIterations;

        const i = this.iteration;
        if (i >= stop) return null;
        if (i < warmup) return Math.min(warmup, stop);

        const k = Math.ceil((i + 1 - warmup) / interval);
        const next = warmup + k * interval;
        return next <= stop ? next : null;
    }

    // Single training step
    async step() {
        if (!this.isTraining || !this.pointCloud) return;
        const stepStart = performance.now();

        // Pick a random camera/image pair
        const index = Math.floor(Math.random() * this.trainCameras.length);
        const camData = this.trainCameras[index];
        const image = this.images[index];

        if (!camData || !image) {
            console.warn(`Missing data for index ${index}`);
            return;
        }

        // Update Training Camera
        this.camera.canvas.width = image.width;
        this.camera.canvas.height = image.height;
        this.camera.set_preset(camData);
        this.camera.update_buffer();

        // Ensure pipelines match image size
        const width = image.width;
        const height = image.height;
        this.ensurePipelines(width, height);

        const nextIteration = this.iteration + 1;
        const warmup = this.densifyPruneConfig.schedule.warmupIterations;
        const interval = Math.max(1, this.densifyPruneConfig.schedule.interval);
        const stop = this.densifyPruneConfig.schedule.stopIterations;
        const shouldDensify =
            this.densifyPruneConfig.schedule.enabled &&
            nextIteration >= warmup &&
            nextIteration <= stop &&
            (nextIteration === warmup || ((nextIteration - warmup) % interval === 0));

        const encoder = this.device.createCommandEncoder({ label: 'trainer-step' });

        // Forward Pass
        this.forwardPass!.encode(encoder);

        // Rasterize Pass (to offscreen texture)
        this.rasterizer!.encode(encoder, width, height);

        // Backward Pass
        const outputView = this.rasterizer!.getOutputTextureView();
        const gtTextureView = image.texture.createView();

        // Get resources
        const forwardResources = this.forwardPass!.getResources();

        // Assemble backward resources
        const backwardResources = {
            splatBuffer: forwardResources.splatBuffer,
            tileOffsetsBuffer: this.rasterizer!.getTileOffsetsBuffer(),
            tileIndicesBuffer: this.forwardPass!.getSortedIndicesBuffer(),
            cameraBuffer: this.camera.uniform_buffer,
            alphaTexture: this.rasterizer!.getAlphaTextureView(),
            nContribTexture: this.rasterizer!.getNContribTextureView(),
        };

        this.backwardPass!.encode(encoder, outputView, gtTextureView, backwardResources);

        // Optimizer Step
        if (this.optimizer) {
            const tileCounts = forwardResources.tileCountsBuffer;
            const gradients = this.backwardPass!.getGradientsBuffer();

            this.optimizer.step(encoder, this.pointCloud, gradients, tileCounts);
        }

        // Submit
        const cmd = encoder.finish();
        if (this.queueGate) {
            await this.queueGate.submit([cmd]);
        } else {
            this.device.queue.submit([cmd]);
            await this.device.queue.onSubmittedWorkDone();
        }

        this.iteration++;
        const stepEnd = performance.now();
        this.stepMs = stepEnd - stepStart;
        const instItersPerSec = this.stepMs > 0 ? 1000 / this.stepMs : 0;
        this.stepItersPerSec = this.stepItersPerSec === 0 ? instItersPerSec : this.stepItersPerSec * 0.9 + instItersPerSec * 0.1;

        if (shouldDensify) {
            await this.runDensifyPruneMultiView();
        }

        if (this.iteration >= this.maxIterations) {
            this.stop();
        }
    }

    private ensurePipelines(width: number, height: number) {
        this.lastViewportWidth = Math.max(1, Math.floor(width));
        this.lastViewportHeight = Math.max(1, Math.floor(height));

        if (!this.forwardPass) {
            this.forwardPass = new TiledForwardPass(this.device, this.pointCloud!, this.camera.uniform_buffer, {
                viewportWidth: width,
                viewportHeight: height,
                renderMode: 'gaussian'
            });
        } else {
            this.forwardPass.setViewport(width, height);
        }

        if (!this.rasterizer) {
            this.rasterizer = new TiledRasterizer({
                device: this.device,
                forwardPass: this.forwardPass,
                format: 'rgba8unorm',
            });
        }
        if (!this.backwardPass) {
            this.backwardPass = new TiledBackwardPass(this.device, this.pointCloud!, {
                viewportWidth: width,
                viewportHeight: height,
                trainingConfig: this.trainingConfig
            });
        } else {
            this.backwardPass.setViewport(width, height);
        }
    }

    // Method for Loss Visualization
    visualizeLoss(context: GPUCanvasContext, cameraIndex: number) {
        if (!this.pointCloud || !this.backwardPass || !this.forwardPass || !this.rasterizer) return;

        // Setup Camera
        const camData = this.trainCameras[cameraIndex];
        const width = camData.width || 0;
        const height = camData.height || 0;

        if (width === 0 || height === 0) return;

        // Update Pipelines if size changed
        this.forwardPass.setViewport(width, height);
        this.rasterizer.encode(this.device.createCommandEncoder(), width, height);
        this.backwardPass.setViewport(width, height);

        // Update Camera Uniforms
        this.camera.set_preset(camData);
        this.camera.update_buffer();
        this.forwardPass.setCameraBuffer(this.camera.uniform_buffer);

        // Prepare Command Encoder
        const encoder = this.device.createCommandEncoder({ label: 'visualize-loss-encoder' });

        // Forward Pass
        this.forwardPass.encode(encoder);

        // Rasterize
        this.rasterizer.encode(encoder, width, height);

        // Compute Loss
        const image = this.images.find(img => img.name === camData.img_name);
        if (!image) {
            console.error(`Image not found for camera ${cameraIndex}`);
            return;
        }

        const targetView = image.texture.createView();
        const predView = this.rasterizer.getOutputTextureView();

        this.backwardPass.computeLossOnly(encoder, predView, targetView);

        // Blit to Screen
        const pass = encoder.beginRenderPass({
            label: 'visualize-loss-blit',
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });

        // Create temp bind group
        const bg = this.device.createBindGroup({
            label: 'vis-loss-bg',
            layout: this.visualizeLossPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.backwardPass.getLossTextureView() },
                { binding: 1, resource: this.debugSampler }
            ]
        });

        pass.setPipeline(this.visualizeLossPipeline);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();

        const cmd = encoder.finish();
        if (this.queueGate) {
            this.queueGate.trySubmit([cmd]);
        } else {
            this.device.queue.submit([cmd]);
        }
    }
}
