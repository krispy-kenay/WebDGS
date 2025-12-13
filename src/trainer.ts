
import { TiledForwardPass } from './renderers/tiled-forward-pass';
import { TiledRasterizer } from './renderers/tiled-rasterizer';
import { TiledBackwardPass, TrainingConfig } from './renderers/tiled-backward-pass';
import { Camera } from './camera/camera';
import { PointCloud, CameraData } from './utils/load';
import { LoadedImage } from './utils/load-images';
import { Optimizer } from './renderers/optimizer';

// Shaders
import blitWGSL from './shaders/blit.wgsl';

export class Trainer {
    public readonly camera: Camera;

    private readonly device: GPUDevice;

    // Components
    private forwardPass: TiledForwardPass | null = null;
    private rasterizer: TiledRasterizer | null = null;
    private backwardPass: TiledBackwardPass | null = null;
    private optimizer: Optimizer | null = null;
    private pointCloud: PointCloud | null = null;

    private trainingConfig: TrainingConfig;

    // Training state
    private isTraining = false;
    private iteration = 0;

    // Debug
    private visualizeLossPipeline: GPURenderPipeline;
    private debugSampler: GPUSampler;

    // Dataset
    private trainCameras: CameraData[] = [];
    private images: LoadedImage[] = [];

    constructor(device: GPUDevice, trainingConfig?: TrainingConfig) {
        this.device = device;
        // Dummy canvas for training camera
        const dummyCanvas = document.createElement('canvas');
        this.camera = new Camera(dummyCanvas, device);

        this.trainingConfig = trainingConfig || {
            lambda_l1: 0.2,
            lambda_l2: 0.0,
            lambda_dssim: 0.2
        };

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
    }

    setPointCloud(pointCloud: PointCloud) {
        this.pointCloud = pointCloud;
        this.optimizer = new Optimizer(this.device, pointCloud);
    }

    setDataset(cameras: CameraData[], images: LoadedImage[]) {
        this.trainCameras = cameras;
        this.images = images;
    }

    start() {
        if (!this.pointCloud || this.trainCameras.length === 0) {
            console.error("Cannot start training: Missing point cloud or dataset.");
            return;
        }
        this.isTraining = true;
        this.iteration = 0;
        console.log("Training started");
    }

    stop() {
        this.isTraining = false;
        console.log("Training stopped");
    }

    getIsTraining() {
        return this.isTraining;
    }

    // Single training step
    async step() {
        if (!this.isTraining || !this.pointCloud) return;

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
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        this.iteration++;

        if (this.iteration % 100 === 0) {
            console.log(`Iteration ${this.iteration}`);
        }
    }

    private ensurePipelines(width: number, height: number) {
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

        this.device.queue.submit([encoder.finish()]);
    }
}
