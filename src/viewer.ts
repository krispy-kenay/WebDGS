
import { TiledForwardPass, RenderMode } from './renderers/tiled-forward-pass';
import { TiledRasterizer } from './renderers/tiled-rasterizer';
import { Camera } from './camera/camera';
import { CameraControl } from './camera/camera-control';
import { PointCloud } from './utils/load';

export class Viewer {
    public readonly camera: Camera;
    public readonly cameraControl: CameraControl;

    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly canvas: HTMLCanvasElement;

    private forwardPass: TiledForwardPass | null = null;
    private rasterizer: TiledRasterizer | null = null;
    private pointCloud: PointCloud | null = null;

    private presentationFormat: GPUTextureFormat;

    constructor(
        device: GPUDevice,
        context: GPUCanvasContext,
        canvas: HTMLCanvasElement,
        format: GPUTextureFormat
    ) {
        this.device = device;
        this.context = context;
        this.canvas = canvas;
        this.presentationFormat = format;

        this.camera = new Camera(canvas, device);
        this.cameraControl = new CameraControl(this.camera);

        // Handle resize internally
        const resizeObserver = new ResizeObserver(() => {
            this.handleResize();
        });
        resizeObserver.observe(canvas);

        // Initial setup
        this.handleResize();
    }

    setPointCloud(pointCloud: PointCloud) {
        this.pointCloud = pointCloud;
        this.forwardPass = new TiledForwardPass(this.device, pointCloud, this.camera.uniform_buffer, {
            viewportWidth: this.canvas.width,
            viewportHeight: this.canvas.height,
            renderMode: 'pointcloud',
        });

        this.rasterizer = new TiledRasterizer({
            device: this.device,
            forwardPass: this.forwardPass,
            format: this.presentationFormat,
        });

        // Ensure camera is updated
        this.camera.on_update_canvas();
    }

    update(dt: number) {
        this.cameraControl.update(dt);
    }

    render(commandEncoder: GPUCommandEncoder) {
        if (!this.forwardPass || !this.rasterizer || !this.pointCloud) {
            return;
        }

        // Forward Pass
        this.forwardPass.encode(commandEncoder);

        // Rasterization
        const swapTexture = this.context.getCurrentTexture();
        const swapView = swapTexture.createView();

        this.rasterizer.encode(commandEncoder, swapTexture.width, swapTexture.height);

        // Blit to screen
        this.rasterizer.blitToTexture(commandEncoder, swapView);
    }

    // Pass-through setters
    setRenderMode(mode: RenderMode) {
        this.forwardPass?.setRenderMode(mode);
    }

    setGaussianScale(value: number) {
        this.forwardPass?.setGaussianScale(value);
    }

    setPointSize(value: number) {
        this.forwardPass?.setPointSize(value);
    }

    // Pass-through getters
    getForwardPass() {
        return this.forwardPass;
    }

    private handleResize() {
        if (!this.canvas) return;
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;

        this.camera.on_update_canvas();
        this.forwardPass?.setViewport(this.canvas.width, this.canvas.height);
    }
}
