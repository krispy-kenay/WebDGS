import './style.css';
import { ForwardPass, RenderMode } from './renderers/forward-pass';
import { Rasterizer } from './renderers/rasterizer';
import { assert } from './utils/util';
import { load } from './utils/load';
import { Camera, load_camera_presets } from './camera/camera';
import { CameraControl } from './camera/camera-control';

type CameraPreset = Awaited<ReturnType<typeof load_camera_presets>>[number];
type StatusState = 'checking' | 'ready' | 'error';

const statusIndicator = document.getElementById('status-indicator') as HTMLElement;
const statusIcon = document.getElementById('status-icon') as HTMLElement;
const statusText = document.getElementById('status-text') as HTMLElement;
const statusDetails = document.getElementById('status-details') as HTMLElement;
const logContainer = document.getElementById('log') as HTMLDivElement;
const rendererStatus = document.getElementById('renderer-status') as HTMLElement;
const enterRendererBtn = document.getElementById('enter-renderer') as HTMLButtonElement;
const renderModeSelect = document.getElementById('render-mode') as HTMLSelectElement;
const gaussianScaleSlider = document.getElementById('gaussian-scale') as HTMLInputElement;
const pointSizeSlider = document.getElementById('point-size') as HTMLInputElement;
const plyInput = document.getElementById('ply-input') as HTMLInputElement;
const colmapInput = document.getElementById('colmap-input') as HTMLInputElement;
const cameraInput = document.getElementById('camera-input') as HTMLInputElement;
const trainStartBtn = document.getElementById('train-start') as HTMLButtonElement;
const trainStopBtn = document.getElementById('train-stop') as HTMLButtonElement;
const controlPanel = document.getElementById('control-panel') as HTMLElement;
const panelCollapseBtn = document.getElementById('panel-collapse') as HTMLButtonElement;
const panelFloatingToggle = document.getElementById('panel-floating-toggle') as HTMLButtonElement;
const fpsIndicator = document.getElementById('fps-indicator') as HTMLElement;

const state = {
  device: null as GPUDevice | null,
  context: null as GPUCanvasContext | null,
  canvas: null as HTMLCanvasElement | null,
  presentationFormat: 'bgra8unorm' as GPUTextureFormat,
  camera: null as Camera | null,
  cameraControl: null as CameraControl | null,
  forwardPass: null as ForwardPass | null,
  rasterizer: null as Rasterizer | null,
  pointCloudLoaded: false,
  camerasLoaded: false,
  renderMode: renderModeSelect.value as RenderMode,
  gaussianScale: parseFloat(gaussianScaleSlider.value) || 1,
  pointSize: parseFloat(pointSizeSlider.value) || 1,
  renderActive: false,
  panelCollapsed: false,
  cameraPresets: [] as CameraPreset[],
};

bootstrap();

async function bootstrap() {
  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  assert(context !== null);

  await initializeWebGPU(canvas, context);
  setupUiHandlers();
  startRenderLoop();
}

async function initializeWebGPU(canvas: HTMLCanvasElement, context: GPUCanvasContext) {
  if (!navigator.gpu) {
    updateStatus('error', 'WebGPU not supported', 'Use a modern Chromium browser with WebGPU enabled.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    updateStatus('error', 'No compatible GPU adapter found', 'Ensure your GPU drivers and browser are up to date.');
    return;
  }

  try {
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    state.device = device;
    state.context = context;
    state.canvas = canvas;
    state.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // Ensure canvas matches display pixels
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    context.configure({
      device,
      format: state.presentationFormat,
      alphaMode: 'opaque',
    });

  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);
  state.camera = camera;
  state.cameraControl = control;
  control.registerKeyboardListeners(window);

    const resizeObserver = new ResizeObserver(() => {
      if (!state.canvas || !state.camera) return;
      state.canvas.width = state.canvas.clientWidth;
      state.canvas.height = state.canvas.clientHeight;
      state.camera.on_update_canvas();
      state.forwardPass?.setViewport(state.canvas.width, state.canvas.height);
    });
    resizeObserver.observe(canvas);

    updateStatus('ready', 'All required WebGPU features available', 'Adapter ready.');
    logMessage('WebGPU initialized. Upload data to begin.', 'success');
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    updateStatus('error', 'Failed to initialize WebGPU', message);
  }
}

function setupUiHandlers() {
  renderModeSelect.addEventListener('change', (event) => {
    const mode = (event.target as HTMLSelectElement).value as RenderMode;
    state.renderMode = mode;
    state.forwardPass?.setRenderMode(mode);
  });

  gaussianScaleSlider.addEventListener('input', (event) => {
    const value = parseFloat((event.target as HTMLInputElement).value);
    state.gaussianScale = value;
    state.forwardPass?.setGaussianScale(value);
  });

  pointSizeSlider.addEventListener('input', (event) => {
    const value = parseFloat((event.target as HTMLInputElement).value);
    state.pointSize = value;
    state.forwardPass?.setPointSize(value);
  });

  plyInput.addEventListener('change', async () => {
    if (!state.device || !state.camera || !state.canvas || !state.context) return;
    const file = plyInput.files?.[0];
    if (!file) return;
    logMessage(`Loading ${file.name}…`);
    try {
      const pointCloud = await load(file, state.device);
      setupForwardPipeline(pointCloud);
      state.pointCloudLoaded = true;
      logMessage(`Loaded ${file.name}`, 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      logMessage(`Failed to load ${file.name}: ${message}`, 'error');
    } finally {
      updateRendererCta();
      plyInput.value = '';
    }
  });

  cameraInput.addEventListener('change', async () => {
    const file = cameraInput.files?.[0];
    if (!file) return;
    logMessage(`Loading camera presets from ${file.name}…`);
    try {
      const presets = await load_camera_presets(file);
      state.cameraPresets = presets;
      state.camerasLoaded = presets.length > 0;
      if (state.camera && presets.length > 0) {
        state.camera.set_preset(presets[0]);
        logMessage(`Loaded ${presets.length} camera presets`, 'success');
      } else if (!presets.length) {
        logMessage('No camera presets found in file.', 'error');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      logMessage(`Failed to load camera presets: ${message}`, 'error');
    } finally {
      updateRendererCta();
      cameraInput.value = '';
    }
  });

  colmapInput.addEventListener('change', () => {
    const files = colmapInput.files;
    if (!files || files.length === 0) return;
    logMessage(`Received ${files.length} COLMAP files. Pipeline integration coming soon.`);
  });

  enterRendererBtn.addEventListener('click', () => {
    if (!canEnterViewer()) {
      logMessage('Load a PLY file and camera presets before toggling the viewer.', 'error');
      return;
    }
    state.renderActive = !state.renderActive;
    if (state.renderActive) {
      document.body.classList.add('render-mode');
      logMessage('Viewer activated. Camera controls enabled.', 'success');
    } else {
      document.body.classList.remove('render-mode');
      logMessage('Viewer paused.', 'info');
    }
    setPanelCollapsed(false);
    updateRendererCta();
  });

  trainStartBtn.addEventListener('click', () => {
    logMessage('Training pipeline queued. (Placeholder)', 'success');
  });
  trainStopBtn.addEventListener('click', () => {
    logMessage('Training stopped. (Placeholder)');
  });

  panelCollapseBtn.addEventListener('click', () => {
    if (!state.renderActive) return;
    setPanelCollapsed(!state.panelCollapsed);
  });
  panelFloatingToggle.addEventListener('click', () => setPanelCollapsed(false));

  document.addEventListener('keydown', (event) => {
    if (state.renderActive && state.panelCollapsed && !event.repeat) {
      state.cameraControl?.setEnabled(true);
    }
    if (!state.cameraPresets.length || !state.camera) return;
    if (event.key >= '0' && event.key <= '9') {
      const idx = parseInt(event.key, 10);
      const preset = state.cameraPresets[idx];
      if (preset) {
        state.camera.set_preset(preset);
        logMessage(`Camera preset ${idx} selected.`);
      }
    }
  });

  updateRendererCta();
}

function setupForwardPipeline(pointCloud: Awaited<ReturnType<typeof load>>) {
  if (!state.device || !state.camera || !state.canvas) return;
  state.forwardPass = new ForwardPass(state.device, pointCloud, state.camera.uniform_buffer, {
    viewportWidth: state.canvas.width,
    viewportHeight: state.canvas.height,
    gaussianScale: state.gaussianScale,
    pointSizePx: state.pointSize,
    renderMode: state.renderMode,
  });
  state.rasterizer = new Rasterizer({
    device: state.device,
    forwardPass: state.forwardPass,
    format: state.presentationFormat,
  });
}

let smoothedFps = 0;
let lastFpsLabelUpdate = 0;

function startRenderLoop() {
  let lastFrameTimestamp = 0;
  const loop = (timestamp: number) => {
    const deltaSeconds = lastFrameTimestamp > 0 ? (timestamp - lastFrameTimestamp) / 1000 : 0;
    if (state.cameraControl) {
      state.cameraControl.setEnabled(state.renderActive && state.panelCollapsed);
      state.cameraControl.update(deltaSeconds);
    }
    if (state.renderActive && state.forwardPass && state.rasterizer && state.device && state.context) {
      if (lastFrameTimestamp > 0) {
        const delta = timestamp - lastFrameTimestamp;
        if (delta > 0) {
          const currentFps = 1000 / delta;
          smoothedFps = smoothedFps === 0 ? currentFps : smoothedFps * 0.9 + currentFps * 0.1;
        }
      }
      if (timestamp - lastFpsLabelUpdate > 200 && fpsIndicator) {
        fpsIndicator.textContent = `${smoothedFps.toFixed(1)} fps`;
        lastFpsLabelUpdate = timestamp;
      }
      const encoder = state.device.createCommandEncoder();
      state.forwardPass.encode(encoder);
      const textureView = state.context.getCurrentTexture().createView();
      state.rasterizer.encode(encoder, textureView);
      state.device.queue.submit([encoder.finish()]);
      updateRendererCta(true);
    } else if (fpsIndicator) {
      fpsIndicator.textContent = '-- fps';
      smoothedFps = 0;
    }
    lastFrameTimestamp = timestamp;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function updateStatus(stateName: StatusState, summary: string, details = '') {
  statusIndicator.dataset.status = stateName;
  statusText.textContent = summary;
  statusDetails.textContent = details;
  if (stateName === 'ready') {
    statusText.style.color = '#4ade80';
    statusIcon.textContent = '✔';
  } else if (stateName === 'error') {
    statusText.style.color = '#f87171';
    statusIcon.textContent = '✕';
  } else {
    statusText.style.color = '#f8fafc';
    statusIcon.textContent = '⏳';
  }
}

function logMessage(message: string, variant: 'info' | 'error' | 'success' = 'info') {
  if (!logContainer) return;
  const entry = document.createElement('p');
  entry.textContent = message;
  entry.classList.add(variant);
  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
}

function updateRendererCta(rendering = false) {
  const ready = canEnterViewer();
  enterRendererBtn.disabled = !ready;
  if (!ready) {
    rendererStatus.textContent = 'Load a splat and camera data to enable the viewer.';
  } else if (!state.renderActive) {
    rendererStatus.textContent = 'Viewer ready. Toggle to start.';
  } else if (rendering) {
    rendererStatus.textContent = 'Rendering…';
  } else {
    rendererStatus.textContent = 'Viewer running.';
  }
}

function canEnterViewer() {
  return Boolean(state.forwardPass && state.camerasLoaded);
}

function setPanelCollapsed(collapsed: boolean) {
  state.panelCollapsed = collapsed;
  controlPanel.classList.toggle('collapsed', collapsed);
  const disableOverlay = collapsed && state.renderActive;
  document.body.classList.toggle('panel-collapsed', disableOverlay);
  panelFloatingToggle.classList.toggle('visible', collapsed && state.renderActive);
  state.cameraControl?.setEnabled(state.renderActive && collapsed);
}
