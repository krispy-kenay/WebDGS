
import './style.css';
import { assert } from './utils/util';
import { load, CameraData, PointCloud } from './utils/load';
import { loadCamera } from './utils/load-camera';
import { loadImages, LoadedImage } from './utils/load-images';
import { Viewer } from './viewer';
import { Trainer } from './trainer';
import { RenderMode } from './renderers/tiled-forward-pass';

type CameraPreset = CameraData;
type StatusState = 'checking' | 'ready' | 'error';

const statusIndicator = document.getElementById('status-indicator') as HTMLElement;
const statusIcon = document.getElementById('status-icon') as HTMLElement;
const statusText = document.getElementById('status-text') as HTMLElement;
const statusDetails = document.getElementById('status-details') as HTMLElement;
const logContainer = document.getElementById('log') as HTMLDivElement;
const rendererStatus = document.getElementById('renderer-status') as HTMLElement;
const enterRendererBtn = document.getElementById('enter-renderer') as HTMLButtonElement;
const renderModeSelect = document.getElementById('render-mode') as HTMLSelectElement;
const cameraChoiceSelect = document.getElementById('camera-choice') as HTMLSelectElement;
const gaussianScaleSlider = document.getElementById('gaussian-scale') as HTMLInputElement;
const pointSizeSlider = document.getElementById('point-size') as HTMLInputElement;
const plyInput = document.getElementById('ply-input') as HTMLInputElement;
const imagesInput = document.getElementById('images-input') as HTMLInputElement;
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

  viewer: null as Viewer | null,
  trainer: null as Trainer | null,

  pointCloudLoaded: false,
  camerasLoaded: false,

  renderMode: renderModeSelect.value as RenderMode,
  gaussianScale: parseFloat(gaussianScaleSlider.value) || 1,
  pointSize: parseFloat(pointSizeSlider.value) || 1,

  renderActive: false,
  panelCollapsed: false,

  trainingPresets: [] as CameraPreset[],
  images: [] as LoadedImage[],

  trainingBusy: false,
  showLoss: false,
  currentCameraIndex: -1,
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
    const requiredFeatures: GPUFeatureName[] = [];

    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    state.device = device;
    state.context = context;
    state.canvas = canvas;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // Ensure canvas matches display pixels
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: 'opaque',
    });

    // Initialize Viewer
    state.viewer = new Viewer(device, context, canvas, presentationFormat);

    // Initialize Trainer
    state.trainer = new Trainer(device);

    state.viewer.cameraControl.registerKeyboardListeners(window);

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
    state.viewer?.setRenderMode(mode);
  });

  cameraChoiceSelect.addEventListener('change', (event) => {
    const value = parseInt((event.target as HTMLSelectElement).value, 10);
    if (!state.viewer) return;

    if (value === -1) {
      state.viewer.camera.reset();
      logMessage('Viewer camera reset to default.');
    } else {
      const preset = state.trainingPresets[value];
      if (preset) {
        state.viewer.camera.set_preset(preset);
        logMessage(`Viewer copied intrinsics from Training Camera ${value}.`);
      }
    }
  });

  const showLossCheckbox = document.getElementById('show-loss') as HTMLInputElement;
  if (showLossCheckbox) {
    showLossCheckbox.addEventListener('change', () => {
      state.showLoss = showLossCheckbox.checked;
    });
  }

  gaussianScaleSlider.addEventListener('input', (event) => {
    const value = parseFloat((event.target as HTMLInputElement).value);
    state.gaussianScale = value;
    state.viewer?.setGaussianScale(value);
  });

  pointSizeSlider.addEventListener('input', (event) => {
    const value = parseFloat((event.target as HTMLInputElement).value);
    state.pointSize = value;
    state.viewer?.setPointSize(value);
  });

  plyInput.addEventListener('change', async () => {
    if (!state.device || !state.viewer || !state.trainer) return;
    const file = plyInput.files?.[0];
    if (!file) return;
    logMessage(`Loading ${file.name}…`);
    try {
      const pointCloud = await load(file, state.device);
      if ('type' in pointCloud) {
        state.viewer.setPointCloud(pointCloud);
        state.trainer.setPointCloud(pointCloud);
        state.pointCloudLoaded = true;
        logMessage(`Loaded ${file.name}`, 'success');
      } else {
        logMessage('Loaded cameras instead of point cloud?', 'error');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      logMessage(`Failed to load ${file.name}: ${message}`, 'error');
    } finally {
      updateRendererCta();
      plyInput.value = '';
    }
  });

  cameraInput.addEventListener('change', async () => {
    if (!state.device) return;
    const fileList = cameraInput.files;
    if (!fileList || fileList.length === 0) return;

    const files = Array.from(fileList);

    logMessage(`Loading camera presets from ${files.length} file(s)...`);
    try {
      const presets = await loadCamera(files);
      state.trainingPresets = presets;
      state.camerasLoaded = presets.length > 0;

      if (state.trainer) {
        state.trainer.setDataset(state.trainingPresets, state.images);
      }

      updateCameraDropdown();
      if (presets.length) {
        logMessage(`Loaded ${presets.length} training camera presets`, 'success');
      } else {
        logMessage('No camera presets found in files.', 'error');
      }

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      logMessage(`Failed to load camera presets: ${message}`, 'error');
    } finally {
      updateRendererCta();
      cameraInput.value = '';
    }
  });

  imagesInput.addEventListener('change', async () => {
    const files = imagesInput.files;
    if (!files || files.length === 0) return;
    logMessage(`Loading ${files.length} images...`);

    try {
      if (!state.device) throw new Error('Device not initialized');
      const loaded = await loadImages(files, state.device);
      state.images = loaded;

      if (state.trainer) {
        state.trainer.setDataset(state.trainingPresets, state.images);
      }

      logMessage(`Successfully loaded ${loaded.length} images.`, 'success');
    } catch (error) {
      logMessage(`Failed to load images: ${error}`, 'error');
    } finally {
      imagesInput.value = '';
    }
  });

  enterRendererBtn.addEventListener('click', () => {
    if (!canEnterViewer()) {
      logMessage('Load a splat before toggling the viewer.', 'error');
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
    if (!state.trainer) return;
    state.trainer.start();
    logMessage('Training started.', 'success');
  });

  trainStopBtn.addEventListener('click', () => {
    if (!state.trainer) return;
    state.trainer.stop();
    logMessage('Training stopped.');
  });

  panelCollapseBtn.addEventListener('click', () => {
    if (!state.renderActive) return;
    setPanelCollapsed(!state.panelCollapsed);
  });
  panelFloatingToggle.addEventListener('click', () => setPanelCollapsed(false));

  document.addEventListener('keydown', (event) => {
    if (state.renderActive && state.panelCollapsed && !event.repeat) {
      state.viewer?.cameraControl.setEnabled(true);
    }
  });

  updateRendererCta();
}

let smoothedFps = 0;
let lastFpsLabelUpdate = 0;

function startRenderLoop() {
  let lastFrameTimestamp = 0;

  const loop = (timestamp: number) => {
    const deltaSeconds = lastFrameTimestamp > 0 ? (timestamp - lastFrameTimestamp) / 1000 : 0;

    // Viewer update
    if (state.viewer) {
      state.viewer.cameraControl.setEnabled(state.renderActive && state.panelCollapsed);
      state.viewer.update(deltaSeconds);
    }

    if (state.renderActive && state.viewer && state.device && state.context) {
      // FPS Stats
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

      // Render Viewer or Loss
      if (state.showLoss && state.trainer && state.currentCameraIndex >= 0) {
        state.trainer.visualizeLoss(state.context, state.currentCameraIndex);
      } else {
        const encoder = state.device.createCommandEncoder();
        state.viewer.render(encoder);
        state.device.queue.submit([encoder.finish()]);
      }

      updateRendererCta(true);
    } else if (fpsIndicator) {
      fpsIndicator.textContent = '-- fps';
      smoothedFps = 0;
    }

    // Trainer Step
    if (state.trainer && state.trainer.getIsTraining() && !state.trainingBusy) {
      state.trainingBusy = true;
      state.trainer.step().finally(() => {
        state.trainingBusy = false;
      });
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
    rendererStatus.textContent = 'Load a splat to enable the viewer.';
  } else if (!state.renderActive) {
    rendererStatus.textContent = 'Viewer ready. Toggle to start.';
  } else if (rendering) {
    rendererStatus.textContent = 'Rendering…';
  } else {
    rendererStatus.textContent = 'Viewer running.';
  }
}

function canEnterViewer() {
  return Boolean(state.pointCloudLoaded && state.viewer);
}

function setPanelCollapsed(collapsed: boolean) {
  state.panelCollapsed = collapsed;
  controlPanel.classList.toggle('collapsed', collapsed);
  const disableOverlay = collapsed && state.renderActive;
  document.body.classList.toggle('panel-collapsed', disableOverlay);
  panelFloatingToggle.classList.toggle('visible', collapsed && state.renderActive);
  state.viewer?.cameraControl.setEnabled(state.renderActive && collapsed);
}

function updateCameraDropdown() {
  cameraChoiceSelect.innerHTML = '';

  const defaultOpt = document.createElement('option');
  defaultOpt.value = '-1';
  defaultOpt.textContent = 'Default Camera';
  cameraChoiceSelect.appendChild(defaultOpt);

  state.trainingPresets.forEach((preset, index) => {
    const opt = document.createElement('option');
    opt.value = index.toString();
    const name = preset.img_name ? preset.img_name : `Camera ${index}`;
    opt.textContent = name;
    cameraChoiceSelect.appendChild(opt);
  });
}
