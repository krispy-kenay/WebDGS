
import './style.css';
import { assert } from './utils/util';
import { load, CameraData, PointCloud } from './utils/load';
import { loadCamera } from './utils/load-camera';
import { loadImages, LoadedImage } from './utils/load-images';
import { Viewer } from './viewer';
import { Trainer, PointCloudSwapRequest } from './trainer';
import { RenderMode } from './renderers/tiled-forward-pass';
import { QueueGate } from './utils/queue-gate';

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
const trainIterationsSlider = document.getElementById('train-iterations') as HTMLInputElement;
const densifyWarmupSlider = document.getElementById('densify-warmup') as HTMLInputElement;
const densifyIntervalSlider = document.getElementById('densify-interval') as HTMLInputElement;
const densifyStopSlider = document.getElementById('densify-stop') as HTMLInputElement;
const lrPosSlider = document.getElementById('lr-pos') as HTMLInputElement;
const lrRotSlider = document.getElementById('lr-rot') as HTMLInputElement;
const lrScaleSlider = document.getElementById('lr-scale') as HTMLInputElement;
const lrOpacitySlider = document.getElementById('lr-opacity') as HTMLInputElement;
const lrColorSlider = document.getElementById('lr-color') as HTMLInputElement;
const lossLambdaL1Slider = document.getElementById('loss-lambda-l1') as HTMLInputElement;
const lossLambdaL2Slider = document.getElementById('loss-lambda-l2') as HTMLInputElement;
const lossLambdaDssimSlider = document.getElementById('loss-lambda-dssim') as HTMLInputElement;

const trainingStatusPill = document.getElementById('training-status') as HTMLElement;
const trainingIterLabel = document.getElementById('training-iter') as HTMLElement;
const trainingIterMaxLabel = document.getElementById('training-iter-max') as HTMLElement;
const trainingProgressBar = document.getElementById('training-progress-bar') as HTMLElement;
const trainingIpsLabel = document.getElementById('training-ips') as HTMLElement;
const trainingGaussiansLabel = document.getElementById('training-gaussians') as HTMLElement;
const trainingNextDensifyLabel = document.getElementById('training-next-densify') as HTMLElement;

const trainIterationsValue = document.getElementById('train-iterations-value') as HTMLElement;
const densifyWarmupValue = document.getElementById('densify-warmup-value') as HTMLElement;
const densifyIntervalValue = document.getElementById('densify-interval-value') as HTMLElement;
const densifyStopValue = document.getElementById('densify-stop-value') as HTMLElement;
const lrPosValue = document.getElementById('lr-pos-value') as HTMLElement;
const lrRotValue = document.getElementById('lr-rot-value') as HTMLElement;
const lrScaleValue = document.getElementById('lr-scale-value') as HTMLElement;
const lrOpacityValue = document.getElementById('lr-opacity-value') as HTMLElement;
const lrColorValue = document.getElementById('lr-color-value') as HTMLElement;
const lossLambdaL1Value = document.getElementById('loss-lambda-l1-value') as HTMLElement;
const lossLambdaL2Value = document.getElementById('loss-lambda-l2-value') as HTMLElement;
const lossLambdaDssimValue = document.getElementById('loss-lambda-dssim-value') as HTMLElement;
const lossLambdaWarning = document.getElementById('loss-lambda-warning') as HTMLElement;
const gaussianScaleValue = document.getElementById('gaussian-scale-value') as HTMLElement;
const pointSizeValue = document.getElementById('point-size-value') as HTMLElement;
const controlPanel = document.getElementById('control-panel') as HTMLElement;
const panelCollapseBtn = document.getElementById('panel-collapse') as HTMLButtonElement;
const panelFloatingToggle = document.getElementById('panel-floating-toggle') as HTMLButtonElement;
const fpsIndicator = document.getElementById('fps-indicator') as HTMLElement;
const trainingIterHud = document.getElementById('training-iter-hud') as HTMLElement;

const state = {
  device: null as GPUDevice | null,
  context: null as GPUCanvasContext | null,
  canvas: null as HTMLCanvasElement | null,

  viewer: null as Viewer | null,
  trainer: null as Trainer | null,
  queueGate: null as QueueGate | null,

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
  swapBusy: false,
  pendingPointCloudSwap: null as PointCloudSwapRequest | null,
  showLoss: false,
  currentCameraIndex: -1,
};

bootstrap();

function formatNumber(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatItersPerSec(v: number): string {
  if (!isFinite(v) || v <= 0) return '--';
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

function bindSliderValue(
  input: HTMLInputElement,
  valueEl: HTMLElement | null,
  format: (v: number) => string = (v) => `${v}`,
) {
  const update = () => {
    const min = parseFloat(input.min || '0');
    const max = parseFloat(input.max || '1');
    const v = parseFloat(input.value);
    const pct = max > min ? ((v - min) / (max - min)) * 100 : 0;
    input.style.setProperty('--pct', `${Math.max(0, Math.min(100, pct))}%`);
    if (valueEl) valueEl.textContent = format(v);
  };
  input.addEventListener('input', update);
  update();
}

function updateTrainingWidget() {
  if (!state.trainer) return;

  const iter = state.trainer.getIteration();
  const maxIter = state.trainer.getMaxIterations();
  const isTraining = state.trainer.getIsTraining();
  const gaussians = state.trainer.getPointCount();
  const ips = state.trainer.getItersPerSec();
  const nextDensify = state.trainer.getNextDensifyPruneIteration();

  trainingIterLabel.textContent = formatNumber(iter);
  trainingIterMaxLabel.textContent = formatNumber(maxIter);
  trainingGaussiansLabel.textContent = gaussians ? formatNumber(gaussians) : '--';
  trainingIpsLabel.textContent = formatItersPerSec(ips);
  trainingNextDensifyLabel.textContent = nextDensify ? formatNumber(nextDensify) : '--';

  if (trainingIterHud) {
    const showHud = state.renderActive && state.panelCollapsed && isTraining;
    trainingIterHud.style.display = showHud ? 'block' : 'none';
    if (showHud) {
      trainingIterHud.textContent = `${formatNumber(iter)} / ${formatNumber(maxIter)}`;
    }
  }

  const pct = maxIter > 0 ? (iter / maxIter) * 100 : 0;
  trainingProgressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;

  if (isTraining) {
    trainingStatusPill.textContent = 'Training';
    trainingStatusPill.className = 'pill pill--active';
  } else if (iter > 0 && iter >= maxIter) {
    trainingStatusPill.textContent = 'Done';
    trainingStatusPill.className = 'pill pill--paused';
  } else {
    trainingStatusPill.textContent = 'Idle';
    trainingStatusPill.className = 'pill pill--muted';
  }
}

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
    state.queueGate = new QueueGate(device.queue, { maxInFlight: 2 });
    state.trainer = new Trainer(device, undefined, state.queueGate);

    state.viewer.cameraControl.registerKeyboardListeners(window);

    updateStatus('ready', 'All required WebGPU features available', 'Adapter ready.');
    logMessage('WebGPU initialized. Upload data to begin.', 'success');
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    updateStatus('error', 'Failed to initialize WebGPU', message);
  }
}

function setupUiHandlers() {
  if (state.trainer) {
    const cfg = state.trainer.getTrainingConfig();
    lossLambdaL1Slider.value = `${cfg.lambda_l1 ?? 0}`;
    lossLambdaL2Slider.value = `${cfg.lambda_l2 ?? 0}`;
    lossLambdaDssimSlider.value = `${cfg.lambda_dssim ?? 0}`;

    const opt = state.trainer.getOptimizerHyperparameters();
    lrPosSlider.value = `${opt.lr_pos ?? 0}`;
    lrRotSlider.value = `${opt.lr_rot ?? 0}`;
    lrScaleSlider.value = `${opt.lr_scale ?? 0}`;
    lrOpacitySlider.value = `${opt.lr_opacity ?? 0}`;
    lrColorSlider.value = `${opt.lr_color ?? 0}`;
  }

  bindSliderValue(trainIterationsSlider, trainIterationsValue, (v) => formatNumber(Math.round(v)));
  bindSliderValue(densifyWarmupSlider, densifyWarmupValue, (v) => formatNumber(Math.round(v)));
  bindSliderValue(densifyIntervalSlider, densifyIntervalValue, (v) => formatNumber(Math.round(v)));
  bindSliderValue(densifyStopSlider, densifyStopValue, (v) => formatNumber(Math.round(v)));
  bindSliderValue(lrPosSlider, lrPosValue, (v) => v.toExponential(2));
  bindSliderValue(lrRotSlider, lrRotValue, (v) => v.toExponential(2));
  bindSliderValue(lrScaleSlider, lrScaleValue, (v) => v.toExponential(2));
  bindSliderValue(lrOpacitySlider, lrOpacityValue, (v) => v.toExponential(2));
  bindSliderValue(lrColorSlider, lrColorValue, (v) => v.toExponential(2));
  bindSliderValue(lossLambdaL1Slider, lossLambdaL1Value, (v) => v.toFixed(2));
  bindSliderValue(lossLambdaL2Slider, lossLambdaL2Value, (v) => v.toFixed(2));
  bindSliderValue(lossLambdaDssimSlider, lossLambdaDssimValue, (v) => v.toFixed(2));
  bindSliderValue(gaussianScaleSlider, gaussianScaleValue, (v) => v.toFixed(2));
  bindSliderValue(pointSizeSlider, pointSizeValue, (v) => formatNumber(Math.round(v)));

  const syncTrainerConfig = () => {
    if (!state.trainer) return;
    const maxIters = parseInt(trainIterationsSlider.value, 10) || 10000;
    state.trainer.setMaxIterations(maxIters);

    state.trainer.setDensifyPruneConfig({
      schedule: {
        enabled: true,
        warmupIterations: Math.max(0, Math.floor(parseInt(densifyWarmupSlider.value, 10) || 0)),
        interval: Math.max(1, Math.floor(parseInt(densifyIntervalSlider.value, 10) || 100)),
        stopIterations: Math.max(1, Math.floor(parseInt(densifyStopSlider.value, 10) || 15000)),
      },
    });
  };

  trainIterationsSlider.addEventListener('input', syncTrainerConfig);
  densifyWarmupSlider.addEventListener('input', syncTrainerConfig);
  densifyIntervalSlider.addEventListener('input', syncTrainerConfig);
  densifyStopSlider.addEventListener('input', syncTrainerConfig);

  const syncOptimizerConfig = () => {
    if (!state.trainer) return;
    state.trainer.setOptimizerHyperparameters({
      lr_pos: parseFloat(lrPosSlider.value) || 0,
      lr_rot: parseFloat(lrRotSlider.value) || 0,
      lr_scale: parseFloat(lrScaleSlider.value) || 0,
      lr_opacity: parseFloat(lrOpacitySlider.value) || 0,
      lr_color: parseFloat(lrColorSlider.value) || 0,
    });
  };

  lrPosSlider.addEventListener('input', syncOptimizerConfig);
  lrRotSlider.addEventListener('input', syncOptimizerConfig);
  lrScaleSlider.addEventListener('input', syncOptimizerConfig);
  lrOpacitySlider.addEventListener('input', syncOptimizerConfig);
  lrColorSlider.addEventListener('input', syncOptimizerConfig);

  const updateLossLambdaWarning = () => {
    const l1 = parseFloat(lossLambdaL1Slider.value) || 0;
    const l2 = parseFloat(lossLambdaL2Slider.value) || 0;
    const dssim = parseFloat(lossLambdaDssimSlider.value) || 0;
    const sum = l1 + l2 + dssim;
    const delta = Math.abs(sum - 1);
    const ok = delta < 1e-3;

    if (ok) {
      lossLambdaWarning.style.display = 'none';
      lossLambdaWarning.textContent = '';
      return;
    }

    lossLambdaWarning.style.display = 'block';
    if (sum === 0) {
      lossLambdaWarning.textContent = 'Loss weights sum to 0.0; gradients will be zero.';
    } else {
      lossLambdaWarning.textContent = `Loss weights should sum to 1.0 (current: ${sum.toFixed(2)}).`;
    }
  };

  const syncLossConfig = () => {
    if (!state.trainer) return;
    state.trainer.setTrainingConfig({
      lambda_l1: parseFloat(lossLambdaL1Slider.value) || 0,
      lambda_l2: parseFloat(lossLambdaL2Slider.value) || 0,
      lambda_dssim: parseFloat(lossLambdaDssimSlider.value) || 0,
    });
    updateLossLambdaWarning();
  };

  lossLambdaL1Slider.addEventListener('input', syncLossConfig);
  lossLambdaL2Slider.addEventListener('input', syncLossConfig);
  lossLambdaDssimSlider.addEventListener('input', syncLossConfig);

  syncTrainerConfig();
  syncOptimizerConfig();
  syncLossConfig();
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
    syncTrainerConfig();
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

async function applyPendingPointCloudSwap() {
  if (state.swapBusy) return;
  if (!state.device || !state.viewer || !state.trainer) return;
  const next = state.pendingPointCloudSwap;
  if (!next) return;

  state.swapBusy = true;
  state.pendingPointCloudSwap = null;
  state.trainingBusy = true;

	  try {
	    if (state.queueGate) {
	      await state.queueGate.waitForIdle();
	    } else {
	      await state.device.queue.onSubmittedWorkDone();
	    }
	    state.viewer.setPointCloud(next.pointCloud);
	    state.viewer.setRenderMode(state.renderMode);
	    state.viewer.setGaussianScale(state.gaussianScale);
	    state.viewer.setPointSize(state.pointSize);
	    state.trainer.applyPointCloudSwap(next);
	  } catch (e) {
	    console.error('Failed to apply point cloud swap:', e);
	  } finally {
    state.swapBusy = false;
    state.trainingBusy = false;
  }
}

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
      if (!state.swapBusy && !state.pendingPointCloudSwap && state.showLoss && state.trainer && state.currentCameraIndex >= 0) {
        state.trainer.visualizeLoss(state.context, state.currentCameraIndex);
      } else if (!state.swapBusy && !state.pendingPointCloudSwap) {
        const gate = state.queueGate;
        if (!gate || gate.canSubmit()) {
          const encoder = state.device.createCommandEncoder();
          state.viewer.render(encoder);
          const cmd = encoder.finish();
          if (gate) {
            gate.trySubmit([cmd]);
          } else {
            state.device.queue.submit([cmd]);
          }
        }
      }

      updateRendererCta(true);
    } else if (fpsIndicator) {
      fpsIndicator.textContent = '-- fps';
      smoothedFps = 0;
    }

    // Trainer Step
    if (state.trainer && !state.pendingPointCloudSwap) {
      const swapReq = state.trainer.consumePointCloudSwapRequest();
      if (swapReq) state.pendingPointCloudSwap = swapReq;
    }
    if (state.pendingPointCloudSwap && !state.swapBusy) {
      applyPendingPointCloudSwap();
    }

    if (state.trainer && state.trainer.getIsTraining() && !state.trainingBusy && !state.swapBusy && !state.pendingPointCloudSwap) {
      state.trainingBusy = true;
      state.trainer.step().finally(() => {
        state.trainingBusy = false;
      });
    }

    updateTrainingWidget();

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
