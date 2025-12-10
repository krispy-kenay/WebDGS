import { vec3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

interface MovementState {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  up: boolean;
  down: boolean;
  rotateLeft: boolean;
  rotateRight: boolean;
}

export class CameraControl {
  private element: HTMLCanvasElement;
  private pointerActive = false;
  private lastX = 0;
  private lastY = 0;
  private pointerId: number | null = null;
  private moveVec = vec3.create();
  private tempVec = vec3.create();
  private localForward = vec3.create();
  private localRight = vec3.create();
  private localUp = vec3.create();
  private readonly worldUp = vec3.create(0, 1, 0);
  private tempQuat = quat.create();
  private tempMat = mat4.create();
  private movementState: MovementState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false,
    rotateLeft: false,
    rotateRight: false,
  };
  private enabled = false;
  private readonly lookSensitivity = 0.003;

  constructor(private readonly camera: Camera) {
    this.element = camera.canvas;
    this.attachPointerListeners();
    this.element.addEventListener('wheel', this.handleWheel, { passive: false });
  }

  registerKeyboardListeners(target: HTMLElement | Window = window) {
    target.addEventListener('keydown', (event: KeyboardEvent) => {
      if (!this.enabled) return;
      if (this.setMovementState(event.code, true)) {
        event.preventDefault();
      }
    });
    target.addEventListener('keyup', (event: KeyboardEvent) => {
      if (!this.enabled) return;
      if (this.setMovementState(event.code, false)) {
        event.preventDefault();
      }
    });
  }

  setEnabled(value: boolean) {
    if (this.enabled === value) return;
    this.enabled = value;
    if (!value) {
      Object.keys(this.movementState).forEach((key) => {
        this.movementState[key as keyof MovementState] = false;
      });
      this.pointerActive = false;
      if (this.pointerId !== null) {
        this.element.releasePointerCapture(this.pointerId);
        this.pointerId = null;
      }
    }
  }

  update(deltaSeconds: number) {
    if (!this.enabled || deltaSeconds <= 0) return;
    const forward = vec3.normalize(this.camera.look, this.localForward);
    const right = vec3.normalize(this.camera.right, this.localRight);
    const up = vec3.normalize(this.camera.up, this.localUp);
    const speed = 4;
    const move = this.moveVec;
    vec3.set(0, 0, 0, move);
    if (this.movementState.forward) vec3.add(move, forward, move);
    if (this.movementState.backward) vec3.subtract(move, forward, move);
    if (this.movementState.left) vec3.subtract(move, right, move);
    if (this.movementState.right) vec3.add(move, right, move);
    if (this.movementState.up) vec3.add(move, up, move);
    if (this.movementState.down) vec3.subtract(move, up, move);

    if (vec3.len(move) > 0) {
      vec3.normalize(move, move);
      vec3.scale(move, speed * deltaSeconds, move);
      vec3.add(this.camera.position, move, this.camera.position);
      this.camera.update_buffer();
    }

    const rollSpeed = 80;
    if (this.movementState.rotateLeft) this.roll(forward, rollSpeed * deltaSeconds);
    if (this.movementState.rotateRight) this.roll(forward, -rollSpeed * deltaSeconds);
  }

  private setMovementState(code: string, value: boolean) {
    switch (code) {
      case 'KeyW':
        this.movementState.forward = value;
        return true;
      case 'KeyS':
        this.movementState.backward = value;
        return true;
      case 'KeyA':
        this.movementState.left = value;
        return true;
      case 'KeyD':
        this.movementState.right = value;
        return true;
      case 'Space':
        this.movementState.up = value;
        return true;
      case 'ControlLeft':
      case 'ControlRight':
        this.movementState.down = value;
        return true;
      case 'KeyQ':
        this.movementState.rotateLeft = value;
        return true;
      case 'KeyE':
        this.movementState.rotateRight = value;
        return true;
      default:
        return false;
    }
  }

  private attachPointerListeners() {
    this.element.addEventListener('pointerdown', (event) => {
      if (!this.enabled || !event.isPrimary || event.button !== 0) return;
      this.pointerActive = true;
      this.pointerId = event.pointerId;
      this.lastX = event.pageX;
      this.lastY = event.pageY;
      this.element.setPointerCapture(event.pointerId);
    });

    this.element.addEventListener('pointermove', (event) => {
      if (!this.enabled || !this.pointerActive || event.pointerId !== this.pointerId) return;
      const xDelta = event.pageX - this.lastX;
      const yDelta = event.pageY - this.lastY;
      this.lastX = event.pageX;
      this.lastY = event.pageY;
      this.rotate(xDelta, yDelta);
    });

    const clearPointer = () => {
      if (this.pointerId !== null) {
        this.element.releasePointerCapture(this.pointerId);
      }
      this.pointerId = null;
      this.pointerActive = false;
    };

    this.element.addEventListener('pointerup', clearPointer);
    this.element.addEventListener('pointercancel', clearPointer);
    this.element.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  private handleWheel = (event: WheelEvent) => {
    if (!this.enabled) return;
    event.preventDefault();
    const move = vec3.scale(this.camera.look, -event.deltaY * 0.002, this.tempVec);
    vec3.add(this.camera.position, move, this.camera.position);
    this.camera.update_buffer();
  };

  private rotate(deltaYawPixels: number, deltaPitchPixels: number) {
    const yaw = deltaYawPixels * this.lookSensitivity;
    const pitch = -deltaPitchPixels * this.lookSensitivity;

    if (yaw !== 0) {
      const yawAxis = vec3.normalize(this.camera.up, this.tempVec);
      if (vec3.len(yawAxis) < 1e-5) {
        vec3.copy(this.worldUp, yawAxis);
      }
      const yawQuat = quat.fromAxisAngle(yawAxis as any, yaw, this.tempQuat);
      const yawMat = mat4.fromQuat(yawQuat, this.tempMat);
      mat4.multiply(this.camera.rotation, yawMat, this.camera.rotation);
    }

    if (pitch !== 0) {
      const axis = vec3.normalize(this.camera.right, this.tempVec);
      if (vec3.len(axis) > 0) {
        const pitchQuat = quat.fromAxisAngle(axis as any, pitch, this.tempQuat);
        const pitchMat = mat4.fromQuat(pitchQuat, this.tempMat);
        mat4.multiply(this.camera.rotation, pitchMat, this.camera.rotation);
      }
    }

    this.camera.update_buffer();
  }

  private roll(axis: Float32Array, angleDegrees: number) {
    if (angleDegrees === 0 || vec3.len(axis) < 1e-5) return;
    const rollQuat = quat.fromAxisAngle(axis as any, angleDegrees * Math.PI / 180, this.tempQuat);
    const rollMat = mat4.fromQuat(rollQuat, this.tempMat);
    mat4.multiply(this.camera.rotation, rollMat, this.camera.rotation);
    this.camera.update_buffer();
  }
}
