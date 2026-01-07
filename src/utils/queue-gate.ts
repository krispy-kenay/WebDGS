export class QueueGate {
  private readonly queue: GPUQueue;
  private readonly maxInFlight: number;
  private inFlight = 0;
  private capacityWaiters: Array<() => void> = [];
  private idleWaiters: Array<() => void> = [];

  constructor(queue: GPUQueue, options?: { maxInFlight?: number }) {
    this.queue = queue;
    this.maxInFlight = Math.max(1, Math.floor(options?.maxInFlight ?? 2));
  }

  getInFlight(): number {
    return this.inFlight;
  }

  canSubmit(): boolean {
    return this.inFlight < this.maxInFlight;
  }

  async waitForIdle(): Promise<void> {
    if (this.inFlight === 0) return;
    await new Promise<void>((resolve) => {
      this.idleWaiters.push(resolve);
    });
  }

  trySubmit(commandBuffers: GPUCommandBuffer[]): boolean {
    if (!this.canSubmit()) return false;
    this.submitInternal(commandBuffers);
    return true;
  }

  async submit(commandBuffers: GPUCommandBuffer[]): Promise<void> {
    await this.waitForCapacity();
    return this.submitInternal(commandBuffers);
  }

  private async waitForCapacity(): Promise<void> {
    if (this.canSubmit()) return;
    await new Promise<void>((resolve) => {
      this.capacityWaiters.push(resolve);
    });
  }

  private submitInternal(commandBuffers: GPUCommandBuffer[]): Promise<void> {
    // If submit throws (device lost / validation), don't mutate state.
    this.queue.submit(commandBuffers);
    this.inFlight++;

    const done = this.queue.onSubmittedWorkDone();
    done.finally(() => {
      this.inFlight = Math.max(0, this.inFlight - 1);
      this.pumpWaiters();
    });
    return done;
  }

  private pumpWaiters(): void {
    while (this.inFlight < this.maxInFlight && this.capacityWaiters.length > 0) {
      const resolve = this.capacityWaiters.shift();
      resolve?.();
    }
    if (this.inFlight === 0 && this.idleWaiters.length > 0) {
      const waiters = this.idleWaiters;
      this.idleWaiters = [];
      for (const resolve of waiters) resolve();
    }
  }
}

