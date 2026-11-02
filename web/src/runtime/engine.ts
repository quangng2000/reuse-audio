import { prepareRuntime, type Runtime } from './setup';
import { createScan } from '../gpu/scan';
import { createFiniteCheck } from '../gpu/finite';
import { upload, readBuffer, requireGpu, type Handle } from '../gpu/tensor';
import { executeFrame } from './execute';
export async function createEngine(url: string, hash: string) {
  const runtime = await prepareRuntime(url, hash);
  try { return new StreamingEngine(runtime, await createScan(runtime.device), await createFiniteCheck(runtime.device)); }
  catch (error) { await releaseRuntime(runtime); throw error; }
}
async function releaseRuntime(runtime: Runtime) {
  runtime.constants.forEach(handle => handle.release());
  for (const session of runtime.sessions.values()) await session.release();
  runtime.device.destroy();
}
export class StreamingEngine {
  private states = new Map<string, Handle>();
  private failed = false;
  private busy = false;
  private disposed = false;
  private deviceFailed = false;
  constructor(private runtime: Runtime, private scan: Awaited<ReturnType<typeof createScan>>,
    private check: Awaited<ReturnType<typeof createFiniteCheck>>) {
    this.reset();
    runtime.device.lost.then(() => { this.failed = true; this.deviceFailed = true; });
    runtime.device.addEventListener('uncapturederror', () => { this.failed = true; this.deviceFailed = true; });
  }
  reset() {
    if (this.busy || this.disposed || this.deviceFailed) throw new Error('Cannot reset a busy/disposed engine or lost GPU device.');
    this.states.forEach(handle => handle.release()); this.states.clear();
    for (const state of this.runtime.bundle.states)
      this.states.set(state.inputName, upload(this.runtime.device, state.shape));
    this.failed = false;
  }
  async process(spectrum: Float32Array): Promise<Float32Array> {
    if (this.busy || this.failed || this.disposed) throw new Error('Engine busy or unavailable; reset required.');
    this.busy = true;
    try {
      const result = await executeFrame({ runtime: this.runtime, states: this.states, spectrum, scan: this.scan, check: this.check });
      if (this.failed) { result.states.forEach(handle => handle.release()); throw new Error('WebGPU device failed.'); }
      this.states.forEach(handle => handle.release()); this.states = result.states;
      return result.spectrum;
    } catch (error) { this.failed = true; throw error; }
    finally { this.busy = false; }
  }
  async dispose() {
    if (this.busy) throw new Error('Wait for the current frame before disposing.');
    if (this.disposed) return;
    this.disposed = true; this.states.forEach(handle => handle.release()); this.states.clear();
    await releaseRuntime(this.runtime);
  }
  /** Diagnostic only: downloads every cache; excluded from latency measurements. */
  async inspectStates() {
    if (this.busy || this.disposed) throw new Error('Cannot inspect a busy/disposed engine.');
    const result: Record<string, Float32Array> = {};
    for (const [name, handle] of this.states)
      result[name] = new Float32Array(await readBuffer(this.runtime.device, requireGpu(handle), handle.tensor.size * 4));
    return result;
  }
}
