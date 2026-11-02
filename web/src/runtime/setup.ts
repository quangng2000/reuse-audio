import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import moduleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url';
import { fetchAsset, loadBundle } from './assets';
import { upload, type Handle } from '../gpu/tensor';
import type { Segment } from './types';
let configured = false;
async function configureGpu() {
  if (configured) throw new Error('Use a new worker for a new engine; ONNX Runtime owns one device per worker.');
  configured = true;
  if (!globalThis.isSecureContext || !navigator.gpu) throw new Error('WebGPU requires a supported browser on HTTPS or localhost.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('No hardware WebGPU adapter is available.');
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: moduleUrl }; ort.env.webgpu.adapter = adapter;
}
async function createSegment(base: string, step: Segment) {
  return ort.InferenceSession.create(await fetchAsset(base, step), {
    executionProviders: ['webgpu'], preferredOutputLocation: 'gpu-buffer',
    graphOptimizationLevel: 'basic',
    logSeverityLevel: 2,
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  });
}
export async function prepareRuntime(url: string, hash: string) {
  const bundle = await loadBundle(url, hash);
  await configureGpu();
  const constants = new Map<string, Handle>(), sessions = new Map<Segment, ort.InferenceSession>();
  let device: GPUDevice | undefined;
  try {
    for (const step of bundle.steps) if (step.kind === 'onnx') sessions.set(step, await createSegment(url, step));
    device = await ort.env.webgpu.device;
    for (const constant of bundle.constants) {
      const data = await fetchAsset(url, constant);
      constants.set(constant.name, upload(device, constant.shape, new Float32Array(data.buffer)));
    }
    return { bundle, device, constants, sessions };
  } catch (error) {
    constants.forEach(handle => handle.release());
    for (const session of sessions.values()) await session.release();
    device?.destroy(); throw error;
  }
}
export type Runtime = Awaited<ReturnType<typeof prepareRuntime>>;
