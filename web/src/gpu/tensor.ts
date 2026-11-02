import { Tensor } from 'onnxruntime-web/webgpu';
export interface Handle { tensor: Tensor; release(): void }
export function upload(device: GPUDevice, shape: readonly number[], data?: Float32Array): Handle {
  const count = shape.reduce((a, b) => a * b, 1);
  if (!shape.every(n => Number.isInteger(n) && n > 0) || count > 32_000_000)
    throw new Error('Invalid GPU tensor shape.');
  if (data && (data.length !== count || !data.every(Number.isFinite))) throw new Error('Invalid FP32 tensor.');
  const buffer = device.createBuffer({ size: Math.ceil(count * 4 / 16) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (data) device.queue.writeBuffer(buffer, 0, new Float32Array(data));
  return { tensor: Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: [...shape] }),
    release: () => buffer.destroy() };
}
export function requireGpu(handle: Handle | undefined): GPUBuffer {
  if (!handle || handle.tensor.location !== 'gpu-buffer') throw new Error('Expected GPU-resident tensor.');
  return handle.tensor.gpuBuffer;
}
export async function readBuffer(device: GPUDevice, source: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  try { await staging.mapAsync(GPUMapMode.READ); return staging.getMappedRange().slice(0); }
  finally { staging.destroy(); }
}
