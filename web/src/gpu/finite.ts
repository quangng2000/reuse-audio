import { readBuffer, requireGpu, type Handle } from './tensor';
const source = `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> failed: atomic<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&values) && (bitcast<u32>(values[id.x]) & 0x7f800000u) == 0x7f800000u) {
    atomicStore(&failed, 1u);
  }
}`;
export async function createFiniteCheck(device: GPUDevice) {
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: device.createShaderModule({ code: source }), entryPoint: 'main' } });
  return async (handles: Handle[]) => {
    const flag = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    for (const handle of handles) {
      const pass = encoder.beginComputePass();
      const entries = [requireGpu(handle), flag].map((buffer, binding) => ({ binding,
        resource: { buffer, size: binding === 0 ? handle.tensor.size * 4 : 4 } }));
      pass.setPipeline(pipeline); pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(Math.ceil(handle.tensor.size / 256)); pass.end();
    }
    device.queue.submit([encoder.finish()]);
    try { if (new Uint32Array(await readBuffer(device, flag, 4))[0]) throw new Error('Nonfinite model output/state; reset required.'); }
    finally { flag.destroy(); }
  };
}
