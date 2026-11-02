import source from './scan.wgsl?raw';
import { requireGpu, upload, type Handle } from './tensor';
import type { Scan } from '../runtime/types';
export async function createScan(device: GPUDevice) {
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: device.createShaderModule({ code: source }), entryPoint: 'main' } });
  return (scan: Scan, tensors: Map<string, Handle>): Handle => {
    const output = upload(device, [1, 161, 256]);
    const names = ['u', 'dt', 'b', 'c', 'z', 'a', 'd'] as const;
    const buffers = [...names.map(name => requireGpu(tensors.get(scan.inputs[name]))), requireGpu(output)];
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(4); pass.end();
    device.queue.submit([encoder.finish()]);
    return output;
  };
}
