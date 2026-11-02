import type { Tensor } from 'onnxruntime-web/webgpu';
import { upload, requireGpu, readBuffer, type Handle } from '../gpu/tensor';
import type { Runtime } from './setup';
import type { createScan } from '../gpu/scan';
import type { createFiniteCheck } from '../gpu/finite';
type ScanRunner = Awaited<ReturnType<typeof createScan>>;
type FiniteCheck = Awaited<ReturnType<typeof createFiniteCheck>>;
export async function executeSteps(runtime: Runtime, values: Map<string, Handle>, scan: ScanRunner) {
  for (const step of runtime.bundle.steps) {
    if (step.kind === 'scan') { values.set(step.output, scan(step, values)); continue; }
    const feeds: Record<string, Tensor> = {};
    for (const input of step.inputs) {
      const handle = values.get(input.name);
      requireGpu(handle); feeds[input.name] = handle!.tensor;
    }
    const output = await runtime.sessions.get(step)!.run(feeds);
    for (const [name, tensor] of Object.entries(output)) values.set(name, { tensor, release: () => tensor.dispose() });
  }
}
export async function executeFrame(args: {
  runtime: Runtime; states: Map<string, Handle>; spectrum: Float32Array; scan: ScanRunner; check: FiniteCheck;
}) {
  const { runtime, states, spectrum, scan, check } = args;
  if (spectrum.length !== 642 || !spectrum.every(Number.isFinite)) throw new Error('Expected 642 finite spectral values.');
  const values = new Map([...runtime.constants, ...states]);
  values.set('magnitude', upload(runtime.device, [1, 321, 1], spectrum.slice(0, 321)));
  values.set('phase', upload(runtime.device, [1, 321, 1], spectrum.slice(321)));
  let retained = new Set<Handle>();
  try {
    await executeSteps(runtime, values, scan);
    const result = await collectResult(runtime, values, check);
    retained = new Set(result.states.values()); return result;
  } finally {
    const existing = new Set([...runtime.constants.values(), ...states.values(), ...retained]);
    for (const handle of new Set(values.values())) if (!existing.has(handle)) handle.release();
  }
}
async function collectResult(runtime: Runtime, values: Map<string, Handle>, check: FiniteCheck) {
  for (const alias of runtime.bundle.aliases ?? []) {
    const source = values.get(alias.source);
    if (!source) throw new Error(`Missing aliased model output: ${alias.source}`);
    values.set(alias.name, source);
  }
  const states = new Map(runtime.bundle.states.map(state => [state.inputName, values.get(state.outputName)!]));
  const outputs = ['enhanced_magnitude', 'enhanced_phase'].map(name => values.get(name)!);
  await check([...states.values(), ...outputs]);
  const spectrum = new Float32Array(642);
  for (const [index, handle] of outputs.entries())
    spectrum.set(new Float32Array(await readBuffer(runtime.device, requireGpu(handle), 321 * 4)), index * 321);
  return { states, spectrum };
}
