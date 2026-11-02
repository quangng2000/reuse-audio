import initDsp, { StreamingDsp } from '../dsp/pkg/reuse_streaming_dsp';
import wasmUrl from '../dsp/pkg/reuse_streaming_dsp_bg.wasm?url';
import type { StreamingEngine } from '../src/runtime/engine';
import { fetchAsset } from '../src/runtime/assets';
import type { Asset, Bundle } from '../src/runtime/types';
export async function validateStates(engine: StreamingEngine, assets: (Asset & { name: string })[]) {
  const manifest: Bundle = await (await fetch('/model/manifest.json')).json();
  const actual = await engine.inspectStates(); let maximum = 0;
  for (const asset of assets) {
    const key = manifest.states.find(state => state.outputName === asset.name)!.inputName;
    const data = await fetchAsset(new URL('/model/manifest.json', location.href).href, asset);
    const expected = new Float32Array(data.buffer);
    for (let i = 0; i < expected.length; i++) {
      const error = Math.abs(actual[key][i] - expected[i]); maximum = Math.max(maximum, error);
      if (!Number.isFinite(error) || error > 0.0003 + 0.0002 * Math.abs(expected[i])) throw new Error(`Cache parity failed: ${key}`);
    }
  }
  return maximum;
}
async function readFloat(path: string) { return new Float32Array(await (await fetch(path)).arrayBuffer()); }
export async function validateAudio(engine: StreamingEngine) {
  await initDsp({ module_or_path: wasmUrl });
  const dsp = new StreamingDsp(), frames = await readFloat('/model/audio-input-frames.f32');
  const expected = await readFloat('/model/audio-expected-output.f32');
  engine.reset(); const audio: number[] = [];
  try {
    for (let i = 0; i < frames.length; i += 640)
      audio.push(...dsp.synthesize(await engine.process(dsp.analyze(frames.slice(i, i + 640)))));
    const maximum = Math.max(...audio.map((value, i) => Math.abs(value - expected[i])));
    if (audio.length !== expected.length || maximum > 0.0003) throw new Error(`Waveform parity failed: ${maximum}`);
    return { maximum, samples: audio.length, timing: await benchmark(engine, dsp, frames.slice(0, 640)) };
  } finally { dsp.free(); }
}
async function benchmark(engine: StreamingEngine, dsp: StreamingDsp, frame: Float32Array) {
  engine.reset(); dsp.reset(); const times = [];
  for (let i = 0; i < 55; i++) {
    const start = performance.now(); dsp.synthesize(await engine.process(dsp.analyze(frame)));
    if (i >= 5) times.push(performance.now() - start);
  }
  const sorted = [...times].sort((a, b) => a - b);
  return { frames: times.length, meanMs: times.reduce((a, b) => a + b) / times.length,
    p95Ms: sorted[Math.ceil(times.length * 0.95) - 1], maxMs: sorted.at(-1),
    over20ms: times.filter(time => time > 20).length };
}
