import initDsp, { StreamingDsp } from '../../dsp/pkg/reuse_streaming_dsp';
import wasmUrl from '../../dsp/pkg/reuse_streaming_dsp_bg.wasm?url';
import { createEngine, type StreamingEngine } from '../runtime/engine';
import { StreamFrames } from '../audio/framing';
export async function createProcessor(url: string, hash: string) {
  await initDsp({ module_or_path: wasmUrl });
  const engine = await createEngine(url, hash), dsp = new StreamingDsp();
  try {
    for (let i = 0; i < 3; i++) dsp.synthesize(await engine.process(dsp.analyze(new Float32Array(640))));
    engine.reset(); dsp.reset(); return new AudioProcessor(engine, dsp);
  } catch (error) { dsp.free(); await engine.dispose(); throw error; }
}
export class AudioProcessor {
  private frames = new StreamFrames();
  private emitted = 0;
  constructor(private engine: StreamingEngine, private dsp: StreamingDsp) {}
  async push(samples: Float32Array) { return this.render(this.frames.push(samples)); }
  async finish() {
    const remaining = this.frames.received - this.emitted;
    const result = await this.render(this.frames.finish());
    result.samples = result.samples.slice(0, remaining);
    return result;
  }
  reset() { this.engine.reset(); this.dsp.reset(); this.frames = new StreamFrames(); this.emitted = 0; }
  async dispose() { await this.engine.dispose(); this.dsp.free(); }
  private async render(frames: Float32Array[]) {
    const audio: number[] = [], frameTimes: number[] = [];
    for (const frame of frames) {
      const start = performance.now();
      const spectrum = await this.engine.process(this.dsp.analyze(frame));
      audio.push(...this.dsp.synthesize(spectrum)); frameTimes.push(performance.now() - start);
    }
    this.emitted += audio.length;
    return { samples: new Float32Array(audio), metrics: { frameTimes,
      frameMs: frameTimes.length ? frameTimes.reduce((a, b) => a + b) / frameTimes.length : 0 } };
  }
}
