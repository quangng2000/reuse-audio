import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { initSync, StreamingDsp } from '../pkg/reuse_streaming_dsp.js';

const fixture = new URL('./fixtures/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', fixture), 'utf8'));
async function load(name) {
  const bytes = await readFile(new URL(name, fixture));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), manifest.files[name].sha256);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
initSync({ module: await readFile(new URL('../pkg/reuse_streaming_dsp_bg.wasm', import.meta.url)) });
const [frames, spectra, expected] = await Promise.all(['frames.f32', 'spectra.f32', 'output.f32'].map(load));
const dsp = new StreamingDsp();
const output = [];
let magnitudeError = 0, phaseError = 0;
for (let index = 0; index < manifest.frames; index++) {
  const spectrum = dsp.analyze(frames.subarray(index * 640, index * 640 + 640));
  const target = spectra.subarray(index * 642, index * 642 + 642);
  for (let bin = 0; bin < 321; bin++) {
    magnitudeError = Math.max(magnitudeError, Math.abs(spectrum[bin] - target[bin]));
    const delta = spectrum[bin + 321] - target[bin + 321];
    phaseError = Math.max(phaseError, Math.abs(Math.atan2(Math.sin(delta), Math.cos(delta))));
  }
  output.push(...dsp.synthesize(target));
}
assert.equal(output.length, expected.length);
const waveformError = Math.max(...output.map((value, index) => Math.abs(value - expected[index])));
assert.ok(magnitudeError < 1e-5 && phaseError < 0.003 && waveformError < 1e-6);
dsp.free();
console.log(JSON.stringify({ magnitudeError, phaseError, waveformError, source: manifest.source, passed: true }));
