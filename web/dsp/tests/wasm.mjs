import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { initSync, StreamingDsp } from '../pkg/reuse_streaming_dsp.js';

initSync({ module: await readFile(new URL('../pkg/reuse_streaming_dsp_bg.wasm', import.meta.url)) });
const dsp = new StreamingDsp();
const input = Float32Array.from({ length: 1600 }, (_, index) =>
  0.1 * Math.sin(index * 0.071) + 0.013 * Math.sin(index * 1.417));
const prefix = input.slice(1, 321).reverse();
const suffix = input.slice(0, -1).reverse();
const padded = Float32Array.from([...prefix, ...input, ...suffix]);
const output = [];
for (let hop = 0; hop < 6; hop++) {
  const spectrum = dsp.analyze(padded.slice(hop * 320, hop * 320 + 640));
  const chunk = dsp.synthesize(spectrum);
  assert.equal(chunk.length, hop === 0 ? 0 : 320);
  output.push(...chunk);
}
const error = Math.max(...output.map((value, index) => Math.abs(value - input[index])));
assert.ok(error < 1e-6, `WASM round-trip maximum error ${error}`);
assert.throws(() => dsp.analyze(new Float32Array(639)));
assert.throws(() => dsp.synthesize(new Float32Array(642).fill(NaN)));
dsp.reset();
assert.equal(dsp.synthesize(new Float32Array(642)).length, 0);
dsp.free();
console.log(JSON.stringify({ wasmRoundTripMaximumError: error, samples: output.length, passed: true }));
