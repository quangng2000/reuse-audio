import { test } from 'node:test';
import assert from 'node:assert/strict';
import { StreamFrames } from '../src/audio/framing';
test('initial centered frame uses reflection without repeating endpoint', () => {
  const frames = new StreamFrames(), input = Float32Array.from({ length: 640 }, (_, i) => i);
  assert.equal(frames.push(input.slice(0, 320)).length, 0);
  const result = frames.push(input.slice(320));
  assert.equal(result.length, 2);
  assert.deepEqual(result[0].slice(0, 4), new Float32Array([320, 319, 318, 317]));
  assert.deepEqual(result[1], input);
});
test('all chunk boundaries preserve centered frames and padded final hop count', () => {
  for (const length of [640, 641, 959, 960, 1377, 8000, 16000]) {
    for (const size of [1, 127, 320]) {
      const stream = new StreamFrames(), input = Float32Array.from({ length }, (_, i) => i);
      const output: Float32Array[] = [];
      for (let start = 0; start < length; start += size) output.push(...stream.push(input.slice(start, start + size)));
      output.push(...stream.finish());
      assert.equal(output.length, 1 + Math.ceil(length / 320));
      assert.ok(output.every(frame => frame.length === 640));
      assert.throws(() => stream.push(new Float32Array()));
    }
  }
});
test('rejects invalid audio and premature finish', () => {
  const stream = new StreamFrames();
  assert.throws(() => stream.push(new Float32Array([NaN])));
  assert.throws(() => stream.push(new Float32Array(321)));
  assert.throws(() => stream.finish());
});
