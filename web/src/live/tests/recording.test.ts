import assert from 'node:assert/strict';
import test from 'node:test';
import { Recordings } from '../recording';

test('recordings share one peak-safe gain without independently normalizing speech', async () => {
  const recording = new Recordings();
  recording.add('input', new Float32Array([0, 0.5, -0.5]));
  recording.add('restored', new Float32Array([0, 2, -2]));
  const result = recording.finish();
  const input = new DataView(await result.input.arrayBuffer());
  const restored = new DataView(await result.restored.arrayBuffer());
  assert.equal(result.playbackGain, 0.495);
  assert.equal(input.getUint32(24, true), 16000);
  assert.equal(input.getUint32(40, true), 6);
  assert.equal(restored.getInt16(46, true), Math.round(0.99 * 32767));
  assert.equal(input.getInt16(46, true), Math.round(0.2475 * 32767));
});

test('rejects nonfinite audio before constructing a downloadable recording', () => {
  const recording = new Recordings();
  assert.throws(() => recording.add('input', new Float32Array([NaN])), /nonfinite/);
  assert.equal(recording.inputSamples, 0);
});
