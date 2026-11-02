import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

function setup() {
  const messages: { type: string; samples?: Float32Array; underrunSamples?: number; startupMs?: number }[] = [];
  let Processor: any;
  class Base { port = { postMessage: (value: any) => messages.push(value), onmessage: undefined }; }
  const source = readFileSync(new URL('../capture-worklet.js', import.meta.url), 'utf8');
  vm.runInNewContext(source, { AudioWorkletProcessor: Base,
    registerProcessor: (_: string, value: any) => { Processor = value; }, Float32Array });
  return { processor: new Processor(), messages };
}

test('worklet captures exactly320 samples and flushes only actual tail on stop', () => {
  const { processor, messages } = setup();
  const output = new Float32Array(128);
  for (let index = 0; index < 3; index++) processor.process([[new Float32Array(128).fill(0.25)]], [[output]]);
  assert.equal(messages.length, 1);
  assert.equal(messages[0].samples?.length, 320);
  assert.ok(messages[0].samples?.every((sample) => sample === 0.25));
  assert.ok(output.every((sample) => sample === 0));
  processor.receive({ type: 'stop' });
  assert.equal(messages[1].samples?.length, 64);
  assert.equal(messages[2].type, 'stopped');
  processor.process([[new Float32Array(128)]], [[output]]);
  assert.equal(messages.length, 3);
});

test('worklet preserves restored samples across128 sample output quanta', () => {
  const { processor } = setup();
  const expected = Float32Array.from({ length: 320 }, (_, index) => index / 320);
  processor.receive({ type: 'audio', samples: expected });
  const received: number[] = [];
  for (let index = 0; index < 3; index++) {
    const output = new Float32Array(128);
    processor.process([[]], [[output]]);
    received.push(...output);
  }
  assert.deepEqual(received.slice(0, 320), [...expected]);
  assert.ok(received.slice(320).every((sample) => sample === 0));
});

test('reports output gaps after playback starts and excludes initial buffering', () => {
  const { processor, messages } = setup();
  const render = () => processor.process([[]], [[new Float32Array(128)]]);
  for (let index = 0; index < 25; index++) render();
  assert.deepEqual({ ...messages.at(-1) }, { type: 'playback', underrunSamples: 0, startupMs: undefined });
  processor.receive({ type: 'audio', samples: new Float32Array(320) });
  for (let index = 0; index < 25; index++) render();
  const report = messages.at(-1)!;
  assert.equal(report.startupMs, 200);
  assert.equal(report.underrunSamples, 2880);
});
