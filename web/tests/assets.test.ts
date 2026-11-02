import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fetchVerified, validateBundle } from '../src/runtime/assets';
import type { Bundle } from '../src/runtime/types';
test('download rejects wrong pin and byte count', async () => {
  const url = 'data:application/octet-stream;base64,AQID';
  await assert.rejects(fetchVerified(url, '0'.repeat(64)), /SHA256 mismatch/);
  await assert.rejects(fetchVerified(url, '0'.repeat(64), 2), /byte count mismatch/);
  await assert.rejects(fetchVerified(url, ''), /trusted SHA256/);
});
test('contract rejects unrelated checkpoints and streaming geometry', () => {
  const base = { schemaVersion: 1, checkpointSha256: '6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade',
    license: 'NSCLv1', researchOnly: true, sampleRate: 16000, fftSize: 640, hopSize: 320,
    layers: 4, states: Array(23).fill({}), steps: [] } as unknown as Bundle;
  assert.equal(validateBundle(base), base);
  for (const patch of [{ sampleRate: 48000 }, { hopSize: 80 }, { checkpointSha256: '0'.repeat(64) },
    { researchOnly: false }, { layers: 2 }, { states: [] }])
    assert.throws(() => validateBundle({ ...base, ...patch }));
});
