import assert from 'node:assert/strict';
import test from 'node:test';
import { BoundedQueue } from '../queue';

test('preserves order and refuses unbounded audio backlog', async () => {
  const queue = new BoundedQueue(2);
  let release!: () => void;
  const wait = new Promise<void>((resolve) => { release = resolve; });
  const order: number[] = [];
  const first = queue.push(async () => { await wait; order.push(1); });
  const second = queue.push(async () => { order.push(2); });
  await assert.rejects(queue.push(async () => undefined), /falling behind/);
  release();
  await Promise.all([first, second]);
  await queue.finish();
  assert.deepEqual(order, [1, 2]);
  assert.equal(queue.pending, 0);
  await assert.rejects(queue.push(async () => undefined), /closed/);
});

test('failed recurrent update prevents queued updates from running', async () => {
  const queue = new BoundedQueue(2);
  let secondRan = false;
  const first = queue.push(async () => { throw new Error('Device lost'); });
  const second = queue.push(async () => { secondRan = true; });
  await assert.rejects(first, /Device lost/);
  await assert.rejects(second, /Device lost/);
  await queue.finish();
  assert.equal(secondRan, false);
  assert.equal(queue.pending, 0);
});
