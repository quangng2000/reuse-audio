// Run after npm run dev -- --port 8891; always uses Chrome's synthetic microphone.
import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { collectReceipt, installAudit } from './instrumentation.mjs';

const seconds = Number(process.argv[2] ?? 10);
assert.ok(seconds === 10 || seconds === 60, 'Choose 10 (manual stop) or 60 (automatic stop).');
const base = process.env.REUSE_DEMO_URL ?? 'http://127.0.0.1:8891/example/';
const prefix = process.env.REUSE_RECEIPT_PREFIX ?? `/tmp/reuse-web-live-${seconds}s`;

async function capture(page) {
  await page.goto(base);
  await page.getByRole('button', { name: 'Start microphone test' }).click();
  await page.locator('.status').filter({ hasText: /^(live|error)$/ }).waitFor({ timeout: 120000 });
  if ((await page.locator('.status').innerText()).toLowerCase() === 'live' && seconds === 10) {
    await page.waitForTimeout(10000);
    if (await page.getByRole('button', { name: 'Stop & listen' }).isEnabled())
      await page.getByRole('button', { name: 'Stop & listen' }).click();
  }
  await page.locator('.status').filter({ hasText: /^(complete|error)$/ }).waitFor({ timeout: seconds * 1000 + 20000 });
}

function validate(receipt) {
  assert.equal(receipt.status, 'complete'); assert.deepEqual(receipt.errors, []); assert.deepEqual(receipt.warnings, []);
  assert.equal(receipt.recordings.length, 2);
  const [input, output] = receipt.recordings; assert.equal(input.samples, output.samples);
  assert.ok(input.samples >= seconds * 16000);
  if (seconds === 60) assert.equal(input.samples, 960000);
  for (const recording of receipt.recordings) {
    assert.equal(recording.sampleRate, 16000); assert.ok(recording.peak <= 1 && Number.isFinite(recording.rms));
  }
  assert.deepEqual(receipt.trackStates, ['ended']); assert.deepEqual(receipt.contextStates, ['closed']);
  assert.deepEqual(receipt.workersTerminated, [true]);
}

async function run(browser) {
  const context = await browser.newContext({ permissions: ['microphone'], viewport: { width: 1280, height: 1100 } });
  const page = await context.newPage(), errors = [];
  page.on('pageerror', error => errors.push(error.message)); await page.addInitScript(installAudit);
  await capture(page);
  const receipt = { timestamp: new Date().toISOString(), test: `Chrome synthetic microphone ${seconds}s`,
    errors, ...await page.evaluate(collectReceipt) };
  writeFileSync(`${prefix}.json`, JSON.stringify(receipt, null, 2) + '\n');
  await page.screenshot({ path: `${prefix}.png`, fullPage: true });
  console.log(JSON.stringify(receipt)); validate(receipt);
}

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: [
  '--enable-unsafe-webgpu', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
] });
try { await run(browser); } finally { await browser.close(); }
