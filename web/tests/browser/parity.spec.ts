import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
test('WebGPU preserves carried-hop reference spectra', async ({ page }) => {
  const hash = createHash('sha256').update(readFileSync('public/model/manifest.json')).digest('hex');
  page.on('console', msg => { if (msg.type() === 'error') console.error(msg.text()); });
  page.on('pageerror', error => console.error(error.message));
  await page.goto('/tests/harness.html');
  await page.waitForFunction(() => 'runValidation' in window);
  const result = await page.evaluate(async pin => {
    const run = (window as unknown as {runValidation(hash: string): Promise<{metrics: {ms: number; magnitude: number; phase: number}[]}>}).runValidation;
    return run(pin);
  }, hash);
  console.log(JSON.stringify(result));
  writeFileSync('test-results/browser-parity.json', JSON.stringify({ manifestSha256: hash, ...result }, null, 2));
  for (const hop of result.metrics) { expect(hop.magnitude).toBeLessThan(0.003); expect(hop.phase).toBeLessThan(0.003); }
});
