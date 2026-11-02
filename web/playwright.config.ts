import { defineConfig } from '@playwright/test';
export default defineConfig({ testDir: './tests/browser', timeout: 180_000,
  use: { channel: 'chrome', headless: true, baseURL: 'http://127.0.0.1:8891',
    launchOptions: { args: ['--enable-unsafe-webgpu'] } },
  webServer: { command: 'npm run dev -- --port 8891', url: 'http://127.0.0.1:8891/tests/harness.html', reuseExistingServer: true },
});
