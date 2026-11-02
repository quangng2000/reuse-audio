import { defineConfig } from 'vite';
export default defineConfig({
  root: '.',
  worker: { format: 'es' },
  build: { outDir: 'dist', rollupOptions: { input: 'example/index.html' } },
  server: { hmr: false, headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' } },
  optimizeDeps: { include: ['onnxruntime-web/webgpu', 'react', 'react-dom/client'] },
});
