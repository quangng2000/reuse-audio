import { createEngine } from '../src/runtime/engine';
import { fetchAsset } from '../src/runtime/assets';
import type { Asset } from '../src/runtime/types';
import { validateAudio, validateStates } from './validation';
type Fixture = Record<'magnitude'|'phase'|'enhanced_magnitude'|'enhanced_phase', Asset>;
async function loadSpectrum(base: string, fixture: Fixture, enhanced = false) {
  const keys = enhanced ? ['enhanced_magnitude', 'enhanced_phase'] : ['magnitude', 'phase'];
  const spectrum = new Float32Array(642);
  for (const [i, key] of keys.entries()) {
    const bytes = await fetchAsset(base, fixture[key as keyof Fixture]);
    spectrum.set(new Float32Array(bytes.buffer), 321 * i);
  }
  return spectrum;
}
async function runValidation(hash: string) {
  const url = new URL('/model/manifest.json', location.href).href;
  const fixture = await (await fetch('/model/parity.json')).json();
  const engine = await createEngine(url, hash), metrics = []; let stateError = 0;
  try {
    for (const hop of fixture.fixtures) {
      const input = await loadSpectrum(url, hop), expected = await loadSpectrum(url, hop, true);
      const start = performance.now(), output = await engine.process(input);
      const error = output.map((value, i) => Math.abs(value - expected[i]));
      metrics.push({ ms: performance.now() - start, magnitude: Math.max(...error.slice(0, 321)), phase: Math.max(...error.slice(321)) });
      if (metrics.length === 1) stateError = await validateStates(engine, fixture.stateFixtures.outputs);
    }
    const audio = await validateAudio(engine);
    const adapter = await navigator.gpu.requestAdapter();
    return { metrics, stateError, audio, browser: navigator.userAgent,
      gpu: adapter ? {vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description} : null };
  } finally { await engine.dispose(); }
}
Object.assign(window, { runValidation });
