import type { Asset, Bundle } from './types';
export async function fetchVerified(url: string, hash: string, bytes?: number): Promise<Uint8Array> {
  if (!/^[a-f0-9]{64}$/.test(hash)) throw new Error('A trusted SHA256 pin is required.');
  const response = await fetch(url, { credentials: 'omit' });
  if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
  const data = await response.arrayBuffer();
  if (bytes !== undefined && data.byteLength !== bytes) throw new Error('Model byte count mismatch.');
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', data));
  const actual = [...digest].map(value => value.toString(16).padStart(2, '0')).join('');
  if (actual !== hash) throw new Error('Model SHA256 mismatch.');
  return new Uint8Array(data);
}
export function validateBundle(bundle: Bundle): Bundle {
  const identity = '6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade';
  if (bundle.schemaVersion !== 1 || bundle.checkpointSha256 !== identity ||
      bundle.license !== 'NSCLv1' || bundle.researchOnly !== true ||
      bundle.sampleRate !== 16000 || bundle.fftSize !== 640 || bundle.hopSize !== 320)
    throw new Error('Unsupported streaming model contract.');
  if (!Number.isInteger(bundle.layers) || bundle.layers < 3 || bundle.layers > 12 ||
      bundle.states.length !== 15 + 2 * bundle.layers) throw new Error('Invalid streaming depth/state count.');
  for (const step of bundle.steps) if (step.kind === 'scan' &&
      (step.bins !== 161 || step.channels !== 256 || step.components !== 16))
    throw new Error('Unsupported selective scan shape.');
  return bundle;
}
export async function loadBundle(url: string, hash: string): Promise<Bundle> {
  return validateBundle(JSON.parse(new TextDecoder().decode(await fetchVerified(url, hash))));
}
export function fetchAsset(base: string, asset: Asset): Promise<Uint8Array> {
  return fetchVerified(new URL(asset.file, base).href, asset.sha256, asset.bytes);
}
