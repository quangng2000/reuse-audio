import { useEffect, useState } from 'react';
import type { LiveRecordings } from '../../src/live';

export function Playback({ result }: { result?: LiveRecordings }) {
  const [urls, setUrls] = useState<string[]>([]);
  useEffect(() => {
    if (!result) { setUrls([]); return; }
    const next = [URL.createObjectURL(result.input), URL.createObjectURL(result.restored)];
    setUrls(next);
    return () => next.forEach(URL.revokeObjectURL);
  }, [result]);
  if (!result || urls.length !== 2) return null;
  return <section className="playback" aria-label="Recorded comparison">
    <h2>Listen back</h2><p>The same playback gain is applied to both recordings.</p>
    {['Original microphone', 'RE-USE restored'].map((name, i) => <article key={name}>
      <div><h3>{name}</h3><a href={urls[i]} download={i ? 'reuse-restored.wav' : 'microphone.wav'}>Save WAV</a></div>
      <audio controls src={urls[i]} aria-label={name} />
    </article>)}
    {result.inputSamples !== result.restoredSamples && <p className="warning">
      Restoration ended early: {(result.restoredSamples / 16000).toFixed(2)} of{' '}
      {(result.inputSamples / 16000).toFixed(2)} seconds were produced.</p>}
  </section>;
}
