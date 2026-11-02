import { useEffect, useRef, useState, type RefObject } from 'react';
import { LiveSession, type LiveMetrics, type LiveRecordings } from '../src/live';

type Mode = 'idle' | 'loading' | 'live' | 'stopping' | 'complete' | 'error';
const manifestUrl = import.meta.env.VITE_MODEL_URL ?? '/model/manifest.json';
const manifestSha256 = import.meta.env.VITE_MODEL_SHA256 ?? '';
type Setters = {
  mode: React.Dispatch<React.SetStateAction<Mode>>;
  metrics: React.Dispatch<React.SetStateAction<LiveMetrics | undefined>>;
  result: React.Dispatch<React.SetStateAction<LiveRecordings | undefined>>;
  error: (value: string) => void;
};

async function startCapture(session: RefObject<LiveSession | undefined>, set: Setters) {
  set.mode('loading'); set.error(''); set.metrics(undefined); set.result(undefined);
  try {
    session.current = await LiveSession.start({ manifestUrl, manifestSha256,
      onMetrics: set.metrics,
      onError: (failure) => { set.error(failure.message); set.mode('error'); },
      onStopped: (audio) => { set.result(audio); set.mode((old) => old === 'error' ? old : 'complete'); },
    });
    set.mode('live');
  } catch (failure) { set.error((failure as Error).message); set.mode('error'); }
}

async function stopCapture(session: RefObject<LiveSession | undefined>, set: Setters) {
  set.mode('stopping');
  await session.current?.stop();
  session.current = undefined;
}

export function useLiveDemo() {
  const session = useRef<LiveSession | undefined>(undefined);
  const [mode, setMode] = useState<Mode>('idle');
  const [metrics, setMetrics] = useState<LiveMetrics>();
  const [result, setResult] = useState<LiveRecordings>();
  const [error, setError] = useState('');
  useEffect(() => () => { void session.current?.stop(); }, []);
  const set = { mode: setMode, metrics: setMetrics, result: setResult, error: setError };
  return { mode, metrics, result, error, configured: /^[a-f0-9]{64}$/i.test(manifestSha256),
    start: () => { void startCapture(session, set); }, stop: () => { void stopCapture(session, set); } };
}
export type Demo = ReturnType<typeof useLiveDemo>;
