import type { LiveMetrics } from '../../src/live';

export function Metrics({ value }: { value?: LiveMetrics }) {
  const duration = (ms?: number) => ms === undefined ? '—' : `${ms.toFixed(1)} ms`;
  return <dl className="metrics">
    <div><dt>Recorded</dt><dd>{value ? `${value.elapsedSeconds.toFixed(1)} s` : '—'}</dd></div>
    <div><dt>Mean frame</dt><dd>{duration(value?.meanFrameMs)}</dd></div>
    <div><dt>Output gaps</dt><dd>{duration(value?.playbackUnderrunMs)}</dd></div>
    <div><dt>Over 20 ms</dt><dd>{value ? `${value.deadlineMisses} / ${value.frames}` : '—'}</dd></div>
  </dl>;
}

export function FrameDetail({ value }: { value?: LiveMetrics }) {
  if (value?.startupMs === undefined) return null;
  return <p className="fineprint">First restored output: {value.startupMs.toFixed(0)} ms after capture started.
    Slowest processing frame: {value.maxFrameMs.toFixed(1)} ms.</p>;
}
