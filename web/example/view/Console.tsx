import type { Demo } from '../controller';
import { Metrics, FrameDetail } from './Metrics';

function Controls({ demo }: { demo: Demo }) {
  return <div className="actions"><button
    disabled={!demo.configured || ['loading', 'live', 'stopping'].includes(demo.mode)} onClick={demo.start}>
    {demo.mode === 'loading' ? 'Loading model…' : 'Start microphone test'}</button>
    <button className="secondary" disabled={demo.mode !== 'live'} onClick={demo.stop}>Stop & listen</button></div>;
}

export function Console({ demo }: { demo: Demo }) {
  return <section className="console" aria-label="Live restoration test">
    <div className="console-heading"><h2>Microphone session</h2>
      <span className={`status ${demo.mode === 'live' ? 'active' : ''}`} role="status">{demo.mode}</span></div>
    <p>No audio uploads. No speaker monitoring while recording.</p>
    {!demo.configured && <p className="warning">Export the streaming browser bundle, then set VITE_MODEL_SHA256
      to its trusted manifest hash before starting.</p>}
    {demo.error && <p className="warning" role="alert">{demo.error}</p>}
    <Controls demo={demo} /><Metrics value={demo.metrics} />
    <FrameDetail value={demo.metrics} />
    <p className="fineprint">A frame represents 20 ms of audio. These processing times exclude microphone,
      playback, network and call buffering. Output gaps count underruns after the first restored sample.
      If inference falls behind, the test stops.</p>
  </section>;
}
