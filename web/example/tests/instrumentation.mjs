/** Test-only lifecycle instrumentation. Never reads the real microphone. */
export function installAudit() {
  window.__liveAudit = { streams: [], contexts: [], workers: [] };
  const capture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (...args) => {
    const stream = await capture(...args); window.__liveAudit.streams.push(stream); return stream;
  };
  const Context = window.AudioContext;
  window.AudioContext = class extends Context {
    constructor(...args) { super(...args); window.__liveAudit.contexts.push(this); }
  };
  const WorkerClass = window.Worker;
  window.Worker = class extends WorkerClass {
    constructor(...args) { super(...args); this.wasTerminated = false; window.__liveAudit.workers.push(this); }
    terminate() { this.wasTerminated = true; super.terminate(); }
  };
}

/** Runs inside the browser: captures WAV headers and explicit resource states. */
export async function collectReceipt() {
  const recordings = await Promise.all([...document.querySelectorAll('audio')].map(async (node) => {
    const data = await (await fetch(node.src)).arrayBuffer(), view = new DataView(data);
    let peak = 0, squared = 0;
    for (let offset = 44; offset < data.byteLength; offset += 2) {
      const sample = view.getInt16(offset, true) / 32768; peak = Math.max(peak, Math.abs(sample)); squared += sample * sample;
    }
    const samples = (data.byteLength - 44) / 2;
    return { label: node.getAttribute('aria-label'), samples, sampleRate: view.getUint32(24, true), peak, rms: Math.sqrt(squared / samples) };
  }));
  return { status: document.querySelector('.status').textContent,
    warnings: [...document.querySelectorAll('.warning')].map(x => x.textContent),
    metrics: [...document.querySelectorAll('.metrics>div')].map(x => x.innerText), recordings,
    details: [...document.querySelectorAll('.fineprint')].map(x => x.textContent),
    trackStates: window.__liveAudit.streams.flatMap(s => s.getTracks().map(t => t.readyState)),
    contextStates: window.__liveAudit.contexts.map(c => c.state),
    workersTerminated: window.__liveAudit.workers.map(w => w.wasTerminated) };
}
