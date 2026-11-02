export function Introduction() {
  return <section className="intro"><p className="eyebrow">Your microphone. Your device.</p>
    <h1>Hear the difference.<br /><span>Measure the delay.</span></h1>
    <p>Test streaming speech restoration with local WebGPU inference.
      Record up to one minute, then compare the original and restored audio.</p></section>;
}

export function ResearchNotice() {
  return <footer><strong>Noncommercial research only · NVIDIA NSCLv1</strong>
    <p>This uses the separate Real-time RE-USE checkpoint. Browser latency and audio parity must be validated
      on each device before considering live-call use. It is not the commercial offline model.</p></footer>;
}
