# Browser streaming research preview

Live microphone enhancement for React, running locally on the browser's GPU.
Rust/WebAssembly implements the audio transforms; TypeScript orchestrates
WebGPU inference and custom WGSL Mamba scans. Audio is never uploaded.

**Experimental, NSCLv1 noncommercial research only.** This uses the separate
four-layer Real-time RE-USE checkpoint, not the commercial offline 30-layer
model. It is not an npm release or a production calling service. Keep the
[NVIDIA license](dsp/LICENSE), [notice](dsp/NOTICE) and
[Mamba attribution](../tools/streaming-export/MAMBA-LICENSE.txt).

## Run the React demo

Requirements: Node 24, Rust with `wasm32-unknown-unknown`, wasm-bindgen CLI
0.2.105, Python with the exporter's pinned dependencies, and a hardware WebGPU
browser. Use HTTPS or localhost. Chrome on Apple M4 Pro is the measured target;
support is capability-checked, not inferred from the browser name.

1. Produce the pinned L4 ONNX model with the
   [streaming exporter](../tools/streaming-export/README.md).
2. Install [browser exporter dependencies](../tools/web-streaming-export/requirements.txt)
   in an isolated environment, then run from the repository root:

```sh
python tools/web-streaming-export/export_web.py \
  --source tools/streaming-export/generated/models/reuse-streaming-l4-fp32.onnx \
  --output web/public/model
```

3. Build the [Rust/WASM DSP](dsp/README.md). The generated files belong in
   `web/dsp/pkg/`; they and model assets are excluded from Git.
4. Copy `.env.example` to `.env.local`, setting `VITE_MODEL_SHA256` to the SHA256
   of your generated `public/model/manifest.json`. Pin this value independently
   when deploying; never fetch the trusted hash from the same untrusted URL.
5. From `web/`, run `npm ci`, then `npm run dev`. Open `/example/` on the displayed URL.

Click **Start microphone**. Initialization warms the shaders before requesting
the microphone. Speak for up to 60 seconds, then stop and compare the two WAVs.
Playback uses one shared peak-safe gain. Speakers are not monitored during
capture, preventing acoustic feedback. Start uses a new worker and fresh caches.

## Integrate with a React/WebRTC prototype

The Vite example imports the source API below. Keep the worker, AudioWorklet,
WASM and model assets in the build; do not execute inference on React's thread.

```ts
import { LiveSession } from './src/live';

const session = await LiveSession.start({
  manifestUrl: '/model/manifest.json',
  manifestSha256: TRUSTED_MANIFEST_SHA256,
  onMetrics: metrics => console.log(metrics),
  onError: error => console.error(error),
});
await sender.replaceTrack(session.stream.getAudioTracks()[0]);
// When finished, including React component cleanup:
const recordings = await session.stop();
```

`stream` is a processed `MediaStream`, ready for a WebRTC audio sender.
This demo does not implement signaling, echo cancellation, Opus transport or a
call service. Browser AGC/noise suppression/echo cancellation are disabled to
isolate model performance. A real call needs a separately tested echo pipeline.

## Contract and failure behavior

- Mono 16 kHz; FFT 640, hop 320 (20 ms), centered startup reflection, exact EOF
  trimming. Startup waits for 640 real samples; compute time is not call latency.
- Original FP32 weights; nine ONNX GPU segments plus eight custom WGSL scans.
  Persistent temporal states stay on GPU. CPU inference fallback is disabled.
- SHA256/size-verified assets; finite input, spectrum and every next-state checks
  before committing an update. Reset clears neural and DSP state together.
- Only three microphone chunks may be pending. Overload stops with an error;
  it does not silently drop recurrent frames or accumulate unbounded latency.
- AudioWorklet reports playback underruns separately from inference deadlines.
  Stop/error releases microphone tracks, the audio context and inference worker.

ONNX Runtime Web is pinned to **1.22.0's JSEP WebGPU path**. The exporter uses
CPU ORT 1.23.2 as an independent reference. Browser-specific rewrites materialize
convolution defaults and replace reverse Slice with explicit Gather indices.
Do not upgrade the runtime without repeating GPU/state/waveform checks.

## Verification

`npm test` checks framing, queue bounds, recording and playback behavior.
`npm run build` type-checks and bundles the demo. See
[validation evidence and reproduction](validation/README.md) for model-backed
browser checks and measured timing. A fast Mac result does not establish mobile
browser latency, sustained thermal behavior, perceptual quality or call quality.
