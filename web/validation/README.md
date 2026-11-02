# Browser validation — 2026-11-01

Measured on **Apple M4 Pro**, hardware Apple/Metal-3 adapter, Chrome 156,
macOS; four-layer FP32 streaming model. Headless Chrome was launched with
`--enable-unsafe-webgpu`. No software GPU or CPU model fallback was allowed.

The [model-backed receipt](m4-pro-chrome-l4.json) records the exact model bundle
manifest hash and browser identity. Three carried spectral hops matched the
original CPU model: maximum magnitude error 1.53e-7, raw phase error 3.46e-5.
Every element of all 23 first-hop caches was compared: maximum error 1.08e-5.
All caches and spectral outputs are also checked for finite values on every hop.

Six audio-derived hops, through Rust/WASM analysis, WebGPU inference and WASM
reconstruction, produced 1,600 samples. Maximum waveform error was **6.56e-5**
against independent NumPy DSP plus the original CPU ONNX model, below the
existing 3e-4 waveform gate. These are synthetic correctness fixtures, not STOI
or perceptual-quality measurements on speech.

The short warmed-up 50-hop test averaged **10.74 ms**, p95 **10.83 ms**, max 16.61 ms;
none exceeded 20 ms. Timing includes DSP, inference, finite-state checking and
spectral readback. Compilation, initialization and diagnostic cache downloads
are excluded. The first cold spectral hop in this run took 51.5 ms.

The separate [60-second microphone receipt](m4-pro-chrome-live-60s.json) covers
Chrome's synthetic microphone, AudioWorklet, worker queue, recording and cleanup:
960,000 input and 960,000 restored samples, 3,001 model hops, mean **11.4 ms**, 13 hops over 20 ms,
and **4ms of output underrun**. The microphone ended, context closed and worker
terminated; no reported errors. Chrome's fake microphone emits nonspeech test
sound, so a nearly silent enhanced result does not establish speech quality.
There is no deliberately added 20 ms playout buffer in this version.

A [built-app smoke test](m4-pro-chrome-built-live-10s.json) also passed, with
11.3 ms mean processing, zero of 505 hops over 20 ms, first output at 56 ms,
4 ms of output gaps and complete resource cleanup. The worker, WASM, model
assets and separate AudioWorklet JavaScript loaded from the production bundle.

This supports a working local-GPU browser prototype on this Mac. It does not
establish iPhone/Android browser performance, live-network call latency, echo
handling, intelligibility or sustained thermal performance beyond this run.

## Reproduce the model-backed check

After building the DSP and exporting the model per [setup](../README.md), use
the exporter's Python environment from the repository root:

```sh
python tools/web-streaming-export/verify.py \
  --source tools/streaming-export/generated/models/reuse-streaming-l4-fp32.onnx \
  --artifact web/public/model --hops 3
python tools/web-streaming-export/verify_audio.py \
  --source tools/streaming-export/generated/models/reuse-streaming-l4-fp32.onnx \
  --artifact web/public/model --fixtures web/dsp/tests/fixtures
python web/tests/prepare_audio.py web/public/model
cd web
npm run test:browser
```

Playwright uses the installed Chrome channel; no browser download is needed
when Chrome is already installed. The test starts Vite on port 8891 if needed
and writes `test-results/browser-parity.json`.
GPU correctness tests require an actual adapter and intentionally fail if one
is unavailable. Portable CI runs Rust/WASM, exporter, queue/framing and build
checks; it does not claim hosted runners provide this hardware validation.

For the full capture/record/cleanup test, serve the demo with the configured
manifest pin, then run from `web/`:

```sh
REUSE_DEMO_URL=http://127.0.0.1:8891/example/ node example/tests/live-check.mjs 60
```

This uses Chrome's fake media device, not the user's microphone. It checks
paired PCM lengths, finite WAVs, status, resource cleanup and actual output
underruns. It writes its receipt and screenshot under `/tmp/`.
