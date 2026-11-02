# Streaming DSP for the browser

Research-only RE-USE adaptation; NVIDIA NSCLv1 applies (see `LICENSE` and
`NOTICE`). This crate contains no learned weights. It implements CPU audio
transforms in Rust/WebAssembly; the neural network uses the separate WebGPU
runtime. WebAssembly is not GPU acceleration.

From this directory:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.105 --locked
cargo test --locked
cargo build --release --locked --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/reuse_streaming_dsp.wasm \
  --target web --out-dir pkg
node tests/wasm.mjs
node tests/wasm-reference.mjs
```

Use the generated JS/TypeScript wrapper from a dedicated inference worker:

```ts
import init, { StreamingDsp } from './pkg/reuse_streaming_dsp.js';
await init();
const dsp = new StreamingDsp();
const spectrum = dsp.analyze(frame); // exactly 640 mono samples at 16 kHz
const restoredSpectrum = await model.process(spectrum);
const samples = dsp.synthesize(restoredSpectrum);
// First call emits nothing; later calls emit 320 samples (20 ms).
```

The spectral array is 321 FP32 `log1p(abs(FFT))` values followed by 321 phases
in radians. Synthesis retains the model's ReLU/expm1, majority-zero sweep
suppression, inverse FFT normalization and weighted overlap-add. The periodic
Hann window is byte-identical to the sealed Android/PyTorch FP32 window
(`ca961f55e3d74499159074e0401c180a88f41c6a4a7d9540fb733cb3d8a91a27`).

The host must provide reflection and framing exactly as in the streaming SDK:
wait for the first 640 real samples, prepend a 320-sample reflected prefix,
advance by 320, reflect at the actual end of input and trim output to the
original sample count. This crate does not resample, frame, or apply peak gain.
Reset DSP and model caches together after errors, discontinuities or a new
recording. Call `free()` when disposing the generated WASM object.

RustFFT uses scalar FP32 here; it is not the Android PocketFFT implementation.
Native and actual WASM tests compare against independent NumPy FFT/WOLA fixtures
using the sealed PyTorch Hann. Regenerate with `python tests/generate_fixtures.py`
(NumPy required). These establish transform/WOLA behavior, not end-to-end neural
parity or browser real-time latency. Validate those separately on target browsers.

Dependencies: RustFFT (MIT OR Apache-2.0), wasm-bindgen (MIT OR Apache-2.0).
The pinned versions and transitive dependencies are recorded in `Cargo.lock`.
