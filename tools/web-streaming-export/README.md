# Experimental WebGPU streaming export

This converts the existing pinned **Real-time RE-USE** FP32 ONNX export into
ONNX segments plus custom frequency-scan steps. It is the **NSCLv1 research
model**, not the commercial offline 30-layer checkpoint. Preserve the upstream
[license](../streaming-export/LICENSE), [notice](../streaming-export/NOTICE), and
[Mamba attribution](../streaming-export/MAMBA-LICENSE.txt) with redistribution.

No training or weight changes occur. The source model hash, checkpoint identity,
state layout and tensor names must match the repository's sealed manifest.
The exporter neither downloads models nor accesses audio remotely.

## Build and check

Create an isolated Python environment and install the exporter dependencies:

```sh
python3 -m venv tools/web-streaming-export/.venv
. tools/web-streaming-export/.venv/bin/activate
pip install -r tools/web-streaming-export/requirements.txt
```

First
produce the original graph using [the streaming exporter](../streaming-export/README.md).
From the repository root:

```sh
python tools/web-streaming-export/export_web.py \
  --source tools/streaming-export/generated/models/reuse-streaming-l4-fp32.onnx \
  --output tools/web-streaming-export/generated/l4
python tools/web-streaming-export/verify.py \
  --source tools/streaming-export/generated/models/reuse-streaming-l4-fp32.onnx \
  --artifact tools/web-streaming-export/generated/l4 --hops 3
PYTHONPATH=tools/web-streaming-export python -m unittest discover \
  -s tools/web-streaming-export/tests -p 'test_*.py' -v
```

Generated artifacts stay out of Git. Serve the entire generated directory to
the browser, including `manifest.json`, ONNX segments, FP32 constants,
`LICENSE`, `NOTICE`, and `MAMBA-LICENSE.txt`. The
small `parity.json` and spectral fixture files support carried-hop checks.

## Partition and execution contract

The original model uses two sequential frequency `Scan` operators per layer.
Those operators are not in ONNX Runtime WebGPU's support list. The exporter:

1. Rewrites PReLU as `Where(Less(x, 0), x * slope, x)`.
2. Rewrites Softplus as `log(1 + exp(-abs(x))) + relu(x)` for finite extremes.
3. Uses ORT **BASIC** constant folding to resolve shapes without provider-specific
   fusions, then divides the topologically ordered graph around each Scan.
4. Exports the negative dynamics matrix `a` and skip vector `d` as FP32 constants.
5. Replaces output Identity nodes with top-level `aliases: [{name, source}]`.
   Apply these name aliases after all steps, sharing the same tensor without
   arithmetic or copies. Original model/state output names remain unchanged.
6. Materializes convolution kernel/stride/dilation/padding defaults from the
   static weight rank for runtimes that require explicit ONNX attributes.
7. Replaces full negative-step Slice reversals with explicit Gather indices to
   retain every frequency bin across browser runtimes with INT64_MIN limits.

The manifest lists ordered `steps` with either `kind: onnx` (file, hash, input
and output name/shape/dtype) or `kind: scan`. Scan inputs are `u, dt, b, c, z, a, d`;
`dt` already includes bias and Softplus, while `z` is the **raw** gate input.
Each scan resets 256 × 16 states to zero and traverses 161 bins in order:

```text
state = exp(dt * a) * state + dt * b * u
output = (sum(state * c, components) + d * u) * z * sigmoid(z)
```

Retain named tensors needed by later steps. The model-level `states` list maps
cache inputs to next-state outputs; carry these across 20 ms hops, and zero all
states on stream reset. Frequency states reset each hop. Input/output spectra
remain `[1, 321, 1]` FP32. DSP stays FFT 640 / hop 320 / 16 kHz mono with the
existing streaming window, startup and EOF conventions.

## What verification establishes

The verifier compares each spectral output and every carried cache against the
original full CPU ONNX graph, using independent NumPy frequency recurrences.
It rejects nonfinite values or differences outside atol `0.0003` / rtol `0.0002`.
The pointwise rewrites preserve the mathematical function but can round
differently. The browser GPU backend needs its own parity, complete audio,
latency and thermal measurements. Passing this CPU check does not establish
browser real-time performance or speech-quality improvement.

Official runtime reference:
[ORT WebGPU operators](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md).
