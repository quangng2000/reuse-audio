"""Portable little-endian FP32 spectral fixtures for carried browser hops."""
import json
import numpy as np
from .common import digest


def spectral_input(generator):
    return {"magnitude": generator.uniform(0, 0.5, (1, 321, 1)).astype(np.float32),
            "phase": generator.uniform(-np.pi, np.pi, (1, 321, 1)).astype(np.float32)}


def save_hop(directory, number, inputs, outputs):
    saved = {}
    for name in ("magnitude", "phase", "enhanced_magnitude", "enhanced_phase"):
        array = (inputs if name in inputs else outputs)[name]
        path = directory / f"hop-{number:02d}-{name}.bin"
        path.write_bytes(array.astype("<f4").tobytes())
        saved[name] = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
                       "dtype": "float32", "shape": list(array.shape)}
    return saved


def save_states(directory, outputs):
    saved = []
    for name, array in outputs.items():
        if not name.startswith("next_"): continue
        path = directory / f"hop-00-state-{len(saved):02d}.bin"
        path.write_bytes(array.astype("<f4").tobytes())
        saved.append({"name": name, "file": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
                      "shape": list(array.shape), "dtype": "float32"})
    return saved


def save_receipt(directory, metrics, fixtures, states):
    receipt = {"schemaVersion": 1, "seed": 20261101, "hops": metrics, "fixtures": fixtures,
               "stateFixtures": {"hop": 0, "outputs": states},
               "stateAtol": 0.0003, "stateRtol": 0.0002,
               "scope": "CPU split graph / original ONNX comparison; not browser, audio quality or latency validation"}
    (directory / "parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
