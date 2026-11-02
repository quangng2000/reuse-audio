"""Materialize browser audio oracles after the original CPU-model verification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_module(path):
    spec = importlib.util.spec_from_file_location("dsp_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(directory):
    root = Path(__file__).resolve().parents[1]
    source = directory / "audio-enhanced-spectra.f32"
    receipt = json.loads((directory / "audio-parity.json").read_text())
    if digest(source) != receipt["enhancedSpectra"]["sha256"]:
        raise ValueError("Original-model spectral reference hash mismatch")
    module = reference_module(root / "dsp/tests/generate_fixtures.py")
    hann = (root / "dsp/src/hann.rs").read_text()
    window = np.array([int(v, 16) for v in re.findall(r"0x[0-9a-f]{8}", hann)], dtype="<u4").view("<f4")
    spectra = np.frombuffer(source.read_bytes(), dtype="<f4").reshape(-1, 642)
    output = directory / "audio-expected-output.f32"
    module.synthesize(spectra, window).astype("<f4").tofile(output)
    frames = directory / "audio-input-frames.f32"
    shutil.copyfile(root / "dsp/tests/fixtures/frames.f32", frames)
    metadata = {p.name: {"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [output, frames]}
    (directory / "audio-reference.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    prepare(parser.parse_args().artifact)
