"""Run carried original/split ONNX inference on independently prepared PCM spectra."""
import argparse
import json
from pathlib import Path
import numpy as np
from support.common import read_source, digest
from support.runtime import SplitRuntime, session, zero_states, carry
from verify import compare


def spectral_rows(fixtures):
    manifest = json.loads((fixtures / "manifest.json").read_text())
    entry, path = manifest["files"]["spectra.f32"], fixtures / "spectra.f32"
    if digest(path) != entry["sha256"]: raise ValueError("PCM-spectrum fixture hash mismatch")
    if entry["shape"][1] != 642: raise ValueError("Expected magnitude/phase fixture rows")
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(entry["shape"])


def verify_audio(source, directory, fixtures):
    _, manifest = read_source(source)
    original, split = session(source), SplitRuntime(directory)
    source_state, split_state = zero_states(manifest), zero_states(manifest)
    enhanced, metrics = [], []
    for row in spectral_rows(fixtures):
        inputs = {"magnitude": row[:321].reshape(1, 321, 1), "phase": row[321:].reshape(1, 321, 1)}
        expected = dict(zip(manifest["outputNames"], original.run(None, source_state | inputs), strict=True))
        actual = split.run(split_state | inputs)
        metrics.append(compare(expected, actual))
        enhanced.append(np.concatenate([expected["enhanced_magnitude"].ravel(), expected["enhanced_phase"].ravel()]))
        source_state, split_state = carry(manifest, expected), carry(manifest, actual)
    return save_audio_receipt(directory, fixtures, metrics, enhanced)


def save_audio_receipt(directory, fixtures, metrics, enhanced):
    path = directory / "audio-enhanced-spectra.f32"
    np.stack(enhanced).astype("<f4").tofile(path)
    receipt = {"scope": "Original CPU ONNX and split CPU model on PCM-derived spectra; not browser validation",
               "inputSpectraSha256": digest(fixtures / "spectra.f32"), "hops": metrics,
               "enhancedSpectra": {"file": path.name, "sha256": digest(path), "shape": [len(enhanced), 642]}}
    (directory / "audio-parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    options = parser.parse_args()
    receipt = verify_audio(options.source, options.artifact, options.fixtures)
    print(json.dumps(receipt["hops"]))


if __name__ == "__main__": main()
