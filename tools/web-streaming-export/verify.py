"""Verify original ONNX versus partitioned execution for carried spectral hops."""
import argparse
import json
from pathlib import Path
import numpy as np
from support.common import read_source
from support.fixtures import spectral_input, save_hop, save_states, save_receipt
from support.runtime import SplitRuntime, session, zero_states, carry


def compare(reference, actual):
    errors = {name: float(np.max(np.abs(value - actual[name]))) for name, value in reference.items()}
    for name, value in reference.items():
        if not np.isfinite(actual[name]).all(): raise ValueError(f"Nonfinite output: {name}")
        if not np.allclose(actual[name], value, atol=0.0003, rtol=0.0002):
            raise ValueError(f"Parity failed for {name}: {errors[name]}")
    return {"magnitudeMaxAbs": errors["enhanced_magnitude"], "phaseMaxAbs": errors["enhanced_phase"],
            "allStateMaxAbs": max(error for name, error in errors.items() if name.startswith("next_"))}


def verify(source, directory, hops):
    _, manifest = read_source(source)
    original, split = session(source), SplitRuntime(directory)
    source_state, split_state = zero_states(manifest), zero_states(manifest)
    generator, metrics, fixtures = np.random.default_rng(20261101), [], []
    for index in range(hops):
        inputs = spectral_input(generator)
        expected = dict(zip(manifest["outputNames"], original.run(None, source_state | inputs), strict=True))
        actual = split.run(split_state | inputs)
        metrics.append(compare(expected, actual))
        fixtures.append(save_hop(directory, index, inputs, expected))
        if index == 0: states = save_states(directory, expected)
        source_state, split_state = carry(manifest, expected), carry(manifest, actual)
    return save_receipt(directory, metrics, fixtures, states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--hops", type=int, default=3)
    options = parser.parse_args()
    if options.hops < 1: parser.error("hops must be positive")
    receipt = verify(options.source, options.artifact, options.hops)
    print(json.dumps(receipt["hops"]))


if __name__ == "__main__": main()
