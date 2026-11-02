"""Build an experimental browser artifact from an existing verified ONNX export."""
import argparse
import json
from pathlib import Path
from support.common import read_source, digest, copy_licenses
from support.partition import partition
from support.rewrite import rewrite, fold_shapes, remove_identities
from support.attributes import explicit_convolutions
from support.reverse import explicit_reversals


def export(source, directory):
    model, original = read_source(source)
    directory.mkdir(parents=True, exist_ok=True)
    copy_licenses(directory)
    model = fold_shapes(rewrite(model), directory / "temporary-folded.onnx")
    aliases = remove_identities(model)
    explicit_convolutions(model)
    explicit_reversals(model)
    steps, constants = partition(model, directory)
    keys = ("modelId", "layers", "sampleRate", "fftSize", "hopSize", "license", "researchOnly", "checkpointSha256", "states")
    manifest = {key: original[key] for key in keys}
    manifest.update(schemaVersion=1, backend="experimental-webgpu", precision="float32", lookahead=0,
                    sourceModelSha256=digest(source), steps=steps, constants=constants, aliases=aliases,
                    inputNames=original["inputNames"], outputNames=original["outputNames"])
    if sum(step["kind"] == "scan" for step in steps) != original["layers"] * 2:
        raise ValueError("Wrong number of frequency scans")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    manifest = export(options.source, options.output)
    print(json.dumps({"layers": manifest["layers"], "steps": len(manifest["steps"]), "output": str(options.output)}))


if __name__ == "__main__": main()
