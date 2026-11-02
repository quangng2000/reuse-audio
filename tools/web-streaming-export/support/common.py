"""Shared artifact validation for the experimental NSCLv1 browser export."""
import hashlib
import json
import shutil
from pathlib import Path
import onnx
from onnx import helper, numpy_helper


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path):
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    verify_pin(manifest)
    if digest(path) != manifest["modelSha256"]:
        raise ValueError("Source ONNX does not match its manifest")
    if manifest["license"] != "NSCLv1" or not manifest["researchOnly"]:
        raise ValueError("Expected the separate research streaming model")
    if manifest["layers"] not in range(3, 13) or manifest["lookahead"] != 0:
        raise ValueError("Unsupported streaming depth or lookahead")
    return onnx.load(path), manifest


def verify_pin(manifest):
    root = Path(__file__).resolve().parents[3]
    name = f"reuse-streaming-l{manifest['layers']}-fp32.manifest.json"
    trusted = json.loads((root / "android/reuse-streaming/model/manifests" / name).read_text())
    fields = ("modelSha256", "checkpointSha256", "sourceRevision", "states", "inputNames", "outputNames")
    if any(manifest[field] != trusted[field] for field in fields):
        raise ValueError("Source differs from the repository's pinned streaming export")


def value_index(model):
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    index = {value.name: value for value in values}
    for value in model.graph.initializer:
        index[value.name] = helper.make_tensor_value_info(value.name, value.data_type, value.dims)
    return index


def describe(value):
    tensor = value.type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"Non-FP32 segment boundary: {value.name}")
    if any(not dimension.HasField("dim_value") for dimension in tensor.shape.dim):
        raise ValueError(f"Non-static segment boundary: {value.name}")
    return {"name": value.name, "shape": [dimension.dim_value for dimension in tensor.shape.dim],
            "dtype": "float32"}


def write_constant(value, directory, number):
    array = numpy_helper.to_array(value).astype("<f4")
    path = directory / f"constant-{number:02d}.bin"
    path.write_bytes(array.tobytes())
    return {"name": value.name, "shape": list(array.shape), "dtype": "float32",
            "file": path.name, "sha256": digest(path), "bytes": path.stat().st_size}


def copy_licenses(directory):
    source = Path(__file__).resolve().parents[3] / "tools/streaming-export"
    for name in ("LICENSE", "NOTICE", "MAMBA-LICENSE.txt"):
        shutil.copyfile(source / name, directory / name)
