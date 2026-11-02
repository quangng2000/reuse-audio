"""CPU verification runtime for ONNX segments and the explicit scan contract."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from .common import digest
from .scans import evaluate_scan


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class SplitRuntime:
    def __init__(self, directory):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.constants, self.sessions = {}, {}
        for entry in self.manifest["constants"]:
            path = directory / entry["file"]
            if digest(path) != entry["sha256"]: raise ValueError("Constant hash mismatch")
            self.constants[entry["name"]] = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(entry["shape"])
        for step in self.manifest["steps"]:
            if step["kind"] != "onnx": continue
            path = directory / step["file"]
            if digest(path) != step["sha256"]: raise ValueError("Segment hash mismatch")
            self.sessions[step["file"]] = session(path)

    def run(self, inputs):
        values = self.constants | inputs
        for step in self.manifest["steps"]:
            if step["kind"] == "scan":
                values[step["output"]] = evaluate_scan({key: values[name] for key, name in step["inputs"].items()})
                continue
            inputs = {entry["name"]: values[entry["name"]] for entry in step["inputs"]}
            outputs = self.sessions[step["file"]].run(None, inputs)
            values.update(zip((entry["name"] for entry in step["outputs"]), outputs, strict=True))
        values.update({alias["name"]: values[alias["source"]] for alias in self.manifest["aliases"]})
        return {name: values[name] for name in self.manifest["outputNames"]}


def zero_states(manifest):
    return {entry["inputName"]: np.zeros(entry["shape"], dtype=np.float32) for entry in manifest["states"]}


def carry(manifest, outputs):
    return {entry["inputName"]: outputs[entry["outputName"]] for entry in manifest["states"]}
