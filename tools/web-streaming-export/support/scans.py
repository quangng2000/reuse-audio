"""Extract the exact zero-initialized Mamba frequency recurrence contract."""
import numpy as np
from onnx import numpy_helper, helper


def captures(node):
    body = next(attribute.g for attribute in node.attribute if attribute.name == "body")
    local = {value.name for value in body.input} | {name for item in body.node for name in item.output}
    return {name for item in body.node for name in item.input if name not in local}


def describe_scan(node, constants, future_inputs):
    validate_axes(node)
    if node.output[0] in future_inputs:
        raise ValueError("Frequency final state is unexpectedly consumed")
    zero = numpy_helper.to_array(constants[node.input[0]])
    if zero.shape != (1, 256, 16) or np.any(zero):
        raise ValueError("Only zero-initialized 161-bin frequency scans are supported")
    names = captures(node)
    dynamics = [name for name in names if tuple(constants[name].dims) == (256, 16)]
    skip = [name for name in names if name.endswith(".D")]
    if len(dynamics) != 1 or len(skip) != 1:
        raise ValueError("Unexpected Mamba recurrence captures")
    inputs = dict(zip(("u", "dt", "b", "c", "z"), node.input[1:], strict=True))
    inputs.update(a=dynamics[0], d=skip[0])
    return {"kind": "scan", "inputs": inputs, "output": node.output[1],
            "bins": 161, "channels": 256, "components": 16}


def validate_axes(node):
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    expected = {"num_scan_inputs": 5, "scan_input_axes": [1] * 5, "scan_output_axes": [1]}
    if any(attributes.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected Scan axes")
    for key in ("scan_input_directions", "scan_output_directions"):
        if any(attributes.get(key, [])): raise ValueError("Reverse Scan direction is not supported")


def evaluate_scan(inputs):
    u, dt, b, c, z = (inputs[key] for key in ("u", "dt", "b", "c", "z"))
    state = np.zeros((u.shape[0], 256, 16), dtype=np.float32)
    result = np.empty_like(u)
    for index in range(u.shape[1]):
        step = dt[:, index, :, None]
        state = np.exp(step * inputs["a"]) * state + step * b[:, index, None, :] * u[:, index, :, None]
        output = (state * c[:, index, None, :]).sum(axis=-1) + inputs["d"] * u[:, index]
        gate = z[:, index] / (1 + np.exp(-z[:, index]))
        result[:, index] = output * gate
    return result
