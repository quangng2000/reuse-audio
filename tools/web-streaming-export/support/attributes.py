"""Materialize ONNX convolution defaults for browser runtime compatibility."""
from onnx import helper
from .common import value_index


def explicit_convolutions(model):
    values = value_index(model)
    for node in model.graph.node:
        if node.op_type != "Conv": continue
        dimensions = values[node.input[1]].type.tensor_type.shape.dim
        if any(not dimension.HasField("dim_value") for dimension in dimensions):
            raise ValueError("Convolution weight shape must be static")
        kernel = [dimension.dim_value for dimension in dimensions[2:]]
        defaults = {"kernel_shape": kernel, "strides": [1] * len(kernel),
                    "dilations": [1] * len(kernel), "pads": [0] * (2 * len(kernel))}
        existing = {attribute.name for attribute in node.attribute}
        node.attribute.extend(helper.make_attribute(name, value) for name, value in defaults.items() if name not in existing)
    return model
