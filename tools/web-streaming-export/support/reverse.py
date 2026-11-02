"""Use explicit Gather indices for full reversals instead of INT64_MIN Slice."""
import numpy as np
from onnx import helper, numpy_helper
from .common import value_index


def gather_reverse(node, constants, values):
    starts, ends, axes, steps = (numpy_helper.to_array(constants[name]).ravel() for name in node.input[1:])
    if len(steps) != 1 or steps[0] != -1 or starts[0] != -1 or ends[0] != np.iinfo(np.int64).min:
        raise ValueError("Only complete single-axis negative Slice reversals are supported")
    dimensions = values[node.input[0]].type.tensor_type.shape.dim
    axis = int(axes[0]) % len(dimensions)
    if not dimensions[axis].HasField("dim_value"): raise ValueError("Reversal axis must be static")
    length = dimensions[axis].dim_value
    name = node.output[0] + "_web_reverse_indices"
    indices = numpy_helper.from_array(np.arange(length - 1, -1, -1, dtype=np.int64), name)
    replacement = helper.make_node("Gather", [node.input[0], name], list(node.output), axis=axis)
    return replacement, indices


def explicit_reversals(model):
    constants = {value.name: value for value in model.graph.initializer}
    values, nodes = value_index(model), []
    for node in model.graph.node:
        negative = node.op_type == "Slice" and len(node.input) == 5
        if negative: negative = np.any(numpy_helper.to_array(constants[node.input[4]]) < 0)
        if not negative:
            nodes.append(node)
            continue
        replacement, indices = gather_reverse(node, constants, values)
        nodes.append(replacement)
        model.graph.initializer.append(indices)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model
