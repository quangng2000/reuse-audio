"""Replace unsupported pointwise ops; retain FP32 learned tensor bytes."""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def prelu(node, zero):
    name, inputs = node.output[0], node.input
    return [helper.make_node("Less", [inputs[0], zero], [name + "_negative"]),
            helper.make_node("Mul", list(inputs), [name + "_scaled"]),
            helper.make_node("Where", [name + "_negative", name + "_scaled", inputs[0]], [name])]


def softplus(node, one):
    # log(1 + exp(-abs(x))) + relu(x) avoids exp(x) overflow.
    name, source = node.output[0], node.input[0]
    return [helper.make_node("Abs", [source], [name + "_abs"]),
            helper.make_node("Neg", [name + "_abs"], [name + "_negative"]),
            helper.make_node("Exp", [name + "_negative"], [name + "_exp"]),
            helper.make_node("Add", [name + "_exp", one], [name + "_offset"]),
            helper.make_node("Log", [name + "_offset"], [name + "_log"]),
            helper.make_node("Relu", [source], [name + "_positive"]),
            helper.make_node("Add", [name + "_log", name + "_positive"], [name])]


def rewrite(model):
    graph, nodes = model.graph, []
    for name, value in (("web_zero", 0), ("web_one", 1)):
        graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name))
    for node in graph.node:
        if node.op_type == "PRelu": nodes.extend(prelu(node, "web_zero"))
        elif node.op_type == "Softplus": nodes.extend(softplus(node, "web_one"))
        else: nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def fold_shapes(model, destination):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(destination)
    options.intra_op_num_threads = 2
    ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    folded = onnx.shape_inference.infer_shapes(onnx.load(destination), data_prop=True)
    destination.unlink()
    return folded


def remove_identities(model):
    aliases = {node.output[0]: node.input[0] for node in model.graph.node if node.op_type == "Identity"}
    for name in aliases:
        while aliases[name] in aliases: aliases[name] = aliases[aliases[name]]
    nodes = [node for node in model.graph.node if node.op_type != "Identity"]
    for node in nodes:
        for position, name in enumerate(node.input): node.input[position] = aliases.get(name, name)
    exported = [{"name": value.name, "source": aliases[value.name]} for value in model.graph.output if value.name in aliases]
    outputs = {}
    for value in model.graph.output:
        value.name = aliases.get(value.name, value.name)
        outputs[value.name] = value
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.output[:]
    model.graph.output.extend(outputs.values())
    return exported
