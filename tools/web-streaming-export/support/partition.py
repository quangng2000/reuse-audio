"""Partition topologically ordered ONNX around custom frequency scans."""
import onnx
from onnx import helper
from .common import describe, digest, value_index, write_constant
from .scans import describe_scan, captures


def future_names(nodes, outputs):
    names = set(outputs)
    for node in nodes:
        names.update(node.input)
        if node.op_type == "Scan": names.update(captures(node))
    return names


def segment_model(nodes, model, future, index):
    produced = {name for node in nodes for name in node.output}
    used = {name for node in nodes for name in node.input if name}
    constants = {value.name: value for value in model.graph.initializer}
    inputs, outputs = sorted(used - produced - constants.keys()), sorted(produced & future)
    graph = helper.make_graph(nodes, "RE-USE browser segment", [index[name] for name in inputs],
                              [index[name] for name in outputs], [constants[name] for name in sorted(used & constants.keys())])
    result = helper.make_model(graph, opset_imports=model.opset_import, producer_name="reuse-web-research")
    result.ir_version = model.ir_version
    onnx.checker.check_model(result, full_check=True)
    return result


def save_segment(nodes, model, future, directory, number):
    segment = segment_model(nodes, model, future, value_index(model))
    path = directory / f"segment-{number:02d}.onnx"
    onnx.save(segment, path)
    return {"kind": "onnx", "file": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
            "inputs": [describe(value) for value in segment.graph.input],
            "outputs": [describe(value) for value in segment.graph.output]}


def partition(model, directory):
    nodes, pending, steps, exported = list(model.graph.node), [], [], {}
    constants = {value.name: value for value in model.graph.initializer}
    outputs = {value.name for value in model.graph.output}
    for position, node in enumerate(nodes):
        if node.op_type != "Scan":
            pending.append(node)
            continue
        future = future_names(nodes[position:], outputs)
        if pending: steps.append(save_segment(pending, model, future, directory, len(steps)))
        pending = []
        step = describe_scan(node, constants, future_names(nodes[position + 1:], outputs))
        for key in ("a", "d"):
            name = step["inputs"][key]
            if name not in exported: exported[name] = write_constant(constants[name], directory, len(exported))
        steps.append(step)
    if pending: steps.append(save_segment(pending, model, outputs, directory, len(steps)))
    return steps, list(exported.values())
