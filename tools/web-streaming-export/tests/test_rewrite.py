"""Numerical edge cases for the exported supported pointwise replacements."""
import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from support.rewrite import rewrite, remove_identities
from support.common import describe


def run_operator(operator, values, slope=None):
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, values.shape)]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, values.shape)
    names, constants = ["input"], []
    if slope is not None:
        names.append("slope")
        constants.append(numpy_helper.from_array(slope, "slope"))
    graph = helper.make_graph([helper.make_node(operator, names, ["output"])], "edge", inputs, [output], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    session = ort.InferenceSession(rewrite(model).SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": values})[0]


class RewriteTests(unittest.TestCase):
    def test_softplus_is_finite_at_extreme_values(self):
        values = np.array([-1000, -100, -20, -1, 0, 1, 20, 100, 1000], dtype=np.float32)
        output = run_operator("Softplus", values)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output, np.logaddexp(values, 0), atol=1e-7, rtol=1e-6)

    def test_prelu_preserves_per_channel_negative_slopes(self):
        values = np.array([[-2, -1, 0, 3], [-4, -1, 1, 5]], dtype=np.float32)
        slope = np.array([[0.25], [-0.5]], dtype=np.float32)
        output = run_operator("PRelu", values, slope)
        np.testing.assert_array_equal(output, [[-0.5, -0.25, 0, 3], [2, 0.5, 1, 5]])

    def test_dynamic_boundaries_are_rejected(self):
        value = helper.make_tensor_value_info("dynamic", TensorProto.FLOAT, [1, "frames"])
        with self.assertRaisesRegex(ValueError, "Non-static"):
            describe(value)

    def test_non_float_boundaries_are_rejected(self):
        value = helper.make_tensor_value_info("integer", TensorProto.INT64, [1])
        with self.assertRaisesRegex(ValueError, "Non-FP32"):
            describe(value)

    def test_chained_identity_outputs_become_aliases_without_arithmetic(self):
        source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        nodes = [helper.make_node("Identity", ["source"], ["middle"]),
                 helper.make_node("Identity", ["middle"], ["output"])]
        model = helper.make_model(helper.make_graph(nodes, "alias", [source], [output]))
        aliases = remove_identities(model)
        self.assertEqual(aliases, [{"name": "output", "source": "source"}])
        self.assertEqual(len(model.graph.node), 0)
        self.assertEqual(model.graph.output[0].name, "source")


if __name__ == "__main__": unittest.main()
