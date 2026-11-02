"""Full negative Slice replacement must retain every spectral bin."""
import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from support.reverse import explicit_reversals


def reversal_model(length):
    values = {"starts": [-1], "ends": [np.iinfo(np.int64).min], "axes": [1], "steps": [-1]}
    initializers = [numpy_helper.from_array(np.array(value, dtype=np.int64), name) for name, value in values.items()]
    source = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, length, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, length, 2])
    node = helper.make_node("Slice", ["input", *values.keys()], ["output"])
    model = helper.make_model(helper.make_graph([node], "reverse", [source], [output], initializers),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


class ReverseTests(unittest.TestCase):
    def test_reversal_keeps_first_and_last_of_all_161_bins(self):
        model = explicit_reversals(reversal_model(161))
        self.assertEqual(model.graph.node[0].op_type, "Gather")
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        source = np.arange(322, dtype=np.float32).reshape(1, 161, 2)
        actual = session.run(None, {"input": source})[0]
        np.testing.assert_array_equal(actual, source[:, ::-1, :])
        self.assertEqual(actual.shape, (1, 161, 2))


if __name__ == "__main__": unittest.main()
