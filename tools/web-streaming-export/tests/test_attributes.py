"""Convolution defaults preserve rank and explicitly supplied attributes."""
import unittest
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from support.attributes import explicit_convolutions


def convolution_model(shape, **attributes):
    weight = numpy_helper.from_array(np.zeros(shape, dtype=np.float32), "weight")
    source = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, shape[1], *([12] * (len(shape) - 2))])
    node = helper.make_node("Conv", ["input", "weight"], ["output"], **attributes)
    return helper.make_model(helper.make_graph([node], "default", [source], [], [weight]))


class AttributeTests(unittest.TestCase):
    def test_two_dimensional_defaults_preserve_existing_dilation(self):
        model = explicit_convolutions(convolution_model([64, 64, 3, 3], dilations=[2, 1]))
        values = {value.name: helper.get_attribute_value(value) for value in model.graph.node[0].attribute}
        self.assertEqual(values["kernel_shape"], [3, 3])
        self.assertEqual(values["strides"], [1, 1])
        self.assertEqual(values["dilations"], [2, 1])
        self.assertEqual(values["pads"], [0, 0, 0, 0])

    def test_one_dimensional_defaults_have_one_spatial_axis(self):
        model = explicit_convolutions(convolution_model([256, 1, 4], group=256, pads=[3, 3]))
        values = {value.name: helper.get_attribute_value(value) for value in model.graph.node[0].attribute}
        self.assertEqual(values["kernel_shape"], [4])
        self.assertEqual(values["strides"], [1])
        self.assertEqual(values["dilations"], [1])
        self.assertEqual(values["pads"], [3, 3])
        self.assertEqual(values["group"], 256)


if __name__ == "__main__": unittest.main()
