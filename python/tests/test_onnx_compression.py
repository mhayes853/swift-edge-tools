# pyright: reportMissingImports=false
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from needle.export.onnx_compression import (  # pyright: ignore[reportMissingImports]
    MatMulNBitsONNXCompressor,
    ONNXQuantizationBits,
)


class ONNXCompressionTests(unittest.TestCase):
    def test_int4_quantizes_constant_matmul_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.onnx"
            destination = directory / "destination.onnx"
            weight = numpy_helper.from_array(
                np.arange(64, dtype=np.float32).reshape(8, 8),
                name="weight",
            )
            graph = helper.make_graph(
                [helper.make_node("MatMul", ("input", "weight"), ("output",))],
                "matmul",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1, 8))],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 8))],
                [weight],
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 21)],
            )
            onnx.save(model, source)

            MatMulNBitsONNXCompressor.int4().compress(
                source,
                destination,
                component="encoder",
            )

            quantized = onnx.load(destination)
            onnx.checker.check_model(quantized)
            nodes = [
                node for node in quantized.graph.node if node.op_type == "MatMulNBits"
            ]
            self.assertEqual(len(nodes), 1)
            bits = next(
                attribute.i
                for attribute in nodes[0].attribute
                if attribute.name == "bits"
            )
            self.assertEqual(bits, 4)

    def test_rejects_unsupported_bit_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 4 or 8 bits"):
            MatMulNBitsONNXCompressor(bits=cast(ONNXQuantizationBits, 3))


if __name__ == "__main__":
    unittest.main()
