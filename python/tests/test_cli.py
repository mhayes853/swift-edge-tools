import unittest
from pathlib import Path
from unittest.mock import patch

import cli
from needle.export.onnx_compression import MatMulNBitsONNXCompressor


class CLITests(unittest.TestCase):
    def test_parser_accepts_only_onnx_export_options(self) -> None:
        arguments = cli.parse_arguments(
            [
                "--source",
                "model",
                "--output",
                "export",
                "--dtype",
                "float16",
                "--quantization",
                "int4",
            ]
        )

        self.assertEqual(arguments.source, "model")
        self.assertEqual(arguments.output, "export")
        self.assertEqual(arguments.dtype, "float16")
        self.assertEqual(arguments.quantization, "int4")

    def test_parser_rejects_int8_quantization(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.parse_arguments(
                [
                    "--output",
                    "export",
                    "--quantization",
                    "int8",
                ]
            )

    def test_main_exports_onnx_with_selected_compression(self) -> None:
        with patch("cli.export_needle_onnx") as export:
            export.return_value = Path("/tmp/needle-onnx")

            result = cli.main(
                [
                    "--source",
                    "model",
                    "--output",
                    "export",
                    "--quantization",
                    "int4",
                ]
            )

        self.assertEqual(result, 0)
        arguments = export.call_args
        self.assertEqual(arguments.args, ("model", "export"))
        self.assertEqual(arguments.kwargs["dtype"], "float32")
        self.assertIsInstance(
            arguments.kwargs["compressor"],
            MatMulNBitsONNXCompressor,
        )


if __name__ == "__main__":
    unittest.main()
