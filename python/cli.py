from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections.abc import Sequence

from needle.export.helpers import DEFAULT_SOURCE
from needle.export.onnx import export_needle_onnx
from needle.export.onnx_compression import MatMulNBitsONNXCompressor, ONNXCompressor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Needle models to ONNX")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--quantization", choices=("int4",))
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(arguments)


def _build_compressor(quantization: str | None) -> ONNXCompressor | None:
    if quantization == "int4":
        return MatMulNBitsONNXCompressor.int4()
    return None


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    with contextlib.redirect_stdout(io.StringIO()):
        output_directory = export_needle_onnx(
            parsed.source,
            parsed.output,
            compressor=_build_compressor(parsed.quantization),
            dtype=parsed.dtype,
        )
    sys.stdout.write(f"{output_directory}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
