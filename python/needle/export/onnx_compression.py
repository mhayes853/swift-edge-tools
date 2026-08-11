from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import QuantFormat

ONNXModelComponent = Literal["encoder", "decoder"]
ONNXQuantizationBits = Literal[4]


class ONNXCompressor(Protocol):
    def compress(
        self,
        source: Path,
        destination: Path,
        *,
        component: ONNXModelComponent,
    ) -> None:
        """Compress source and write the resulting ONNX model to destination."""


@dataclass(frozen=True)
class MatMulNBitsONNXCompressor:
    bits: ONNXQuantizationBits

    def __post_init__(self) -> None:
        if self.bits != 4:
            raise ValueError("MatMulNBits quantization supports only 4 bits")

    @classmethod
    def int4(cls) -> MatMulNBitsONNXCompressor:
        return cls(bits=4)

    def compress(
        self,
        source: Path,
        destination: Path,
        *,
        component: ONNXModelComponent,
    ) -> None:
        _ = component
        model = onnx.load(source, load_external_data=True)
        config = DefaultWeightOnlyQuantConfig(
            bits=self.bits,
            block_size=128,
            is_symmetric=True,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
        quantizer = MatMulNBitsQuantizer(model, algo_config=config)
        quantizer.process()
        cast(ONNXModel, quantizer.model).save_model_to_file(
            destination,
            use_external_data_format=True,
        )
