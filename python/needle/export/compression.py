from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from coreai_opt import ExportBackend
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.quantization import QuantizationSpec, Quantizer, QuantizerConfig


class NeedleCompressor(Protocol):
    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
        *,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    ) -> torch.nn.Module:
        return module


@dataclass(frozen=True)
class CoreAIOptCompressor:
    backend: ExportBackend
    quantizer_config: QuantizerConfig | None = None
    palettizer_config: KMeansPalettizerConfig | None = None

    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
        *,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    ) -> torch.nn.Module:
        compressed_module = module
        if self.palettizer_config is not None:
            palettizer = KMeansPalettizer(
                compressed_module,
                _with_quantized_lookup_tables(
                    self.palettizer_config,
                    self.quantizer_config,
                ),
            )
            palettizer.prepare(cast(tuple[torch.Tensor], sample_args))
            compressed_module = palettizer.finalize(backend=self.backend)
        if self.quantizer_config is not None and self.palettizer_config is None:
            quantizer = Quantizer(compressed_module, self.quantizer_config)
            quantizer.prepare(sample_args, dynamic_shapes=dynamic_shapes)
            compressed_module = quantizer.finalize(backend=self.backend)
        compressed_module.eval()
        return compressed_module


def _with_quantized_lookup_tables(
    palettizer_config: KMeansPalettizerConfig,
    quantizer_config: QuantizerConfig | None,
) -> KMeansPalettizerConfig:
    if quantizer_config is None:
        return palettizer_config
    quantizer_global_config = quantizer_config.global_config
    palettizer_global_config = palettizer_config.global_config
    if quantizer_global_config is None or palettizer_global_config is None:
        return palettizer_config
    quantization_spec = _weight_quantization_spec(quantizer_global_config.op_state_spec)
    if quantization_spec is None:
        return palettizer_config
    palettization_state_spec = palettizer_global_config.op_state_spec
    if palettization_state_spec is None:
        return palettizer_config
    state_spec = {
        name: _with_quantized_lookup_table(spec, quantization_spec)
        for name, spec in palettization_state_spec.items()
    }
    global_config = palettizer_global_config.model_copy(
        update={"op_state_spec": state_spec}
    )
    return palettizer_config.model_copy(update={"global_config": global_config})


def _weight_quantization_spec(
    state_spec: dict[str, QuantizationSpec | None] | None,
) -> QuantizationSpec | None:
    if state_spec is None:
        return None
    return state_spec.get("weight")


def _with_quantized_lookup_table(
    spec: PalettizationSpec | None,
    quantization_spec: QuantizationSpec,
) -> PalettizationSpec | None:
    if spec is None:
        return None
    return spec.model_copy(update={"lut_qspec": quantization_spec})


@dataclass(frozen=True)
class CoreAIQuantizerCompressor(CoreAIOptCompressor):
    def __init__(
        self,
        config: QuantizerConfig | None = None,
        *,
        palettizer_config: KMeansPalettizerConfig | None = None,
    ):
        super().__init__(
            backend=ExportBackend.CoreAI,
            quantizer_config=config,
            palettizer_config=palettizer_config,
        )


@dataclass(frozen=True)
class CoreMLQuantizerCompressor(CoreAIOptCompressor):
    def __init__(
        self,
        config: QuantizerConfig | None = None,
        *,
        palettizer_config: KMeansPalettizerConfig | None = None,
    ):
        super().__init__(
            backend=ExportBackend.CoreML,
            quantizer_config=config,
            palettizer_config=palettizer_config,
        )
