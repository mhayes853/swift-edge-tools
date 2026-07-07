from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from coreai_opt import ExportBackend
from coreai_opt.quantization import Quantizer, QuantizerConfig


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
class CoreAIOptQuantizerCompressor:
    config: QuantizerConfig
    backend: ExportBackend

    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
        *,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    ) -> torch.nn.Module:
        quantizer = Quantizer(module, self.config)
        quantizer.prepare(sample_args, dynamic_shapes=dynamic_shapes)
        finalized_module = quantizer.finalize(backend=self.backend)
        finalized_module.eval()
        return finalized_module


@dataclass(frozen=True)
class CoreAIQuantizerCompressor(CoreAIOptQuantizerCompressor):
    def __init__(self, config: QuantizerConfig):
        super().__init__(config=config, backend=ExportBackend.CoreAI)


@dataclass(frozen=True)
class CoreMLQuantizerCompressor(CoreAIOptQuantizerCompressor):
    def __init__(self, config: QuantizerConfig):
        super().__init__(config=config, backend=ExportBackend.CoreML)
