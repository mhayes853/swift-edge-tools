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
    ) -> torch.nn.Module: ...


@dataclass(frozen=True)
class CoreAIQuantizerCompressor:
    config: QuantizerConfig

    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
        *,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    ) -> torch.nn.Module:
        quantizer = Quantizer(module, self.config)
        quantizer.prepare(sample_args, dynamic_shapes=dynamic_shapes)
        finalized_module = quantizer.finalize(backend=ExportBackend.CoreAI)
        finalized_module.eval()
        return finalized_module