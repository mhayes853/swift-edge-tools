from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from coreai_opt import ExportBackend
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
from coreai_opt.quantization import Quantizer, QuantizerConfig


class NeedleCompressor(Protocol):
    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
    ) -> torch.nn.Module: ...


@dataclass(frozen=True)
class CoreAIQuantizerCompressor:
    config: QuantizerConfig

    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
    ) -> torch.nn.Module:
        quantizer = Quantizer(module, self.config)
        quantizer.prepare(sample_args)
        finalized_module = quantizer.finalize(backend=ExportBackend.CoreAI)
        finalized_module.eval()
        return finalized_module


@dataclass(frozen=True)
class CoreAIKMeansPalettizerCompressor:
    config: KMeansPalettizerConfig

    def compress(
        self,
        module: torch.nn.Module,
        sample_args: tuple[torch.Tensor, ...],
    ) -> torch.nn.Module:
        palettizer = KMeansPalettizer(module, self.config)
        palettizer.prepare(cast(tuple[torch.Tensor], sample_args))
        finalized_module = palettizer.finalize(backend=ExportBackend.CoreAI)
        finalized_module.eval()
        return finalized_module
