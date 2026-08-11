from __future__ import annotations

import torch

from .needle_configuration import NeedleModelConfiguation


def explicit_cache_shape(
    configuration: NeedleModelConfiguation,
    *,
    length: int | None = None,
) -> tuple[int, int, int, int]:
    return (
        configuration.decoder_layers,
        configuration.decoder_max_length if length is None else length,
        configuration.kv_heads,
        configuration.attention_head_dimensions,
    )


def empty_decoder_caches(
    configuration: NeedleModelConfiguation,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = explicit_cache_shape(configuration, length=length)
    return (
        torch.zeros(shape, dtype=dtype, device=device),
        torch.zeros(shape, dtype=dtype, device=device),
    )
