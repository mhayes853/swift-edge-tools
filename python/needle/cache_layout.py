from __future__ import annotations

import torch

from .decoder_strategy import DecoderExportStrategy
from .needle_configuration import NeedleModelConfiguation


def self_attention_state_names(
    configuration: NeedleModelConfiguation,
) -> tuple[str, ...]:
    return tuple(
        name
        for index in range(configuration.decoder_layers)
        for name in (
            f"self_attention_key_cache_{index}",
            f"self_attention_value_cache_{index}",
        )
    )


def decoder_state_names(
    configuration: NeedleModelConfiguation,
    strategy: DecoderExportStrategy,
) -> tuple[str, ...]:
    if not strategy.stateful:
        return ()
    if not strategy.cross_attention_cache_states:
        return self_attention_state_names(configuration)
    return tuple(
        name
        for index in range(configuration.decoder_layers)
        for name in (
            f"self_attention_key_cache_{index}",
            f"self_attention_value_cache_{index}",
            f"cross_attention_key_cache_{index}",
            f"cross_attention_value_cache_{index}",
        )
    )


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


def self_attention_state_shape(
    configuration: NeedleModelConfiguation,
) -> tuple[int, int, int]:
    return (
        configuration.decoder_max_length,
        configuration.kv_heads,
        configuration.attention_head_dimensions,
    )


def cross_attention_state_shape(
    configuration: NeedleModelConfiguation,
) -> tuple[int, int, int, int]:
    return (
        1,
        configuration.kv_heads,
        configuration.encoder_max_length,
        configuration.attention_head_dimensions,
    )


def decoder_state_shape(
    state_name: str,
    configuration: NeedleModelConfiguation,
) -> tuple[int, ...]:
    if state_name.startswith("self_attention_"):
        return self_attention_state_shape(configuration)
    if state_name.startswith("cross_attention_"):
        return cross_attention_state_shape(configuration)
    raise ValueError(f"Unknown decoder state: {state_name}")


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


def cross_attention_source(state_name: str) -> tuple[str, int]:
    kind = "k" if "_key_" in state_name else "v"
    try:
        layer_index = int(state_name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid cross-attention state name: {state_name}") from error
    return f"encoder_projected_{kind}", layer_index
