from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any

import torch


def next_cache_offset(
    cache_offset: torch.Tensor, query_length: int, max_tokens: int
) -> torch.Tensor:
    return torch.minimum(
        cache_offset + torch.tensor(query_length, device=cache_offset.device),
        torch.tensor(max_tokens, device=cache_offset.device, dtype=cache_offset.dtype),
    )


def decoder_self_mask(
    query_length: int,
    max_tokens: int,
    cache_offset: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    query_positions = (
        torch.arange(query_length, device=device, dtype=cache_offset.dtype)
        + cache_offset
    )
    key_positions = torch.arange(max_tokens, device=device, dtype=cache_offset.dtype)
    final_cache_offset = torch.minimum(
        cache_offset
        + torch.tensor(query_length, device=device, dtype=cache_offset.dtype),
        torch.tensor(max_tokens, device=device, dtype=cache_offset.dtype),
    )
    valid_key_positions = (key_positions < final_cache_offset).to(dtype=dtype)
    causal_positions = (query_positions[:, None] >= key_positions[None, :]).to(
        dtype=dtype
    )
    allowed_positions = causal_positions * valid_key_positions[None, :]
    return ((1.0 - allowed_positions) * -10000.0)[None, None, :, :]


def forward_decoder(
    *,
    decoder: Any,
    x: torch.Tensor,
    cross_mask: torch.Tensor,
    encoder_projected_k: torch.Tensor,
    encoder_projected_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_offset: torch.Tensor,
    rope_frequencies: Any,
    self_attention_adapter: Callable[
        [
            Any,
            torch.Tensor,
            torch.Tensor,
            Any,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
) -> torch.Tensor:
    final_cache_offset = next_cache_offset(cache_offset, x.shape[1], k_cache.shape[3])
    rope = rope_frequencies.from_positions(
        inverse=typing.cast(torch.Tensor, decoder.inverse_rope),
        positions=torch.arange(x.shape[1], device=x.device, dtype=cache_offset.dtype)
        + cache_offset.to(device=x.device),
        dtype=x.dtype,
    )
    self_mask = decoder_self_mask(
        query_length=x.shape[1],
        max_tokens=k_cache.shape[3],
        cache_offset=final_cache_offset,
        device=x.device,
        dtype=x.dtype,
    )
    updated_keys = [torch.empty(0)]
    updated_values = [torch.empty(0)]
    updated_keys.clear()
    updated_values.clear()
    for index, layer in enumerate(decoder.layers):
        x, updated_k, updated_v = forward_decoder_block(
            layer=layer,
            x=x,
            self_mask=self_mask,
            cross_mask=cross_mask,
            encoder_projected_k=encoder_projected_k[index],
            encoder_projected_v=encoder_projected_v[index],
            rope=rope,
            layer_k_cache=k_cache[index],
            layer_v_cache=v_cache[index],
            cache_offset=cache_offset,
            self_attention_adapter=self_attention_adapter,
        )
        updated_keys.append(updated_k)
        updated_values.append(updated_v)
    k_cache.copy_(torch.stack(updated_keys, dim=0))
    v_cache.copy_(torch.stack(updated_values, dim=0))
    return decoder.norm(x)


def forward_decoder_block(
    *,
    layer: Any,
    x: torch.Tensor,
    self_mask: torch.Tensor,
    cross_mask: torch.Tensor,
    encoder_projected_k: torch.Tensor,
    encoder_projected_v: torch.Tensor,
    rope: Any,
    layer_k_cache: torch.Tensor | None,
    layer_v_cache: torch.Tensor | None,
    cache_offset: torch.Tensor | None,
    self_attention_adapter: Callable[
        [
            Any,
            torch.Tensor,
            torch.Tensor,
            Any,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layer_k_cache is None or layer_v_cache is None or cache_offset is None:
        raise ValueError("Decoder self-attention requires KV cache state.")
    self_normed = layer.input_layernorm(x)
    self_attention, updated_k, updated_v = self_attention_adapter(
        layer,
        self_normed,
        self_mask,
        rope,
        layer_k_cache,
        layer_v_cache,
        cache_offset,
    )
    hidden_state = torch.clip(
        x
        + torch.sigmoid(layer.self_attn_gate).to(self_attention.dtype) * self_attention,
        min=-65500.0,
        max=65500.0,
    )
    cross_normed = layer.encoder_attn_layer_norm(hidden_state)
    cross_attention = layer.encoder_attn.forward_with_projected_kv(
        q=cross_normed,
        projected_k=encoder_projected_k,
        projected_v=encoder_projected_v,
        mask=cross_mask,
    )
    return (
        torch.clip(
            hidden_state
            + torch.sigmoid(layer.cross_attn_gate).to(cross_attention.dtype)
            * cross_attention,
            min=-65500.0,
            max=65500.0,
        ),
        updated_k,
        updated_v,
    )


def forward_decoder_self_attention(
    layer: Any,
    q: torch.Tensor,
    mask: torch.Tensor,
    rope: Any,
    layer_k_cache: torch.Tensor,
    layer_v_cache: torch.Tensor,
    cache_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return forward_attention_with_kv_cache(
        attention=layer.self_attn,
        q=q,
        kv=q,
        mask=mask,
        rope=rope,
        layer_k_cache=layer_k_cache,
        layer_v_cache=layer_v_cache,
        cache_offset=cache_offset,
        update_kv_cache=update_kv_cache_index_copy,
    )


def forward_attention_with_kv_cache(
    *,
    attention: Any,
    q: torch.Tensor,
    kv: torch.Tensor,
    mask: torch.Tensor | None,
    rope: Any,
    layer_k_cache: torch.Tensor,
    layer_v_cache: torch.Tensor,
    cache_offset: torch.Tensor,
    update_kv_cache: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected_k, projected_v = project_kv_for_cache(attention, kv=kv, rope=rope)
    updated_k, updated_v = update_kv_cache(
        layer_k_cache,
        layer_v_cache,
        projected_k,
        projected_v,
        cache_offset,
    )
    output = attention.forward_with_projected_kv(
        q=q,
        projected_k=updated_k,
        projected_v=updated_v,
        mask=mask,
        rope=rope,
    )
    return output, updated_k, updated_v


def project_kv_for_cache(
    attention: Any, *, kv: torch.Tensor, rope: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_k, projected_v = attention.project_kv(kv)
    return rope.apply(projected_k), projected_v


def update_kv_cache_index_copy(
    layer_k_cache: torch.Tensor,
    layer_v_cache: torch.Tensor,
    projected_k: torch.Tensor,
    projected_v: torch.Tensor,
    cache_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = projected_k.shape[2]
    indices = (
        cache_offset.to(device=projected_k.device)
        + torch.arange(
            seq_len,
            device=projected_k.device,
            dtype=cache_offset.dtype,
        )
    ).to(torch.long)
    updated_k = torch.index_copy(layer_k_cache, 2, indices, projected_k)
    updated_v = torch.index_copy(layer_v_cache, 2, indices, projected_v)
    return updated_k, updated_v
