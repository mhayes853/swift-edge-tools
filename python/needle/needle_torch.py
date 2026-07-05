import math
import typing

import torch
from torch import nn

from .needle_configuration import NeedleModelConfiguation


class Needle(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.configuration = configuration
        self.model = _NeedleSimpleAttentionNetwork(configuration)
        self.lm_head = (
            None
            if configuration.tie_word_embeddings
            else nn.Linear(
                configuration.dimensions,
                configuration.vocabulary_size,
                bias=False,
            )
        )
        object.__setattr__(self, "_encoder_export", NeedleEncoder(self))
        object.__setattr__(self, "_decoder_export", NeedleDecoder(self))

    @property
    def encoder(self) -> "NeedleEncoder":
        return typing.cast(NeedleEncoder, self._encoder_export)

    @property
    def decoder(self) -> "NeedleDecoder":
        return typing.cast(NeedleDecoder, self._decoder_export)

    def reset(self) -> None:
        self.decoder.reset()

    def forward(
        self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        cross_attention_mask, encoder_projected_k, encoder_projected_v = self.encoder(
            encoder_input_ids
        )
        self.reset()
        return self.decoder(
            decoder_input_ids,
            cross_attention_mask,
            encoder_projected_k,
            encoder_projected_v,
        )


class NeedleEncoder(nn.Module):
    def __init__(self, needle: Needle):
        super().__init__()
        self.configuration = needle.configuration
        self.model = needle.model

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_mask = _padding_mask(input_ids, self.configuration.pad_token_id)
        encoder_output = self.model.encode(input_ids, mask=encoder_mask)
        encoder_projected_k, encoder_projected_v = self.model.project_encoder_kv(
            encoder_output
        )
        return encoder_mask, encoder_projected_k, encoder_projected_v


class NeedleDecoder(nn.Module):
    def __init__(self, needle: Needle):
        super().__init__()
        self.configuration = needle.configuration
        self.model = needle.model
        self.lm_head = needle.lm_head
        self.register_buffer(
            "k_cache",
            torch.zeros(
                (
                    self.configuration.decoder_layers,
                    1,
                    self.configuration.kv_heads,
                    self.configuration.encoder_max_length,
                    self.configuration.attention_head_dimensions,
                )
            ),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                (
                    self.configuration.decoder_layers,
                    1,
                    self.configuration.kv_heads,
                    self.configuration.encoder_max_length,
                    self.configuration.attention_head_dimensions,
                )
            ),
        )
        self.register_buffer("cache_offset", torch.zeros((), dtype=torch.int32))

    def reset(self) -> None:
        typing.cast(torch.Tensor, self.k_cache).zero_()
        typing.cast(torch.Tensor, self.v_cache).zero_()
        typing.cast(torch.Tensor, self.cache_offset).zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"NeedleDecoder.forward() expects batch size 1, got {input_ids.shape[0]}"
            )

        hidden_state = self.model.decode(
            input_ids,
            cross_mask=cross_attention_mask,
            encoder_projected_k=encoder_projected_k,
            encoder_projected_v=encoder_projected_v,
            k_cache=typing.cast(torch.Tensor, self.k_cache),
            v_cache=typing.cast(torch.Tensor, self.v_cache),
            cache_offset=typing.cast(torch.Tensor, self.cache_offset),
        )
        typing.cast(torch.Tensor, self.cache_offset).add_(input_ids.shape[1])
        return _project_logits(
            hidden_state,
            self.lm_head,
            self.model.embed_tokens.weight,
        )


class _RoPEFrequencies:
    @staticmethod
    def inverse(dimensions: int, theta: float) -> torch.Tensor:
        positions = torch.arange(0, dimensions, 2, dtype=torch.float32)
        return 1.0 / (theta ** (positions / dimensions))

    def __init__(self, inverse: torch.Tensor, seq_len: int, dtype: torch.dtype):
        positions = torch.arange(0, seq_len, dtype=torch.float32, device=inverse.device)
        self._initialize(inverse, positions, dtype)

    @classmethod
    def from_positions(
        cls, inverse: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype
    ) -> "_RoPEFrequencies":
        rope = cls.__new__(cls)
        rope._initialize(
            inverse, positions.to(device=inverse.device, dtype=torch.float32), dtype
        )
        return rope

    def _initialize(
        self, inverse: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype
    ) -> None:
        frequencies = positions[:, None] * inverse[None, :]
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        self.sin = torch.sin(embeddings)[None, :].to(device=inverse.device, dtype=dtype)
        self.cos = torch.cos(embeddings)[None, :].to(device=inverse.device, dtype=dtype)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        sin = self.sin.unsqueeze(1)
        cos = self.cos.unsqueeze(1)
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return (x * cos) + (rotated * sin)


class _NeedleSimpleAttentionNetwork(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            num_embeddings=configuration.vocabulary_size,
            embedding_dim=configuration.dimensions,
        )
        self.encoder = _NeedleEncoder(configuration)
        self.decoder = _NeedleDecoder(configuration)
        self.embed_scale = math.sqrt(float(configuration.dimensions))

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed_tokens(x) * self.embed_scale, mask=mask)

    def decode(
        self,
        x: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_offset: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(x) * self.embed_scale
        return self.decoder(
            x,
            cross_mask=cross_mask,
            encoder_projected_k=encoder_projected_k,
            encoder_projected_v=encoder_projected_v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_offset=cache_offset,
        )

    def project_encoder_kv(
        self,
        encoder_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [
            typing.cast(_NeedleAttention, layer.encoder_attn).project_kv(encoder_output)
            for layer in self.decoder.layers
        ]
        projected_k = torch.stack([k for k, _ in projected], dim=0)
        projected_v = torch.stack([v for _, v in projected], dim=0)
        return projected_k, projected_v


class _NeedleDecoder(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _NeedleDecoderBlock(configuration)
                for _ in range(0, configuration.decoder_layers)
            ]
        )
        self.norm = _ZCRMSNorm.from_configuration(configuration)
        self.register_buffer(
            "inverse_rope",
            _RoPEFrequencies.inverse(
                dimensions=configuration.attention_head_dimensions,
                theta=configuration.rope_theta,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_offset: torch.Tensor,
    ) -> torch.Tensor:
        next_cache_offset = _next_cache_offset(
            cache_offset, x.shape[1], k_cache.shape[3]
        )
        rope = _RoPEFrequencies.from_positions(
            inverse=typing.cast(torch.Tensor, self.inverse_rope),
            positions=torch.arange(
                x.shape[1], device=x.device, dtype=cache_offset.dtype
            )
            + cache_offset.to(device=x.device),
            dtype=x.dtype,
        )
        self_mask = _decoder_self_mask(
            query_length=x.shape[1],
            max_tokens=k_cache.shape[3],
            cache_offset=next_cache_offset,
            device=x.device,
            dtype=x.dtype,
        )
        updated_keys: list[torch.Tensor] = []
        updated_values: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            x, updated_k, updated_v = layer(
                x,
                self_mask=self_mask,
                cross_mask=cross_mask,
                encoder_projected_k=encoder_projected_k[index],
                encoder_projected_v=encoder_projected_v[index],
                rope=rope,
                layer_k_cache=k_cache[index],
                layer_v_cache=v_cache[index],
                cache_offset=cache_offset,
            )
            updated_keys.append(updated_k)
            updated_values.append(updated_v)
        k_cache.copy_(torch.stack(updated_keys, dim=0))
        v_cache.copy_(torch.stack(updated_values, dim=0))
        return self.norm(x)


class _NeedleDecoderBlock(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.input_layernorm = _ZCRMSNorm.from_configuration(configuration)
        self.self_attn = _NeedleAttention(configuration)
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        self.encoder_attn_layer_norm = _ZCRMSNorm.from_configuration(configuration)
        self.encoder_attn = _NeedleAttention(configuration)
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        rope: _RoPEFrequencies,
        layer_k_cache: torch.Tensor | None,
        layer_v_cache: torch.Tensor | None,
        cache_offset: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_k_cache is None or layer_v_cache is None or cache_offset is None:
            raise ValueError("Decoder self-attention requires KV cache state.")
        self_normed = self.input_layernorm(x)
        self_attention, updated_k, updated_v = self.self_attn.forward_with_kv_cache(
            q=self_normed,
            kv=self_normed,
            mask=self_mask,
            rope=rope,
            layer_k_cache=layer_k_cache,
            layer_v_cache=layer_v_cache,
            cache_offset=cache_offset,
        )
        hidden_state = _gated_residual(x, self.self_attn_gate, self_attention)
        cross_normed = self.encoder_attn_layer_norm(hidden_state)
        cross_attention = self.encoder_attn.forward_with_projected_kv(
            q=cross_normed,
            projected_k=encoder_projected_k,
            projected_v=encoder_projected_v,
            mask=cross_mask,
        )
        return (
            _gated_residual(hidden_state, self.cross_attn_gate, cross_attention),
            updated_k,
            updated_v,
        )


class _NeedleEncoder(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _NeedleEncoderBlock(configuration)
                for _ in range(0, configuration.encoder_layers)
            ]
        )
        self.final_norm = _ZCRMSNorm.from_configuration(configuration)
        self.register_buffer(
            "inverse_rope",
            _RoPEFrequencies.inverse(
                dimensions=configuration.attention_head_dimensions,
                theta=configuration.rope_theta,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rope = _RoPEFrequencies(
            inverse=typing.cast(torch.Tensor, self.inverse_rope),
            seq_len=x.shape[1],
            dtype=x.dtype,
        )
        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope)
        return self.final_norm(x)


class _NeedleEncoderBlock(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.input_layernorm = _ZCRMSNorm.from_configuration(configuration)
        self.self_attn = _NeedleAttention(configuration)
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, rope: _RoPEFrequencies
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        attention = self.self_attn(q=normed, kv=normed, mask=mask, rope=rope)
        return _gated_residual(x, self.attn_gate, attention)


class _NeedleAttention(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.heads = configuration.attention_heads
        self.kv_heads = configuration.kv_heads
        self.head_dimensions = configuration.attention_head_dimensions
        self.scale = math.sqrt(1.0 / float(configuration.attention_head_dimensions))

        self.q_norm = _ZCRMSNorm(
            dimensions=configuration.attention_head_dimensions,
            eps=configuration.rms_norm_eps,
        )
        self.k_norm = _ZCRMSNorm(
            dimensions=configuration.attention_head_dimensions,
            eps=configuration.rms_norm_eps,
        )

        self.q_proj = nn.Linear(
            configuration.hidden_dimensions, configuration.hidden_dimensions, bias=False
        )
        self.k_proj = nn.Linear(
            configuration.hidden_dimensions, configuration.kv_dimensions, bias=False
        )
        self.v_proj = nn.Linear(
            configuration.hidden_dimensions, configuration.kv_dimensions, bias=False
        )
        self.out_proj = nn.Linear(
            configuration.hidden_dimensions, configuration.hidden_dimensions, bias=False
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None,
        rope: _RoPEFrequencies | None = None,
    ) -> torch.Tensor:
        projected_k, projected_v = self.project_kv(kv)
        if rope is not None:
            projected_k = rope.apply(projected_k)
        return self.forward_with_projected_kv(
            q=q,
            projected_k=projected_k,
            projected_v=projected_v,
            mask=mask,
            rope=rope,
        )

    def forward_with_kv_cache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None,
        rope: _RoPEFrequencies,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        cache_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_k, projected_v = self.project_kv(kv)
        projected_k = rope.apply(projected_k)
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
        output = self.forward_with_projected_kv(
            q=q,
            projected_k=updated_k,
            projected_v=updated_v,
            mask=mask,
            rope=rope,
        )
        return output, updated_k, updated_v

    def forward_with_projected_kv(
        self,
        q: torch.Tensor,
        projected_k: torch.Tensor,
        projected_v: torch.Tensor,
        mask: torch.Tensor | None,
        rope: _RoPEFrequencies | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = q.shape
        q = self.q_norm(
            self.q_proj(q)
            .view(batch, seq_len, self.heads, self.head_dimensions)
            .transpose(1, 2)
        )

        if rope is not None:
            q = rope.apply(q)

        keys = projected_k
        values = projected_v

        if self.heads != self.kv_heads:
            repeat = self.heads // self.kv_heads
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)

        output = nn.functional.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.heads * self.head_dimensions)
        )
        return self.out_proj(output)

    def project_kv(self, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, kv_len, _ = kv.shape
        keys = (
            self.k_proj(kv)
            .view(batch, kv_len, self.kv_heads, self.head_dimensions)
            .transpose(1, 2)
        )
        values = (
            self.v_proj(kv)
            .view(batch, kv_len, self.kv_heads, self.head_dimensions)
            .transpose(1, 2)
        )
        keys = self.k_norm(keys)
        return (keys, values)


class _ZCRMSNorm(nn.Module):
    @staticmethod
    def from_configuration(configuration: NeedleModelConfiguation):
        return _ZCRMSNorm(
            dimensions=configuration.dimensions, eps=configuration.rms_norm_eps
        )

    def __init__(self, dimensions: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dimensions))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rms_norm(x, (x.shape[-1],), 1 + self.weight.to(x.dtype), self.eps)


def _gated_residual(
    x: torch.Tensor, gate: torch.Tensor, sublayer: torch.Tensor
) -> torch.Tensor:
    return torch.clip(
        x + torch.sigmoid(gate).to(sublayer.dtype) * sublayer, min=-65500.0, max=65500.0
    )


def _padding_mask(ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (ids != pad_token_id)[..., None, None, :]


def _project_logits(
    hidden_state: torch.Tensor,
    lm_head: nn.Linear | None,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    if lm_head is not None:
        return lm_head(hidden_state)
    return nn.functional.linear(hidden_state, embedding_weight)



def _next_cache_offset(
    cache_offset: torch.Tensor, query_length: int, max_tokens: int
) -> torch.Tensor:
    return torch.minimum(
        cache_offset + torch.tensor(query_length, device=cache_offset.device),
        torch.tensor(max_tokens, device=cache_offset.device, dtype=cache_offset.dtype),
    )


def _decoder_self_mask(
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
    next_cache_offset = torch.minimum(
        cache_offset
        + torch.tensor(query_length, device=device, dtype=cache_offset.dtype),
        torch.tensor(max_tokens, device=device, dtype=cache_offset.dtype),
    )
    valid_key_positions = key_positions < next_cache_offset
    causal_positions = query_positions[:, None] >= key_positions[None, :]
    return torch.where(
        causal_positions & valid_key_positions[None, :],
        torch.tensor(0, device=device, dtype=dtype),
        torch.tensor(-10000, device=device, dtype=dtype),
    )[None, None, :, :]
