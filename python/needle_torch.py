import math
import typing

import torch
from torch import nn

from .needle_configuration import NeedleModelConfiguation

_NEEDLE_MAX_TOKENS = 1024


class Needle(nn.Module):
    def __init__(self, configuration: NeedleModelConfiguation):
        super().__init__()
        self.lm_head = nn.Embedding(
            num_embeddings=configuration.vocabulary_size,
            embedding_dim=configuration.dimensions,
        )
        self.model = _NeedleSimpleAttentionNetwork(configuration)
        self.register_buffer(
            "k_cache",
            torch.zeros(
                configuration.decoder_layers,
                1,
                _NEEDLE_MAX_TOKENS,
                configuration.dimensions,
            ),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                configuration.decoder_layers,
                1,
                _NEEDLE_MAX_TOKENS,
                configuration.dimensions,
            ),
        )


class _RoPEFrequencies:
    @staticmethod
    def inverse(dimensions: int, theta: float) -> torch.Tensor:
        positions = torch.arange(0, dimensions, 2, dtype=torch.float32)
        return 1.0 / (theta ** (positions / dimensions))

    def __init__(self, inverse: torch.Tensor, seq_len: int, dtype: torch.dtype):
        positions = torch.arange(0, seq_len, dtype=torch.float32)
        frequencies = positions[:, None] * inverse[None, :]
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        self.sin = torch.sin(embeddings)[None, :].to(dtype)
        self.cos = torch.cos(embeddings)[None, :].to(dtype)

    def apply(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[2]
        sin = self.sin[..., offset : offset + seq_len, :].unsqueeze(1)
        cos = self.cos[..., offset : offset + seq_len, :].unsqueeze(1)
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

    def forward(
        self,
        x: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(x) * self.embed_scale
        return self.decoder(
            x,
            self_mask=_causal_mask(x.shape[1]),
            cross_mask=cross_mask,
            encoder_projected_k=encoder_projected_k,
            encoder_projected_v=encoder_projected_v,
            k_cache=k_cache,
            v_cache=v_cache,
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
        self.inverse_rope = _RoPEFrequencies.inverse(
            dimensions=configuration.attention_head_dimensions,
            theta=configuration.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        rope = _RoPEFrequencies(
            inverse=self.inverse_rope,
            seq_len=k_cache[0].shape[2] + x.shape[1],
            dtype=x.dtype,
        )
        layers = enumerate(zip(self.layers, k_cache, v_cache))
        for index, (layer, k_cache, v_cache) in layers:
            x = layer(
                x,
                self_mask=self_mask,
                cross_mask=cross_mask,
                encoder_projected_k=encoder_projected_k[index],
                encoder_projected_v=encoder_projected_v[index],
                rope=rope,
                k_cache=k_cache,
                v_cache=v_cache,
            )
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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        self_normed = self.input_layernorm(x)
        self_attention = self.self_attn(
            q=self_normed,
            kv=self_normed,
            mask=self_mask,
            rope=rope,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        hidden_state = _gated_residual(x, self.self_attn_gate, self_attention)
        cross_normed = self.encoder_attn_layer_norm(hidden_state)
        cross_attention = self.encoder_attn(
            q=cross_normed,
            projected_k=encoder_projected_k,
            projected_v=encoder_projected_v,
            mask=cross_mask,
        )
        return _gated_residual(hidden_state, self.cross_attn_gate, cross_attention)


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
        self.inverse_rope = _RoPEFrequencies.inverse(
            dimensions=configuration.attention_head_dimensions,
            theta=configuration.rope_theta,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rope = _RoPEFrequencies(
            inverse=self.inverse_rope, seq_len=x.shape[1], dtype=x.dtype
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
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_k, projected_v = self.project_kv(kv)
        if rope is not None:
            offset = 0 if k_cache is None else k_cache.shape[2]
            projected_k = rope.apply(projected_k, offset)
        return self.forward_with_projected_kv(
            q=q,
            projected_k=projected_k,
            projected_v=projected_v,
            mask=mask,
            rope=rope,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    def forward_with_projected_kv(
        self,
        q: torch.Tensor,
        projected_k: torch.Tensor,
        projected_v: torch.Tensor,
        mask: torch.Tensor | None,
        rope: _RoPEFrequencies | None = None,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = q.shape
        q = self.q_norm(
            self.q_proj(q)
            .view(batch, seq_len, self.heads, self.head_dimensions)
            .transpose(1, 2)
        )

        cache_offset = 0 if k_cache is None else k_cache.shape[2]
        if rope is not None:
            q = rope.apply(q, offset=cache_offset)

        if k_cache is not None and v_cache is not None:
            end = cache_offset + seq_len
            k_cache[:, :, cache_offset:end, :] = projected_k
            v_cache[:, :, cache_offset:end, :] = projected_v
            keys = k_cache[:, :, :end, :]
            values = v_cache[:, :, :end, :]
        else:
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
        return torch.rms_norm(x, x.shape, 1 + self.weight.to(x.dtype), self.eps)


def _causal_mask(seq_len: int) -> torch.Tensor:
    return torch.ones((seq_len, seq_len), dtype=torch.bool).tril()[None, None, :, :]


def _gated_residual(
    x: torch.Tensor, gate: torch.Tensor, sublayer: torch.Tensor
) -> torch.Tensor:
    return torch.clip(
        x + torch.sigmoid(gate).to(sublayer.dtype) * sublayer, min=-65500.0, max=65500.0
    )


def _padding_mask(ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (ids != pad_token_id)[..., None, None, :]
