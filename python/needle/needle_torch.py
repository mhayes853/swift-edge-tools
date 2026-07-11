import math
import typing

import torch
from torch import nn

from .needle_configuration import NeedleModelConfiguation


class NeedleModel(nn.Module):
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.model = _NeedleSimpleAttentionNetwork(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.lm_head = (
            None
            if configuration.tie_word_embeddings
            else nn.Linear(
                configuration.dimensions,
                configuration.vocabulary_size,
                bias=False,
            )
        )
        self.configuration = configuration


class Needle:
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool = True,
    ):
        self.configuration = configuration
        self._weights = NeedleModel(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.encoder = NeedleEncoder(self._weights)
        self.decoder = NeedleDecoder(self._weights)

    def state_dict(self, *args: typing.Any, **kwargs: typing.Any):
        return self._weights.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: typing.Any, **kwargs: typing.Any):
        return self._weights.load_state_dict(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self._weights.parameters(recurse=recurse)

    def to(self, *args: typing.Any, **kwargs: typing.Any) -> "Needle":
        self.decoder.to(*args, **kwargs)
        return self

    def eval(self) -> "Needle":
        self.encoder.eval()
        self.decoder.eval()
        return self


class NeedleEncoder(nn.Module):
    def __init__(self, model: NeedleModel):
        super().__init__()
        self.configuration = model.configuration
        self.model = model.model

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_dtype = typing.cast(torch.Tensor, self.model.embed_tokens.weight).dtype
        encoder_mask = _padding_attention_mask(
            input_ids,
            self.configuration.pad_token_id,
            dtype=model_dtype,
        )
        encoder_output = self.model.encode(input_ids, mask=encoder_mask)
        encoder_projected_k, encoder_projected_v = self.model.project_encoder_kv(
            encoder_output
        )
        return encoder_mask, encoder_projected_k, encoder_projected_v


class NeedleDecoder(nn.Module):
    def __init__(self, model: NeedleModel):
        super().__init__()
        self.configuration = model.configuration
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_state, updated_key_cache, updated_value_cache = self.model.decode(
            input_ids,
            cache_position=cache_position,
            self_mask=self_attention_mask,
            cross_mask=cross_attention_mask,
            encoder_projected_k=encoder_projected_k,
            encoder_projected_v=encoder_projected_v,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        logits = _project_logits(
            hidden_state,
            self.lm_head,
            self.model.embed_tokens.weight,
        )
        return logits, updated_key_cache, updated_value_cache


class _RoPEFrequencies:
    @staticmethod
    def inverse(dimensions: int, theta: float) -> torch.Tensor:
        positions = torch.arange(0, dimensions, 2, dtype=torch.float32)
        return 1.0 / (theta ** (positions / dimensions))

    @staticmethod
    def table(dimensions: int, theta: float, length: int) -> torch.Tensor:
        inverse = _RoPEFrequencies.inverse(dimensions, theta)
        positions = torch.arange(0, length, dtype=torch.float32)
        frequencies = positions[:, None] * inverse[None, :]
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return torch.stack((torch.sin(embeddings), torch.cos(embeddings)), dim=0)

    def __init__(self, inverse: torch.Tensor, seq_len: int, dtype: torch.dtype):
        positions = torch.arange(0, seq_len, dtype=torch.float32, device=inverse.device)
        frequencies = positions[:, None] * inverse[None, :]
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        self.sin = torch.sin(embeddings)[None, :].to(device=inverse.device, dtype=dtype)
        self.cos = torch.cos(embeddings)[None, :].to(device=inverse.device, dtype=dtype)

    @classmethod
    def from_table(
        cls, table: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype
    ) -> "_RoPEFrequencies":
        rope = cls.__new__(cls)
        selected = table[:, positions.to(dtype=torch.long)]
        rope.sin = selected[0].unsqueeze(0).to(dtype=dtype)
        rope.cos = selected[1].unsqueeze(0).to(dtype=dtype)
        return rope

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        sin = self.sin.unsqueeze(1)
        cos = self.cos.unsqueeze(1)
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return (x * cos) + (rotated * sin)


class _NeedleSimpleAttentionNetwork(nn.Module):
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            num_embeddings=configuration.vocabulary_size,
            embedding_dim=configuration.dimensions,
        )
        self.encoder = _NeedleEncoder(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.decoder = _NeedleDecoder(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.embed_scale = math.sqrt(configuration.dimensions)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed_tokens(x) * self.embed_scale, mask=mask)

    def decode(
        self,
        x: torch.Tensor,
        cache_position: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embed_tokens(x) * self.embed_scale
        return self.decoder(
            x,
            cache_position=cache_position,
            self_mask=self_mask,
            cross_mask=cross_mask,
            encoder_projected_k=encoder_projected_k,
            encoder_projected_v=encoder_projected_v,
            key_cache=key_cache,
            value_cache=value_cache,
        )

    def project_encoder_kv(
        self,
        encoder_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [
            typing.cast(_NeedleAttention, layer.encoder_attn).expand_kv_heads(
                *typing.cast(_NeedleAttention, layer.encoder_attn).project_kv(
                    encoder_output
                )
            )
            for layer in self.decoder.layers
        ]
        projected_k = torch.stack([k for k, _ in projected], dim=0)
        projected_v = torch.stack([v for _, v in projected], dim=0)
        return projected_k, projected_v


class _NeedleDecoder(nn.Module):
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _NeedleDecoderBlock(
                    configuration,
                    use_native_sdpa=use_native_sdpa,
                )
                for _ in range(configuration.decoder_layers)
            ]
        )
        self.norm = _ZCRMSNorm.from_configuration(configuration)
        self.register_buffer(
            "rope_table",
            _RoPEFrequencies.table(
                dimensions=configuration.attention_head_dimensions,
                theta=configuration.rope_theta,
                length=configuration.encoder_max_length,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache_position: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rope = _RoPEFrequencies.from_table(
            table=typing.cast(torch.Tensor, self.rope_table),
            positions=cache_position,
            dtype=x.dtype,
        )

        updated_keys = [torch.empty(0, device=x.device, dtype=x.dtype)]
        updated_values = [torch.empty(0, device=x.device, dtype=x.dtype)]
        updated_keys.clear()
        updated_values.clear()
        for index, layer in enumerate(self.layers):
            x, updated_key, updated_value = layer(
                x=x,
                self_mask=self_mask,
                cross_mask=cross_mask,
                encoder_projected_k=encoder_projected_k[index],
                encoder_projected_v=encoder_projected_v[index],
                rope=rope,
                layer_key_cache=key_cache[index],
                layer_value_cache=value_cache[index],
            )
            updated_keys.append(updated_key)
            updated_values.append(updated_value)
        return (
            self.norm(x),
            torch.stack(updated_keys, dim=0),
            torch.stack(updated_values, dim=0),
        )


class _NeedleDecoderBlock(nn.Module):
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.input_layernorm = _ZCRMSNorm.from_configuration(configuration)
        self.self_attn = _NeedleAttention(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        self.encoder_attn_layer_norm = _ZCRMSNorm.from_configuration(configuration)
        self.encoder_attn = _NeedleAttention(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        rope: _RoPEFrequencies,
        layer_key_cache: torch.Tensor,
        layer_value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_normed = self.input_layernorm(x)
        self_attention, updated_key, updated_value = (
            self.self_attn.forward_with_kv_cache(
                q=self_normed,
                kv=self_normed,
                mask=self_mask,
                rope=rope,
                layer_key_cache=layer_key_cache,
                layer_value_cache=layer_value_cache,
            )
        )
        hidden_state = _gated_residual(x, self.self_attn_gate, self_attention)
        cross_attention = self.encoder_attn.forward_with_projected_kv(
            q=self.encoder_attn_layer_norm(hidden_state),
            projected_k=encoder_projected_k,
            projected_v=encoder_projected_v,
            mask=cross_mask,
        )
        return (
            _gated_residual(hidden_state, self.cross_attn_gate, cross_attention),
            updated_key,
            updated_value,
        )


class _NeedleEncoder(nn.Module):
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _NeedleEncoderBlock(
                    configuration,
                    use_native_sdpa=use_native_sdpa,
                )
                for _ in range(configuration.encoder_layers)
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
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.input_layernorm = _ZCRMSNorm.from_configuration(configuration)
        self.self_attn = _NeedleAttention(
            configuration,
            use_native_sdpa=use_native_sdpa,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, rope: _RoPEFrequencies
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        attention = self.self_attn(q=normed, kv=normed, mask=mask, rope=rope)
        return _gated_residual(x, self.attn_gate, attention)


class _NeedleAttention(nn.Module):
    def __init__(
        self,
        configuration: NeedleModelConfiguation,
        *,
        use_native_sdpa: bool,
    ):
        super().__init__()
        self.heads = configuration.attention_heads
        self.use_native_sdpa = use_native_sdpa
        self.kv_heads = configuration.kv_heads
        self.head_dimensions = configuration.attention_head_dimensions

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
        projected_k, projected_v = self.expand_kv_heads(projected_k, projected_v)
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
        layer_key_cache: torch.Tensor,
        layer_value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_k, projected_v = self.project_kv(kv)
        projected_k = rope.apply(projected_k)
        projected_k, projected_v = self.expand_kv_heads(projected_k, projected_v)
        updated_key = _update_kv_cache(layer_key_cache, projected_k)
        updated_value = _update_kv_cache(layer_value_cache, projected_v)
        active_k = updated_key.transpose(0, 1).unsqueeze(0)
        active_v = updated_value.transpose(0, 1).unsqueeze(0)
        output = self.forward_with_projected_kv(
            q=q,
            projected_k=active_k.to(dtype=q.dtype),
            projected_v=active_v.to(dtype=q.dtype),
            mask=mask,
            rope=rope,
        )
        return output, updated_key, updated_value

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

        if self.use_native_sdpa:
            output = nn.functional.scaled_dot_product_attention(
                q,
                projected_k,
                projected_v,
                attn_mask=mask,
                dropout_p=0.0,
            )
        else:
            attention_scores = torch.matmul(q, projected_k.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(q.shape[-1])
            if mask is not None:
                attention_scores = attention_scores + mask
            attention_probabilities = torch.softmax(attention_scores, dim=-1)
            output = torch.matmul(attention_probabilities, projected_v)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.heads * self.head_dimensions)
        )
        return self.out_proj(output)

    def expand_kv_heads(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.heads == self.kv_heads:
            return keys, values
        repeat = self.heads // self.kv_heads
        return (
            keys.repeat_interleave(repeat, dim=1),
            values.repeat_interleave(repeat, dim=1),
        )

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


def _update_kv_cache(
    layer_cache: torch.Tensor, projected: torch.Tensor
) -> torch.Tensor:
    current = projected[0].transpose(0, 1)
    return torch.cat((layer_cache[1:], current), dim=0)


def _gated_residual(
    x: torch.Tensor, gate: torch.Tensor, sublayer: torch.Tensor
) -> torch.Tensor:
    return torch.clip(
        x + torch.sigmoid(gate).to(sublayer.dtype) * sublayer, min=-65500.0, max=65500.0
    )


def _padding_attention_mask(
    ids: torch.Tensor,
    pad_token_id: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    disallowed = (ids == pad_token_id).to(dtype=dtype)
    return disallowed[:, None, None, :] * disallowed.new_tensor(-65500.0)


def _project_logits(
    hidden_state: torch.Tensor,
    lm_head: nn.Linear | None,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    if lm_head is not None:
        return lm_head(hidden_state)
    return nn.functional.linear(hidden_state, embedding_weight)
