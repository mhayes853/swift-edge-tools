from __future__ import annotations

import subprocess
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from coreai.runtime import AIModelAssetMetadata
from coremltools.models import MLModel

import needle.export_helpers as export_helpers

from . import Needle, NeedleModelConfiguation
from .needle_compression import NeedleCompressor
from .needle_torch import NeedleDecoder, _RoPEFrequencies, _project_logits
from .needle_torch_helpers import forward_attention_with_kv_cache, forward_decoder

_ENCODER_INPUT_NAMES = ["input_ids"]
_ENCODER_OUTPUT_NAMES = [
    "cross_attention_mask",
    "encoder_projected_k",
    "encoder_projected_v",
]
_DECODER_INPUT_NAMES = [
    "input_ids",
    "cross_attention_mask",
    "encoder_projected_k",
    "encoder_projected_v",
]
_DECODER_OUTPUT_NAMES = ["logits"]
_DECODER_STATE_NAMES = ["keyCache", "valueCache", "cacheOffset"]


class _CoreMLEncoderWrapper(torch.nn.Module):
    def __init__(self, needle: Needle):
        super().__init__()
        self.configuration = needle.configuration
        self.model: Any = needle.model
        self.model_dtype = typing.cast(
            torch.Tensor, needle.model.embed_tokens.weight
        ).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_output = self.model.encode(input_ids, mask=None)
        encoder_projected_k, encoder_projected_v = self.model.project_encoder_kv(
            encoder_output
        )
        cross_attention_mask = torch.zeros(
            (1, 1, 1, input_ids.shape[1]),
            device=encoder_output.device,
            dtype=self.model_dtype,
        )
        return (
            cross_attention_mask.to(dtype=torch.float16),
            encoder_projected_k.to(dtype=torch.float16),
            encoder_projected_v.to(dtype=torch.float16),
        )


class _CoreMLDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: NeedleDecoder):
        super().__init__()
        self.configuration = decoder.configuration
        self.model: Any = decoder.model
        self.lm_head: Any = decoder.lm_head
        self.embedding_weight = typing.cast(
            torch.Tensor, decoder.model.embed_tokens.weight
        )
        self.register_buffer(
            "keyCache",
            torch.zeros(
                typing.cast(torch.Tensor, decoder.k_cache).shape,
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "valueCache",
            torch.zeros(
                typing.cast(torch.Tensor, decoder.v_cache).shape,
                dtype=torch.float16,
            ),
        )
        self.register_buffer("cacheOffset", torch.zeros((1,), dtype=torch.float16))

    def reset(self) -> None:
        typing.cast(torch.Tensor, self.keyCache).zero_()
        typing.cast(torch.Tensor, self.valueCache).zero_()
        typing.cast(torch.Tensor, self.cacheOffset).zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
    ) -> torch.Tensor:
        model_dtype = self.embedding_weight.dtype
        key_cache = typing.cast(torch.Tensor, self.keyCache).to(dtype=model_dtype)
        value_cache = typing.cast(torch.Tensor, self.valueCache).to(dtype=model_dtype)
        cache_offset = typing.cast(torch.Tensor, self.cacheOffset).to(dtype=torch.int32)

        hidden_state = self._decode_with_coreml_cache(
            input_ids=input_ids,
            cross_attention_mask=cross_attention_mask.to(dtype=model_dtype),
            encoder_projected_k=encoder_projected_k.to(dtype=model_dtype),
            encoder_projected_v=encoder_projected_v.to(dtype=model_dtype),
            key_cache=key_cache,
            value_cache=value_cache,
            cache_offset=cache_offset,
        )
        typing.cast(torch.Tensor, self.keyCache).copy_(
            key_cache.to(dtype=typing.cast(torch.Tensor, self.keyCache).dtype)
        )
        typing.cast(torch.Tensor, self.valueCache).copy_(
            value_cache.to(dtype=typing.cast(torch.Tensor, self.valueCache).dtype)
        )
        typing.cast(torch.Tensor, self.cacheOffset).copy_(
            (cache_offset + torch.full_like(cache_offset, input_ids.shape[1])).to(
                dtype=typing.cast(torch.Tensor, self.cacheOffset).dtype
            )
        )
        return _project_logits(
            hidden_state,
            self.lm_head,
            self.embedding_weight,
        ).to(dtype=torch.float16)

    def _decode_with_coreml_cache(
        self,
        *,
        input_ids: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor,
        encoder_projected_v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_offset: torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self.model.embed_tokens(input_ids) * self.model.embed_scale
        decoder = self.model.decoder
        return forward_decoder(
            decoder=decoder,
            x=hidden_state,
            cross_mask=cross_attention_mask,
            encoder_projected_k=encoder_projected_k,
            encoder_projected_v=encoder_projected_v,
            k_cache=key_cache,
            v_cache=value_cache,
            cache_offset=cache_offset,
            rope_frequencies=_RoPEFrequencies,
            self_attention_adapter=self._forward_layer_self_attention,
        )

    def _forward_layer_self_attention(
        self,
        layer: typing.Any,
        q: torch.Tensor,
        mask: torch.Tensor,
        rope: _RoPEFrequencies,
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
            update_kv_cache=lambda layer_k_cache, layer_v_cache, projected_k, projected_v, cache_offset: (
                self._update_kv_cache(
                    layer_k_cache=layer_k_cache,
                    layer_v_cache=layer_v_cache,
                    projected_k=projected_k,
                    projected_v=projected_v,
                    cache_offset=cache_offset,
                )
            ),
        )

    def _update_kv_cache(
        self,
        *,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        projected_k: torch.Tensor,
        projected_v: torch.Tensor,
        cache_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_k = layer_k_cache.squeeze(0)
        cache_v = layer_v_cache.squeeze(0)
        update_k = projected_k.squeeze(0)
        update_v = projected_v.squeeze(0)

        seq_len = update_k.shape[1]
        positions = torch.arange(
            cache_k.shape[1],
            device=update_k.device,
            dtype=cache_offset.dtype,
        )
        token_positions = cache_offset.to(device=update_k.device) + torch.arange(
            seq_len,
            device=update_k.device,
            dtype=cache_offset.dtype,
        )
        scatter = positions[None, :] == token_positions[:, None]
        scatter_values = scatter.to(dtype=update_k.dtype)
        write_mask = scatter.any(dim=0)[None, :, None]

        expanded_k = torch.einsum("hsd,st->htd", update_k, scatter_values)
        expanded_v = torch.einsum("hsd,st->htd", update_v, scatter_values)
        updated_k = torch.where(write_mask, expanded_k, cache_k).unsqueeze(0)
        updated_v = torch.where(write_mask, expanded_v, cache_v).unsqueeze(0)
        return updated_k, updated_v


def export_needle_coreml(
    source: str,
    output_directory: str | Path,
    *,
    compressor: NeedleCompressor | None = None,
    model_metadata: AIModelAssetMetadata | None = None,
) -> Path:
    source_files = export_helpers.resolve_model_source(source)
    configuration = export_helpers.load_configuration(source_files.configuration_path)
    needle = _load_needle_coreml_model(
        configuration,
        source_files.weights_path,
    )

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    encoder_model, decoder_model = convert_needle_coreml_models(
        needle,
        configuration,
        compressor=compressor,
    )
    _persist_model(
        encoder_model,
        output_directory=output_directory,
        name="encoder",
        metadata=model_metadata,
    )
    _persist_model(
        decoder_model,
        output_directory=output_directory,
        name="decoder",
        metadata=model_metadata,
    )
    export_helpers.copy_bundle_resources(source_files, output_directory)
    return output_directory


def _load_needle_coreml_model(
    configuration: NeedleModelConfiguation,
    weights_path: str | Path,
) -> Needle:
    needle = export_helpers.load_needle_model(configuration, weights_path)
    needle = needle.to(dtype=torch.float32)
    needle.encoder.to(dtype=torch.float32)
    needle.decoder.to(dtype=torch.float32)
    return needle


def convert_needle_coreml_models(
    needle: Needle,
    configuration: NeedleModelConfiguation,
    *,
    compressor: NeedleCompressor | None = None,
) -> tuple[MLModel, MLModel]:
    encoder_sample = (
        export_helpers.sample_encoder_input(
            configuration,
            export_helpers.DEFAULT_ENCODER_SAMPLE_LENGTH,
        ),
    )
    encoder_shapes = export_helpers.encoder_dynamic_shapes(configuration)
    encoder_module = _prepare_module_for_coreml_export(
        _CoreMLEncoderWrapper(needle),
        encoder_sample,
        compressor=compressor,
        dynamic_shapes=encoder_shapes,
    )
    encoder_model = typing.cast(
        MLModel,
        ct.convert(
            export_helpers.export_program(
                encoder_module,
                encoder_sample,
                dynamic_shapes=encoder_shapes,
                decomposition_table={},
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
            inputs=_coreml_input_tensor_types(_ENCODER_INPUT_NAMES, encoder_sample),
            outputs=_coreml_output_tensor_types(
                _ENCODER_OUTPUT_NAMES,
                encoder_module(*encoder_sample),
            ),
        ),
    )
    encoder_model = _rename_model_features(
        encoder_model,
        input_names=_ENCODER_INPUT_NAMES,
        output_names=_ENCODER_OUTPUT_NAMES,
    )

    needle.decoder.reset()
    encoder_outputs = encoder_module(*encoder_sample)
    decoder_sample = (
        torch.full(
            (1, export_helpers.DEFAULT_DECODER_SAMPLE_LENGTH),
            fill_value=configuration.decoder_start_token_id,
            dtype=torch.long,
        ),
        *encoder_outputs,
    )
    decoder_shapes = export_helpers.decoder_dynamic_shapes()
    decoder_module = _prepare_module_for_coreml_export(
        _CoreMLDecoderWrapper(needle.decoder),
        decoder_sample,
        compressor=compressor,
        dynamic_shapes=decoder_shapes,
    )
    decoder_model = typing.cast(
        MLModel,
        ct.convert(
            export_helpers.export_program(
                decoder_module,
                decoder_sample,
                dynamic_shapes=decoder_shapes,
                decomposition_table={},
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
            inputs=_coreml_input_tensor_types(_DECODER_INPUT_NAMES, decoder_sample),
            outputs=_coreml_output_tensor_types(
                _DECODER_OUTPUT_NAMES,
                (decoder_module(*decoder_sample),),
            ),
            states=_coreml_state_types(
                typing.cast(_CoreMLDecoderWrapper, decoder_module)
            ),
        ),
    )
    decoder_model = _rename_model_features(
        decoder_model,
        input_names=_DECODER_INPUT_NAMES,
        output_names=_DECODER_OUTPUT_NAMES,
        state_names=_DECODER_STATE_NAMES,
    )
    return encoder_model, decoder_model


def _prepare_module_for_coreml_export(
    module: torch.nn.Module,
    sample_args: tuple[torch.Tensor, ...],
    *,
    compressor: NeedleCompressor | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
) -> torch.nn.Module:
    if compressor is None:
        return module
    return compressor.compress(
        module,
        sample_args,
        dynamic_shapes=dynamic_shapes,
    )


def _numpy_dtype(dtype: torch.dtype) -> Any:
    if dtype in (torch.int32,):
        return np.int32
    if dtype in (torch.int64, torch.long):
        return np.int32
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    raise ValueError(f"Unsupported CoreML tensor dtype: {dtype}")


def _coreml_input_tensor_types(
    names: Sequence[str],
    tensors: Sequence[torch.Tensor],
) -> list[ct.TensorType]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    return [
        ct.TensorType(
            name=name,
            dtype=_numpy_dtype(tensor.dtype),
        )
        for name, tensor in zip(names, tensors, strict=True)
    ]


def _coreml_output_tensor_types(
    names: Sequence[str],
    tensors: Sequence[torch.Tensor],
) -> list[ct.TensorType]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    return [
        ct.TensorType(
            name=name,
            dtype=_numpy_dtype(tensor.dtype),
        )
        for name, tensor in zip(names, tensors, strict=True)
    ]


def _coreml_state_types(module: _CoreMLDecoderWrapper) -> list[ct.StateType]:
    key_cache = typing.cast(torch.Tensor, module.keyCache)
    value_cache = typing.cast(torch.Tensor, module.valueCache)
    cache_offset = typing.cast(torch.Tensor, module.cacheOffset)
    return [
        ct.StateType(
            name="keyCache",
            wrapped_type=typing.cast(
                Any,
                ct.TensorType(
                    shape=tuple(key_cache.shape),
                    dtype=_numpy_dtype(key_cache.dtype),
                ),
            ),
        ),
        ct.StateType(
            name="valueCache",
            wrapped_type=typing.cast(
                Any,
                ct.TensorType(
                    shape=tuple(value_cache.shape),
                    dtype=_numpy_dtype(value_cache.dtype),
                ),
            ),
        ),
        ct.StateType(
            name="cacheOffset",
            wrapped_type=typing.cast(
                Any,
                ct.TensorType(
                    shape=tuple(cache_offset.shape),
                    dtype=_numpy_dtype(cache_offset.dtype),
                ),
            ),
        ),
    ]


def _rename_model_features(
    model: MLModel,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    state_names: Sequence[str] = (),
) -> MLModel:
    spec = model.get_spec()
    _rename_feature_collection(spec.description.input, input_names)
    _rename_feature_collection(spec.description.output, output_names)
    _rename_feature_collection(spec.description.state, state_names)
    return MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)


def _rename_feature_collection(features: Sequence[Any], names: Sequence[str]) -> None:
    if len(features) != len(names):
        raise ValueError(f"Expected {len(names)} features, got {len(features)}")
    for feature, name in zip(features, names, strict=True):
        feature.name = name


def _persist_model(
    model: MLModel,
    *,
    output_directory: Path,
    name: str,
    metadata: AIModelAssetMetadata | None = None,
) -> None:
    path = output_directory / f"{name}.mlpackage"
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], check=True)
    model = _apply_model_metadata(model, metadata)
    model.save(str(path))


def _apply_model_metadata(
    model: MLModel,
    metadata: AIModelAssetMetadata | None,
) -> MLModel:
    if metadata is None:
        return model
    model.author = metadata.author
    model.short_description = metadata.model_description
    model.license = metadata.license
    model.user_defined_metadata.update(
        dict(getattr(metadata, "creator_defined_metadata", {}))
    )
    return model
