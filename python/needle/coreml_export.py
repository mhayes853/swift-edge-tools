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
from .needle_torch import NeedleDecoder, _project_logits

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
        self.register_buffer("cacheOffset", torch.zeros((), dtype=torch.float16))

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

        hidden_state = self.model.decode(
            input_ids,
            cross_mask=cross_attention_mask,
            encoder_projected_k=encoder_projected_k.to(dtype=model_dtype),
            encoder_projected_v=encoder_projected_v.to(dtype=model_dtype),
            k_cache=key_cache,
            v_cache=value_cache,
            cache_offset=cache_offset,
        )
        typing.cast(torch.Tensor, self.keyCache).copy_(
            key_cache.to(dtype=typing.cast(torch.Tensor, self.keyCache).dtype)
        )
        typing.cast(torch.Tensor, self.valueCache).copy_(
            value_cache.to(dtype=typing.cast(torch.Tensor, self.valueCache).dtype)
        )
        typing.cast(torch.Tensor, self.cacheOffset).copy_(
            (cache_offset + input_ids.shape[1]).to(
                dtype=typing.cast(torch.Tensor, self.cacheOffset).dtype
            )
        )
        return _project_logits(hidden_state, self.lm_head, self.embedding_weight)


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
    if next(needle.parameters()).dtype == torch.bfloat16:
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
        needle.encoder,
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
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        ),
    )
    encoder_model = _rename_model_features(
        encoder_model,
        input_names=_ENCODER_INPUT_NAMES,
        output_names=_ENCODER_OUTPUT_NAMES,
    )

    needle.decoder.reset()
    decoder_sample = export_helpers.sample_decoder_inputs(needle, configuration)
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
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
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


def _coreml_state_types(module: _CoreMLDecoderWrapper) -> list[ct.StateType]:
    key_cache = typing.cast(torch.Tensor, module.keyCache)
    value_cache = typing.cast(torch.Tensor, module.valueCache)
    cache_offset = typing.cast(torch.Tensor, module.cacheOffset)
    return [
        ct.StateType(
            name="keyCache",
            wrapped_type=typing.cast(
                Any,
                ct.TensorType(shape=tuple(key_cache.shape), dtype=np.float16),
            ),
        ),
        ct.StateType(
            name="valueCache",
            wrapped_type=typing.cast(
                Any,
                ct.TensorType(
                    shape=tuple(value_cache.shape),
                    dtype=np.float16,
                ),
            ),
        ),
        ct.StateType(
            name="cacheOffset",
            wrapped_type=typing.cast(
                Any,
                ct.TensorType(
                    shape=tuple(cache_offset.shape),
                    dtype=np.float16,
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
