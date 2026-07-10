from __future__ import annotations

import subprocess
import typing
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import coremltools as ct
import needle.export_helpers as export_helpers
import numpy as np
import torch
from coreai.runtime import AIModelAssetMetadata
from coremltools.models import MLModel

from . import Needle, NeedleDecoder, NeedleModelConfiguation
from .needle_compression import NeedleCompressor

_ENCODER_INPUT_NAMES = ["input_ids"]
_ENCODER_OUTPUT_NAMES = [
    "cross_attention_mask",
    "encoder_projected_k",
    "encoder_projected_v",
]
_DECODER_INPUT_NAMES = [
    "input_ids",
    "cache_position",
    "self_attention_mask",
    "cross_attention_mask",
    "encoder_projected_k",
    "encoder_projected_v",
    "key_cache",
    "value_cache",
]
_DECODER_OUTPUT_NAMES = ["logits", "updated_key_cache", "updated_value_cache"]


class _CoreMLDecoder(torch.nn.Module):
    def __init__(self, decoder: NeedleDecoder):
        super().__init__()
        self.decoder = decoder

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
        return self.decoder.forward_with_cache(
            input_ids,
            cache_position,
            self_attention_mask,
            cross_attention_mask,
            encoder_projected_k,
            encoder_projected_v,
            key_cache,
            value_cache,
        )


def coreml_operation_histogram(model: MLModel) -> Counter[str]:
    """Counts operations in a converted model, including operations in nested blocks."""
    histogram = Counter[str]()

    def count_operations(operations: Sequence[Any]) -> None:
        for operation in operations:
            histogram[operation.op_type] += 1
            for block in operation.blocks:
                count_operations(block.operations)

    program = model._mil_program
    if program is None:
        raise ValueError("CoreML model does not retain its MIL program")
    for function in program.functions.values():
        count_operations(function.operations)
    return histogram


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
            configuration.encoder_max_length,
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
                dtype_overrides=dict.fromkeys(_ENCODER_OUTPUT_NAMES, np.float16),
            ),
        ),
    )
    encoder_model = _rename_model_features(
        encoder_model,
        input_names=_ENCODER_INPUT_NAMES,
        output_names=_ENCODER_OUTPUT_NAMES,
    )

    encoder_outputs = encoder_module(*encoder_sample)
    decoder_sample = (
        *export_helpers.sample_decoder_inputs_from_encoder_outputs(
            configuration,
            encoder_outputs,
        ),
        *export_helpers.empty_decoder_caches(configuration, dtype=torch.float32),
    )
    decoder_shapes = (
        *export_helpers.decoder_dynamic_shapes(configuration),
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.STATIC,
        },
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.STATIC,
        },
    )
    decoder_module = _prepare_module_for_coreml_export(
        _CoreMLDecoder(needle.decoder),
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
            inputs=_coreml_input_tensor_types(
                _DECODER_INPUT_NAMES,
                decoder_sample,
                dtype_overrides=dict.fromkeys(
                    (
                        "self_attention_mask",
                        "cross_attention_mask",
                        "encoder_projected_k",
                        "encoder_projected_v",
                        "key_cache",
                        "value_cache",
                    ),
                    np.float16,
                ),
            ),
            outputs=_coreml_output_tensor_types(
                _DECODER_OUTPUT_NAMES,
                decoder_module(*decoder_sample),
                dtype_overrides={
                    "updated_key_cache": np.float16,
                    "updated_value_cache": np.float16,
                },
            ),
        ),
    )
    decoder_model = _rename_model_features(
        decoder_model,
        input_names=_DECODER_INPUT_NAMES,
        output_names=_DECODER_OUTPUT_NAMES,
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
    *,
    dtype_overrides: Mapping[str, Any] | None = None,
) -> list[ct.TensorType]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    dtype_overrides = dtype_overrides or {}
    return [
        ct.TensorType(
            name=name,
            dtype=dtype_overrides.get(name, _numpy_dtype(tensor.dtype)),
        )
        for name, tensor in zip(names, tensors, strict=True)
    ]


def _coreml_output_tensor_types(
    names: Sequence[str],
    tensors: Sequence[torch.Tensor],
    *,
    dtype_overrides: Mapping[str, Any] | None = None,
) -> list[ct.TensorType]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    dtype_overrides = dtype_overrides or {}
    return [
        ct.TensorType(
            name=name,
            dtype=dtype_overrides.get(name, _numpy_dtype(tensor.dtype)),
        )
        for name, tensor in zip(names, tensors, strict=True)
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
    renamed_model = MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
    renamed_model._mil_program = model._mil_program
    return renamed_model


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
