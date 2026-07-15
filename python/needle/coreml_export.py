from __future__ import annotations

import subprocess
import typing
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
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

_COREML_COMPILE_PLATFORMS = {
    "macos": "macOS",
    "ios": "iOS",
    "watchos": "watchOS",
    "tvos": "tvOS",
    "maccatalyst": "macCatalyst",
    "visionos": "visionOS",
}


class CoreMLComputeUnits(str, Enum):
    ALL = "all"
    CPU_ONLY = "cpu-only"
    CPU_AND_GPU = "cpu-and-gpu"
    CPU_AND_NE = "cpu-and-ne"

    @property
    def uses_native_sdpa(self) -> bool:
        return self in {self.CPU_ONLY, self.CPU_AND_GPU}


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
    compute_units: CoreMLComputeUnits = CoreMLComputeUnits.ALL,
    compile_platforms: Sequence[str] = (),
) -> Path:
    compile_platforms = _validated_compile_platforms(compile_platforms)
    source_files, configuration, output_directory = export_helpers.prepare_export(
        source,
        output_directory,
    )
    needle = _load_needle_coreml_model(
        configuration,
        source_files.weights_path,
        use_native_sdpa=compute_units.uses_native_sdpa,
    )

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
        compile_platforms=compile_platforms,
    )
    _persist_model(
        decoder_model,
        output_directory=output_directory,
        name="decoder",
        metadata=model_metadata,
        compile_platforms=compile_platforms,
    )
    export_helpers.copy_bundle_resources(source_files, output_directory)
    return output_directory


def _load_needle_coreml_model(
    configuration: NeedleModelConfiguation,
    weights_path: str | Path,
    *,
    use_native_sdpa: bool,
) -> Needle:
    needle = export_helpers.load_needle_model(
        configuration,
        weights_path,
        use_native_sdpa=use_native_sdpa,
    )
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
    encoder_spec = export_helpers.encoder_export_spec(configuration)
    encoder_sample = (
        export_helpers.sample_encoder_input(
            configuration,
            configuration.encoder_max_length,
        ),
    )
    encoder_module = export_helpers.prepare_module_for_export(
        needle.encoder,
        encoder_sample,
        compressor=compressor,
        dynamic_shapes=encoder_spec.dynamic_shapes,
    )
    encoder_model = typing.cast(
        MLModel,
        ct.convert(
            export_helpers.export_program(
                encoder_module,
                encoder_sample,
                dynamic_shapes=encoder_spec.dynamic_shapes,
                decomposition_table={},
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
            inputs=_coreml_input_tensor_types(encoder_spec.input_names, encoder_sample),
            outputs=_coreml_output_tensor_types(
                encoder_spec.output_names,
                encoder_module(*encoder_sample),
                dtype_overrides=dict.fromkeys(encoder_spec.output_names, np.float16),
            ),
        ),
    )
    encoder_model = _rename_model_features(
        encoder_model,
        input_names=encoder_spec.input_names,
        output_names=encoder_spec.output_names,
    )

    encoder_outputs = encoder_module(*encoder_sample)
    decoder_sample = (
        *export_helpers.sample_decoder_inputs_from_encoder_outputs(
            configuration,
            encoder_outputs,
        ),
        *export_helpers.empty_decoder_caches(configuration, dtype=torch.float32),
    )
    decoder_spec = export_helpers.decoder_export_spec(configuration)
    decoder_module = export_helpers.prepare_module_for_export(
        needle.decoder,
        decoder_sample,
        compressor=compressor,
        dynamic_shapes=decoder_spec.dynamic_shapes,
    )
    decoder_model = typing.cast(
        MLModel,
        ct.convert(
            export_helpers.export_program(
                decoder_module,
                decoder_sample,
                dynamic_shapes=decoder_spec.dynamic_shapes,
                decomposition_table={},
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
            inputs=_coreml_input_tensor_types(
                decoder_spec.input_names,
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
                decoder_spec.output_names,
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
        input_names=decoder_spec.input_names,
        output_names=decoder_spec.output_names,
    )
    return encoder_model, decoder_model


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
    compile_platforms: Sequence[str] = (),
) -> None:
    export_helpers.persist_export_artifact(
        model,
        output_directory=output_directory,
        name=name,
        extension=".mlpackage",
        metadata=metadata,
        compile_platforms=compile_platforms,
        save=lambda artifact, path, metadata: _save_model(
            artifact,
            path,
            metadata=metadata,
        ),
        compile_artifact=lambda source_path, output_directory, platforms: (
            _compile_model(
                source_path,
                output_directory=output_directory,
                platforms=platforms,
            )
        ),
    )


def _save_model(
    model: MLModel,
    path: Path,
    *,
    metadata: AIModelAssetMetadata | None = None,
) -> None:
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], check=True)
    model = _apply_model_metadata(model, metadata)
    model.save(str(path))


def _compile_model(
    source_path: Path,
    *,
    output_directory: Path,
    platforms: Sequence[str],
) -> None:
    for platform in platforms:
        platform_output_directory = output_directory / "compiled" / platform
        platform_output_directory.mkdir(parents=True, exist_ok=True)
        command = [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(source_path),
            str(platform_output_directory),
            "--platform",
            platform,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            output = "\n".join(
                value for value in [error.stdout.strip(), error.stderr.strip()] if value
            )
            raise RuntimeError(
                f"Failed to compile CoreML model {source_path.name} for {platform}: {output}"
            ) from error


def _validated_compile_platforms(platforms: Sequence[str]) -> tuple[str, ...]:
    normalized_platforms = tuple(
        _COREML_COMPILE_PLATFORMS.get(platform.casefold(), "") for platform in platforms
    )
    if invalid_platforms := [
        platform
        for platform, normalized in zip(platforms, normalized_platforms, strict=True)
        if not normalized
    ]:
        supported = ", ".join(_COREML_COMPILE_PLATFORMS.values())
        invalid = ", ".join(invalid_platforms)
        raise ValueError(
            f"Unsupported CoreML compile platform(s): {invalid}. Expected: {supported}"
        )
    if len(set(normalized_platforms)) != len(normalized_platforms):
        raise ValueError("CoreML compile platforms must not contain duplicates")
    return normalized_platforms


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
