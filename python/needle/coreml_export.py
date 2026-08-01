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
from .cache_layout import (  # pyright: ignore[reportMissingImports]
    decoder_state_shape,
)
from .decoder_strategy import (  # pyright: ignore[reportMissingImports]
    DecoderExportStrategy,
)
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

    @property
    def coremltools_value(self) -> ct.ComputeUnit:
        return {
            self.ALL: ct.ComputeUnit.ALL,
            self.CPU_ONLY: ct.ComputeUnit.CPU_ONLY,
            self.CPU_AND_GPU: ct.ComputeUnit.CPU_AND_GPU,
            self.CPU_AND_NE: ct.ComputeUnit.CPU_AND_NE,
        }[self]


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
    compute_units: CoreMLComputeUnits | None = None,
    encoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_NE,
    decoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_GPU,
    decoder_strategy: DecoderExportStrategy | None = None,
    compile_platforms: Sequence[str] = (),
) -> Path:
    compile_platforms = _validated_compile_platforms(compile_platforms)
    resolved_decoder_strategy = decoder_strategy or DecoderExportStrategy.coreml()
    if resolved_decoder_strategy.dynamic_cache and compressor is not None:
        raise ValueError("Dynamic CoreML state does not support compression")
    if compute_units is not None:
        encoder_compute_units = compute_units
        decoder_compute_units = compute_units
    source_files, configuration, output_directory = export_helpers.prepare_export(
        source,
        output_directory,
    )
    needle = _load_needle_coreml_model(
        configuration,
        source_files.weights_path,
        encoder_use_native_sdpa=encoder_compute_units.uses_native_sdpa,
        decoder_strategy=resolved_decoder_strategy,
    )

    encoder_model, decoder_model = convert_needle_coreml_models(
        needle,
        configuration,
        compressor=compressor,
        encoder_compute_units=encoder_compute_units,
        decoder_compute_units=decoder_compute_units,
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
    export_helpers.write_exported_decoder_length(output_directory, configuration)
    return output_directory


def _load_needle_coreml_model(
    configuration: NeedleModelConfiguation,
    weights_path: str | Path,
    *,
    encoder_use_native_sdpa: bool,
    decoder_strategy: DecoderExportStrategy,
) -> Needle:
    needle = export_helpers.load_needle_model(
        configuration,
        weights_path,
        encoder_use_native_sdpa=encoder_use_native_sdpa,
        decoder_strategy=decoder_strategy,
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
    encoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_NE,
    decoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_GPU,
) -> tuple[MLModel, MLModel]:
    strategy = needle.decoder.strategy
    sequence_lengths = _coreml_sequence_lengths(configuration.encoder_max_length)
    uses_dynamic_encoder = len(sequence_lengths) > 1
    encoder_spec = export_helpers.encoder_export_spec(
        configuration,
        dynamic_buffers=uses_dynamic_encoder,
    )
    encoder_shape_overrides = (
        {
            "input_ids": _coreml_enumerated_shape(
                [(1, length) for length in sequence_lengths]
            )
        }
        if uses_dynamic_encoder
        else {}
    )
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
            _coreml_source_model(
                encoder_module,
                encoder_sample,
                dynamic_shapes=encoder_spec.dynamic_shapes,
                use_torchscript=uses_dynamic_encoder and compressor is None,
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=encoder_compute_units.coremltools_value,
            skip_model_load=True,
            inputs=_coreml_input_tensor_types(
                encoder_spec.input_names,
                encoder_sample,
                shape_overrides=encoder_shape_overrides,
            ),
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
    decoder_prefix = export_helpers.sample_decoder_inputs_from_encoder_outputs(
        configuration,
        encoder_outputs,
    )
    uses_cross_attention_states = strategy.cross_attention_cache_states
    decoder_sample: tuple[torch.Tensor, ...] = (
        decoder_prefix[:4] if uses_cross_attention_states else decoder_prefix
    )
    if not strategy.stateful:
        decoder_sample = (
            *decoder_prefix,
            *export_helpers.empty_decoder_caches(
                configuration,
                dtype=torch.float32,
            ),
        )
    decoder_spec = export_helpers.decoder_export_spec(
        configuration,
        strategy,
        dynamic_buffers=uses_dynamic_encoder,
    )
    enumerated_cross_attention_shapes = {
        "cross_attention_mask": _coreml_enumerated_shape(
            [(1, 1, 1, length) for length in sequence_lengths]
        ),
        "encoder_projected_k": _coreml_enumerated_shape(
            [
                (
                    configuration.decoder_layers,
                    1,
                    configuration.kv_heads,
                    length,
                    configuration.attention_head_dimensions,
                )
                for length in sequence_lengths
            ]
        ),
        "encoder_projected_v": _coreml_enumerated_shape(
            [
                (
                    configuration.decoder_layers,
                    1,
                    configuration.kv_heads,
                    length,
                    configuration.attention_head_dimensions,
                )
                for length in sequence_lengths
            ]
        ),
    }
    encoder_range = ct.RangeDim(
        lower_bound=1,
        upper_bound=configuration.encoder_max_length,
        default=configuration.encoder_max_length,
    )
    ranged_cross_attention_shapes = {
        "cross_attention_mask": (1, 1, 1, encoder_range),
        "encoder_projected_k": (
            configuration.decoder_layers,
            1,
            configuration.kv_heads,
            encoder_range,
            configuration.attention_head_dimensions,
        ),
        "encoder_projected_v": (
            configuration.decoder_layers,
            1,
            configuration.kv_heads,
            encoder_range,
            configuration.attention_head_dimensions,
        ),
    }
    cross_attention_shape_overrides = (
        ranged_cross_attention_shapes
        if uses_dynamic_encoder and strategy.dynamic_cache
        else enumerated_cross_attention_shapes
        if uses_dynamic_encoder
        else {}
    )
    cache_shape_overrides = (
        {
            "self_attention_mask": (
                1,
                1,
                1,
                ct.RangeDim(
                    lower_bound=1,
                    upper_bound=configuration.decoder_max_length,
                    default=configuration.decoder_max_length,
                ),
            )
        }
        if strategy.dynamic_cache
        else {}
    )
    decoder_shape_overrides = {
        **cross_attention_shape_overrides,
        **cache_shape_overrides,
    }
    decoder_module = export_helpers.prepare_module_for_export(
        needle.decoder,
        decoder_sample,
        compressor=compressor,
        dynamic_shapes=decoder_spec.dynamic_shapes,
    )
    decoder_outputs = decoder_module(*decoder_sample)
    if isinstance(decoder_outputs, torch.Tensor):
        decoder_outputs = (decoder_outputs,)
    states = (
        [
            ct.StateType(
                wrapped_type=typing.cast(
                    type[Any],
                    ct.TensorType(
                        shape=decoder_state_shape(state_name, configuration),
                        dtype=np.float16,
                    ),
                ),
                name=buffer_name,
            )
            for state_name, buffer_name in zip(
                decoder_spec.state_names,
                needle.decoder.state_buffer_names,
                strict=True,
            )
        ]
        if strategy.stateful
        else None
    )
    decoder_model = typing.cast(
        MLModel,
        ct.convert(
            _coreml_source_model(
                decoder_module,
                decoder_sample,
                dynamic_shapes=decoder_spec.dynamic_shapes,
                use_torchscript=(uses_dynamic_encoder or strategy.stateful)
                and compressor is None,
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=decoder_compute_units.coremltools_value,
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
                shape_overrides=decoder_shape_overrides,
            ),
            outputs=_coreml_output_tensor_types(
                decoder_spec.output_names,
                decoder_outputs,
                dtype_overrides={
                    "updated_key_cache": np.float16,
                    "updated_value_cache": np.float16,
                },
            ),
            states=states,
        ),
    )
    decoder_model = _rename_model_features(
        decoder_model,
        input_names=decoder_spec.input_names,
        output_names=decoder_spec.output_names,
        state_names=decoder_spec.state_names,
    )
    return encoder_model, decoder_model


def _coreml_source_model(
    module: torch.nn.Module,
    sample: tuple[torch.Tensor, ...],
    *,
    dynamic_shapes: tuple[Any, ...],
    use_torchscript: bool,
) -> Any:
    if use_torchscript:
        return torch.jit.trace(
            module,
            sample,
            strict=False,
            check_trace=False,
        )
    return export_helpers.export_program(
        module,
        sample,
        dynamic_shapes=dynamic_shapes,
        decomposition_table={},
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
    shape_overrides: Mapping[str, Any] | None = None,
) -> list[ct.TensorType]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    dtype_overrides = dtype_overrides or {}
    shape_overrides = shape_overrides or {}
    tensor_types = []
    for name, tensor in zip(names, tensors, strict=True):
        arguments = {
            "name": name,
            "dtype": dtype_overrides.get(name, _numpy_dtype(tensor.dtype)),
            "shape": shape_overrides.get(name, tuple(tensor.shape)),
        }
        tensor_types.append(ct.TensorType(**arguments))
    return tensor_types


def _coreml_sequence_lengths(maximum: int) -> tuple[int, ...]:
    buckets = (128, 256, 512, 1024)
    return tuple(length for length in buckets if length < maximum) + (maximum,)


def _coreml_enumerated_shape(shapes: Sequence[tuple[int, ...]]) -> Any:
    if len(shapes) == 1:
        return shapes[0]
    return ct.EnumeratedShapes(shapes=list(shapes), default=shapes[-1])


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
    _rename_state_features(spec, state_names)
    renamed_model = MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
    renamed_model._mil_program = model._mil_program
    return renamed_model


def _rename_feature_collection(features: Sequence[Any], names: Sequence[str]) -> None:
    if len(features) != len(names):
        raise ValueError(f"Expected {len(names)} features, got {len(features)}")
    for feature, name in zip(features, names, strict=True):
        feature.name = name


def _rename_state_features(spec: Any, names: Sequence[str]) -> None:
    # CoreMLTools sanitizes nested buffer names in ML Program inputs. Renaming
    # only the state descriptors leaves the runtime state bindings invalid.
    states = spec.description.state
    if len(states) != len(names):
        raise ValueError(f"Expected {len(names)} states, got {len(states)}")
    for state, name in zip(states, names, strict=True):
        source_name = state.name
        state.name = name
        for function in spec.mlProgram.functions.values():
            for function_input in function.inputs:
                if function_input.name == source_name:
                    function_input.name = name
            for block in function.block_specializations.values():
                for index, output in enumerate(block.outputs):
                    if output == source_name:
                        block.outputs[index] = name
                for operation in block.operations:
                    for argument in operation.inputs.values():
                        for binding in argument.arguments:
                            if binding.HasField("name") and binding.name == source_name:
                                binding.name = name
                    for output in operation.outputs:
                        if output.name == source_name:
                            output.name = name


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
