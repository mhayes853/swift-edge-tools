from __future__ import annotations

import subprocess
import typing
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from coreai.runtime import AIModelAssetMetadata
from coremltools.models import MLModel

from ..cache_layout import decoder_state_shape, empty_decoder_caches
from ..decoder_strategy import DecoderExportStrategy
from ..needle_configuration import NeedleModelConfiguation
from ..needle_torch import Needle
from . import helpers
from .helpers import NeedleCompressor

_COREML_COMPILE_PLATFORMS = {
    "macos": "macOS",
    "ios": "iOS",
    "watchos": "watchOS",
    "tvos": "tvOS",
    "maccatalyst": "macCatalyst",
    "visionos": "visionOS",
}

_NUMPY_DTYPES = {
    torch.int32: np.int32,
    torch.int64: np.int32,
    torch.float16: np.float16,
    torch.float32: np.float32,
}

_DECODER_FLOAT16_INPUTS = (
    "self_attention_mask",
    "cross_attention_mask",
    "encoder_projected_k",
    "encoder_projected_v",
    "key_cache",
    "value_cache",
)

_DECODER_FLOAT16_OUTPUTS = ("updated_key_cache", "updated_value_cache")


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
    source_files, configuration, output_directory = helpers.prepare_export(
        source,
        output_directory,
    )
    needle = helpers.load_needle_model(
        configuration,
        source_files.weights_path,
        encoder_use_native_sdpa=encoder_compute_units.uses_native_sdpa,
        decoder_strategy=resolved_decoder_strategy,
    ).to(dtype=torch.float32)

    models = convert_needle_coreml_models(
        needle,
        configuration,
        compressor=compressor,
        encoder_compute_units=encoder_compute_units,
        decoder_compute_units=decoder_compute_units,
    )
    for name, model in zip(("encoder", "decoder"), models, strict=True):
        _persist_model(
            model,
            output_directory=output_directory,
            name=name,
            metadata=model_metadata,
            compile_platforms=compile_platforms,
        )
    helpers.copy_bundle_resources(source_files, output_directory)
    helpers.write_exported_decoder_length(output_directory, configuration)
    return output_directory


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

    encoder_spec = helpers.encoder_export_spec(
        configuration,
        dynamic_buffers=uses_dynamic_encoder,
    )
    encoder_sample = (
        helpers.sample_encoder_input(
            configuration,
            configuration.encoder_max_length,
        ),
    )
    encoder_model, encoder_outputs = _convert_component(
        needle.encoder,
        encoder_sample,
        spec=encoder_spec,
        compressor=compressor,
        compute_units=encoder_compute_units,
        use_torchscript=uses_dynamic_encoder and compressor is None,
        input_shape_overrides=(
            {
                "input_ids": _coreml_enumerated_shape(
                    [(1, length) for length in sequence_lengths]
                )
            }
            if uses_dynamic_encoder
            else None
        ),
        output_dtype_overrides=dict.fromkeys(encoder_spec.output_names, np.float16),
    )

    decoder_spec = helpers.decoder_export_spec(
        configuration,
        strategy,
        dynamic_buffers=uses_dynamic_encoder,
    )
    decoder_model, _ = _convert_component(
        needle.decoder,
        _decoder_sample(configuration, strategy, encoder_outputs),
        spec=decoder_spec,
        compressor=compressor,
        compute_units=decoder_compute_units,
        use_torchscript=(uses_dynamic_encoder or strategy.stateful)
        and compressor is None,
        input_dtype_overrides=dict.fromkeys(_DECODER_FLOAT16_INPUTS, np.float16),
        input_shape_overrides=_decoder_shape_overrides(
            configuration,
            strategy,
            sequence_lengths,
            uses_dynamic_encoder=uses_dynamic_encoder,
        ),
        output_dtype_overrides=dict.fromkeys(_DECODER_FLOAT16_OUTPUTS, np.float16),
        states=(
            _decoder_states(needle, configuration, decoder_spec)
            if strategy.stateful
            else None
        ),
    )
    return encoder_model, decoder_model


# MARK: - Conversion


def _convert_component(
    module: torch.nn.Module,
    sample: tuple[torch.Tensor, ...],
    *,
    spec: helpers.ModuleExportSpec,
    compressor: NeedleCompressor | None,
    compute_units: CoreMLComputeUnits,
    use_torchscript: bool,
    input_dtype_overrides: Mapping[str, Any] | None = None,
    input_shape_overrides: Mapping[str, Any] | None = None,
    output_dtype_overrides: Mapping[str, Any] | None = None,
    states: list[ct.StateType] | None = None,
) -> tuple[MLModel, tuple[torch.Tensor, ...]]:
    dynamic_shapes = _compression_dynamic_shapes(
        spec.dynamic_shapes,
        compressor=compressor,
    )
    module = helpers.prepare_module_for_export(
        module,
        sample,
        compressor=compressor,
        dynamic_shapes=dynamic_shapes,
    )
    outputs = module(*sample)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    model = typing.cast(
        MLModel,
        ct.convert(
            _coreml_source_model(
                module,
                sample,
                dynamic_shapes=dynamic_shapes,
                use_torchscript=use_torchscript,
            ),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=compute_units.coremltools_value,
            skip_model_load=True,
            inputs=_coreml_tensor_types(
                spec.input_names,
                sample,
                dtype_overrides=input_dtype_overrides,
                shape_overrides=input_shape_overrides,
                with_shapes=True,
            ),
            outputs=_coreml_tensor_types(
                spec.output_names,
                outputs,
                dtype_overrides=output_dtype_overrides,
                with_shapes=False,
            ),
            states=states,
        ),
    )
    return _renaming_states(model, spec.state_names), outputs


def _decoder_sample(
    configuration: NeedleModelConfiguation,
    strategy: DecoderExportStrategy,
    encoder_outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    prefix = helpers.sample_decoder_inputs_from_encoder_outputs(
        configuration,
        typing.cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            encoder_outputs,
        ),
    )
    if not strategy.stateful:
        return (*prefix, *empty_decoder_caches(configuration, dtype=torch.float32))
    return prefix[:4] if strategy.cross_attention_cache_states else prefix


def _decoder_shape_overrides(
    configuration: NeedleModelConfiguation,
    strategy: DecoderExportStrategy,
    sequence_lengths: Sequence[int],
    *,
    uses_dynamic_encoder: bool,
) -> dict[str, Any]:
    overrides = (
        _cross_attention_shape_overrides(
            configuration,
            sequence_lengths,
            ranged=strategy.dynamic_cache,
        )
        if uses_dynamic_encoder
        else {}
    )
    if strategy.dynamic_cache:
        overrides["self_attention_mask"] = (
            1,
            1,
            1,
            _coreml_range(configuration.decoder_max_length),
        )
    return overrides


def _cross_attention_shape_overrides(
    configuration: NeedleModelConfiguation,
    sequence_lengths: Sequence[int],
    *,
    ranged: bool,
) -> dict[str, Any]:
    def projected_shape(length: Any) -> tuple[Any, ...]:
        return (
            configuration.decoder_layers,
            1,
            configuration.kv_heads,
            length,
            configuration.attention_head_dimensions,
        )

    if ranged:
        length = _coreml_range(configuration.encoder_max_length)
        return {
            "cross_attention_mask": (1, 1, 1, length),
            "encoder_projected_k": projected_shape(length),
            "encoder_projected_v": projected_shape(length),
        }
    return {
        "cross_attention_mask": _coreml_enumerated_shape(
            [(1, 1, 1, length) for length in sequence_lengths]
        ),
        "encoder_projected_k": _coreml_enumerated_shape(
            [projected_shape(length) for length in sequence_lengths]
        ),
        "encoder_projected_v": _coreml_enumerated_shape(
            [projected_shape(length) for length in sequence_lengths]
        ),
    }


def _decoder_states(
    needle: Needle,
    configuration: NeedleModelConfiguation,
    spec: helpers.ModuleExportSpec,
) -> list[ct.StateType]:
    return [
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
            spec.state_names,
            needle.decoder.state_buffer_names,
            strict=True,
        )
    ]


def _compression_dynamic_shapes(
    dynamic_shapes: tuple[dict[int, Any], ...],
    *,
    compressor: NeedleCompressor | None,
) -> tuple[dict[int, Any], ...]:
    if compressor is None:
        return dynamic_shapes
    return tuple(
        {
            axis: (
                torch.export.Dim.STATIC
                if dimension is torch.export.Dim.STATIC
                else torch.export.Dim.AUTO
            )
            for axis, dimension in shape.items()
        }
        for shape in dynamic_shapes
    )


def _coreml_source_model(
    module: torch.nn.Module,
    sample: tuple[torch.Tensor, ...],
    *,
    dynamic_shapes: tuple[Any, ...],
    use_torchscript: bool,
) -> Any:
    if use_torchscript:
        return torch.jit.trace(module, sample, strict=False, check_trace=False)
    return helpers.export_program(
        module,
        sample,
        dynamic_shapes=dynamic_shapes,
        decomposition_table={},
    )


# MARK: - Tensor Types


def _coreml_tensor_types(
    names: Sequence[str],
    tensors: Sequence[torch.Tensor],
    *,
    with_shapes: bool,
    dtype_overrides: Mapping[str, Any] | None = None,
    shape_overrides: Mapping[str, Any] | None = None,
) -> list[ct.TensorType]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    dtypes = dtype_overrides or {}
    shapes = shape_overrides or {}
    return [
        ct.TensorType(
            name=name,
            dtype=dtypes.get(name, _numpy_dtype(tensor.dtype)),
            **({"shape": shapes.get(name, tuple(tensor.shape))} if with_shapes else {}),
        )
        for name, tensor in zip(names, tensors, strict=True)
    ]


def _numpy_dtype(dtype: torch.dtype) -> Any:
    try:
        return _NUMPY_DTYPES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported CoreML tensor dtype: {dtype}") from None


def _coreml_sequence_lengths(maximum: int) -> tuple[int, ...]:
    buckets = (128, 256, 512, 1024)
    return tuple(length for length in buckets if length < maximum) + (maximum,)


def _coreml_enumerated_shape(shapes: Sequence[tuple[int, ...]]) -> Any:
    if len(shapes) == 1:
        return shapes[0]
    return ct.EnumeratedShapes(shapes=list(shapes), default=shapes[-1])


def _coreml_range(maximum: int) -> ct.RangeDim:
    return ct.RangeDim(lower_bound=1, upper_bound=maximum, default=maximum)


# MARK: - State Renaming


def _renaming_states(model: MLModel, names: Sequence[str]) -> MLModel:
    # Input and output names already come from the converted tensor types, but
    # CoreMLTools sanitizes nested buffer names in ML Program inputs. Renaming
    # only the state descriptors leaves the runtime state bindings invalid.
    if not names:
        return model
    spec = model.get_spec()
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
    renamed_model = MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
    renamed_model._mil_program = model._mil_program
    return renamed_model


# MARK: - Persistence


def _persist_model(
    model: MLModel,
    *,
    output_directory: Path,
    name: str,
    metadata: AIModelAssetMetadata | None = None,
    compile_platforms: Sequence[str] = (),
) -> None:
    helpers.persist_export_artifact(
        model,
        output_directory=output_directory,
        name=name,
        extension=".mlpackage",
        metadata=metadata,
        compile_platforms=compile_platforms,
        save=_save_model,
        compile_artifact=_compile_model,
    )


def _save_model(
    model: MLModel,
    path: Path,
    metadata: AIModelAssetMetadata | None,
) -> None:
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], check=True)
    _apply_model_metadata(model, metadata).save(str(path))


def _compile_model(
    source_path: Path,
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
                f"Failed to compile CoreML model {source_path.name} "
                f"for {platform}: {output}"
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
