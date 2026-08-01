from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from . import export_helpers
from .decoder_strategy import (  # pyright: ignore[reportMissingImports]
    DecoderExportStrategy,
)
from .needle_configuration import NeedleModelConfiguation
from .needle_torch import Needle
from .onnx_compression import ONNXCompressor, ONNXModelComponent


def export_needle_onnx(
    source: str,
    output_directory: str | Path,
    *,
    compressor: ONNXCompressor | None = None,
    dtype: str = "float32",
    decoder_strategy: DecoderExportStrategy | None = None,
    opset_version: int = 21,
    external_data: bool = True,
) -> Path:
    source_files, configuration, output_directory = export_helpers.prepare_export(
        source,
        output_directory,
    )
    export_configuration = replace(configuration, dtype=dtype)
    needle = export_helpers.load_needle_model(
        export_configuration,
        source_files.weights_path,
        decoder_strategy=decoder_strategy or DecoderExportStrategy.onnx(),
    )
    convert_needle_onnx_models(
        needle,
        export_configuration,
        output_directory,
        compressor=compressor,
        opset_version=opset_version,
        external_data=external_data,
    )
    export_helpers.copy_bundle_resources(source_files, output_directory)
    export_helpers.write_exported_decoder_length(
        output_directory,
        export_configuration,
    )
    _write_exported_configuration_dtype(output_directory, dtype=dtype)
    return output_directory


def _write_exported_configuration_dtype(
    output_directory: Path,
    *,
    dtype: str,
) -> None:
    configuration_path = output_directory / "configuration.json"
    try:
        configuration = json.loads(configuration_path.read_text())
        configuration["dtype"] = dtype
        configuration["torch_dtype"] = dtype
        configuration_path.write_text(json.dumps(configuration))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Failed to update exported configuration {configuration_path}: {error}"
        ) from error


def convert_needle_onnx_models(
    needle: Needle,
    configuration: NeedleModelConfiguation,
    output_directory: str | Path,
    *,
    compressor: ONNXCompressor | None = None,
    opset_version: int = 21,
    external_data: bool = True,
) -> tuple[Path, Path]:
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    encoder_spec = export_helpers.encoder_export_spec(
        configuration,
        dynamic_buffers=True,
    )
    encoder_sample = (
        export_helpers.sample_encoder_input(
            configuration,
            configuration.encoder_max_length,
        ),
    )
    decoder_spec = export_helpers.decoder_export_spec(
        configuration,
        needle.decoder.strategy,
        dynamic_buffers=True,
    )
    decoder_sample = (
        *export_helpers.sample_decoder_inputs(needle, configuration),
        *export_helpers.empty_decoder_caches(
            configuration,
            dtype=next(needle.parameters()).dtype,
        ),
    )

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary_directory = Path(temporary_name)
        encoder_path = _export_onnx_component(
            needle.encoder,
            encoder_sample,
            spec=encoder_spec,
            component="encoder",
            temporary_directory=temporary_directory,
            output_directory=output_directory,
            compressor=compressor,
            opset_version=opset_version,
            external_data=external_data,
        )
        decoder_path = _export_onnx_component(
            needle.decoder,
            decoder_sample,
            spec=decoder_spec,
            component="decoder",
            temporary_directory=temporary_directory,
            output_directory=output_directory,
            compressor=compressor,
            opset_version=opset_version,
            external_data=external_data,
        )

    return encoder_path, decoder_path


def _export_onnx_component(
    module: torch.nn.Module,
    sample_args: tuple[torch.Tensor, ...],
    *,
    spec: export_helpers.ModuleExportSpec,
    component: ONNXModelComponent,
    temporary_directory: Path,
    output_directory: Path,
    compressor: ONNXCompressor | None,
    opset_version: int,
    external_data: bool,
) -> Path:
    destination = output_directory / f"{component}.onnx"
    export_path = (
        destination if compressor is None else temporary_directory / destination.name
    )
    module.eval()
    torch.onnx.export(
        module,
        sample_args,
        export_path,
        dynamo=True,
        input_names=spec.input_names,
        output_names=spec.output_names,
        dynamic_shapes=spec.dynamic_shapes,
        opset_version=opset_version,
        external_data=external_data,
    )
    if compressor is not None:
        compressor.compress(export_path, destination, component=component)
    return destination
