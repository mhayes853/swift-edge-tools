from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import needle.export_helpers as export_helpers
import torch
from coreai.authoring.asset import AIProgram
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter

from . import Needle, NeedleModelConfiguation
from .needle_compression import NeedleCompressor


def export_needle_coreai(
    source: str,
    output_directory: str | Path,
    *,
    compressor: NeedleCompressor | None = None,
    model_metadata: AIModelAssetMetadata | None = None,
    compile_platforms: Sequence[str] = (),
) -> Path:
    source_files = export_helpers.resolve_model_source(source)
    configuration = export_helpers.load_configuration(source_files.configuration_path)
    needle = export_helpers.load_needle_model(
        configuration,
        source_files.weights_path,
        use_native_sdpa=True,
    )

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    encoder_program, decoder_program = convert_needle_coreai_programs(
        needle,
        configuration,
        compressor=compressor,
    )
    _persist_program(
        encoder_program,
        output_directory=output_directory,
        name="encoder",
        metadata=model_metadata,
        compile_platforms=compile_platforms,
    )
    _persist_program(
        decoder_program,
        output_directory=output_directory,
        name="decoder",
        metadata=model_metadata,
        compile_platforms=compile_platforms,
    )
    export_helpers.copy_bundle_resources(source_files, output_directory)
    return output_directory


def convert_needle_coreai_programs(
    needle: Needle,
    configuration: NeedleModelConfiguation,
    *,
    compressor: NeedleCompressor | None = None,
) -> tuple[AIProgram, AIProgram]:
    encoder_sample = (
        export_helpers.sample_encoder_input(
            configuration,
            configuration.encoder_max_length,
        ),
    )
    decoder_sample = (
        *export_helpers.sample_decoder_inputs(needle, configuration),
        *export_helpers.empty_decoder_caches(configuration, dtype=torch.float32),
    )
    encoder_shapes = export_helpers.encoder_dynamic_shapes(configuration)
    encoder_module = _prepare_module_for_coreai_export(
        needle.encoder,
        encoder_sample,
        compressor=compressor,
        dynamic_shapes=encoder_shapes,
    )

    encoder_program = (
        TorchConverter()
        .add_pytorch_module(
            encoder_module,
            export_fn=lambda module: export_helpers.export_program(
                module,
                encoder_sample,
                dynamic_shapes=encoder_shapes,
            ),
            input_names=["input_ids"],
            output_names=[
                "cross_attention_mask",
                "encoder_projected_k",
                "encoder_projected_v",
            ],
        )
        .to_coreai()
    )
    encoder_program.optimize()

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
    decoder_module = _prepare_module_for_coreai_export(
        needle.decoder,
        decoder_sample,
        compressor=compressor,
        dynamic_shapes=decoder_shapes,
    )
    decoder_program = (
        TorchConverter()
        .add_pytorch_module(
            decoder_module,
            export_fn=lambda module: export_helpers.export_program(
                module,
                decoder_sample,
                dynamic_shapes=decoder_shapes,
            ),
            input_names=[
                "input_ids",
                "cache_position",
                "self_attention_mask",
                "cross_attention_mask",
                "encoder_projected_k",
                "encoder_projected_v",
                "key_cache",
                "value_cache",
            ],
            output_names=["logits", "updated_key_cache", "updated_value_cache"],
        )
        .to_coreai()
    )
    decoder_program.optimize()

    return encoder_program, decoder_program


def _prepare_module_for_coreai_export(
    module: torch.nn.Module,
    sample_args: tuple[torch.Tensor, ...],
    *,
    compressor: NeedleCompressor | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
) -> torch.nn.Module:
    if compressor is None:
        return module
    return compressor.compress(module, sample_args, dynamic_shapes=dynamic_shapes)


def _persist_program(
    program: AIProgram,
    *,
    output_directory: Path,
    name: str,
    metadata: AIModelAssetMetadata | None = None,
    compile_platforms: Sequence[str] = (),
) -> None:
    if compile_platforms:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_asset_path = Path(temporary_directory) / f"{name}.aimodel"
            _save_program(program, source_asset_path, metadata=metadata)
            _compile_model_asset(
                source_asset_path,
                output_directory=output_directory,
                platforms=compile_platforms,
            )
        return

    _save_program(program, output_directory / f"{name}.aimodel", metadata=metadata)


def _save_program(
    program: AIProgram,
    path: Path,
    *,
    metadata: AIModelAssetMetadata | None = None,
) -> None:
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], check=True)
    program.save_asset(path, metadata=metadata)


def _compile_model_asset(
    source_asset_path: Path,
    *,
    output_directory: Path,
    platforms: Sequence[str],
) -> None:
    command = [
        "xcrun",
        "coreai-build",
        "compile",
        str(source_asset_path),
        "--output",
        str(output_directory),
    ]
    for platform in platforms:
        command.extend(["--platform", platform])

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        output = "\n".join(
            value for value in [error.stdout.strip(), error.stderr.strip()] if value
        )
        raise RuntimeError(
            f"Failed to compile CoreAI model asset {source_asset_path.name}: {output}"
        ) from error
