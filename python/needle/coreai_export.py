from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

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
    source_files, configuration, output_directory = export_helpers.prepare_export(
        source,
        output_directory,
    )
    needle = export_helpers.load_needle_model(
        configuration,
        source_files.weights_path,
        use_native_sdpa=True,
    )

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
    encoder_spec = export_helpers.encoder_export_spec(configuration)
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
    encoder_module = export_helpers.prepare_module_for_export(
        needle.encoder,
        encoder_sample,
        compressor=compressor,
        dynamic_shapes=encoder_spec.dynamic_shapes,
    )

    encoder_program = (
        TorchConverter()
        .add_pytorch_module(
            encoder_module,
            export_fn=lambda module: export_helpers.export_program(
                module,
                encoder_sample,
                dynamic_shapes=encoder_spec.dynamic_shapes,
            ),
            input_names=encoder_spec.input_names,
            output_names=encoder_spec.output_names,
        )
        .to_coreai()
    )
    encoder_program.optimize()

    decoder_spec = export_helpers.decoder_export_spec(configuration)
    decoder_module = export_helpers.prepare_module_for_export(
        needle.decoder,
        decoder_sample,
        compressor=compressor,
        dynamic_shapes=decoder_spec.dynamic_shapes,
    )
    decoder_program = (
        TorchConverter()
        .add_pytorch_module(
            decoder_module,
            export_fn=lambda module: export_helpers.export_program(
                module,
                decoder_sample,
                dynamic_shapes=decoder_spec.dynamic_shapes,
            ),
            input_names=decoder_spec.input_names,
            output_names=decoder_spec.output_names,
        )
        .to_coreai()
    )
    decoder_program.optimize()

    return encoder_program, decoder_program


def _persist_program(
    program: AIProgram,
    *,
    output_directory: Path,
    name: str,
    metadata: AIModelAssetMetadata | None = None,
    compile_platforms: Sequence[str] = (),
) -> None:
    export_helpers.persist_export_artifact(
        program,
        output_directory=output_directory,
        name=name,
        extension=".aimodel",
        metadata=metadata,
        compile_platforms=compile_platforms,
        save=lambda artifact, path, metadata: _save_program(
            artifact,
            path,
            metadata=metadata,
        ),
        compile_artifact=lambda source_path, output_directory, platforms: (
            _compile_model_asset(
                source_path,
                output_directory=output_directory,
                platforms=platforms,
            )
        ),
    )


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
