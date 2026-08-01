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
from .decoder_strategy import (  # pyright: ignore[reportMissingImports]
    DecoderExportStrategy,
)
from .needle_compression import NeedleCompressor


class _CoreAIEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cross_attention_mask, encoder_projected_k, encoder_projected_v = self.encoder(
            input_ids
        )
        return (
            cross_attention_mask.to(dtype=torch.float16),
            encoder_projected_k.to(dtype=torch.float16),
            encoder_projected_v.to(dtype=torch.float16),
        )


class _CoreAIDecoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.decoder = decoder
        self.dtype = dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        encoder_projected_k: torch.Tensor | None = None,
        encoder_projected_v: torch.Tensor | None = None,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_k = (
            encoder_projected_k.to(dtype=self.dtype)
            if encoder_projected_k is not None
            else None
        )
        projected_v = (
            encoder_projected_v.to(dtype=self.dtype)
            if encoder_projected_v is not None
            else None
        )
        return self.decoder(
            input_ids,
            cache_position,
            self_attention_mask.to(dtype=self.dtype),
            cross_attention_mask.to(dtype=self.dtype),
            projected_k,
            projected_v,
            key_cache,
            value_cache,
        )


def export_needle_coreai(
    source: str,
    output_directory: str | Path,
    *,
    compressor: NeedleCompressor | None = None,
    model_metadata: AIModelAssetMetadata | None = None,
    decoder_strategy: DecoderExportStrategy | None = None,
    compile_platforms: Sequence[str] = (),
) -> Path:
    source_files, configuration, output_directory = export_helpers.prepare_export(
        source,
        output_directory,
    )
    resolved_decoder_strategy = decoder_strategy or DecoderExportStrategy.coreai()
    needle = export_helpers.load_needle_model(
        configuration,
        source_files.weights_path,
        encoder_use_native_sdpa=True,
        decoder_strategy=resolved_decoder_strategy,
    )
    needle = needle.to(dtype=torch.float16)

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
    export_helpers.write_exported_decoder_length(output_directory, configuration)
    return output_directory


def convert_needle_coreai_programs(
    needle: Needle,
    configuration: NeedleModelConfiguation,
    *,
    compressor: NeedleCompressor | None = None,
) -> tuple[AIProgram, AIProgram]:
    strategy = needle.decoder.strategy
    model_dtype = next(needle.parameters()).dtype
    uses_dynamic_buffers = True
    encoder_spec = export_helpers.encoder_export_spec(
        configuration,
        dynamic_buffers=uses_dynamic_buffers,
    )
    encoder_sample = (
        export_helpers.sample_encoder_input(
            configuration,
            configuration.encoder_max_length,
        ),
    )
    encoder_module = export_helpers.prepare_module_for_export(
        _CoreAIEncoder(needle.encoder),
        encoder_sample,
        compressor=compressor,
        dynamic_shapes=encoder_spec.dynamic_shapes,
    )

    with torch.no_grad():
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
                dtype=model_dtype,
            ),
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

    decoder_spec = export_helpers.decoder_export_spec(
        configuration,
        strategy,
        dynamic_buffers=uses_dynamic_buffers,
    )
    decoder_module = export_helpers.prepare_module_for_export(
        _CoreAIDecoder(needle.decoder, dtype=model_dtype),
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
            state_names=decoder_spec.state_names,
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
