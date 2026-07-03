from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from coreai.authoring.asset import AIProgram
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from huggingface_hub import snapshot_download

from . import Needle, NeedleModelConfiguation
from .needle_compression import NeedleCompressor
from .torch_utils import load_state_dict, torch_dtype

DEFAULT_SOURCE = "Cactus-Compute/needle-hf"
_CONFIG_FILENAMES = ("configuration.json", "config.json")
_TOKENIZER_FILENAMES = ("tokenizer.model", "tokenizer.json")
_DEFAULT_ENCODER_SAMPLE_LENGTH = 4
_DEFAULT_DECODER_SAMPLE_LENGTH = 1


def export_needle_coreai(
    source: str,
    output_directory: Union[str, Path],
    *,
    compressor: NeedleCompressor | None = None,
    model_metadata: AIModelAssetMetadata | None = None,
) -> Path:
    source_files = _resolve_model_source(source)
    configuration = _load_configuration(source_files.configuration_path)
    needle = _load_needle_model(configuration, source_files.weights_path)

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    encoder_program, decoder_program = convert_needle_coreai_programs(
        needle,
        configuration,
        compressor=compressor,
    )
    _save_program(
        encoder_program,
        output_directory / "encoder.aimodel",
        metadata=model_metadata,
    )
    _save_program(
        decoder_program,
        output_directory / "decoder.aimodel",
        metadata=model_metadata,
    )
    _copy_bundle_resources(source_files, output_directory)
    return output_directory


def convert_needle_coreai_programs(
    needle: Needle,
    configuration: NeedleModelConfiguation,
    *,
    compressor: NeedleCompressor | None = None,
) -> tuple[AIProgram, AIProgram]:
    encoder_sample = (
        _sample_encoder_input(configuration, _DEFAULT_ENCODER_SAMPLE_LENGTH),
    )
    decoder_sample = _sample_decoder_inputs(needle, configuration)
    encoder_module = _prepare_module_for_coreai_export(
        needle.encoder,
        encoder_sample,
        compressor=compressor,
    )

    encoder_program = (
        TorchConverter()
        .add_pytorch_module(
            encoder_module,
            export_fn=lambda module: _export_program(module, encoder_sample),
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

    needle.decoder.reset()
    decoder_module = _prepare_module_for_coreai_export(
        needle.decoder,
        decoder_sample,
        compressor=compressor,
    )
    decoder_program = (
        TorchConverter()
        .add_pytorch_module(
            decoder_module,
            export_fn=lambda module: _export_program(module, decoder_sample),
            input_names=[
                "input_ids",
                "cross_attention_mask",
                "encoder_projected_k",
                "encoder_projected_v",
            ],
            state_names=["keyCache", "valueCache", "cacheOffset"],
            output_names=["logits"],
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
) -> torch.nn.Module:
    if compressor is None:
        return module
    return compressor.compress(module, sample_args)


@dataclass(frozen=True)
class _ModelSourceFiles:
    directory: Path
    configuration_path: Path
    tokenizer_path: Path
    weights_path: Path


def _resolve_model_source(source: str) -> _ModelSourceFiles:
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise ValueError(f"Model source must be a directory: {source_path}")
        directory = source_path.resolve()
    else:
        directory = Path(
            snapshot_download(
                repo_id=source,
                allow_patterns=[
                    "config.json",
                    "configuration.json",
                    "model.safetensors",
                    "*.pkl",
                    "tokenizer.model",
                    "tokenizer.json",
                ],
            )
        )
    return _ModelSourceFiles(
        directory=directory,
        configuration_path=_resolve_configuration_path(directory),
        tokenizer_path=_resolve_tokenizer_path(directory),
        weights_path=_resolve_weights_path(directory),
    )


def _load_configuration(
    configuration_path: Union[str, Path],
) -> NeedleModelConfiguation:
    return NeedleModelConfiguation.from_file(configuration_path)


def _load_needle_model(
    configuration: NeedleModelConfiguation,
    weights_path: Union[str, Path],
) -> Needle:
    model = Needle(configuration)
    dtype = torch_dtype(configuration.resolved_dtype)
    model = model.to(dtype=dtype)
    model.encoder.to(dtype=dtype)
    model.decoder.to(dtype=dtype)
    state_dict = load_state_dict(weights_path)
    if configuration.tie_word_embeddings:
        state_dict.pop("lm_head.weight", None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _copy_bundle_resources(
    source_files: _ModelSourceFiles,
    output_directory: Union[str, Path],
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_files.configuration_path, output_directory / "configuration.json"
    )
    shutil.copy2(
        source_files.tokenizer_path, output_directory / source_files.tokenizer_path.name
    )


def _resolve_configuration_path(directory: Path) -> Path:
    return _resolve_named_file(directory, _CONFIG_FILENAMES, label="configuration")


def _resolve_tokenizer_path(directory: Path) -> Path:
    return _resolve_named_file(directory, _TOKENIZER_FILENAMES, label="tokenizer")


def _resolve_weights_path(directory: Path) -> Path:
    safetensors_path = directory / "model.safetensors"
    if safetensors_path.exists():
        return safetensors_path

    pickle_paths = sorted(directory.glob("*.pkl"))
    if pickle_paths:
        return pickle_paths[0]

    raise FileNotFoundError(f"No supported weights found in {directory}")


def _resolve_named_file(directory: Path, names: tuple[str, ...], *, label: str) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No supported {label} file found in {directory}")


def _sample_encoder_input(
    configuration: NeedleModelConfiguation,
    sequence_length: int,
) -> torch.Tensor:
    sequence_length = max(1, sequence_length)
    input_ids = torch.full(
        (1, sequence_length),
        fill_value=configuration.decoder_start_token_id,
        dtype=torch.long,
    )
    if sequence_length > 1:
        input_ids[0, -1] = configuration.pad_token_id
    return input_ids


def _sample_decoder_inputs(
    needle: Needle,
    configuration: NeedleModelConfiguation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder_input = _sample_encoder_input(configuration, _DEFAULT_ENCODER_SAMPLE_LENGTH)
    with torch.no_grad():
        cross_attention_mask, encoder_projected_k, encoder_projected_v = needle.encoder(
            encoder_input
        )

    decoder_input = torch.full(
        (1, _DEFAULT_DECODER_SAMPLE_LENGTH),
        fill_value=configuration.decoder_start_token_id,
        dtype=torch.long,
    )
    return (
        decoder_input,
        cross_attention_mask,
        encoder_projected_k,
        encoder_projected_v,
    )


def _export_program(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.export.ExportedProgram:
    module.eval()
    return torch.export.export(module, args=args).run_decompositions(get_decomp_table())


def _save_program(
    program: AIProgram,
    path: Path,
    *,
    metadata: AIModelAssetMetadata | None = None,
) -> None:
    if path.exists():
        shutil.rmtree(path)
    program.save_asset(path, metadata=metadata)
