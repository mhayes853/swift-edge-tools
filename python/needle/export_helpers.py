from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from coreai_torch import get_decomp_table
from huggingface_hub import snapshot_download

from . import Needle, NeedleModelConfiguation
from .torch_utils import load_state_dict, torch_dtype

DEFAULT_SOURCE = "Cactus-Compute/needle-hf"
CONFIG_FILENAMES = ("configuration.json", "config.json")
TOKENIZER_FILENAMES = ("tokenizer.model", "tokenizer.json")
DEFAULT_ENCODER_SAMPLE_LENGTH = 4
DEFAULT_DECODER_SAMPLE_LENGTH = 1


@dataclass(frozen=True)
class ModelSourceFiles:
    directory: Path
    configuration_path: Path
    tokenizer_path: Path
    weights_path: Path


def resolve_model_source(source: str) -> ModelSourceFiles:
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
    return ModelSourceFiles(
        directory=directory,
        configuration_path=resolve_configuration_path(directory),
        tokenizer_path=resolve_tokenizer_path(directory),
        weights_path=resolve_weights_path(directory),
    )


def load_configuration(configuration_path: str | Path) -> NeedleModelConfiguation:
    return NeedleModelConfiguation.from_file(configuration_path)


def load_needle_model(
    configuration: NeedleModelConfiguation,
    weights_path: str | Path,
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


def copy_bundle_resources(
    source_files: ModelSourceFiles,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_files.configuration_path,
        output_directory / "configuration.json",
    )
    shutil.copy2(
        source_files.tokenizer_path,
        output_directory / source_files.tokenizer_path.name,
    )


def resolve_configuration_path(directory: Path) -> Path:
    return resolve_named_file(directory, CONFIG_FILENAMES, label="configuration")


def resolve_tokenizer_path(directory: Path) -> Path:
    return resolve_named_file(directory, TOKENIZER_FILENAMES, label="tokenizer")


def resolve_weights_path(directory: Path) -> Path:
    safetensors_path = directory / "model.safetensors"
    if safetensors_path.exists():
        return safetensors_path

    pickle_paths = sorted(directory.glob("*.pkl"))
    if pickle_paths:
        return pickle_paths[0]

    raise FileNotFoundError(f"No supported weights found in {directory}")


def resolve_named_file(directory: Path, names: tuple[str, ...], *, label: str) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No supported {label} file found in {directory}")


def sample_encoder_input(
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


def sample_decoder_inputs(
    needle: Needle,
    configuration: NeedleModelConfiguation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder_input = sample_encoder_input(configuration, DEFAULT_ENCODER_SAMPLE_LENGTH)
    with torch.no_grad():
        cross_attention_mask, encoder_projected_k, encoder_projected_v = needle.encoder(
            encoder_input
        )

    decoder_input = torch.full(
        (1, DEFAULT_DECODER_SAMPLE_LENGTH),
        fill_value=configuration.decoder_start_token_id,
        dtype=torch.long,
    )
    return (
        decoder_input,
        cross_attention_mask,
        encoder_projected_k,
        encoder_projected_v,
    )


def encoder_dynamic_shapes(
    configuration: NeedleModelConfiguation,
) -> tuple[dict[int, Any], ...]:
    _ = configuration
    return ({0: torch.export.Dim.STATIC, 1: torch.export.Dim.AUTO},)


def decoder_dynamic_shapes() -> tuple[dict[int, Any], ...]:
    return (
        {0: torch.export.Dim.STATIC, 1: torch.export.Dim.STATIC},
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.AUTO,
        },
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.AUTO,
            4: torch.export.Dim.STATIC,
        },
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.AUTO,
            4: torch.export.Dim.STATIC,
        },
    )


def export_program(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
) -> torch.export.ExportedProgram:
    module.eval()
    return torch.export.export(
        module,
        args=args,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    ).run_decompositions(get_decomp_table())
