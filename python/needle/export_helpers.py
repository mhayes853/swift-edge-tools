from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from coreai_torch import get_decomp_table
from huggingface_hub import snapshot_download

from . import Needle, NeedleModelConfiguation
from .cache_layout import (  # pyright: ignore[reportMissingImports]
    decoder_state_names,
    empty_decoder_caches as make_empty_decoder_caches,
)
from .decoder_strategy import (  # pyright: ignore[reportMissingImports]
    DecoderExportStrategy,
)
from .needle_compression import NeedleCompressor
from .torch_utils import load_state_dict, torch_dtype

DEFAULT_SOURCE = "Cactus-Compute/needle"
CONFIG_FILENAMES = ("configuration.json", "config.json")
TOKENIZER_FILENAMES = ("tokenizer.model", "tokenizer.json")
DEFAULT_ENCODER_SAMPLE_LENGTH = 4
DEFAULT_DECODER_SAMPLE_LENGTH = 1

Artifact = TypeVar("Artifact")


@dataclass(frozen=True)
class ModuleExportSpec:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_shapes: tuple[Any, ...]
    state_names: tuple[str, ...] = ()


def prepare_module_for_export(
    module: torch.nn.Module,
    sample_args: tuple[torch.Tensor, ...],
    *,
    compressor: NeedleCompressor | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
) -> torch.nn.Module:
    if compressor is None:
        return module
    return compressor.compress(module, sample_args, dynamic_shapes=dynamic_shapes)


def empty_decoder_caches(
    configuration: NeedleModelConfiguation,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the centralized decoder cache layout."""
    return make_empty_decoder_caches(
        configuration,
        dtype=dtype,
        device=device,
    )


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
                    "*.safetensors",
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


def prepare_export(
    source: str,
    output_directory: str | Path,
) -> tuple[ModelSourceFiles, NeedleModelConfiguation, Path]:
    source_files = resolve_model_source(source)
    configuration = load_configuration(source_files.configuration_path)
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    return source_files, configuration, output_directory


def load_needle_model(
    configuration: NeedleModelConfiguation,
    weights_path: str | Path,
    *,
    decoder_strategy: DecoderExportStrategy,
    encoder_use_native_sdpa: bool = True,
) -> Needle:
    model = Needle(
        configuration,
        decoder_strategy=decoder_strategy,
        encoder_use_native_sdpa=encoder_use_native_sdpa,
    )
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


def write_exported_decoder_length(
    output_directory: Path,
    configuration: NeedleModelConfiguation,
) -> None:
    configuration_path = resolve_configuration_path(output_directory)
    try:
        exported_configuration = json.loads(configuration_path.read_text())
        if not isinstance(exported_configuration, dict):
            raise ValueError("Expected a JSON object")
        exported_configuration["decoder_max_length"] = configuration.decoder_max_length
        configuration_path.write_text(json.dumps(exported_configuration))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Failed to update exported configuration {configuration_path}: {error}"
        ) from error


def resolve_weights_path(directory: Path) -> Path:
    safetensors_paths = sorted(directory.glob("*.safetensors"))
    if safetensors_paths:
        return safetensors_paths[0]

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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    encoder_input = sample_encoder_input(
        configuration, configuration.encoder_max_length
    )
    with torch.no_grad():
        encoder_outputs = needle.encoder(encoder_input)

    return sample_decoder_inputs_from_encoder_outputs(configuration, encoder_outputs)


def sample_decoder_inputs_from_encoder_outputs(
    configuration: NeedleModelConfiguation,
    encoder_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    decoder_input = torch.full(
        (1, DEFAULT_DECODER_SAMPLE_LENGTH),
        fill_value=configuration.decoder_start_token_id,
        dtype=torch.long,
    )
    cache_position = torch.zeros((1,), dtype=torch.int32)
    cross_attention_mask, encoder_projected_k, encoder_projected_v = encoder_outputs
    self_attention_mask = torch.full(
        (
            1,
            1,
            DEFAULT_DECODER_SAMPLE_LENGTH,
            configuration.decoder_max_length,
        ),
        fill_value=-65500.0,
        dtype=cross_attention_mask.dtype,
    )
    self_attention_mask[..., 0] = 0
    return (
        decoder_input,
        cache_position,
        self_attention_mask,
        cross_attention_mask,
        encoder_projected_k,
        encoder_projected_v,
    )


def encoder_export_spec(
    configuration: NeedleModelConfiguation,
    *,
    dynamic_buffers: bool = False,
) -> ModuleExportSpec:
    return ModuleExportSpec(
        input_names=("input_ids",),
        output_names=(
            "cross_attention_mask",
            "encoder_projected_k",
            "encoder_projected_v",
        ),
        dynamic_shapes=encoder_dynamic_shapes(
            configuration,
            dynamic_buffers=dynamic_buffers,
        ),
    )


def decoder_export_spec(
    configuration: NeedleModelConfiguation,
    strategy: DecoderExportStrategy,
    *,
    dynamic_buffers: bool = False,
) -> ModuleExportSpec:
    base_input_names = (
        "input_ids",
        "cache_position",
        "self_attention_mask",
        "cross_attention_mask",
    )
    cross_input_names = ("encoder_projected_k", "encoder_projected_v")
    cache_input_names = ("key_cache", "value_cache")
    input_names = (
        *base_input_names,
        *(() if strategy.cross_attention_cache_states else cross_input_names),
        *(() if strategy.stateful else cache_input_names),
    )
    cache_output_names = (
        ("key_cache_delta", "value_cache_delta")
        if strategy.returns_cache_deltas
        else ("updated_key_cache", "updated_value_cache")
    )
    return ModuleExportSpec(
        input_names=input_names,
        output_names=(
            ("logits",) if strategy.stateful else ("logits", *cache_output_names)
        ),
        dynamic_shapes=decoder_dynamic_shapes(
            configuration,
            strategy,
            dynamic_buffers=dynamic_buffers,
        ),
        state_names=decoder_state_names(configuration, strategy),
    )


def encoder_dynamic_shapes(
    configuration: NeedleModelConfiguation,
    *,
    dynamic_buffers: bool = False,
) -> tuple[dict[int, Any], ...]:
    sequence_length = _bounded_dimension(
        "encoder_sequence_length",
        configuration.encoder_max_length,
        dynamic=dynamic_buffers,
    )
    return ({0: torch.export.Dim.STATIC, 1: sequence_length},)


def decoder_dynamic_shapes(
    configuration: NeedleModelConfiguation,
    strategy: DecoderExportStrategy,
    *,
    dynamic_buffers: bool = False,
) -> tuple[dict[int, Any], ...]:
    encoder_length = _bounded_dimension(
        "encoder_sequence_length",
        configuration.encoder_max_length,
        dynamic=dynamic_buffers,
    )
    decoder_length = _bounded_dimension(
        "decoder_cache_length",
        configuration.decoder_max_length,
        dynamic=strategy.dynamic_cache,
    )
    shapes = (
        {0: torch.export.Dim.STATIC, 1: torch.export.Dim.STATIC},
        {0: torch.export.Dim.STATIC},
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: decoder_length,
        },
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: encoder_length,
        },
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: encoder_length,
            4: torch.export.Dim.STATIC,
        },
        {
            0: torch.export.Dim.STATIC,
            1: torch.export.Dim.STATIC,
            2: torch.export.Dim.STATIC,
            3: encoder_length,
            4: torch.export.Dim.STATIC,
        },
        {
            0: torch.export.Dim.STATIC,
            1: decoder_length,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.STATIC,
        },
        {
            0: torch.export.Dim.STATIC,
            1: decoder_length,
            2: torch.export.Dim.STATIC,
            3: torch.export.Dim.STATIC,
        },
    )
    if strategy.cross_attention_cache_states:
        return (*shapes[:4], *(() if strategy.stateful else shapes[6:]))
    return shapes[:6] if strategy.stateful else shapes


def _bounded_dimension(name: str, maximum: int, *, dynamic: bool) -> Any:
    if not dynamic or maximum <= 1:
        return torch.export.Dim.STATIC
    return torch.export.Dim(name, min=1, max=maximum)


def persist_export_artifact(
    artifact: Artifact,
    *,
    output_directory: Path,
    name: str,
    extension: str,
    metadata: Any | None,
    compile_platforms: Sequence[str],
    save: Callable[[Artifact, Path, Any | None], None],
    compile_artifact: Callable[[Path, Path, Sequence[str]], None],
) -> None:
    output_path = output_directory / f"{name}{extension}"
    if compile_platforms:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / output_path.name
            save(artifact, source_path, metadata)
            compile_artifact(source_path, output_directory, compile_platforms)
        return
    save(artifact, output_path, metadata)


def export_program(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    decomposition_table: dict[Any, Any] | None = None,
    draft: bool = False,
) -> torch.export.ExportedProgram:
    module.eval()
    export = torch.export.draft_export if draft else torch.export.export
    exported_program = export(
        module,
        args=args,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    if decomposition_table is None:
        decomposition_table = get_decomp_table()
    return exported_program.run_decompositions(decomposition_table)
