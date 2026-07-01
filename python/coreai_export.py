from __future__ import annotations

import argparse
import shutil
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from coreai.authoring.asset import AIProgram
from coreai_torch import TorchConverter, get_decomp_table
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file

from swift_needle import Needle, NeedleModelConfiguation

DEFAULT_SOURCE = "Cactus-Compute/needle-hf"
_CONFIG_FILENAMES = ("configuration.json", "config.json")
_TOKENIZER_FILENAMES = ("tokenizer.model", "tokenizer.json")
_DEFAULT_ENCODER_SAMPLE_LENGTH = 4
_DEFAULT_DECODER_SAMPLE_LENGTH = 1

StateDictPayload = Mapping[str, torch.Tensor] | Mapping[str, object] | torch.nn.Module


@dataclass(frozen=True)
class ModelSourceFiles:
    directory: Path
    configuration_path: Path
    tokenizer_path: Path
    weights_path: Path


def export_needle_coreai(
    source: str,
    output_directory: str | Path,
) -> Path:
    source_files = resolve_model_source(source)
    configuration = load_configuration(source_files.configuration_path)
    needle = load_needle_model(configuration, source_files.weights_path)

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    encoder_program, decoder_program = export_programs(
        needle,
        configuration,
    )
    _save_program(encoder_program, output_directory / "encoder.aimodel")
    _save_program(decoder_program, output_directory / "decoder.aimodel")
    copy_bundle_resources(source_files, output_directory)
    return output_directory


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
        configuration_path=_resolve_configuration_path(directory),
        tokenizer_path=_resolve_tokenizer_path(directory),
        weights_path=_resolve_weights_path(directory),
    )


def load_configuration(configuration_path: str | Path) -> NeedleModelConfiguation:
    return NeedleModelConfiguation.from_file(configuration_path)


def load_needle_model(
    configuration: NeedleModelConfiguation, weights_path: str | Path
) -> Needle:
    model = Needle(configuration)
    dtype = _torch_dtype(configuration.resolved_dtype)
    model = model.to(dtype=dtype)
    model.encoder.to(dtype=dtype)
    model.decoder.to(dtype=dtype)
    state_dict = load_model_state_dict(weights_path)
    if configuration.tie_word_embeddings:
        state_dict.pop("lm_head.weight", None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_model_state_dict(weights_path: str | Path) -> dict[str, torch.Tensor]:
    weights_path = Path(weights_path)
    if weights_path.suffix == ".safetensors":
        state_dict = load_safetensors_file(str(weights_path), device="cpu")
    elif weights_path.suffix == ".pkl":
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(payload)
    else:
        raise ValueError(f"Unsupported weights file: {weights_path}")

    return _normalize_state_dict(state_dict)


def export_programs(
    needle: Needle,
    configuration: NeedleModelConfiguation,
) -> tuple[AIProgram, AIProgram]:
    encoder_sample = (
        _sample_encoder_input(configuration, _DEFAULT_ENCODER_SAMPLE_LENGTH),
    )
    decoder_sample = _sample_decoder_inputs(
        needle,
        configuration,
    )

    encoder_program = (
        TorchConverter()
        .add_pytorch_module(
            needle.encoder,
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
    decoder_program = (
        TorchConverter()
        .add_pytorch_module(
            needle.decoder,
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


def copy_bundle_resources(
    source_files: ModelSourceFiles, output_directory: str | Path
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_files.configuration_path, output_directory / "configuration.json"
    )
    shutil.copy2(
        source_files.tokenizer_path, output_directory / source_files.tokenizer_path.name
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Needle CoreAI models")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_directory = export_needle_coreai(arguments.source, arguments.output)
    print(output_directory)


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


def _extract_state_dict(payload: StateDictPayload) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if _is_tensor_state_dict(payload):
            return typing.cast(dict[str, torch.Tensor], dict(payload))
        for key in ("state_dict", "model", "module", "model_state_dict"):
            if key in payload:
                nested_payload = payload[key]
                if isinstance(nested_payload, (Mapping, torch.nn.Module)):
                    return _extract_state_dict(nested_payload)
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    raise ValueError("Could not extract a state_dict from the provided weights")


def _is_tensor_state_dict(payload: Mapping[str, object]) -> bool:
    return bool(payload) and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    )


def _normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {
                key.removeprefix(prefix): value for key, value in normalized.items()
            }
    return normalized


def _sample_encoder_input(
    configuration: NeedleModelConfiguation, sequence_length: int
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
    module: torch.nn.Module, args: tuple[torch.Tensor, ...]
) -> torch.export.ExportedProgram:
    return torch.export.export(module.eval(), args=args).run_decompositions(
        get_decomp_table()
    )


def _save_program(program: AIProgram, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    program.save_asset(path)


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


if __name__ == "__main__":
    main()
