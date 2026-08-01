from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from coreai.runtime import AIModelAssetMetadata
from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.quantization import QuantizerConfig
from needle.coreai_export import export_needle_coreai
from needle.coreml_export import CoreMLComputeUnits, export_needle_coreml
from needle.decoder_strategy import (  # pyright: ignore[reportMissingImports]
    AttentionImplementation,
    DecoderExportStrategy,
)
from needle.export_helpers import DEFAULT_SOURCE
from needle.needle_compression import (
    CoreAIQuantizerCompressor,
    CoreMLQuantizerCompressor,
    NeedleCompressor,
)
from needle.onnx_compression import MatMulNBitsONNXCompressor, ONNXCompressor
from needle.onnx_export import export_needle_onnx

_QUANTIZER_PRESETS = ("w4", "w4_per_block", "w8")
_PALETTIZER_N_BITS = (1, 2, 3, 4, 6, 8)
_COMPUTE_UNITS = tuple(compute_units.value for compute_units in CoreMLComputeUnits)


class CLIBackend(Enum):
    COREAI = "coreai"
    COREML = "coreml"
    ONNX = "onnx"


_BACKENDS = {
    "coreai": CLIBackend.COREAI,
    "coreml": CLIBackend.COREML,
    "onnx": CLIBackend.ONNX,
}


@dataclass(frozen=True)
class CompressionConfig:
    quantizer: QuantizerConfig | None = None
    palettizer: KMeansPalettizerConfig | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Needle models")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="CoreAI")
    parser.add_argument("--onnx-quantization", choices=("int4", "int8"))
    parser.add_argument(
        "--onnx-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--compute-units",
        choices=_COMPUTE_UNITS,
        help="Legacy override that applies the same compute units to both components.",
    )
    parser.add_argument(
        "--encoder-compute-units",
        choices=_COMPUTE_UNITS,
        default=CoreMLComputeUnits.CPU_AND_NE.value,
    )
    parser.add_argument(
        "--decoder-compute-units",
        choices=_COMPUTE_UNITS,
        default=CoreMLComputeUnits.CPU_AND_GPU.value,
    )
    decoder_experiments = parser.add_argument_group("decoder experiments")
    decoder_experiments.add_argument(
        "--coreml-decoder-attention",
        choices=("automatic", "native", "decomposed"),
        default="automatic",
        help="Override the CoreML decoder attention lowering strategy.",
    )
    decoder_experiments.add_argument(
        "--decoder-profile",
        choices=("default", "reference"),
        default="default",
        help="Use the optimized backend contract or explicit-cache reference contract.",
    )
    decoder_experiments.add_argument(
        "--experimental-coreml-dynamic-cache",
        action="store_true",
        help="Limit CoreML self-attention to a dynamic active cache prefix.",
    )
    parser.add_argument(
        "--compile-platform",
        action="append",
        default=[],
        dest="compile_platforms",
        metavar="PLATFORM",
    )

    parser.add_argument("--authoring-metadata")
    parser.add_argument("--authoring-author")
    parser.add_argument("--authoring-description")
    parser.add_argument("--authoring-license")
    parser.add_argument(
        "--authoring-custom",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )

    parser.add_argument("--quantizer-config")
    parser.add_argument("--quantizer-preset", choices=_QUANTIZER_PRESETS)
    parser.add_argument("--quantizer-execution-mode", choices=("graph", "eager"))
    parser.add_argument("--palettizer-config")
    parser.add_argument("--palettizer-n-bits", choices=_PALETTIZER_N_BITS, type=int)
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(arguments)


def _load_structured_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        contents = path.read_text()
    except OSError as error:
        raise ValueError(f"Failed to read {path}: {error}") from error

    try:
        if suffix == ".json":
            data = json.loads(contents)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(contents)
        else:
            raise ValueError(
                f"Unsupported file format for {path}. Expected JSON or YAML."
            )
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Failed to parse {path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping content in {path}.")
    return data


def _parse_key_value_flag(value: str, *, flag_name: str) -> tuple[str, str]:
    key, separator, raw_value = value.partition("=")
    if not separator or not key:
        raise ValueError(f"{flag_name} values must be in KEY=VALUE format")
    return key, raw_value


def _build_authoring_metadata(
    parsed: argparse.Namespace,
) -> AIModelAssetMetadata | None:
    has_metadata_inputs = any(
        [
            parsed.authoring_metadata,
            parsed.authoring_author,
            parsed.authoring_description,
            parsed.authoring_license,
            parsed.authoring_custom,
        ]
    )
    if not has_metadata_inputs:
        return None

    metadata = AIModelAssetMetadata()
    if parsed.authoring_metadata:
        data = _load_structured_file(parsed.authoring_metadata)
        _apply_metadata_mapping(metadata, data)

    if parsed.authoring_author is not None:
        metadata.author = parsed.authoring_author
    if parsed.authoring_description is not None:
        metadata.model_description = parsed.authoring_description
    if parsed.authoring_license is not None:
        metadata.license = parsed.authoring_license

    for item in parsed.authoring_custom:
        key, value = _parse_key_value_flag(item, flag_name="--authoring-custom")
        metadata.set_custom(key, value)

    return metadata


def _apply_metadata_mapping(
    metadata: AIModelAssetMetadata,
    data: dict[str, Any],
) -> None:
    supported_keys = {
        "author",
        "model_description",
        "license",
        "creator_defined_metadata",
    }
    unknown_keys = set(data) - supported_keys
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unsupported authoring metadata keys: {keys}")

    if "author" in data:
        metadata.author = _expect_string(data["author"], key="author")
    if "model_description" in data:
        metadata.model_description = _expect_string(
            data["model_description"], key="model_description"
        )
    if "license" in data:
        metadata.license = _expect_string(data["license"], key="license")
    if "creator_defined_metadata" in data:
        custom_metadata = data["creator_defined_metadata"]
        if not isinstance(custom_metadata, dict):
            raise ValueError("creator_defined_metadata must be a mapping")
        for key, value in custom_metadata.items():
            metadata.set_custom(
                _expect_string(key, key="creator_defined_metadata key"),
                _expect_string(value, key=f"creator_defined_metadata[{key!r}]"),
            )


def _build_compression_config(parsed: argparse.Namespace) -> CompressionConfig | None:
    has_quantizer_inputs = any(
        [
            parsed.quantizer_config,
            parsed.quantizer_preset,
            parsed.quantizer_execution_mode,
        ]
    )
    has_palettizer_inputs = any(
        [
            parsed.palettizer_config,
            parsed.palettizer_n_bits is not None,
        ]
    )
    if not has_quantizer_inputs and not has_palettizer_inputs:
        return None
    if parsed.quantizer_config and parsed.quantizer_preset:
        raise ValueError(
            "--quantizer-config and --quantizer-preset are mutually exclusive"
        )
    if parsed.palettizer_config and parsed.palettizer_n_bits is not None:
        raise ValueError(
            "--palettizer-config and --palettizer-n-bits are mutually exclusive"
        )
    return CompressionConfig(
        quantizer=_build_quantizer_config(parsed) if has_quantizer_inputs else None,
        palettizer=_build_palettizer_config(parsed) if has_palettizer_inputs else None,
    )


def _parse_compute_units(value: str) -> CoreMLComputeUnits:
    return CoreMLComputeUnits(value)


def _parse_backend(value: str) -> CLIBackend:
    backend = _BACKENDS.get(value.casefold())
    if backend is None:
        supported = ", ".join(backend.value for backend in CLIBackend)
        raise ValueError(f"Unsupported backend {value!r}. Expected one of: {supported}")
    return backend


def _build_compressor(parsed: argparse.Namespace) -> NeedleCompressor | None:
    compression_config = _build_compression_config(parsed)
    if compression_config is None:
        return None
    backend = _parse_backend(parsed.backend)
    if backend == CLIBackend.COREAI:
        return CoreAIQuantizerCompressor(
            compression_config.quantizer,
            palettizer_config=compression_config.palettizer,
        )
    if backend == CLIBackend.COREML:
        return CoreMLQuantizerCompressor(
            compression_config.quantizer,
            palettizer_config=compression_config.palettizer,
        )
    raise ValueError(
        "CoreAI/CoreML quantizer and palettizer options cannot be used with ONNX"
    )


def _build_onnx_compressor(parsed: argparse.Namespace) -> ONNXCompressor | None:
    if parsed.onnx_quantization is None:
        return None
    if _parse_backend(parsed.backend) != CLIBackend.ONNX:
        raise ValueError("--onnx-quantization requires --backend onnx")
    if parsed.onnx_quantization == "int4":
        return MatMulNBitsONNXCompressor.int4()
    return MatMulNBitsONNXCompressor.int8()


def _build_quantizer_config(parsed: argparse.Namespace) -> QuantizerConfig:
    if parsed.quantizer_config:
        config = _load_compression_config(parsed.quantizer_config, QuantizerConfig)
    elif parsed.quantizer_preset:
        config = getattr(QuantizerConfig.presets, parsed.quantizer_preset)()
    else:
        config = QuantizerConfig()

    if parsed.quantizer_execution_mode is not None:
        config.set_execution_mode(parsed.quantizer_execution_mode)
    return config


def _build_palettizer_config(parsed: argparse.Namespace) -> KMeansPalettizerConfig:
    if parsed.palettizer_config:
        return _load_compression_config(
            parsed.palettizer_config,
            KMeansPalettizerConfig,
        )
    palettization_spec = PalettizationSpec(n_bits=parsed.palettizer_n_bits)
    return KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(
            op_input_spec={},
            op_output_spec={},
            op_state_spec={
                "weight": palettization_spec,
                "in_proj_weight": palettization_spec,
            },
        )
    )


def _load_compression_config(path: str | Path, config_type: type[Any]):
    data = _load_structured_file(path)
    config_key = getattr(config_type, "_CONFIG_KEY", None)
    if isinstance(config_key, str) and config_key in data:
        return config_type.from_dict(data)
    return config_type(**data)


def _expect_string(value: Any, *, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _export_for_backend(
    backend: CLIBackend,
    source: str,
    output: str,
    *,
    compressor: NeedleCompressor | None = None,
    onnx_compressor: ONNXCompressor | None = None,
    onnx_dtype: str = "float32",
    model_metadata: AIModelAssetMetadata | None = None,
    compute_units: CoreMLComputeUnits | None = None,
    encoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_NE,
    decoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_GPU,
    decoder_use_native_sdpa: bool | None = None,
    decoder_profile: str = "default",
    experimental_coreml_dynamic_cache: bool = False,
    compile_platforms: Sequence[str] = (),
) -> Path:
    coreml_attention = (
        AttentionImplementation.NATIVE
        if decoder_use_native_sdpa
        else AttentionImplementation.DECOMPOSED
    )
    if backend == CLIBackend.COREAI:
        decoder_strategy = (
            DecoderExportStrategy.reference()
            if decoder_profile == "reference"
            else DecoderExportStrategy.coreai()
        )
        return export_needle_coreai(
            source,
            output,
            compressor=compressor,
            model_metadata=model_metadata,
            decoder_strategy=decoder_strategy,
            compile_platforms=compile_platforms,
        )
    if backend == CLIBackend.COREML:
        decoder_strategy = (
            DecoderExportStrategy.reference(
                attention=coreml_attention,
                use_native_gqa=False,
            )
            if decoder_profile == "reference"
            else DecoderExportStrategy.coreml(
                attention=coreml_attention,
                dynamic_cache=experimental_coreml_dynamic_cache,
            )
        )
        return _export_needle_coreml(
            source,
            output,
            compressor=compressor,
            model_metadata=model_metadata,
            compute_units=compute_units,
            encoder_compute_units=encoder_compute_units,
            decoder_compute_units=decoder_compute_units,
            decoder_strategy=decoder_strategy,
            compile_platforms=compile_platforms,
        )
    if backend == CLIBackend.ONNX:
        decoder_strategy = (
            DecoderExportStrategy.reference()
            if decoder_profile == "reference"
            else DecoderExportStrategy.onnx()
        )
        return export_needle_onnx(
            source,
            output,
            compressor=onnx_compressor,
            dtype=onnx_dtype,
            decoder_strategy=decoder_strategy,
        )


def _export_needle_coreml(
    source: str,
    output: str,
    *,
    compressor: NeedleCompressor | None = None,
    model_metadata: AIModelAssetMetadata | None = None,
    compute_units: CoreMLComputeUnits | None = None,
    encoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_NE,
    decoder_compute_units: CoreMLComputeUnits = CoreMLComputeUnits.CPU_AND_GPU,
    decoder_strategy: DecoderExportStrategy | None = None,
    compile_platforms: Sequence[str] = (),
) -> Path:
    return export_needle_coreml(
        source,
        output,
        compressor=compressor,
        model_metadata=model_metadata,
        compute_units=compute_units,
        encoder_compute_units=encoder_compute_units,
        decoder_compute_units=decoder_compute_units,
        decoder_strategy=decoder_strategy,
        compile_platforms=compile_platforms,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    try:
        backend = _parse_backend(parsed.backend)
        compressor = _build_compressor(parsed)
        onnx_compressor = _build_onnx_compressor(parsed)
        model_metadata = _build_authoring_metadata(parsed)
        compute_units = (
            _parse_compute_units(parsed.compute_units)
            if parsed.compute_units is not None
            else None
        )
        encoder_compute_units = _parse_compute_units(parsed.encoder_compute_units)
        decoder_compute_units = _parse_compute_units(parsed.decoder_compute_units)
        decoder_use_native_sdpa = {
            "automatic": None,
            "native": True,
            "decomposed": False,
        }[parsed.coreml_decoder_attention]
        if parsed.experimental_coreml_dynamic_cache and backend != CLIBackend.COREML:
            raise ValueError(
                "Experimental dynamic decoder caches are only supported by CoreML"
            )
    except ValueError as error:
        parser.error(str(error))

    with contextlib.redirect_stdout(io.StringIO()):
        output_directory = _export_for_backend(
            backend,
            parsed.source,
            parsed.output,
            compressor=compressor,
            onnx_compressor=onnx_compressor,
            onnx_dtype=parsed.onnx_dtype,
            model_metadata=model_metadata,
            compute_units=compute_units,
            encoder_compute_units=encoder_compute_units,
            decoder_compute_units=decoder_compute_units,
            decoder_use_native_sdpa=decoder_use_native_sdpa,
            decoder_profile=parsed.decoder_profile,
            experimental_coreml_dynamic_cache=(
                parsed.experimental_coreml_dynamic_cache
            ),
            compile_platforms=parsed.compile_platforms,
        )
    sys.stdout.write(f"{output_directory}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
