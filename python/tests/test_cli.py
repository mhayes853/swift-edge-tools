from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import patch

import torch
import yaml
from coreai.authoring.asset import AIModelAsset
from coreai.runtime import AIModelAssetMetadata
from coreai_opt.palettization import KMeansPalettizerConfig, PalettizationSpec
from coreai_opt.quantization import QuantizerConfig

from cli import (
    _build_authoring_metadata,
    _build_compression_config,
    _parse_backend,
    main,
    parse_arguments,
)
from needle import DecoderExportStrategy, Needle, NeedleModelConfiguation
from needle.coreml_export import CoreMLComputeUnits


class CLITests(unittest.TestCase):
    def test_build_authoring_metadata_supports_files_and_flag_overrides(self) -> None:
        directory_ctx = tempfile.TemporaryDirectory()
        try:
            directory = Path(directory_ctx.name)
            cases = [
                (
                    "json",
                    directory / "metadata.json",
                    json.dumps(
                        {
                            "author": "File Author",
                            "model_description": "File Description",
                            "license": "File License",
                            "creator_defined_metadata": {"source": "file"},
                        }
                    ),
                    [
                        "--authoring-author",
                        "Flag Author",
                        "--authoring-custom",
                        "owner=cli",
                    ],
                    {
                        "author": "Flag Author",
                        "model_description": "File Description",
                        "license": "File License",
                        "custom": {"source": "file", "owner": "cli"},
                    },
                ),
                (
                    "yaml",
                    directory / "metadata.yaml",
                    yaml.safe_dump(
                        {
                            "author": "YAML Author",
                            "model_description": "YAML Description",
                            "license": "YAML License",
                            "creator_defined_metadata": {"suite": "yaml"},
                        }
                    ),
                    [],
                    {
                        "author": "YAML Author",
                        "model_description": "YAML Description",
                        "license": "YAML License",
                        "custom": {"suite": "yaml"},
                    },
                ),
            ]

            for _, metadata_path, contents, extra_args, expected in cases:
                with self.subTest(metadata_path=metadata_path.name):
                    metadata_path.write_text(contents)
                    parsed = parse_arguments(
                        [
                            "--output",
                            str(directory / "out"),
                            "--authoring-metadata",
                            str(metadata_path),
                            *extra_args,
                        ]
                    )
                    metadata = self._unwrap_metadata(_build_authoring_metadata(parsed))

                    self.assertEqual(metadata.author, expected["author"])
                    self.assertEqual(
                        metadata.model_description,
                        expected["model_description"],
                    )
                    self.assertEqual(metadata.license, expected["license"])
                    self.assertEqual(
                        self._custom_metadata(metadata),
                        expected["custom"],
                    )
        finally:
            directory_ctx.cleanup()

    def test_build_compression_config_supports_quantizer_inputs(self) -> None:
        directory_ctx = tempfile.TemporaryDirectory()
        try:
            directory = Path(directory_ctx.name)
            config_path = directory / "quantizer.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "quantization_config": {
                            "execution_mode": "eager",
                        }
                    }
                )
            )

            cases = [
                (
                    "preset",
                    [
                        "--output",
                        str(directory / "out"),
                        "--quantizer-preset",
                        "w8",
                        "--quantizer-execution-mode",
                        "eager",
                    ],
                ),
                (
                    "yaml",
                    [
                        "--output",
                        str(directory / "out"),
                        "--quantizer-config",
                        str(config_path),
                    ],
                ),
                (
                    "quantized and palettized",
                    [
                        "--output",
                        str(directory / "out"),
                        "--quantizer-preset",
                        "w8",
                        "--palettizer-n-bits",
                        "4",
                    ],
                ),
            ]

            for name, arguments in cases:
                with self.subTest(case=name):
                    parsed = parse_arguments(arguments)
                    compression_config = self._unwrap_value(
                        _build_compression_config(parsed)
                    )
                    quantizer_config = self._unwrap_quantizer_config(
                        compression_config.quantizer
                    )
                    if name == "quantized and palettized":
                        palettizer_config = self._unwrap_palettizer_config(
                            compression_config.palettizer
                        )
                        global_config = palettizer_config.global_config
                        if global_config is None:
                            self.fail("Expected a global palettizer configuration")
                        state_spec = global_config.op_state_spec
                        if state_spec is None:
                            self.fail("Expected a weight palettization specification")
                        weight_spec = state_spec["weight"]
                        self.assertIsInstance(weight_spec, PalettizationSpec)
                        self.assertEqual(cast(PalettizationSpec, weight_spec).n_bits, 4)
                    else:
                        self.assertEqual(quantizer_config.execution_mode.value, "eager")
        finally:
            directory_ctx.cleanup()

    def test_build_compression_config_loads_palettizer_config(self) -> None:
        directory_ctx = tempfile.TemporaryDirectory()
        try:
            directory = Path(directory_ctx.name)
            config_path = directory / "palettizer.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "global_config": {
                            "op_input_spec": {},
                            "op_output_spec": {},
                            "op_state_spec": {"weight": {"n_bits": 6}},
                        }
                    }
                )
            )
            parsed = parse_arguments(
                [
                    "--output",
                    str(directory / "out"),
                    "--palettizer-config",
                    str(config_path),
                ]
            )

            compression_config = self._unwrap_value(_build_compression_config(parsed))
            palettizer_config = self._unwrap_palettizer_config(
                compression_config.palettizer
            )
            global_config = palettizer_config.global_config
            if global_config is None:
                self.fail("Expected a global palettizer configuration")
            state_spec = global_config.op_state_spec
            if state_spec is None:
                self.fail("Expected a weight palettization specification")
            weight_spec = state_spec["weight"]
            self.assertIsInstance(weight_spec, PalettizationSpec)
            self.assertEqual(cast(PalettizationSpec, weight_spec).n_bits, 6)
        finally:
            directory_ctx.cleanup()

    def test_build_compression_config_rejects_conflicting_palettizer_inputs(
        self,
    ) -> None:
        parsed = parse_arguments(
            [
                "--output",
                "./build/coreai-export",
                "--palettizer-config",
                "palettizer.json",
                "--palettizer-n-bits",
                "4",
            ]
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _build_compression_config(parsed)

    def test_main_passes_compile_platforms_to_export(self) -> None:
        with patch("cli.export_needle_coreai") as export_needle_coreai_mock:
            export_needle_coreai_mock.return_value = Path("/tmp/coreai-export")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "./build/coreai-export",
                        "--compile-platform",
                        "macOS",
                        "--compile-platform",
                        "iOS",
                    ]
                )

        self.assertEqual(exit_code, 0)
        export_needle_coreai_mock.assert_called_once()
        self.assertEqual(
            export_needle_coreai_mock.call_args.kwargs["compile_platforms"],
            ["macOS", "iOS"],
        )

    def test_parse_backend_supports_case_insensitive_engine_names(self) -> None:
        self.assertEqual(_parse_backend("CoreAI").value, "coreai")
        self.assertEqual(_parse_backend("coreml").value, "coreml")
        self.assertEqual(_parse_backend("OnNx").value, "onnx")

    def test_main_passes_onnx_quantization_to_export(self) -> None:
        with patch("cli.export_needle_onnx") as export_needle_onnx_mock:
            export_needle_onnx_mock.return_value = Path("/tmp/onnx-export")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "./build/onnx-export",
                        "--backend",
                        "onnx",
                        "--onnx-quantization",
                        "int4",
                    ]
                )

        self.assertEqual(exit_code, 0)
        compressor = export_needle_onnx_mock.call_args.kwargs["compressor"]
        self.assertEqual(compressor.bits, 4)

    def test_main_selects_backend_specific_cache_state_defaults(self) -> None:
        with patch("cli.export_needle_coreai") as coreai_export:
            coreai_export.return_value = Path("/tmp/coreai-export")
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    ["--output", "./build/coreai-export", "--backend", "coreai"]
                )

        self.assertEqual(exit_code, 0)
        coreai_strategy = coreai_export.call_args.kwargs["decoder_strategy"]
        self.assertTrue(coreai_strategy.stateful)
        self.assertFalse(coreai_strategy.cross_attention_cache_states)

        with patch("cli._export_needle_coreml") as coreml_export:
            coreml_export.return_value = Path("/tmp/coreml-export")
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    ["--output", "./build/coreml-export", "--backend", "coreml"]
                )

        self.assertEqual(exit_code, 0)
        coreml_strategy = coreml_export.call_args.kwargs["decoder_strategy"]
        self.assertTrue(coreml_strategy.stateful)
        self.assertTrue(coreml_strategy.cross_attention_cache_states)
        self.assertFalse(coreml_strategy.uses_native_attention)

    def test_reference_decoder_profile_uses_explicit_caches(self) -> None:
        with patch("cli._export_needle_coreml") as coreml_export:
            coreml_export.return_value = Path("/tmp/coreml-export")
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--output",
                        "./build/coreml-export",
                        "--backend",
                        "coreml",
                        "--decoder-profile",
                        "reference",
                    ]
                )

        self.assertEqual(exit_code, 0)
        strategy = coreml_export.call_args.kwargs["decoder_strategy"]
        self.assertFalse(strategy.stateful)
        self.assertFalse(strategy.cross_attention_cache_states)

    def test_main_passes_compile_platforms_to_coreml_export(self) -> None:
        with patch("cli._export_needle_coreml") as export_needle_coreml_mock:
            export_needle_coreml_mock.return_value = Path("/tmp/coreml-export")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "./build/coreml-export",
                        "--backend",
                        "coreml",
                        "--compile-platform",
                        "watchOS",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            export_needle_coreml_mock.call_args.kwargs["compile_platforms"],
            ["watchOS"],
        )

    def test_main_passes_compute_units_to_coreml_export(self) -> None:
        with patch("cli._export_needle_coreml") as export_needle_coreml_mock:
            export_needle_coreml_mock.return_value = Path("/tmp/coreml-export")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "./build/coreml-export",
                        "--backend",
                        "coreml",
                        "--compute-units",
                        "cpu-and-gpu",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            export_needle_coreml_mock.call_args.kwargs["compute_units"],
            CoreMLComputeUnits.CPU_AND_GPU,
        )

    def test_main_dispatches_coreml_backend_case_insensitively(self) -> None:
        with (
            patch("cli.export_needle_coreai") as export_needle_coreai_mock,
            patch("cli._export_needle_coreml") as export_needle_coreml_mock,
        ):
            export_needle_coreml_mock.return_value = Path("/tmp/coreml-export")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "./build/coreml-export",
                        "--backend",
                        "coreml",
                    ]
                )

        self.assertEqual(exit_code, 0)
        export_needle_coreai_mock.assert_not_called()
        export_needle_coreml_mock.assert_called_once()

    def test_main_persists_authoring_metadata(self) -> None:
        source_ctx = tempfile.TemporaryDirectory()
        output_ctx = tempfile.TemporaryDirectory()
        try:
            source_directory = Path(source_ctx.name)
            output_directory = Path(output_ctx.name)

            (source_directory / "configuration.json").write_text(
                json.dumps(
                    {
                        "vocab_size": 16,
                        "d_model": 8,
                        "hidden_size": 8,
                        "num_attention_heads": 2,
                        "num_kv_heads": 1,
                        "num_encoder_layers": 1,
                        "num_decoder_layers": 1,
                        "num_hidden_layers": 1,
                        "pad_token_id": 0,
                        "decoder_start_token_id": 1,
                        "tie_word_embeddings": True,
                        "torch_dtype": "float32",
                    }
                )
            )
            (source_directory / "tokenizer.json").write_text("{}")

            configuration = NeedleModelConfiguation.from_file(
                source_directory / "configuration.json"
            )
            needle_instance = Needle(
                configuration,
                decoder_strategy=DecoderExportStrategy.reference(),
            )
            torch.save(needle_instance.state_dict(), source_directory / "weights.pkl")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--source",
                        str(source_directory),
                        "--output",
                        str(output_directory),
                        "--authoring-author",
                        "CLI Test",
                        "--authoring-description",
                        "CLI metadata",
                        "--authoring-license",
                        "BSD-3-Clause",
                        "--authoring-custom",
                        "suite=cli",
                    ]
                )

            self.assertEqual(exit_code, 0)
            result = Path(stdout.getvalue().strip())
            encoder_asset = AIModelAsset.load(result / "encoder.aimodel")
            self.assertEqual(encoder_asset.metadata.author, "CLI Test")
            self.assertEqual(
                encoder_asset.metadata.model_description,
                "CLI metadata",
            )
            self.assertEqual(encoder_asset.metadata.license, "BSD-3-Clause")
            self.assertEqual(
                self._custom_metadata(encoder_asset.metadata)["suite"],
                "cli",
            )
        finally:
            output_ctx.cleanup()
            source_ctx.cleanup()

    def _custom_metadata(self, metadata: AIModelAssetMetadata) -> dict[str, str]:
        return cast(
            dict[str, str],
            object.__getattribute__(metadata, "creator_defined_metadata"),
        )

    def _unwrap_metadata(
        self, metadata: AIModelAssetMetadata | None
    ) -> AIModelAssetMetadata:
        self.assertIsNotNone(metadata)
        return cast(AIModelAssetMetadata, metadata)

    def _unwrap_quantizer_config(self, config: object) -> QuantizerConfig:
        self.assertIsInstance(config, QuantizerConfig)
        return cast(QuantizerConfig, config)

    def _unwrap_palettizer_config(self, config: object) -> KMeansPalettizerConfig:
        self.assertIsInstance(config, KMeansPalettizerConfig)
        return cast(KMeansPalettizerConfig, config)

    def _unwrap_value(self, value: _T | None) -> _T:
        self.assertIsNotNone(value)
        return cast(_T, value)


_T = TypeVar("_T")


if __name__ == "__main__":
    unittest.main()
