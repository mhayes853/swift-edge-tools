from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import TypeVar, cast

import torch
import yaml
from coreai.authoring.asset import AIModelAsset
from coreai.runtime import AIModelAssetMetadata
from coreai_opt.palettization import KMeansPalettizerConfig
from coreai_opt.quantization import QuantizerConfig

from cli import (
    _build_authoring_metadata,
    _build_compression_config,
    main,
    parse_arguments,
)
from needle import Needle, NeedleModelConfiguation


class CLITests(unittest.TestCase):
    def test_main_runs_end_to_end_on_local_bundle(self) -> None:
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
            needle_instance = Needle(configuration)
            torch.save(needle_instance.state_dict(), source_directory / "weights.pkl")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--source",
                        str(source_directory),
                        "--output",
                        str(output_directory),
                    ]
                )

            self.assertEqual(exit_code, 0)
            result = Path(stdout.getvalue().strip())
            self.assertTrue((result / "encoder.aimodel").exists())
            self.assertTrue((result / "decoder.aimodel").exists())
            self.assertTrue((result / "configuration.json").exists())
            self.assertTrue((result / "tokenizer.json").exists())
        finally:
            output_ctx.cleanup()
            source_ctx.cleanup()

    def test_build_authoring_metadata_supports_json_file_and_flag_overrides(
        self,
    ) -> None:
        directory_ctx = tempfile.TemporaryDirectory()
        try:
            directory = Path(directory_ctx.name)
            metadata_path = directory / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "author": "File Author",
                        "model_description": "File Description",
                        "license": "File License",
                        "creator_defined_metadata": {"source": "file"},
                    }
                )
            )

            parsed = parse_arguments(
                [
                    "--output",
                    str(directory / "out"),
                    "--authoring-metadata",
                    str(metadata_path),
                    "--authoring-author",
                    "Flag Author",
                    "--authoring-custom",
                    "owner=cli",
                ]
            )
            metadata = _build_authoring_metadata(parsed)

            self.assertIsNotNone(metadata)
            metadata = self._unwrap_metadata(metadata)
            self.assertEqual(metadata.author, "Flag Author")
            self.assertEqual(metadata.model_description, "File Description")
            self.assertEqual(metadata.license, "File License")
            self.assertEqual(self._custom_metadata(metadata)["source"], "file")
            self.assertEqual(self._custom_metadata(metadata)["owner"], "cli")
        finally:
            directory_ctx.cleanup()

    def test_build_authoring_metadata_supports_yaml_file(self) -> None:
        directory_ctx = tempfile.TemporaryDirectory()
        try:
            directory = Path(directory_ctx.name)
            metadata_path = directory / "metadata.yaml"
            metadata_path.write_text(
                yaml.safe_dump(
                    {
                        "author": "YAML Author",
                        "model_description": "YAML Description",
                        "license": "YAML License",
                        "creator_defined_metadata": {"suite": "yaml"},
                    }
                )
            )

            parsed = parse_arguments(
                [
                    "--output",
                    str(directory / "out"),
                    "--authoring-metadata",
                    str(metadata_path),
                ]
            )
            metadata = _build_authoring_metadata(parsed)

            self.assertIsNotNone(metadata)
            metadata = self._unwrap_metadata(metadata)
            self.assertEqual(metadata.author, "YAML Author")
            self.assertEqual(metadata.model_description, "YAML Description")
            self.assertEqual(metadata.license, "YAML License")
            self.assertEqual(self._custom_metadata(metadata)["suite"], "yaml")
        finally:
            directory_ctx.cleanup()

    def test_build_compression_config_supports_quantizer_preset_and_execution_mode(
        self,
    ) -> None:
        parsed = parse_arguments(
            [
                "--output",
                "/tmp/out",
                "--quantizer-preset",
                "w8",
                "--quantizer-execution-mode",
                "eager",
            ]
        )

        config = _build_compression_config(parsed)

        config = self._unwrap_quantizer_config(config)
        self.assertEqual(config.execution_mode.value, "eager")

    def test_build_compression_config_supports_quantizer_yaml_file(self) -> None:
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

            parsed = parse_arguments(
                [
                    "--output",
                    str(directory / "out"),
                    "--quantizer-config",
                    str(config_path),
                ]
            )
            config = _build_compression_config(parsed)

            config = self._unwrap_quantizer_config(config)
            self.assertEqual(config.execution_mode.value, "eager")
        finally:
            directory_ctx.cleanup()

    def test_build_compression_config_supports_palettizer_json_file(self) -> None:
        directory_ctx = tempfile.TemporaryDirectory()
        try:
            directory = Path(directory_ctx.name)
            config_path = directory / "palettizer.json"
            config_path.write_text(
                json.dumps(
                    {
                        "kmeans_palettization_config": {
                            "global_config": {
                                "rounding_precision": 6,
                            }
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
            config = _build_compression_config(parsed)

            config = self._unwrap_palettizer_config(config)
            global_config = self._unwrap_value(config.global_config)
            self.assertEqual(global_config.rounding_precision, 6)
        finally:
            directory_ctx.cleanup()

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
            needle_instance = Needle(configuration)
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

    def test_main_rejects_quantizer_and_palettizer_options_together(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "--output",
                    "/tmp/out",
                    "--quantizer-preset",
                    "w8",
                    "--palettizer-preset",
                    "w4",
                ]
            )

    def _custom_metadata(self, metadata: AIModelAssetMetadata) -> dict[str, str]:
        return cast(dict[str, str], getattr(metadata, "creator_defined_metadata"))

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
