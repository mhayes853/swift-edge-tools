import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
from coreai.authoring.asset import AIModelAsset
from coreai.runtime import AIModelAssetMetadata
from coreai_opt.palettization import KMeansPalettizerConfig
from coreai_opt.quantization import QuantizerConfig

from needle import DecoderExportStrategy, Needle, NeedleModelConfiguation
from needle.coreai_export import convert_needle_coreai_programs, export_needle_coreai
from needle.export_helpers import (
    encoder_dynamic_shapes,
    export_program,
    load_needle_model,
    sample_encoder_input,
)
from needle.needle_compression import CoreAIQuantizerCompressor


class CoreAIExportTests(unittest.TestCase):
    def test_export_needle_coreai_runs_end_to_end_on_local_bundle(self) -> None:
        self._assert_export_runs_end_to_end()

    def test_convert_needle_coreai_uses_native_sdpa(self) -> None:
        configuration = NeedleModelConfiguation(
            vocabulary_size=16,
            dimensions=8,
            hidden_dimensions=8,
            attention_heads=2,
            kv_heads=1,
            encoder_layers=1,
            decoder_layers=1,
            hidden_layers=1,
            max_seq_len=16,
            dtype="float32",
        )
        needle = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.coreai(),
        )
        exported_programs = []

        def capture_exported_program(*args, **kwargs):
            program = export_program(*args, **kwargs)
            exported_programs.append(program)
            return program

        with patch(
            "needle.coreai_export.export_helpers.export_program",
            side_effect=capture_exported_program,
        ):
            convert_needle_coreai_programs(needle, configuration)

        self.assertEqual(len(exported_programs), 2)
        for program in exported_programs:
            self.assertIn(
                "scaled_dot_product_attention",
                program.graph_module.code,
            )

    def test_export_program_uses_dynamic_encoder_sequence_length(self) -> None:
        configuration = NeedleModelConfiguation(
            vocabulary_size=16,
            dimensions=8,
            hidden_dimensions=8,
            attention_heads=2,
            kv_heads=1,
            encoder_layers=1,
            decoder_layers=1,
            hidden_layers=1,
            max_seq_len=32,
            dtype="float32",
        )
        needle = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.reference(),
        )
        sample = (sample_encoder_input(configuration, 4),)

        exported_program = export_program(
            needle.encoder,
            sample,
            dynamic_shapes=encoder_dynamic_shapes(
                configuration,
                dynamic_buffers=True,
            ),
        )

        self.assertEqual(len(exported_program.range_constraints), 1)
        self.assertEqual(
            tuple(
                exported_program.module()(sample_encoder_input(configuration, 8))[
                    0
                ].shape
            ),
            (1, 1, 1, 8),
        )

    def test_load_needle_model_uses_configuration_dtype(self) -> None:
        source_ctx = tempfile.TemporaryDirectory()
        try:
            source_directory = Path(source_ctx.name)
            configuration = NeedleModelConfiguation(
                vocabulary_size=16,
                dimensions=8,
                hidden_dimensions=8,
                attention_heads=2,
                kv_heads=1,
                encoder_layers=1,
                decoder_layers=1,
                hidden_layers=1,
                max_seq_len=32,
                dtype="bfloat16",
            )
            strategy = DecoderExportStrategy.reference()
            needle = Needle(configuration, decoder_strategy=strategy)
            torch.save(needle.state_dict(), source_directory / "weights.pkl")

            loaded = load_needle_model(
                configuration,
                source_directory / "weights.pkl",
                decoder_strategy=strategy,
            )

            self.assertEqual(next(loaded.parameters()).dtype, torch.bfloat16)
        finally:
            source_ctx.cleanup()

    def test_export_needle_coreai_supports_quantizer_compressor(self) -> None:
        compressor = CoreAIQuantizerCompressor(QuantizerConfig.presets.w8())
        self._assert_export_runs_end_to_end(compressor=compressor)

    def test_export_needle_coreai_supports_quantizer_and_palettizer_compressor(
        self,
    ) -> None:
        compressor = CoreAIQuantizerCompressor(
            QuantizerConfig.presets.w8(),
            palettizer_config=KMeansPalettizerConfig(),
        )
        self._assert_export_runs_end_to_end(compressor=compressor)

    def test_export_needle_coreai_supports_ahead_of_time_compilation(self) -> None:
        def compile_model_asset(
            source_asset_path: Path,
            *,
            output_directory: Path,
            platforms: tuple[str, ...] | list[str],
        ) -> None:
            self.assertEqual(list(platforms), ["macOS"])
            compiled_path = output_directory / f"{source_asset_path.stem}.h16c.aimodelc"
            compiled_path.mkdir(parents=True)
            (compiled_path / "metadata.json").write_text("{}")

        with patch(
            "needle.coreai_export._compile_model_asset",
            side_effect=compile_model_asset,
        ) as compile_model_asset_mock:
            self._assert_export_runs_end_to_end(
                compile_platforms=["macOS"],
                expected_model_paths=[
                    "encoder.h16c.aimodelc",
                    "decoder.h16c.aimodelc",
                ],
                unexpected_model_paths=[
                    "encoder.aimodel",
                    "decoder.aimodel",
                ],
            )

        self.assertEqual(compile_model_asset_mock.call_count, 2)

    def test_export_needle_coreai_persists_model_metadata(self) -> None:
        metadata = AIModelAssetMetadata()
        metadata.author = "Needle Tests"
        metadata.model_description = "Needle CoreAI export metadata test"
        metadata.license = "A totally legal license"
        metadata.set_custom("suite", "coreai_export")

        def assert_metadata(result: Path) -> None:
            encoder_asset = AIModelAsset.load(result / "encoder.aimodel")
            decoder_asset = AIModelAsset.load(result / "decoder.aimodel")
            self._assert_metadata_matches(encoder_asset.metadata, metadata)
            self._assert_metadata_matches(decoder_asset.metadata, metadata)

        self._assert_export_runs_end_to_end(
            model_metadata=metadata,
            post_export_assertions=assert_metadata,
        )

    def _assert_export_runs_end_to_end(
        self,
        compressor=None,
        model_metadata: AIModelAssetMetadata | None = None,
        compile_platforms: list[str] | None = None,
        post_export_assertions: Callable[[Path], None] | None = None,
        expected_model_paths: list[str] | None = None,
        unexpected_model_paths: list[str] | None = None,
    ) -> None:
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

            result = export_needle_coreai(
                str(source_directory),
                output_directory,
                compressor=compressor,
                model_metadata=model_metadata,
                compile_platforms=compile_platforms or [],
            )

            expected_model_paths = expected_model_paths or [
                "encoder.aimodel",
                "decoder.aimodel",
            ]
            unexpected_model_paths = unexpected_model_paths or []

            self.assertEqual(result, output_directory.resolve())
            for model_path in expected_model_paths:
                self.assertTrue((result / model_path).exists())
            for model_path in unexpected_model_paths:
                self.assertFalse((result / model_path).exists())
            self.assertTrue((result / "configuration.json").exists())
            self.assertTrue((result / "tokenizer.json").exists())
            if post_export_assertions is not None:
                post_export_assertions(result)
        finally:
            output_ctx.cleanup()
            source_ctx.cleanup()

    def _assert_metadata_matches(
        self,
        actual: AIModelAssetMetadata,
        expected: AIModelAssetMetadata,
    ) -> None:
        self.assertEqual(actual.author, expected.author)
        self.assertEqual(actual.model_description, expected.model_description)
        self.assertEqual(actual.license, expected.license)
        self.assertEqual(
            self._custom_metadata(actual),
            self._custom_metadata(expected),
        )

    def _custom_metadata(self, metadata: AIModelAssetMetadata) -> dict[str, str]:
        return cast(
            dict[str, str],
            object.__getattribute__(metadata, "creator_defined_metadata"),
        )


if __name__ == "__main__":
    unittest.main()
