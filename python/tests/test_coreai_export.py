import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable, cast

import torch
from coreai.authoring.asset import AIModelAsset
from coreai.runtime import AIModelAssetMetadata
from coreai_opt.quantization import QuantizerConfig

from needle import Needle, NeedleModelConfiguation
from needle.needle_compression import CoreAIQuantizerCompressor
from needle.coreai_export import (
    _encoder_dynamic_shapes,
    _export_program,
    _load_needle_model,
    _sample_encoder_input,
    export_needle_coreai,
)


class CoreAIExportTests(unittest.TestCase):
    def test_export_needle_coreai_runs_end_to_end_on_local_bundle(self) -> None:
        self._assert_export_runs_end_to_end()

    def test_export_program_supports_dynamic_encoder_sequence_length(self) -> None:
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
        needle = Needle(configuration)
        sample = (_sample_encoder_input(configuration, 4),)

        exported_program = _export_program(
            needle.encoder,
            sample,
            dynamic_shapes=_encoder_dynamic_shapes(configuration),
        )

        self.assertEqual(len(exported_program.range_constraints), 1)

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
            needle = Needle(configuration)
            torch.save(needle.state_dict(), source_directory / "weights.pkl")

            loaded = _load_needle_model(configuration, source_directory / "weights.pkl")

            self.assertEqual(next(loaded.parameters()).dtype, torch.bfloat16)
        finally:
            source_ctx.cleanup()

    def test_export_needle_coreai_supports_quantizer_compressor(self) -> None:
        compressor = CoreAIQuantizerCompressor(QuantizerConfig.presets.w8())
        self._assert_export_runs_end_to_end(compressor=compressor)

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
        post_export_assertions: Callable[[Path], None] | None = None,
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
            needle_instance = Needle(configuration)
            torch.save(needle_instance.state_dict(), source_directory / "weights.pkl")

            result = export_needle_coreai(
                str(source_directory),
                output_directory,
                compressor=compressor,
                model_metadata=model_metadata,
            )

            self.assertEqual(result, output_directory.resolve())
            self.assertTrue((result / "encoder.aimodel").exists())
            self.assertTrue((result / "decoder.aimodel").exists())
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
        return cast(dict[str, str], getattr(metadata, "creator_defined_metadata"))


if __name__ == "__main__":
    unittest.main()
