import json
import tempfile
import unittest
from pathlib import Path

from coreai_opt.palettization import KMeansPalettizerConfig
from coreai_opt.quantization import QuantizerConfig
import torch

from needle import Needle, NeedleModelConfiguation
from needle.coreai_export import export_needle_coreai


class CoreAIExportTests(unittest.TestCase):
    def test_export_needle_coreai_runs_end_to_end_on_local_bundle(self) -> None:
        self._assert_export_runs_end_to_end()

    def test_export_needle_coreai_supports_quantization(self) -> None:
        self._assert_export_runs_end_to_end(
            compression_config=QuantizerConfig.presets.w8()
        )

    def test_export_needle_coreai_supports_palettization(self) -> None:
        self._assert_export_runs_end_to_end(
            compression_config=KMeansPalettizerConfig.presets.w4()
        )

    def _assert_export_runs_end_to_end(self, compression_config=None) -> None:
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
            torch.save(
                needle_instance.state_dict(), source_directory / "weights.pkl"
            )

            result = export_needle_coreai(
                str(source_directory),
                output_directory,
                compression_config=compression_config,
            )

            self.assertEqual(result, output_directory.resolve())
            self.assertTrue((result / "encoder.aimodel").exists())
            self.assertTrue((result / "decoder.aimodel").exists())
            self.assertTrue((result / "configuration.json").exists())
            self.assertTrue((result / "tokenizer.json").exists())
        finally:
            output_ctx.cleanup()
            source_ctx.cleanup()


if __name__ == "__main__":
    unittest.main()
