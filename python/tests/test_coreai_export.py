import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from needle import DecoderExportStrategy, Needle, NeedleModelConfiguation
from needle.export.coreai import convert_needle_coreai_programs, export_needle_coreai
from needle.export.helpers import export_program


class CoreAIExportTests(unittest.TestCase):
    def test_export_needle_coreai_runs_end_to_end_on_local_bundle(self) -> None:
        with (
            tempfile.TemporaryDirectory() as source_name,
            tempfile.TemporaryDirectory() as output_name,
        ):
            source_directory = Path(source_name)
            output_directory = Path(output_name)

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
            torch.save(
                Needle(
                    configuration,
                    decoder_strategy=DecoderExportStrategy.reference(),
                ).state_dict(),
                source_directory / "weights.pkl",
            )

            result = export_needle_coreai(str(source_directory), output_directory)

            self.assertEqual(result, output_directory.resolve())
            self.assertTrue((result / "encoder.aimodel").exists())
            self.assertTrue((result / "decoder.aimodel").exists())
            self.assertTrue((result / "configuration.json").exists())
            self.assertTrue((result / "tokenizer.json").exists())

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
            "needle.export.coreai.helpers.export_program",
            side_effect=capture_exported_program,
        ):
            convert_needle_coreai_programs(needle, configuration)

        self.assertEqual(len(exported_programs), 2)
        for program in exported_programs:
            self.assertIn("scaled_dot_product_attention", program.graph_module.code)


if __name__ == "__main__":
    unittest.main()
