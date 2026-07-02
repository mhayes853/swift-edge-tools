from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from cli import main
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
            torch.save(
                needle_instance.state_dict(), source_directory / "weights.pkl"
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["--source", str(source_directory), "--output", str(output_directory)]
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


if __name__ == "__main__":
    unittest.main()