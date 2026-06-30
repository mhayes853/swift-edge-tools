import json
import tempfile
import unittest
from pathlib import Path

import torch

from coreai_export import (
    ModelSourceFiles,
    copy_bundle_resources,
    load_model_state_dict,
    resolve_model_source,
)
from swift_needle.needle_configuration import NeedleModelConfiguation


class CoreAIExportTests(unittest.TestCase):
    def test_load_configuration_from_hf_style_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "vocab_size": 8192,
                        "d_model": 512,
                        "hidden_size": 512,
                        "num_attention_heads": 8,
                        "num_kv_heads": 4,
                        "num_encoder_layers": 12,
                        "num_decoder_layers": 8,
                        "num_hidden_layers": 8,
                        "rope_theta": 10000.0,
                        "rms_norm_eps": 1e-6,
                        "pad_token_id": 0,
                        "decoder_start_token_id": 1,
                        "tie_word_embeddings": True,
                        "torch_dtype": "bfloat16",
                    }
                )
            )

            configuration = NeedleModelConfiguation.from_file(config_path)

        self.assertEqual(configuration.vocabulary_size, 8192)
        self.assertEqual(configuration.dimensions, 512)
        self.assertEqual(configuration.hidden_dimensions, 512)
        self.assertEqual(configuration.attention_heads, 8)
        self.assertEqual(configuration.kv_heads, 4)
        self.assertEqual(configuration.encoder_layers, 12)
        self.assertEqual(configuration.decoder_layers, 8)
        self.assertEqual(configuration.hidden_layers, 8)
        self.assertEqual(configuration.dtype, "bfloat16")

    def test_resolve_model_source_accepts_local_bundle_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "configuration.json").write_text("{}")
            (model_directory / "tokenizer.json").write_text("{}")
            (model_directory / "weights.pkl").write_bytes(b"pickle")

            resolved = resolve_model_source(directory)

        self.assertEqual(resolved.configuration_path.name, "configuration.json")
        self.assertEqual(resolved.tokenizer_path.name, "tokenizer.json")
        self.assertEqual(resolved.weights_path.name, "weights.pkl")

    def test_copy_bundle_resources_writes_configuration_json(self) -> None:
        with tempfile.TemporaryDirectory() as source_directory:
            with tempfile.TemporaryDirectory() as output_directory:
                source_directory = Path(source_directory)
                config_path = source_directory / "config.json"
                tokenizer_path = source_directory / "tokenizer.model"
                weights_path = source_directory / "model.safetensors"
                config_path.write_text('{"d_model": 512}')
                tokenizer_path.write_bytes(b"tokenizer")
                weights_path.write_bytes(b"weights")

                copy_bundle_resources(
                    ModelSourceFiles(
                        directory=source_directory,
                        configuration_path=config_path,
                        tokenizer_path=tokenizer_path,
                        weights_path=weights_path,
                    ),
                    output_directory,
                )

                output_directory = Path(output_directory)
                self.assertTrue((output_directory / "configuration.json").exists())
                self.assertTrue((output_directory / "tokenizer.model").exists())

    def test_load_model_state_dict_accepts_pickled_state_dict_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "weights.pkl"
            expected = {"model.embed_tokens.weight": torch.zeros((2, 2))}
            torch.save({"state_dict": expected}, weights_path)

            state_dict = load_model_state_dict(weights_path)

        self.assertEqual(tuple(state_dict["model.embed_tokens.weight"].shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
