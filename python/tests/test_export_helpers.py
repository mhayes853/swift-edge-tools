import tempfile
import unittest
from pathlib import Path

import torch

from needle import Needle, NeedleModelConfiguation
from needle.export.helpers import (
    decoder_export_spec,
    encoder_export_spec,
    load_needle_model,
)


def mock_configuration(dtype: str = "float32") -> NeedleModelConfiguation:
    return NeedleModelConfiguation(
        vocabulary_size=16,
        dimensions=8,
        hidden_dimensions=8,
        attention_heads=2,
        kv_heads=1,
        encoder_layers=1,
        decoder_layers=1,
        hidden_layers=1,
        max_seq_len=32,
        dtype=dtype,
    )


class ExportHelpersTests(unittest.TestCase):
    def test_export_specs_define_the_onnx_contract(self) -> None:
        configuration = mock_configuration()

        self.assertEqual(encoder_export_spec(configuration).input_names, ("input_ids",))
        self.assertEqual(
            decoder_export_spec(configuration).input_names,
            (
                "input_ids",
                "cache_position",
                "self_attention_mask",
                "cross_attention_mask",
                "encoder_projected_k",
                "encoder_projected_v",
                "key_cache",
                "value_cache",
            ),
        )
        self.assertEqual(
            decoder_export_spec(configuration).output_names,
            ("logits", "key_cache_delta", "value_cache_delta"),
        )

    def test_load_needle_model_uses_configuration_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as source_name:
            source_directory = Path(source_name)
            configuration = mock_configuration(dtype="bfloat16")
            needle = Needle(configuration)
            torch.save(needle.state_dict(), source_directory / "weights.pkl")

            loaded = load_needle_model(
                configuration,
                source_directory / "weights.pkl",
            )

            self.assertEqual(next(loaded.parameters()).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
