import tempfile
import unittest
from pathlib import Path

import torch

from needle import DecoderExportStrategy, Needle, NeedleModelConfiguation
from needle.export.helpers import (
    encoder_dynamic_shapes,
    export_program,
    load_needle_model,
    sample_encoder_input,
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
    def test_export_program_uses_dynamic_encoder_sequence_length(self) -> None:
        configuration = mock_configuration()
        needle = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.reference(),
        )
        sample = (sample_encoder_input(configuration, 4),)

        exported_program = export_program(
            needle.encoder,
            sample,
            dynamic_shapes=encoder_dynamic_shapes(configuration, dynamic_buffers=True),
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
        with tempfile.TemporaryDirectory() as source_name:
            source_directory = Path(source_name)
            configuration = mock_configuration(dtype="bfloat16")
            strategy = DecoderExportStrategy.reference()
            needle = Needle(configuration, decoder_strategy=strategy)
            torch.save(needle.state_dict(), source_directory / "weights.pkl")

            loaded = load_needle_model(
                configuration,
                source_directory / "weights.pkl",
                decoder_strategy=strategy,
            )

            self.assertEqual(next(loaded.parameters()).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
