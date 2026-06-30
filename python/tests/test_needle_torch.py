import typing
import unittest

import torch

from swift_needle.needle_configuration import NeedleModelConfiguation
from swift_needle.needle_torch import Needle, NeedleDecoder


def mock_configuration() -> NeedleModelConfiguation:
    return NeedleModelConfiguation(
        vocabulary_size=32,
        dimensions=8,
        hidden_dimensions=8,
        attention_heads=2,
        kv_heads=1,
        encoder_layers=1,
        decoder_layers=1,
        hidden_layers=2,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        decoder_start_token_id=1,
        tie_word_embeddings=False,
        max_seq_len=16,
    )


class NeedleTorchTests(unittest.TestCase):
    def test_encode_decode_and_reset(self) -> None:
        model = Needle(mock_configuration())
        decoder = typing.cast(NeedleDecoder, model.decoder)
        encoder_input_ids = torch.tensor([[1, 2, 3, 0]])
        decoder_input_ids = torch.tensor([[1, 2]])

        cross_attention_mask, encoder_projected_k, encoder_projected_v = model.encoder(
            encoder_input_ids
        )
        logits = decoder(
            decoder_input_ids,
            cross_attention_mask,
            encoder_projected_k,
            encoder_projected_v,
        )

        self.assertEqual(tuple(cross_attention_mask.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(encoder_projected_k.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(encoder_projected_v.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(logits.shape), (1, 2, 32))
        self.assertEqual(
            tuple(typing.cast(torch.Tensor, decoder.k_cache).shape), (1, 1, 1, 16, 4)
        )
        self.assertEqual(int(typing.cast(torch.Tensor, decoder.cache_offset).item()), 2)

        model.reset()

        self.assertEqual(int(typing.cast(torch.Tensor, decoder.cache_offset).item()), 0)

    def test_forward_runs_encode_then_decode(self) -> None:
        model = Needle(mock_configuration())
        decoder = typing.cast(NeedleDecoder, model.decoder)
        logits = model(
            torch.tensor([[1, 2, 3, 0]]),
            torch.tensor([[1, 2]]),
        )

        self.assertEqual(tuple(logits.shape), (1, 2, 32))
        self.assertEqual(int(typing.cast(torch.Tensor, decoder.cache_offset).item()), 2)

    def test_decode_rejects_non_unit_batch(self) -> None:
        model = Needle(mock_configuration())
        cross_attention_mask, encoder_projected_k, encoder_projected_v = model.encoder(
            torch.tensor([[1, 2, 3, 0]])
        )

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            model.decoder(
                torch.tensor([[1], [2]]),
                cross_attention_mask,
                encoder_projected_k,
                encoder_projected_v,
            )


if __name__ == "__main__":
    unittest.main()
