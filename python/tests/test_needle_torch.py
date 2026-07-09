import typing
import unittest

import torch

from needle import Needle, NeedleDecoder
from needle.needle_configuration import NeedleModelConfiguation


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
        decoder: NeedleDecoder = model.decoder
        encoder_input_ids = torch.tensor([[1, 2, 3, 0]])
        decoder_input_ids = torch.tensor([[1, 2]])

        cross_attention_mask, encoder_projected_k, encoder_projected_v = model.encoder(
            encoder_input_ids
        )
        self_attention_mask = torch.zeros((1, 1, 2, 16))
        logits = decoder(
            decoder_input_ids,
            torch.zeros((1,), dtype=torch.int32),
            self_attention_mask,
            cross_attention_mask,
            encoder_projected_k,
            encoder_projected_v,
        )

        self.assertEqual(tuple(cross_attention_mask.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(encoder_projected_k.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(encoder_projected_v.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(logits.shape), (1, 2, 32))
        self.assertEqual(
            tuple(typing.cast(torch.Tensor, decoder.key_cache).shape), (1, 16, 1, 4)
        )
        self.assertEqual(
            tuple(typing.cast(torch.Tensor, decoder.value_cache).shape),
            (1, 16, 1, 4),
        )
        self.assertFalse(hasattr(decoder, "cache_offset"))

        model.reset()

        key_cache = typing.cast(torch.Tensor, decoder.key_cache)
        value_cache = typing.cast(torch.Tensor, decoder.value_cache)
        self.assertTrue(torch.equal(key_cache, torch.zeros_like(key_cache)))
        self.assertTrue(torch.equal(value_cache, torch.zeros_like(value_cache)))

    def test_forward_runs_encode_then_decode(self) -> None:
        model = Needle(mock_configuration())
        decoder: NeedleDecoder = model.decoder
        logits = model(
            torch.tensor([[1, 2, 3, 0]]),
            torch.tensor([[1, 2]]),
        )

        self.assertEqual(tuple(logits.shape), (1, 2, 32))
        self.assertFalse(hasattr(decoder, "cache_offset"))

    def test_decode_rejects_non_unit_batch(self) -> None:
        model = Needle(mock_configuration())
        cross_attention_mask, encoder_projected_k, encoder_projected_v = model.encoder(
            torch.tensor([[1, 2, 3, 0]])
        )

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            model.decoder(
                torch.tensor([[1], [2]]),
                torch.zeros((1,), dtype=torch.int32),
                torch.zeros((1, 1, 1, 16)),
                cross_attention_mask,
                encoder_projected_k,
                encoder_projected_v,
            )


if __name__ == "__main__":
    unittest.main()
