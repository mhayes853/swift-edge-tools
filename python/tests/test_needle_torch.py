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
    def test_encode_decode_with_explicit_caches(self) -> None:
        model = Needle(mock_configuration())
        decoder: NeedleDecoder = model.decoder
        encoder_input_ids = torch.tensor([[1, 2, 3, 0]])
        decoder_input_ids = torch.tensor([[1]])

        cross_attention_mask, encoder_projected_k, encoder_projected_v = model.encoder(
            encoder_input_ids
        )
        self_attention_mask = torch.zeros((1, 1, 1, 16))
        key_cache = torch.zeros((1, 16, 2, 4))
        value_cache = torch.zeros((1, 16, 2, 4))
        logits, updated_key_cache, updated_value_cache = decoder(
            decoder_input_ids,
            torch.zeros((1,), dtype=torch.int32),
            self_attention_mask,
            cross_attention_mask,
            encoder_projected_k,
            encoder_projected_v,
            key_cache,
            value_cache,
        )

        self.assertEqual(tuple(cross_attention_mask.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(encoder_projected_k.shape), (1, 1, 2, 4, 4))
        self.assertEqual(tuple(encoder_projected_v.shape), (1, 1, 2, 4, 4))
        self.assertEqual(tuple(logits.shape), (1, 1, 32))
        self.assertEqual(tuple(updated_key_cache.shape), (1, 16, 2, 4))
        self.assertEqual(tuple(updated_value_cache.shape), (1, 16, 2, 4))

    def test_needle_wraps_export_modules(self) -> None:
        model = Needle(mock_configuration())

        self.assertNotIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model.encoder, torch.nn.Module)
        self.assertIsInstance(model.decoder, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
