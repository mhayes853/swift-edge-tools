import unittest

import torch

from needle import Needle, NeedleDecoder, NeedleModelConfiguation


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
    def test_decoder_context_defaults_to_canonical_length(self) -> None:
        self.assertEqual(NeedleModelConfiguation().encoder_max_length, 1024)
        self.assertEqual(NeedleModelConfiguation().decoder_max_length, 512)
        self.assertEqual(mock_configuration().decoder_max_length, 16)
        self.assertEqual(
            NeedleModelConfiguation.from_json_object(
                {"max_seq_len": 1024, "max_dec_len": 128}
            ).decoder_max_length,
            128,
        )

    def test_encode_decode_returns_explicit_cache_deltas(self) -> None:
        configuration = mock_configuration()
        model = Needle(configuration)
        decoder: NeedleDecoder = model.decoder
        encoder_outputs = model.encoder(torch.tensor([[1, 2, 3, 0]]))
        key_cache = torch.zeros((1, 16, 1, 4))
        value_cache = torch.zeros((1, 16, 1, 4))

        logits, key_delta, value_delta = decoder(
            torch.tensor([[1]]),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1, 1, 1, configuration.decoder_max_length)),
            *encoder_outputs,
            key_cache,
            value_cache,
        )

        cross_attention_mask, encoder_projected_k, encoder_projected_v = (
            encoder_outputs
        )
        self.assertEqual(tuple(cross_attention_mask.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(encoder_projected_k.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(encoder_projected_v.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(logits.shape), (1, 1, 32))
        self.assertEqual(tuple(key_delta.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(value_delta.shape), (1, 1, 1, 4))

    def test_cache_delta_uses_requested_position(self) -> None:
        configuration = mock_configuration()
        model = Needle(configuration)
        encoder_outputs = model.encoder(torch.tensor([[1, 2, 3, 0]]))
        key_cache = torch.zeros((1, 16, 1, 4))
        value_cache = torch.zeros((1, 16, 1, 4))

        _, key_delta, value_delta = model.decoder(
            torch.tensor([[1]]),
            torch.tensor([5], dtype=torch.int32),
            torch.zeros((1, 1, 1, configuration.decoder_max_length)),
            *encoder_outputs,
            key_cache,
            value_cache,
        )

        self.assertGreater(torch.count_nonzero(key_delta).item(), 0)
        self.assertGreater(torch.count_nonzero(value_delta).item(), 0)
        self.assertEqual(torch.count_nonzero(key_cache).item(), 0)
        self.assertEqual(torch.count_nonzero(value_cache).item(), 0)

    def test_needle_exposes_unregistered_export_modules(self) -> None:
        model = Needle(mock_configuration())

        self.assertIsInstance(model.encoder, torch.nn.Module)
        self.assertIsInstance(model.decoder, torch.nn.Module)
        self.assertTrue(
            all(key.startswith(("model.", "lm_head.")) for key in model.state_dict()),
            model.state_dict().keys(),
        )


if __name__ == "__main__":
    unittest.main()
