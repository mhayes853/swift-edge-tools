import typing
import unittest

import torch

from needle import DecoderExportStrategy, Needle, NeedleDecoder
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

    def test_encode_decode_with_explicit_caches(self) -> None:
        model = Needle(
            mock_configuration(),
            decoder_strategy=DecoderExportStrategy.reference(),
        )
        decoder: NeedleDecoder = model.decoder
        encoder_input_ids = torch.tensor([[1, 2, 3, 0]])
        decoder_input_ids = torch.tensor([[1]])

        cross_attention_mask, encoder_projected_k, encoder_projected_v = model.encoder(
            encoder_input_ids
        )
        self_attention_mask = torch.zeros((1, 1, 1, 16))
        key_cache = torch.zeros((1, 16, 1, 4))
        value_cache = torch.zeros((1, 16, 1, 4))
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

        # Caches and the encoder handoff stay at kv_heads; expansion happens in attention.
        self.assertEqual(tuple(cross_attention_mask.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(encoder_projected_k.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(encoder_projected_v.shape), (1, 1, 1, 4, 4))
        self.assertEqual(tuple(logits.shape), (1, 1, 32))
        self.assertEqual(tuple(updated_key_cache.shape), (1, 16, 1, 4))
        self.assertEqual(tuple(updated_value_cache.shape), (1, 16, 1, 4))

    def test_decoder_can_store_kv_caches_in_buffers(self) -> None:
        configuration = mock_configuration()
        stateless = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.reference(),
        )
        stateful = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.coreai(),
        )
        stateful.load_state_dict(stateless.state_dict())

        encoder_input_ids = torch.tensor([[1, 2, 3, 0]])
        encoder_outputs = stateless.encoder(encoder_input_ids)
        decoder_prefix = (
            torch.tensor([[1]]),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1, 1, 1, configuration.decoder_max_length)),
            *encoder_outputs,
        )
        key_cache = torch.zeros((1, 16, 1, 4))
        value_cache = torch.zeros((1, 16, 1, 4))

        stateless_logits, updated_key, updated_value = stateless.decoder(
            *decoder_prefix,
            key_cache,
            value_cache,
        )
        stateful_logits = stateful.decoder(*decoder_prefix)

        self.assertIsInstance(stateful_logits, torch.Tensor)
        torch.testing.assert_close(stateful_logits, stateless_logits)
        layer = stateful.decoder.model.decoder.layers[0]
        torch.testing.assert_close(
            layer.self_attention_key_cache,
            updated_key[0],
        )
        torch.testing.assert_close(
            layer.self_attention_value_cache,
            updated_value[0],
        )
        self.assertNotIn(
            "self_attention_key_cache_0",
            dict(stateful.decoder.named_buffers(recurse=False)),
        )

    def test_stateless_decoder_can_return_cache_deltas(self) -> None:
        configuration = mock_configuration()
        full_cache_model = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.reference(),
        )
        delta_model = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.onnx(),
        )
        delta_model.load_state_dict(full_cache_model.state_dict())
        encoder_outputs = full_cache_model.encoder(torch.tensor([[1, 2, 3, 0]]))
        decoder_prefix = (
            torch.tensor([[1]]),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1, 1, 1, configuration.decoder_max_length)),
            *encoder_outputs,
        )
        key_cache = torch.zeros((1, 16, 1, 4))
        value_cache = torch.zeros((1, 16, 1, 4))

        full_logits, updated_key, updated_value = full_cache_model.decoder(
            *decoder_prefix,
            key_cache,
            value_cache,
        )
        delta_logits, key_delta, value_delta = delta_model.decoder(
            *decoder_prefix,
            key_cache,
            value_cache,
        )

        torch.testing.assert_close(delta_logits, full_logits)
        torch.testing.assert_close(key_delta, updated_key[:, :1])
        torch.testing.assert_close(value_delta, updated_value[:, :1])
        self.assertEqual(tuple(key_delta.shape), (1, 1, 1, 4))
        self.assertEqual(tuple(value_delta.shape), (1, 1, 1, 4))

    def test_per_layer_self_and_cross_attention_states_match_stateless_decode(
        self,
    ) -> None:
        configuration = mock_configuration()
        stateless = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.reference(
                use_native_gqa=False,
            ),
        )
        stateful = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.coreml(),
        )
        stateful.load_state_dict(stateless.state_dict())
        encoder_outputs = stateless.encoder(torch.tensor([[1, 2, 3, 0]]))
        cross_attention_mask, encoder_projected_k, encoder_projected_v = encoder_outputs
        encoder_length = encoder_projected_k.shape[-2]
        layer = stateful.decoder.model.decoder.layers[0]
        cross_key_cache = typing.cast(torch.Tensor, layer.cross_attention_key_cache)
        cross_value_cache = typing.cast(
            torch.Tensor,
            layer.cross_attention_value_cache,
        )
        cross_key_cache[..., :encoder_length, :] = encoder_projected_k[0]
        cross_value_cache[..., :encoder_length, :] = encoder_projected_v[0]
        decoder_prefix = (
            torch.tensor([[1]]),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1, 1, 1, configuration.decoder_max_length)),
            cross_attention_mask,
        )
        key_cache = torch.zeros((1, 16, 1, 4))
        value_cache = torch.zeros((1, 16, 1, 4))

        stateless_logits, updated_key, updated_value = stateless.decoder(
            *decoder_prefix,
            encoder_projected_k,
            encoder_projected_v,
            key_cache,
            value_cache,
        )
        stateful_logits = stateful.decoder(*decoder_prefix)

        torch.testing.assert_close(stateful_logits, stateless_logits)
        torch.testing.assert_close(
            layer.self_attention_key_cache,
            updated_key[0],
        )
        torch.testing.assert_close(
            layer.self_attention_value_cache,
            updated_value[0],
        )
        self.assertEqual(
            stateful.decoder.state_names,
            (
                "self_attention_key_cache_0",
                "self_attention_value_cache_0",
                "cross_attention_key_cache_0",
                "cross_attention_value_cache_0",
            ),
        )
        self.assertEqual(
            stateful.decoder.state_buffer_names,
            (
                "model.decoder.layers.0.self_attention_key_cache",
                "model.decoder.layers.0.self_attention_value_cache",
                "model.decoder.layers.0.cross_attention_key_cache",
                "model.decoder.layers.0.cross_attention_value_cache",
            ),
        )

    def test_stateful_decoder_uses_the_active_cache_prefix(self) -> None:
        configuration = mock_configuration()
        stateless = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.reference(
                use_native_gqa=False,
            ),
        )
        stateful = Needle(
            configuration,
            decoder_strategy=DecoderExportStrategy.coreml(dynamic_cache=True),
        )
        stateful.load_state_dict(stateless.state_dict())

        encoder_outputs = stateless.encoder(torch.tensor([[1, 2, 3, 0]]))
        cross_attention_mask, encoder_projected_k, encoder_projected_v = encoder_outputs
        encoder_length = encoder_projected_k.shape[-2]
        layer = stateful.decoder.model.decoder.layers[0]
        cross_key_cache = typing.cast(torch.Tensor, layer.cross_attention_key_cache)
        cross_value_cache = typing.cast(
            torch.Tensor,
            layer.cross_attention_value_cache,
        )
        cross_key_cache[..., :encoder_length, :] = encoder_projected_k[0]
        cross_value_cache[..., :encoder_length, :] = encoder_projected_v[0]
        active_length = 8
        decoder_prefix = (
            torch.tensor([[1]]),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1, 1, 1, active_length)),
            cross_attention_mask,
        )
        key_cache = torch.zeros((1, active_length, 1, 4))
        value_cache = torch.zeros((1, active_length, 1, 4))

        stateless_logits, updated_key, updated_value = stateless.decoder(
            *decoder_prefix,
            encoder_projected_k,
            encoder_projected_v,
            key_cache,
            value_cache,
        )
        stateful_logits = stateful.decoder(*decoder_prefix)

        torch.testing.assert_close(stateful_logits, stateless_logits)
        self_key_cache = typing.cast(torch.Tensor, layer.self_attention_key_cache)
        self_value_cache = typing.cast(
            torch.Tensor,
            layer.self_attention_value_cache,
        )
        torch.testing.assert_close(
            self_key_cache[:active_length],
            updated_key[0],
        )
        torch.testing.assert_close(
            self_value_cache[:active_length],
            updated_value[0],
        )
        self.assertEqual(torch.count_nonzero(self_key_cache[active_length:]), 0)
        self.assertEqual(torch.count_nonzero(self_value_cache[active_length:]), 0)

    def test_attention_strategy_is_selectable(self) -> None:
        encoder_input_ids = torch.tensor([[1, 2, 3, 0]])

        native = torch.export.export(
            Needle(
                mock_configuration(),
                decoder_strategy=DecoderExportStrategy.reference(),
                encoder_use_native_sdpa=True,
            ).encoder,
            (encoder_input_ids,),
            strict=False,
        )
        canonical = torch.export.export(
            Needle(
                mock_configuration(),
                decoder_strategy=DecoderExportStrategy.reference(),
                encoder_use_native_sdpa=False,
            ).encoder,
            (encoder_input_ids,),
            strict=False,
        )

        self.assertIn("scaled_dot_product_attention", native.graph_module.code)
        self.assertNotIn("scaled_dot_product_attention", canonical.graph_module.code)
        self.assertIn("torch.ops.aten.matmul", canonical.graph_module.code)
        self.assertIn("torch.ops.aten.softmax", canonical.graph_module.code)

    def test_needle_exposes_unregistered_export_modules(self) -> None:
        model = Needle(
            mock_configuration(),
            decoder_strategy=DecoderExportStrategy.reference(),
        )

        self.assertIsInstance(model.encoder, torch.nn.Module)
        self.assertIsInstance(model.decoder, torch.nn.Module)
        self.assertTrue(
            all(key.startswith(("model.", "lm_head.")) for key in model.state_dict()),
            model.state_dict().keys(),
        )


if __name__ == "__main__":
    unittest.main()
