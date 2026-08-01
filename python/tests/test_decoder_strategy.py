import unittest

from needle import NeedleModelConfiguation
from needle.cache_layout import (  # pyright: ignore[reportMissingImports]
    decoder_state_names,
    decoder_state_shape,
)
from needle.decoder_strategy import (  # pyright: ignore[reportMissingImports]
    ActiveCacheStrategy,
    AttentionImplementation,
    CacheLayout,
    CacheOutput,
    DecoderExportStrategy,
)


class DecoderStrategyTests(unittest.TestCase):
    def test_backend_defaults_capture_measured_fast_paths(self) -> None:
        onnx = DecoderExportStrategy.onnx()
        self.assertEqual(onnx.cache_layout, CacheLayout.EXPLICIT)
        self.assertEqual(onnx.cache_output, CacheOutput.DELTA)
        self.assertEqual(onnx.active_cache, ActiveCacheStrategy.ADAPTIVE_RUNTIME)
        self.assertTrue(onnx.uses_native_attention)
        self.assertTrue(onnx.use_native_gqa)

        coreml = DecoderExportStrategy.coreml()
        self.assertEqual(coreml.cache_layout, CacheLayout.PER_LAYER_STATE)
        self.assertTrue(coreml.cross_attention_cache_states)
        self.assertEqual(coreml.attention, AttentionImplementation.DECOMPOSED)
        self.assertFalse(coreml.use_native_gqa)

        coreai = DecoderExportStrategy.coreai()
        self.assertEqual(coreai.cache_layout, CacheLayout.PER_LAYER_STATE)
        self.assertFalse(coreai.cross_attention_cache_states)
        self.assertTrue(coreai.uses_native_attention)
        self.assertTrue(coreai.use_native_gqa)

    def test_cache_layout_owns_state_names_and_shapes(self) -> None:
        configuration = NeedleModelConfiguation(
            dimensions=8,
            attention_heads=2,
            kv_heads=1,
            decoder_layers=2,
            max_seq_len=16,
        )
        strategy = DecoderExportStrategy.coreml()

        self.assertEqual(
            decoder_state_names(configuration, strategy),
            (
                "self_attention_key_cache_0",
                "self_attention_value_cache_0",
                "cross_attention_key_cache_0",
                "cross_attention_value_cache_0",
                "self_attention_key_cache_1",
                "self_attention_value_cache_1",
                "cross_attention_key_cache_1",
                "cross_attention_value_cache_1",
            ),
        )
        self.assertEqual(
            decoder_state_shape("self_attention_key_cache_0", configuration),
            (16, 1, 4),
        )
        self.assertEqual(
            decoder_state_shape("cross_attention_key_cache_0", configuration),
            (1, 1, 16, 4),
        )

    def test_arbitrary_strategy_construction_is_not_public(self) -> None:
        with self.assertRaises(TypeError):
            DecoderExportStrategy()  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
