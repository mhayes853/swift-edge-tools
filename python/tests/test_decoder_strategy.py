import unittest

from needle import NeedleModelConfiguation
from needle.cache_layout import (  # pyright: ignore[reportMissingImports]
    decoder_state_names,
    decoder_state_shape,
)
from needle.decoder_strategy import (  # pyright: ignore[reportMissingImports]
    DecoderExportStrategy,
)


class DecoderStrategyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
