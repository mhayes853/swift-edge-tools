from __future__ import annotations

import unittest

import numpy as np

from needle.benchmark.interface import (  # pyright: ignore[reportMissingImports]
    BenchmarkRunner,
)
from scripts.benchmark_needle import run_benchmark


class _ONNXRunner(BenchmarkRunner):
    name = "onnx"
    stateful = False
    cross_attention_stateful = False
    cache_shape = ()

    def __init__(self) -> None:
        self.cache_lengths: list[int] = []
        self.dtypes = {
            "input_ids": np.int64,
            "cache_position": np.int64,
            "self_attention_mask": np.float32,
            "cross_attention_mask": np.float32,
            "encoder_projected_k": np.float32,
            "encoder_projected_v": np.float32,
            "key_cache": np.float32,
            "value_cache": np.float32,
            "logits": np.float32,
            "key_cache_delta": np.float32,
            "value_cache_delta": np.float32,
        }
        self.feed_dtypes = self.dtypes

    def reset_state(self) -> None:
        pass

    def encode(self, input_ids: np.ndarray) -> dict[str, np.ndarray]:
        encoder_length = input_ids.shape[1]
        projected_shape = (1, 1, 1, encoder_length, 2)
        return {
            "cross_attention_mask": np.zeros(
                (1, 1, 1, encoder_length), dtype=np.float32
            ),
            "encoder_projected_k": np.zeros(projected_shape, dtype=np.float32),
            "encoder_projected_v": np.zeros(projected_shape, dtype=np.float32),
        }

    def initialize_cross_attention_state(
        self,
        encoder_outputs: dict[str, np.ndarray],
    ) -> None:
        pass

    def decode(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.cache_lengths.append(inputs["key_cache"].shape[1])
        delta_shape = (inputs["key_cache"].shape[0], 1, 1, 2)
        return {
            "logits": np.zeros((1, 1, 8), dtype=np.float32),
            "key_cache_delta": np.full(delta_shape, len(self.cache_lengths)),
            "value_cache_delta": np.full(delta_shape, len(self.cache_lengths)),
        }


class BenchmarkNeedleTests(unittest.TestCase):
    def test_onnx_cache_grows_without_reducing_maximum_context(self) -> None:
        runner = _ONNXRunner()
        configuration = {
            "max_seq_len": 8,
            "decoder_max_length": 8,
            "decoder_start_token_id": 1,
            "pad_token_id": 0,
            "num_decoder_layers": 1,
            "num_attention_heads": 1,
            "num_kv_heads": 1,
            "d_model": 2,
            "vocab_size": 8,
        }

        results = run_benchmark(
            runner,
            configuration,
            prompt_tokens=4,
            encoder_length=4,
            cache_length=8,
            initial_cache_length=2,
            generate=5,
            warmup=0,
            repeats=1,
            layout="index",
        )

        self.assertEqual(runner.cache_lengths, [2, 2, 4, 4, 8])
        self.assertEqual(results["cache_shape"], [1, 2, 1, 2])
        self.assertEqual(results["final_cache_shape"], [1, 8, 1, 2])
        self.assertEqual(results["maximum_cache_shape"], [1, 8, 1, 2])


if __name__ == "__main__":
    unittest.main()
