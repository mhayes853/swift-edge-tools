from __future__ import annotations

import statistics
import time
from collections.abc import Mapping
from typing import TypedDict

import numpy as np
from numpy.typing import DTypeLike

from .interface import (  # pyright: ignore[reportMissingImports]
    BenchmarkRunner,
    DTypeSpec,
    TensorMap,
)


class BenchmarkResult(TypedDict):
    backend: str
    prompt_tokens: int
    encoder_length: int
    generated_tokens: int
    cache_shape: list[int]
    maximum_cache_shape: list[int]
    final_cache_shape: list[int]
    ttft_ms: float
    encode_ms: float
    cross_cache_init_ms: float
    first_step_ms: float
    model_decode_step_ms: float
    cache_update_ms: float
    decode_step_ms: float
    decode_tps: float
    decoder_bytes_per_step: int
    encoder_output_bytes: int


class _PassResult(TypedDict):
    encode_seconds: float
    cross_cache_init_seconds: float
    first_step_seconds: float
    steady_step_seconds: list[float]
    model_step_seconds: list[float]
    cache_update_seconds: list[float]
    decoder_bytes_per_step: int
    final_cache_length: int
    encoder_bytes: int


def _configuration_int(
    configuration: Mapping[str, object],
    key: str,
    default: int | None = None,
) -> int:
    value = configuration.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer configuration value for {key}")
    return value


def self_attention_mask(step: int, max_length: int, layout: str) -> np.ndarray:
    mask = np.full((1, 1, 1, max_length), -65500.0, dtype=np.float32)
    if layout == "shift":
        mask[..., max(0, max_length - step - 1) :] = 0.0
    else:
        mask[..., : step + 1] = 0.0
    return mask


def total_bytes(
    tensors: Mapping[str, np.ndarray],
    dtypes: Mapping[str, DTypeSpec] | None = None,
) -> int:
    """Return bytes as the backend consumes them, not as NumPy holds them."""
    total = 0
    for name, value in tensors.items():
        array = np.asarray(value)
        dtype = (dtypes or {}).get(name)
        itemsize = np.dtype(dtype).itemsize if dtype is not None else array.itemsize
        total += array.size * itemsize
    return total


def run_benchmark(
    runner: BenchmarkRunner,
    configuration: Mapping[str, object],
    *,
    prompt_tokens: int,
    encoder_length: int | None,
    cache_length: int | None,
    initial_cache_length: int | None,
    generate: int,
    warmup: int,
    repeats: int,
    layout: str,
) -> BenchmarkResult:
    configured_encoder_length = configuration.get("max_seq_len")
    if not isinstance(configured_encoder_length, int):
        configured_encoder_length = configuration.get("max_position_embeddings")
    if not isinstance(configured_encoder_length, int):
        configured_encoder_length = 1024
    if encoder_length is None:
        encoder_length = configured_encoder_length

    configured_cache_length = configuration.get("decoder_max_length")
    runner_cache_length = runner.cache_shape[1] if len(runner.cache_shape) > 1 else None
    if cache_length is None and isinstance(runner_cache_length, int):
        cache_length = runner_cache_length
    if cache_length is None and isinstance(configured_cache_length, int):
        cache_length = configured_cache_length
    if cache_length is None:
        cache_length = min(configured_encoder_length, 512)

    if initial_cache_length is None:
        initial_cache_length = (
            min(cache_length, 128) if runner.name == "onnx" else cache_length
        )
    if initial_cache_length < 1 or initial_cache_length > cache_length:
        raise ValueError("Initial cache length must be between 1 and cache length")
    if prompt_tokens > encoder_length:
        raise ValueError("Prompt length cannot exceed encoder length")
    if generate > cache_length:
        raise ValueError("Generated token count cannot exceed cache length")
    start_token = _configuration_int(configuration, "decoder_start_token_id", 1)
    pad_token = _configuration_int(configuration, "pad_token_id", 0)
    layers = _configuration_int(configuration, "num_decoder_layers")
    heads = _configuration_int(configuration, "num_attention_heads")
    dimensions = _configuration_int(configuration, "d_model")
    vocabulary_size = _configuration_int(configuration, "vocab_size")
    head_dimensions = dimensions // heads

    rng = np.random.default_rng(0)
    prompt = rng.integers(4, vocabulary_size, size=prompt_tokens)
    input_ids = np.full((1, encoder_length), pad_token, dtype=np.int64)
    input_ids[0, :prompt_tokens] = prompt

    cache_heads = (
        heads
        if layout == "shift"
        else _configuration_int(configuration, "num_kv_heads")
    )

    def cache_shape(active_length: int) -> tuple[int, int, int, int]:
        return (
            layers,
            active_length,
            cache_heads,
            head_dimensions,
        )

    initial_cache_shape = cache_shape(initial_cache_length)
    maximum_cache_shape = cache_shape(cache_length)

    dtypes = runner.dtypes
    feed_dtypes = runner.feed_dtypes

    def dtype_for(name: str) -> DTypeLike:
        """Return allocation dtype, which may differ from the declared dtype."""
        dtype = feed_dtypes.get(name)
        return np.float32 if dtype is None else dtype

    def empty_caches(active_length: int) -> tuple[np.ndarray, np.ndarray]:
        shape = cache_shape(active_length)
        return (
            np.zeros(shape, dtype=dtype_for("key_cache")),
            np.zeros(shape, dtype=dtype_for("value_cache")),
        )

    def grow_cache(cache: np.ndarray, active_length: int) -> np.ndarray:
        grown = np.zeros(cache_shape(active_length), dtype=cache.dtype)
        grown[:, : cache.shape[1], ...] = cache
        return grown

    def one_pass(collect: bool) -> _PassResult:
        encode_start = time.perf_counter()
        encoder_outputs = runner.encode(input_ids)
        encode_end = time.perf_counter()

        # Native consumers retain backend tensors. Cast once here rather than on
        # each step so this benchmark follows the same allocation pattern.
        encoder_outputs = {
            name: value.astype(dtype_for(name), copy=False)
            for name, value in encoder_outputs.items()
        }
        runner.reset_state()
        cross_cache_init_start = time.perf_counter()
        runner.initialize_cross_attention_state(encoder_outputs)
        cross_cache_init_end = time.perf_counter()
        active_cache_length = initial_cache_length
        key_cache, value_cache = empty_caches(active_cache_length)
        step_times = []
        model_step_times = []
        cache_update_times = []
        decoder_bytes = 0

        token_buffer = np.zeros((1, 1), dtype=dtype_for("input_ids"))
        token_buffer[0, 0] = start_token
        position_buffer = np.zeros((1,), dtype=dtype_for("cache_position"))
        mask_buffer = np.zeros(
            (1, 1, 1, active_cache_length),
            dtype=dtype_for("self_attention_mask"),
        )

        for step in range(generate):
            if step >= active_cache_length:
                active_cache_length = min(cache_length, active_cache_length * 2)
                if not runner.stateful:
                    key_cache = grow_cache(key_cache, active_cache_length)
                    value_cache = grow_cache(value_cache, active_cache_length)
                mask_buffer = np.zeros(
                    (1, 1, 1, active_cache_length),
                    dtype=dtype_for("self_attention_mask"),
                )

            position_buffer[0] = step
            np.copyto(
                mask_buffer,
                self_attention_mask(step, active_cache_length, layout),
                casting="unsafe",
            )
            inputs: TensorMap = {
                "input_ids": token_buffer,
                "cache_position": position_buffer,
                "self_attention_mask": mask_buffer,
                "cross_attention_mask": encoder_outputs["cross_attention_mask"],
            }
            if not runner.cross_attention_stateful:
                inputs["encoder_projected_k"] = encoder_outputs["encoder_projected_k"]
                inputs["encoder_projected_v"] = encoder_outputs["encoder_projected_v"]
            if not runner.stateful:
                inputs["key_cache"] = key_cache
                inputs["value_cache"] = value_cache
            step_start = time.perf_counter()
            outputs = runner.decode(inputs)
            model_step_end = time.perf_counter()

            if collect:
                decoder_bytes += total_bytes(inputs, dtypes) + total_bytes(
                    outputs, dtypes
                )
            if not runner.stateful and "key_cache_delta" in outputs:
                key_cache[:, step : step + 1] = outputs["key_cache_delta"]
                value_cache[:, step : step + 1] = outputs["value_cache_delta"]
            elif not runner.stateful:
                np.copyto(key_cache, outputs["updated_key_cache"], casting="unsafe")
                np.copyto(
                    value_cache,
                    outputs["updated_value_cache"],
                    casting="unsafe",
                )
            cache_update_end = time.perf_counter()
            model_step_times.append(model_step_end - step_start)
            cache_update_times.append(cache_update_end - model_step_end)
            step_times.append(cache_update_end - step_start)
            token_buffer[0, 0] = np.argmax(
                np.asarray(outputs["logits"], dtype=np.float32)
            )

        return {
            "encode_seconds": encode_end - encode_start,
            "cross_cache_init_seconds": cross_cache_init_end - cross_cache_init_start,
            "first_step_seconds": step_times[0],
            "steady_step_seconds": step_times[1:],
            "model_step_seconds": model_step_times[1:],
            "cache_update_seconds": cache_update_times[1:],
            "decoder_bytes_per_step": decoder_bytes // generate,
            "final_cache_length": active_cache_length,
            "encoder_bytes": total_bytes(encoder_outputs, runner.dtypes),
        }

    for _ in range(warmup):
        one_pass(collect=False)

    passes = [one_pass(collect=True) for _ in range(repeats)]
    ttfts = [
        result["encode_seconds"]
        + result["cross_cache_init_seconds"]
        + result["first_step_seconds"]
        for result in passes
    ]
    steady = [value for result in passes for value in result["steady_step_seconds"]]
    model_steady = [
        value for result in passes for value in result["model_step_seconds"]
    ]
    cache_updates = [
        value for result in passes for value in result["cache_update_seconds"]
    ]

    return {
        "backend": runner.name,
        "prompt_tokens": prompt_tokens,
        "encoder_length": encoder_length,
        "generated_tokens": generate,
        "cache_shape": list(initial_cache_shape),
        "maximum_cache_shape": list(maximum_cache_shape),
        "final_cache_shape": list(
            cache_shape(max(result["final_cache_length"] for result in passes))
        ),
        "ttft_ms": statistics.median(ttfts) * 1000,
        "encode_ms": statistics.median(result["encode_seconds"] for result in passes)
        * 1000,
        "cross_cache_init_ms": statistics.median(
            result["cross_cache_init_seconds"] for result in passes
        )
        * 1000,
        "first_step_ms": statistics.median(
            result["first_step_seconds"] for result in passes
        )
        * 1000,
        "model_decode_step_ms": statistics.median(model_steady) * 1000,
        "cache_update_ms": statistics.median(cache_updates) * 1000,
        "decode_step_ms": statistics.median(steady) * 1000,
        "decode_tps": 1.0 / statistics.median(steady),
        "decoder_bytes_per_step": passes[0]["decoder_bytes_per_step"],
        "encoder_output_bytes": passes[0]["encoder_bytes"],
    }
