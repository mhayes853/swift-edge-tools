#!/usr/bin/env python3
"""Measure Needle TTFT and decode throughput for an exported bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from needle.benchmark import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    RUNNERS,
    CoreAIRunner,
    CoreMLRunner,
    OnnxRunner,
    run_benchmark,
    self_attention_mask,
    total_bytes,
)

__all__ = [
    "RUNNERS",
    "CoreAIRunner",
    "CoreMLRunner",
    "OnnxRunner",
    "load_configuration",
    "main",
    "run_benchmark",
    "self_attention_mask",
    "total_bytes",
]


def load_configuration(bundle: Path) -> dict[str, Any]:
    configuration_path = bundle / "configuration.json"
    try:
        configuration = json.loads(configuration_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to load model configuration at {configuration_path}"
        ) from error
    if not isinstance(configuration, dict):
        raise ValueError(f"Expected a JSON object at {configuration_path}")
    return configuration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--backend", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--encoder-length", type=int)
    parser.add_argument(
        "--cache-length",
        type=int,
        help="Maximum decoder cache length (defaults to the exported configuration).",
    )
    parser.add_argument(
        "--initial-cache-length",
        type=int,
        help="Initial active cache length (defaults to 128 for ONNX).",
    )
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layout", choices=("shift", "index"), default="index")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args()

    bundle = Path(parsed.bundle).expanduser().resolve()
    configuration = load_configuration(bundle)
    runner = RUNNERS[parsed.backend](bundle)
    results = run_benchmark(
        runner,
        configuration,
        prompt_tokens=parsed.prompt_tokens,
        encoder_length=parsed.encoder_length,
        cache_length=parsed.cache_length,
        initial_cache_length=parsed.initial_cache_length,
        generate=parsed.generate,
        warmup=parsed.warmup,
        repeats=parsed.repeats,
        layout=parsed.layout,
    )

    if parsed.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"backend           {results['backend']}")
    print(f"initial cache     {results['cache_shape']}")
    print(f"final cache       {results['final_cache_shape']}")
    print(f"maximum cache     {results['maximum_cache_shape']}")
    print(f"prompt/encoder    {results['prompt_tokens']}/{results['encoder_length']}")
    print(f"TTFT              {results['ttft_ms']:.1f} ms")
    print(f"  encode          {results['encode_ms']:.1f} ms")
    print(f"  cross KV init   {results['cross_cache_init_ms']:.1f} ms")
    print(f"  first step      {results['first_step_ms']:.1f} ms")
    print(f"model decode      {results['model_decode_step_ms']:.2f} ms")
    print(f"cache update      {results['cache_update_ms']:.3f} ms")
    print(f"decode step       {results['decode_step_ms']:.2f} ms")
    print(f"decode TPS        {results['decode_tps']:.1f}")
    print(f"decoder I/O/step  {results['decoder_bytes_per_step'] / 2**20:.2f} MiB")
    print(f"encoder outputs   {results['encoder_output_bytes'] / 2**20:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
