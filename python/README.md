# Swift Needle Python Export

This directory contains the Python-side Needle model and CoreAI, CoreML, and
ONNX export workflows.

## Requirements

- Python 3.9+
- `torch`
- `coreai-torch`
- `huggingface_hub`
- `onnx`
- `onnxruntime`
- `onnxscript`
- `safetensors`
- `sentencepiece`
- `tokenizers`

Install dependencies from this directory:

```bash
pip install -e .
```

## Export Inputs

The export script accepts either:

1. A Hugging Face repo id, defaulting to `Cactus-Compute/needle`
2. A local model directory

The source directory or repo snapshot must contain:

- `config.json` or `configuration.json`
- a `.safetensors` weights file (preferred) or a `.pkl` weights file
- `tokenizer.model` or `tokenizer.json`

## Export Output

Every output bundle contains `configuration.json` and `tokenizer.model` or
`tokenizer.json`. CoreAI exports `encoder.aimodel` and `decoder.aimodel`, CoreML
exports `.mlpackage` models, and ONNX exports `encoder.onnx` and `decoder.onnx`.

If CoreAI ahead-of-time compilation is enabled, the output contains compiled
`encoder.*.aimodelc` and `decoder.*.aimodelc` assets instead of the source
`.aimodel` directories.

## CLI Usage

From `python/`:

```bash
python3 cli.py --output ./build/coreai-export
```

With a custom source:

```bash
python3 cli.py --source /path/to/needle-bundle --output ./build/coreai-export
```

With authoring metadata flags and quantization:

```bash
python3 cli.py \
  --source Cactus-Compute/needle \
  --output ./build/coreai-export \
  --authoring-author "Needle" \
  --authoring-description "CoreAI export" \
  --authoring-license "BSD-3-Clause" \
  --authoring-custom suite=cli \
  --quantizer-preset w4_per_block
```

For CoreAI int4 exports, prefer per-block weight quantization
(`--quantizer-preset w4_per_block`). Exports remain unquantized unless compression
is requested explicitly.

Needle uses independent encoder and decoder limits. The encoder supports up to
1024 tokens; decoder caches default to the canonical 512-token context. ONNX
exports use dynamic encoder and decoder-cache dimensions. CoreAI exports use a
dynamic encoder dimension and a static 512-token cache state, including
compressed exports. CoreML exports enumerate encoder lengths 128, 256, 512, and
1024 and use a static 512-token cache state.

CoreAI and CoreML decoders are stateful by default. Each decoder layer owns
separate self-attention K/V states, avoiding a full-cache stack at the end of
every decode. CoreML also stores cross-attention K/V in read-only decoder states
by default; CoreAI keeps those tensors as inputs because its current read-only
state workaround is slower. Native runtimes initialize CoreML
`cross_attention_{key,value}_cache_N` through `MLState` before the first decoder
call. Stateful Apple exports always use the measured per-layer layout. Use
`--decoder-profile reference` when an explicit full-cache interface is needed
for compatibility or parity testing.

With ahead-of-time compilation:

```bash
python3 cli.py \
  --source Cactus-Compute/needle \
  --output ./build/coreai-export \
  --compile-platform macOS
```

With an ONNX export:

```bash
python3 cli.py \
  --backend onnx \
  --source Cactus-Compute/needle \
  --output ./build/onnx-export
```

ONNX weight-only quantization uses ONNX Runtime's `MatMulNBits` representation:

```bash
python3 cli.py --backend onnx --onnx-quantization int4 --output ./build/onnx-int4
python3 cli.py --backend onnx --onnx-quantization int8 --output ./build/onnx-int8
```

The ONNX decoder returns only `key_cache_delta` and `value_cache_delta`, each
containing the newly generated row for every layer. The runtime appends these
small outputs to its local cache tensors instead of receiving complete updated
caches from the model. The benchmark includes that append in measured per-token
latency.

The ONNX benchmark retains the configured 512-position maximum cache but starts
with 128 active positions by default, growing to 256 and 512 only when needed:

```bash
python3 scripts/benchmark_needle.py ./build/onnx-export \
  --backend onnx \
  --encoder-length 128 \
  --generate 32
```

Use `--initial-cache-length 512` for a fixed-size baseline. `--cache-length`
sets the maximum capacity rather than the initial active size. Apple benchmark
results report `cross_cache_init_ms` separately and include it in TTFT.

For custom compression, use the Python API:

```python
from pathlib import Path

from needle import ONNXModelComponent, export_needle_onnx


class CustomCompressor:
    def compress(
        self,
        source: Path,
        destination: Path,
        *,
        component: ONNXModelComponent,
    ) -> None:
        # Apply a caller-defined ONNX transformation and write destination.
        compress_model(source, destination, component=component)


export_needle_onnx(
    "Cactus-Compute/needle",
    "./build/onnx-custom",
    compressor=CustomCompressor(),
)
```

With the preferred CoreML CPU/GPU decoder placement and decomposed GQA:

```bash
python3 cli.py \
  --backend coreml \
  --source Cactus-Compute/needle \
  --output ./build/coreml-export \
  --compute-units cpu-and-gpu
```

With ahead-of-time CoreML compilation for watchOS:

```bash
python3 cli.py \
  --backend coreml \
  --source Cactus-Compute/needle \
  --output ./build/coreml-watchos-export \
  --compile-platform watchOS
```

This produces `compiled/watchOS/encoder.mlmodelc` and
`compiled/watchOS/decoder.mlmodelc`. Multiple `--compile-platform` flags produce
one compiled model pair per platform. CoreML Needle requires the watchOS compiled
artifacts at runtime; source `.mlpackage` files are only emitted when no compile
platform is requested.

With metadata and quantizer config files:

```bash
python3 cli.py \
  --source Cactus-Compute/needle \
  --output ./build/coreai-export \
  --authoring-metadata ./metadata.yaml \
  --quantizer-config ./quantizer.yaml
```

The CLI accepts JSON or YAML for:

- `--authoring-metadata`
- `--quantizer-config`

Ahead-of-time compilation flags:

- `--compile-platform PLATFORM`

ONNX quantization flags:

- `--onnx-quantization {int4,int8}`

Decoder and compute-unit flags:

- `--decoder-profile {default,reference}`
- `--experimental-coreml-dynamic-cache`
- `--coreml-decoder-attention {automatic,native,decomposed}`
- `--compute-units {all,cpu-only,cpu-and-gpu,cpu-and-ne}`
- `--encoder-compute-units {all,cpu-only,cpu-and-gpu,cpu-and-ne}`
- `--decoder-compute-units {all,cpu-only,cpu-and-gpu,cpu-and-ne}`

CoreML defaults to CPU+ANE for the encoder and CPU+GPU for the decoder. The
legacy `--compute-units` flag overrides both components; the component-specific
flags take precedence. The CoreML decoder uses decomposed grouped attention by
default because it avoids native SDPA's K/V head expansion. Use
`--coreml-decoder-attention` to override the decoder strategy independently of
placement. CoreAI always exports native SDPA; compute-unit selection is a runtime
specialization preference rather than a graph change.

The Python implementation resolves these choices into one immutable
`DecoderExportStrategy` before model construction. Use the named
`reference()`, `onnx()`, `coreml()`, and `coreai()` constructors rather than
combining cache booleans. ONNX uses explicit adaptive caches with delta outputs,
CoreML uses per-layer self/cross states with decomposed attention, and CoreAI
uses per-layer self states with cross K/V inputs and native GQA. Cache names and
shapes live in `needle/cache_layout.py`; backend benchmark adapters live under
`needle/benchmark/`.

`--experimental-coreml-dynamic-cache` retains the fixed 512-position CoreML
state while allowing the self-attention mask to select a dynamic active prefix.
Benchmark with `--initial-cache-length 128` to start at 128 positions and grow
to 256/512. Dynamic CoreML state currently requires an uncompressed export.

Supported compression convenience flags:

- `--quantizer-preset {w4,w4_per_block,w8}`
- `--quantizer-execution-mode {graph,eager}`

Authoring metadata flags:

- `--authoring-author`
- `--authoring-description`
- `--authoring-license`
- `--authoring-custom KEY=VALUE`

## End-To-End Shell Script

The repository also includes a runnable shell wrapper:

```bash
./scripts/export_needle_coreai_e2e.sh
```

```bash
./scripts/export_needle_coreml_e2e.sh
./scripts/export_needle_onnx_e2e.sh
```

With an explicit source and output:

```bash
./scripts/export_needle_coreai_e2e.sh /path/to/needle-bundle ./build/coreai-export
./scripts/export_needle_coreml_e2e.sh /path/to/needle-bundle ./build/coreml-export
./scripts/export_needle_onnx_e2e.sh /path/to/needle-bundle ./build/onnx-export int8
```

Defaults:

- source: `Cactus-Compute/needle`
- CoreAI output: `python/build/coreai-export`
- CoreML output: `python/build/coreml-export`
- ONNX output: `python/build/onnx-export`

## Tests

Run the Python tests from `python/`:

```bash
python3 -m unittest
```
