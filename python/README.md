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
  --quantizer-preset w8 \
  --quantizer-execution-mode eager
```

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

With CoreML CPU/GPU execution and native SDPA:

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

Compute-unit flags:

- `--compute-units {all,cpu-only,cpu-and-gpu,cpu-and-ne}`

For CoreML, `all` and `cpu-and-ne` select the ANE-compatible decomposed
attention graph. `cpu-only` and `cpu-and-gpu` select native SDPA. CoreAI
always exports native SDPA; this option does not alter its graph.

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
