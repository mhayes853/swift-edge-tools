# Needle Python exporter

This directory contains the PyTorch implementation of Needle and its ONNX
export pipeline.

## Setup

From this directory, create an environment and install the package:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Export

Export a portable float32 bundle from the default Hugging Face source:

```sh
.venv/bin/needle-export --output ./build/onnx-export
```

Local model directories containing a configuration, tokenizer, and weights are
also supported:

```sh
.venv/bin/needle-export \
  --source /path/to/needle-bundle \
  --output ./build/onnx-export
```

The output bundle contains `encoder.onnx`, `decoder.onnx`,
`configuration.json`, and the source tokenizer. The decoder uses explicit key
and value caches and returns one-position cache deltas.

Supported options are:

- `--dtype {float32,float16,bfloat16}` (default: `float32`)
- `--quantization {int4}`

For example:

```sh
.venv/bin/needle-export \
  --output ./build/onnx-int4 \
  --quantization int4
```

The end-to-end helper script installs missing dependencies when needed and
accepts source, output, and optional quantization as positional arguments:

```sh
./scripts/export_needle_onnx_e2e.sh \
  Cactus-Compute/needle \
  ./build/onnx-export \
  int4
```

## Tests

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```
