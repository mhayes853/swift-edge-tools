# Swift Needle Python Export

This directory contains the Python-side Needle model, along with a CoreAI export workflow.

## Requirements

- Python 3.9+
- `torch`
- `coreai-torch`
- `huggingface_hub`
- `safetensors`
- `sentencepiece`
- `tokenizers`

Install dependencies from this directory:

```bash
pip install -e .
```

## Export Inputs

The export script accepts either:

1. A Hugging Face repo id, defaulting to `Cactus-Compute/needle-hf`
2. A local model directory

The source directory or repo snapshot must contain:

- `config.json` or `configuration.json`
- `model.safetensors` or a `.pkl` weights file
- `tokenizer.model` or `tokenizer.json`

## Export Output

The output bundle contains:

- `encoder.aimodel`
- `decoder.aimodel`
- `configuration.json`
- `tokenizer.model` or `tokenizer.json`

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
  --source Cactus-Compute/needle-hf \
  --output ./build/coreai-export \
  --authoring-author "Needle" \
  --authoring-description "CoreAI export" \
  --authoring-license "BSD-3-Clause" \
  --authoring-custom suite=cli \
  --quantizer-preset w8 \
  --quantizer-execution-mode eager
```

With metadata and palettization config files:

```bash
python3 cli.py \
  --source Cactus-Compute/needle-hf \
  --output ./build/coreai-export \
  --authoring-metadata ./metadata.yaml \
  --palettizer-config ./palettizer.yaml
```

The CLI accepts JSON or YAML for:

- `--authoring-metadata`
- `--quantizer-config`
- `--palettizer-config`

Supported compression convenience flags:

- `--quantizer-preset {w4,w4_per_block,w8}`
- `--quantizer-execution-mode {graph,eager}`
- `--palettizer-preset {w4,w6,w8}`

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

With an explicit source and output:

```bash
./scripts/export_needle_coreai_e2e.sh /path/to/needle-bundle ./build/coreai-export
```

Defaults:

- source: `Cactus-Compute/needle-hf`
- output: `python/build/coreai-export`

## Tests

Run the Python tests from `python/`:

```bash
python3 -m unittest tests.test_needle_torch tests.test_torch_utils tests.test_coreai_export tests.test_cli
```
