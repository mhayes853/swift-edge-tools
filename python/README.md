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
python3 coreai_export.py --output ./build/coreai-export
```

With a custom source:

```bash
python3 coreai_export.py --source /path/to/needle-bundle --output ./build/coreai-export
```

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
python3 -m unittest tests.test_needle_torch tests.test_coreai_export
```
