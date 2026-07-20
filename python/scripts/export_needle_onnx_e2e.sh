#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -x "$PYTHON_DIR/.venv/bin/python" ]]; then
	PYTHON="$PYTHON_DIR/.venv/bin/python"
else
	PYTHON="python3"
fi

if ! "$PYTHON" -c 'import onnxruntime' >/dev/null 2>&1; then
	"$PYTHON" -m pip install -e "$PYTHON_DIR"
fi

SOURCE="${1:-Cactus-Compute/needle}"
OUTPUT="${2:-$PYTHON_DIR/build/onnx-export}"
QUANTIZATION="${3:-}"
ARGS=(--backend onnx --source "$SOURCE" --output "$OUTPUT")
if [[ -n "$QUANTIZATION" ]]; then
	ARGS+=(--onnx-quantization "$QUANTIZATION")
fi

"$PYTHON" "$PYTHON_DIR/cli.py" "${ARGS[@]}"

printf 'Exported Needle ONNX bundle to %s\n' "$OUTPUT"
