#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -x "$PYTHON_DIR/.venv/bin/python" ]]; then
  PYTHON="$PYTHON_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

SOURCE="${1:-Cactus-Compute/needle-hf}"
OUTPUT="${2:-$PYTHON_DIR/build/coreai-export}"

"$PYTHON" "$PYTHON_DIR/cli.py" --source "$SOURCE" --output "$OUTPUT"

printf 'Exported Needle CoreAI bundle to %s\n' "$OUTPUT"