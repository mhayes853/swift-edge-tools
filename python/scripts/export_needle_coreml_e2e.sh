#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -x "$PYTHON_DIR/.venv/bin/python" ]]; then
	PYTHON="$PYTHON_DIR/.venv/bin/python"
else
	PYTHON="python3"
fi

if ! "$PYTHON" -c 'import coreai_opt, coremltools' >/dev/null 2>&1; then
	"$PYTHON" -m pip install -e "$PYTHON_DIR"
fi

SOURCE="${1:-Cactus-Compute/needle-hf}"
OUTPUT="${2:-$PYTHON_DIR/build/coreml-export}"
if (($# >= 3)); then
	EXTRA_ARGS=("${@:3}")
else
	EXTRA_ARGS=()
fi

if ((${#EXTRA_ARGS[@]} > 0)); then
	"$PYTHON" "$PYTHON_DIR/cli.py" \
		--backend CoreML \
		--source "$SOURCE" \
		--output "$OUTPUT" \
		"${EXTRA_ARGS[@]}"
else
	"$PYTHON" "$PYTHON_DIR/cli.py" \
		--backend CoreML \
		--source "$SOURCE" \
		--output "$OUTPUT"
fi

printf 'Exported Needle CoreML bundle to %s\n' "$OUTPUT"
