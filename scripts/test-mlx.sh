#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

filter="MLX"
extra_arguments=()

usage() {
  cat <<'EOF'
Usage: scripts/test-mlx.sh [options] [-- <additional swift test arguments>]

Run MLX tests from the command line on macOS. The script enables the MLX and
XGrammar traits, makes MLX's Metal library available to the SwiftPM test
runner, and opts into MLX inference tests.

Options:
  --filter FILTER         Test filter (default: MLX).
  --all-tests             Run all tests built with the MLX and XGrammar traits.
  -h, --help              Show this help.

Examples:
  scripts/test-mlx.sh
  scripts/test-mlx.sh --filter 'MLX Filters Masked Tokens'
  scripts/test-mlx.sh -- --skip-build
EOF
}

while (($# > 0)); do
  case "$1" in
  --filter)
    filter="${2:?--filter requires a value}"
    shift 2
    ;;
  --all-tests)
    filter=""
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --)
    shift
    extra_arguments+=("$@")
    break
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if [[ "$(uname)" != "Darwin" ]]; then
  echo "MLX tests require macOS." >&2
  exit 1
fi

swift_arguments=(
  --disable-default-traits
  --traits MLX,XGrammar
)

bin_path="$(swift build --show-bin-path "${swift_arguments[@]}")"
metal_library="$bin_path/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
metal_library_link="$root/default.metallib"

if [[ ! -f "$metal_library" ]]; then
  echo "Unable to find MLX's Metal library at $metal_library" >&2
  exit 1
fi
if [[ -e "$metal_library_link" || -L "$metal_library_link" ]]; then
  echo "Refusing to replace existing $metal_library_link" >&2
  exit 1
fi

ln -s "$metal_library" "$metal_library_link"
cleanup() {
  unlink "$metal_library_link"
}
trap cleanup EXIT

command=(swift test "${swift_arguments[@]}")
if [[ -n "$filter" ]]; then
  command+=(--filter "$filter")
fi
if ((${#extra_arguments[@]} > 0)); then
  command+=("${extra_arguments[@]}")
fi

echo "+ EDGE_TOOLS_ENABLE_MLX_TESTS=1 ${command[*]}"
EDGE_TOOLS_ENABLE_MLX_TESTS=1 "${command[@]}"
