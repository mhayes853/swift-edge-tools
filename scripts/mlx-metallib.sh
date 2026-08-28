#!/usr/bin/env bash
set -euo pipefail

# Builds MLX's Metal library and prints its path.
#
# Only Swift Build compiles MLX's Metal shaders into `mlx-swift_Cmlx.bundle`; the native build
# system emits every other resource bundle but skips that one. Swift 6.3's Swift Build in turn
# cannot resolve the modules our binary targets vend, so it can only build mlx-swift on its own,
# away from the rest of the graph. The caller resolves its own dependencies first and then runs
# its suites under whichever build system it likes, because MLX loads `default.metallib` from the
# working directory rather than the build directory.

usage() {
  cat <<'EOF'
Usage: scripts/mlx-metallib.sh [package-path]

Build MLX's Metal library from the mlx-swift checkout the package has already
resolved, and print the path to it.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_path="$(cd "${1:-$root}" && pwd)"
checkout="$package_path/.build/checkouts/mlx-swift"
scratch="$package_path/.build/mlx-metallib"

if [[ ! -d "$checkout" ]]; then
  echo "Unable to find an mlx-swift checkout at $checkout." >&2
  echo "Resolve $package_path with the MLX trait enabled before running this script." >&2
  exit 1
fi

swift build \
  --build-system swiftbuild \
  --package-path "$checkout" \
  --target Cmlx \
  --scratch-path "$scratch" >&2

metal_library="$(
  find "$scratch" -name default.metallib -path '*mlx-swift_Cmlx.bundle*' -print -quit
)"
if [[ -z "$metal_library" ]]; then
  echo "Unable to find MLX's Metal library under $scratch" >&2
  exit 1
fi

echo "$metal_library"
