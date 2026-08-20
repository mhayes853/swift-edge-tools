#!/usr/bin/env bash

set -euo pipefail

package_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository="https://huggingface.co/Cactus-Compute/needle2/resolve/16f97bcfe1b005d0d969d2d71ea30236224c9e83"
output_directory="${2:-$package_directory/dist/native}"
artifact="${1:-}"
workspace="$(mktemp -d)"
trap 'rm -rf "$workspace"' EXIT

repository_artifact="$package_directory/../../bin/needle2-2.0.3.artifactbundle.zip"
if [[ -z "$artifact" && -f "$repository_artifact" ]]; then
  artifact="$repository_artifact"
fi

case "$(uname -s)-$(uname -m)" in
Darwin-arm64)
  platform="macos-arm64"
  checksum="4e1895b2ac286eea76f632d4c3be6d116bea75b92a81884002028769ebb40806"
  shared_name="libneedle2.dylib"
  ;;
Linux-x86_64)
  platform="linux-x86_64"
  checksum="3f32e00cf26751c13659b7e8749e4b39ffe23573e45ec0594f2bd657b7b50995"
  shared_name="libneedle2.so"
  ;;
Linux-aarch64)
  platform="linux-arm64"
  checksum="78970de7a23d0e7443ecd7d8e930c5d9d8465f287e4ca393c8acbe44171ba391"
  shared_name="libneedle2.so"
  ;;
*)
  echo "Unsupported native Needle 2 host: $(uname -s) $(uname -m)" >&2
  exit 1
  ;;
esac

library="$workspace/libneedle.a"
if [[ -n "$artifact" ]]; then
  if [[ ! -f "$artifact" ]]; then
    echo "Needle 2 artifact bundle not found: $artifact" >&2
    exit 1
  fi
  unzip -p "$artifact" \
    "needle2.artifactbundle/$platform/libneedle.a" > "$library"
else
  curl --fail --location --silent --show-error \
    "$repository/$platform/libneedle.a" --output "$library"
fi

actual_checksum="$(shasum --algorithm 256 "$library" | cut -d ' ' -f 1)"
if [[ "$actual_checksum" != "$checksum" ]]; then
  echo "Unexpected Needle 2 $platform checksum: $actual_checksum" >&2
  exit 1
fi

node_include="${NODE_INCLUDE:-$(dirname "$(command -v node)")/../include/node}"
if [[ ! -f "$node_include/node_api.h" ]]; then
  echo "Node headers not found at $node_include; set NODE_INCLUDE explicitly." >&2
  exit 1
fi

cc="${CC:-clang}"
cxx="${CXX:-clang++}"
native_directory="$package_directory/native"
mkdir -p "$output_directory"

"$cc" -std=c11 -fPIC -I"$native_directory" -I"$node_include" \
  -c "$native_directory/needle2_node.c" -o "$workspace/needle2_node.o"

if [[ "$platform" == "macos-arm64" ]]; then
  "$cxx" -dynamiclib -Wl,-force_load,"$library" -lc++ \
    -o "$output_directory/$shared_name"
  "$cxx" -bundle -undefined dynamic_lookup \
    "$workspace/needle2_node.o" "$library" -lc++ \
    -o "$output_directory/needle2.node"
else
  "$cxx" -shared -Wl,--whole-archive "$library" -Wl,--no-whole-archive \
    -lstdc++ -ldl -pthread -o "$output_directory/$shared_name"
  "$cxx" -shared "$workspace/needle2_node.o" "$library" \
    -lstdc++ -ldl -pthread -o "$output_directory/needle2.node"
fi

printf 'Built %s and %s\n' \
  "$output_directory/$shared_name" "$output_directory/needle2.node"
