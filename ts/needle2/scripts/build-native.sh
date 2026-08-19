#!/usr/bin/env bash

set -euo pipefail

package_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository="https://huggingface.co/Cactus-Compute/needle2/resolve/b7ce80e07a8c76fb30a3a78db1e4aea7d72198da"
output_directory="${2:-$package_directory/dist/native}"
artifact="${1:-}"
workspace="$(mktemp -d)"
trap 'rm -rf "$workspace"' EXIT

repository_artifact="$package_directory/../../bin/needle2-2.0.2.artifactbundle.zip"
if [[ -z "$artifact" && -f "$repository_artifact" ]]; then
  artifact="$repository_artifact"
fi

case "$(uname -s)-$(uname -m)" in
Darwin-arm64)
  platform="macos-arm64"
  checksum="1d071ab8337da74ae1b168e766474ee0b8f86039ec4768c97f2cc66322ccc7b5"
  shared_name="libneedle2.dylib"
  ;;
Linux-x86_64)
  platform="linux-x86_64"
  checksum="150b4bad3909f5be4d1889e57ca1486ab77ca01014dc2b52b143a7d575312b02"
  shared_name="libneedle2.so"
  ;;
Linux-aarch64)
  platform="linux-arm64"
  checksum="289fee626efe9f3367a085ce631a0ca3661854c6717acd06d2d93fd34e7912a5"
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
