#!/usr/bin/env bash

set -euo pipefail

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT="${1:-$ROOT_DIRECTORY/bin/needle2-2.0.1.artifactbundle.zip}"
OUTPUT_DIRECTORY="${2:-$ROOT_DIRECTORY/ts/needle2/dist/native}"
NODE_INCLUDE="${NODE_INCLUDE:-$(dirname "$(command -v node)")/../include/node}"
WORKSPACE="$(mktemp -d)"
trap 'rm -rf "$WORKSPACE"' EXIT

if [[ ! -f "$ARTIFACT" ]]; then
  echo "Needle 2 artifact bundle not found: $ARTIFACT" >&2
  exit 1
fi

case "$(uname -s)" in
Darwin)
  platform="macos-arm64"
  shared_name="libneedle2.dylib"
  ;;
Linux)
  platform="linux-x86_64"
  shared_name="libneedle2.so"
  ;;
*)
  echo "Unsupported host: $(uname -s)" >&2
  exit 1
  ;;
esac

unzip -q "$ARTIFACT" -d "$WORKSPACE" \
  "needle2.artifactbundle/$platform/libneedle.a" \
  "needle2.artifactbundle/include/needle.h"
LIBRARY="$WORKSPACE/needle2.artifactbundle/$platform/libneedle.a"
UPSTREAM_INCLUDE="$WORKSPACE/needle2.artifactbundle/include"
mkdir -p "$OUTPUT_DIRECTORY"

CC="${CC:-clang}"
CXX="${CXX:-clang++}"
NATIVE_DIRECTORY="$ROOT_DIRECTORY/ts/needle2/native"
CFLAGS=(-std=c11 -fPIC -I"$NATIVE_DIRECTORY" -I"$UPSTREAM_INCLUDE")

"$CC" "${CFLAGS[@]}" -I/usr/local/include -I"$NODE_INCLUDE" \
  -c "$NATIVE_DIRECTORY/needle2_node.c" -o "$WORKSPACE/needle2_node.o"

if [[ "$platform" == "macos-arm64" ]]; then
  "$CXX" -dynamiclib -Wl,-force_load,"$LIBRARY" -lc++ \
    -o "$OUTPUT_DIRECTORY/$shared_name"
  "$CXX" -bundle -undefined dynamic_lookup \
    "$WORKSPACE/needle2_node.o" "$LIBRARY" -lc++ \
    -o "$OUTPUT_DIRECTORY/needle2.node"
else
  "$CXX" -shared -Wl,--whole-archive "$LIBRARY" -Wl,--no-whole-archive \
    -lstdc++ -ldl -pthread -o "$OUTPUT_DIRECTORY/$shared_name"
  "$CXX" -shared "$WORKSPACE/needle2_node.o" "$LIBRARY" \
    -lstdc++ -ldl -pthread -o "$OUTPUT_DIRECTORY/needle2.node"
fi

printf 'Built %s and %s\n' "$OUTPUT_DIRECTORY/$shared_name" "$OUTPUT_DIRECTORY/needle2.node"
