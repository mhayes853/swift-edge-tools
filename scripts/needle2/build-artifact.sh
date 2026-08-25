#!/bin/bash

set -euo pipefail

version="2.0.3"
revision="16f97bcfe1b005d0d969d2d71ea30236224c9e83"
repository="https://huggingface.co/Cactus-Compute/needle2/resolve/$revision"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(dirname "$(dirname "$script_directory")")"
if [[ $# -eq 0 ]]; then
  output="$repository_directory/bin/needle2-$version.artifactbundle.zip"
elif [[ $1 = /* ]]; then
  output="$1"
else
  output="$(pwd)/$1"
fi
workspace="$(mktemp -d)"
bundle="$workspace/needle2.artifactbundle"
checksum_file="$script_directory/checksums-$version.txt"

trap 'rm -rf "$workspace"' EXIT

mkdir -p "$bundle/include"
curl --fail --location --silent --show-error \
  "$repository/ios-arm64/needle.h" \
  --output "$bundle/include/needle.h"
curl --fail --location --silent --show-error \
  "$repository/LICENSE" \
  --output "$bundle/LICENSE"

cp "$script_directory/module.modulemap" "$bundle/include/module.modulemap"

android_arm64_triples="$(printf 'aarch64-unknown-linux-android%s,' {28..36})"
android_armv7_triples="$(printf 'armv7-unknown-linux-androideabi%s,' {28..36})"
android_riscv64_triples="$(printf 'riscv64-unknown-linux-android%s,' {35..36})"
targets=(
  "android-arm64|${android_arm64_triples%,}"
  "android-armv7|${android_armv7_triples%,}"
  "android-riscv64|${android_riscv64_triples%,}"
  "ios-arm64|arm64-apple-ios"
  "ios-sim-arm64|arm64-apple-ios-simulator"
  "linux-arm64|aarch64-unknown-linux-gnu"
  "linux-armv7|armv7-unknown-linux-gnueabihf"
  "linux-mipsel|mipsel-unknown-linux-gnu"
  "linux-riscv64|riscv64-unknown-linux-gnu"
  "linux-x86_64|x86_64-unknown-linux-gnu"
  "macos-arm64|arm64-apple-macosx"
  "tvos-arm64|arm64-apple-tvos"
  "watchos-arm64|arm64_32-apple-watchos"
  "windows-arm64|aarch64-unknown-windows-msvc"
  "windows-x86_64|x86_64-unknown-windows-msvc"
)

variants=()
for target in "${targets[@]}"; do
  IFS='|' read -r platform triples <<<"$target"
  mkdir -p "$bundle/$platform"
  curl --fail --location --silent --show-error \
    "$repository/$platform/libneedle.a" \
    --output "$bundle/$platform/libneedle.a"
  variants+=("$platform|$triples")
done

if ! (cd "$bundle" && shasum --algorithm 256 --check "$checksum_file"); then
  echo "Needle 2 $version no longer matches its locked artifacts at $revision." >&2
  echo "To upgrade, update version, revision, and the corresponding checksum manifest together." >&2
  exit 1
fi

VARIANTS="$(printf '%s\n' "${variants[@]}")" \
  BUNDLE="$bundle" \
  VERSION="$version" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

variants = []
for entry in os.environ["VARIANTS"].splitlines():
    path, triples = entry.split("|", 1)
    variants.append({
        "path": f"{path}/libneedle.a",
        "staticLibraryMetadata": {"headerPaths": ["include"]},
        "supportedTriples": triples.split(","),
    })

Path(os.environ["BUNDLE"], "info.json").write_text(json.dumps({
    "artifacts": {
        "CNeedle2": {
            "type": "staticLibrary",
            "variants": variants,
            "version": os.environ["VERSION"],
        }
    },
    "schemaVersion": "1.0",
}, indent=2) + "\n")
PY

mkdir -p "$(dirname "$output")"
find "$bundle" -exec touch -t 202001010000 {} +
rm -f "$output"
(cd "$workspace" && COPYFILE_DISABLE=1 zip -X -q -r "$output" needle2.artifactbundle)
echo "Monolithic checksum: $(swift package compute-checksum "$output")"
"$repository_directory/scripts/partition-artifact-bundle.py" "$bundle" "$output"
