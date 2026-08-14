#!/bin/bash

set -euo pipefail

version="2.0.1"
revision="17a803d95928ba33d3e9a0160e024d9565b5c3f2"
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

platforms=(
  android-arm64
  android-armv7
  android-riscv64
  ios-arm64
  ios-sim-arm64
  linux-arm64
  linux-armv7
  linux-mipsel
  linux-riscv64
  linux-x86_64
  macos-arm64
  tvos-arm64
  watchos-arm64
  windows-arm64
  windows-x86_64
)

for platform in "${platforms[@]}"; do
  mkdir -p "$bundle/$platform"
  curl --fail --location --silent --show-error \
    "$repository/$platform/libneedle.a" \
    --output "$bundle/$platform/libneedle.a"
done

if ! (cd "$bundle" && shasum --algorithm 256 --check "$checksum_file"); then
  echo "Needle 2 $version no longer matches its locked artifacts at $revision." >&2
  echo "To upgrade, update version, revision, and the corresponding checksum manifest together." >&2
  exit 1
fi

cp "$script_directory/info.json" "$bundle/info.json"
mkdir -p "$(dirname "$output")"
find "$bundle" -exec touch -t 202001010000 {} +
rm -f "$output"
(cd "$workspace" && COPYFILE_DISABLE=1 zip -X -q -r "$output" needle2.artifactbundle)
swift package compute-checksum "$output"
