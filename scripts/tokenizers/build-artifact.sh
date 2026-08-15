#!/usr/bin/env bash

set -euo pipefail

version="0.1.0"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(dirname "$(dirname "$script_directory")")"
crate_directory="$repository_directory/rust/tokenizers"
output="${1:-$repository_directory/bin/tokenizers-$version.artifactbundle.zip}"
workspace="$(mktemp -d)"
bundle="$workspace/CTokenizers.artifactbundle"

trap 'rm -rf "$workspace"' EXIT

if [[ "$output" != /* ]]; then
  output="$(pwd)/$output"
fi

targets=(
  "aarch64-apple-darwin|macos-arm64|arm64-apple-macosx|stable"
  "x86_64-apple-darwin|macos-x86_64|x86_64-apple-macosx|stable"
  "aarch64-apple-ios|ios-arm64|arm64-apple-ios|stable"
  "aarch64-apple-ios-sim|ios-sim-arm64|arm64-apple-ios-simulator|stable"
  "x86_64-apple-ios|ios-sim-x86_64|x86_64-apple-ios-simulator|stable"
  "aarch64-apple-tvos|tvos-arm64|arm64-apple-tvos|nightly"
  "aarch64-apple-tvos-sim|tvos-sim-arm64|arm64-apple-tvos-simulator|nightly"
  "aarch64-apple-watchos|watchos-arm64|arm64-apple-watchos|nightly"
  "arm64_32-apple-watchos|watchos-arm64_32|arm64_32-apple-watchos|nightly"
  "aarch64-apple-watchos-sim|watchos-sim-arm64|arm64-apple-watchos-simulator|nightly"
  "aarch64-apple-visionos|visionos-arm64|arm64-apple-xros|nightly"
  "aarch64-apple-visionos-sim|visionos-sim-arm64|arm64-apple-xros-simulator|nightly"
  "x86_64-unknown-linux-gnu|linux-x86_64|x86_64-unknown-linux-gnu|stable"
  "aarch64-unknown-linux-gnu|linux-arm64|aarch64-unknown-linux-gnu|stable"
  "aarch64-linux-android|android-arm64|aarch64-unknown-linux-android|stable"
  "armv7-linux-androideabi|android-armv7|armv7-unknown-linux-androideabi|stable"
  "x86_64-linux-android|android-x86_64|x86_64-unknown-linux-android|stable"
  "x86_64-pc-windows-msvc|windows-x86_64|x86_64-unknown-windows-msvc|stable"
  "aarch64-pc-windows-msvc|windows-arm64|aarch64-unknown-windows-msvc|stable"
)
if [[ -n "${RUST_TARGETS:-}" ]]; then
  read -r -a requested <<<"$RUST_TARGETS"
  filtered=()
  for entry in "${targets[@]}"; do
    for rust_target in "${requested[@]}"; do
      if [[ "${entry%%|*}" == "$rust_target" ]]; then
        filtered+=("$entry")
      fi
    done
  done
  if [[ ${#filtered[@]} -ne ${#requested[@]} ]]; then
    echo "RUST_TARGETS names a target this script does not know how to package." >&2
    exit 1
  fi
  targets=("${filtered[@]}")
fi

export IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-15.0}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export TVOS_DEPLOYMENT_TARGET="${TVOS_DEPLOYMENT_TARGET:-17.0}"
export WATCHOS_DEPLOYMENT_TARGET="${WATCHOS_DEPLOYMENT_TARGET:-10.0}"
export XROS_DEPLOYMENT_TARGET="${XROS_DEPLOYMENT_TARGET:-1.0}"

mkdir -p "$bundle/include"
cp "$script_directory/include/tokenizers.h" "$bundle/include/tokenizers.h"
cp "$script_directory/include/module.modulemap" "$bundle/include/module.modulemap"
cp "$crate_directory/LICENSE" "$bundle/LICENSE"
cp "$crate_directory/THIRD_PARTY_NOTICES.md" "$bundle/THIRD_PARTY_NOTICES.md"

variants=()
for entry in "${targets[@]}"; do
  IFS='|' read -r rust_target variant triple toolchain <<<"$entry"

  case "$rust_target" in
  *-apple-*)
    if [[ "$(uname -s)" != "Darwin" ]]; then
      echo "Building $rust_target requires macOS." >&2
      exit 1
    fi
    ;;
  esac

  case "$rust_target" in
  *-windows-msvc) library="edgetools_tokenizers.lib" ;;
  *) library="libedgetools_tokenizers.a" ;;
  esac

  if [[ "$toolchain" == "nightly" ]]; then
    cargo +nightly -Zbuild-std=std,panic_abort build \
      --manifest-path "$crate_directory/Cargo.toml" --locked --release --target "$rust_target"
  else
    cargo build \
      --manifest-path "$crate_directory/Cargo.toml" --locked --release --target "$rust_target"
  fi

  mkdir -p "$bundle/$variant"
  cp "$crate_directory/target/$rust_target/release/$library" "$bundle/$variant/$library"
  variants+=("$variant|$triple|$library")
done

VARIANTS="$(
  IFS=$'\n'
  echo "${variants[*]}"
)" BUNDLE="$bundle" VERSION="$version" python3 - <<'PY'
import json
import os
from pathlib import Path

variants = []
for entry in os.environ["VARIANTS"].splitlines():
    path, triple, library = entry.split("|", 2)
    variants.append({
        "path": f"{path}/{library}",
        "supportedTriples": [triple],
        "staticLibraryMetadata": {
            "headerPaths": ["include"],
            "moduleMapPath": "include/module.modulemap",
        },
    })

Path(os.environ["BUNDLE"], "info.json").write_text(json.dumps({
    "schemaVersion": "1.0",
    "artifacts": {
        "CTokenizers": {
            "type": "staticLibrary",
            "version": os.environ["VERSION"],
            "variants": variants,
        }
    },
}, indent=2) + "\n")
PY

mkdir -p "$(dirname "$output")"
find "$bundle" -exec touch -t 202001010000 {} +
rm -f "$output"
(cd "$workspace" && COPYFILE_DISABLE=1 zip -X -q -r "$output" CTokenizers.artifactbundle)
swift package compute-checksum "$output"
