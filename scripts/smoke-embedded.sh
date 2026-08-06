#!/usr/bin/env bash

# Builds and runs the Embedded Swift smoke executable against the traitless EdgeTools core.
#
# This is the gate for embedded support. Compiling the library on its own proves very little:
# embedded restrictions are only diagnosed once a concrete engine and tool force specialization,
# and missing runtime symbols only appear when an executable links. The smoke package drives the
# public API end to end so those failures surface, then runs the result to catch traps.
#
# Embedded stdlib modules are validated against the exact compiler that produced them, so the
# selected toolchain must be the same development snapshot as the Swift SDK. The snapshot is
# resolved from the WebAssembly SDK feed rather than the toolchain feed, because the SDK is the
# scarcer artifact.
#
# Pass --print-snapshot to resolve the snapshot without building, which is how CI derives both
# the toolchain to install and its cache keys.

set -euo pipefail

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIRECTORY="$ROOT_DIRECTORY/swift/EmbeddedSmoke"
BRANCH="${EMBEDDED_SWIFT_BRANCH:-6.4.x}"
SNAPSHOT="${EMBEDDED_SWIFT_SNAPSHOT:-}"
FEED="https://www.swift.org/api/v1/install/dev/$BRANCH/wasm-sdk.json"
MODULE_NAME="EdgeToolsEmbeddedSmoke.wasm"
PRINT_SNAPSHOT=0

if [[ "${1:-}" == "--print-snapshot" ]]; then
	PRINT_SNAPSHOT=1
	shift
fi

if [[ -z "$SNAPSHOT" ]]; then
	SNAPSHOT="$(curl -fsSL "$FEED" | jq -e -r '.[0].dir')"
fi

if ((PRINT_SNAPSHOT)); then
	echo "$SNAPSHOT"
	exit 0
fi

SDK_ID="${SNAPSHOT}_wasm-embedded"

if ! swift sdk list | grep -qx "$SDK_ID"; then
	metadata="$(curl -fsSL "$FEED" | jq -e --arg dir "$SNAPSHOT" '.[] | select(.dir == $dir)')"
	swift sdk install \
		"https://download.swift.org/swift-$BRANCH-branch/wasm-sdk/$SNAPSHOT/$(jq -r .download <<<"$metadata")" \
		--checksum "$(jq -r .checksum <<<"$metadata")"
fi

cd "$SMOKE_DIRECTORY"

# NB: The smoke package pins the EdgeTools dependency to no traits, so this builds the traitless
# core as well without a separate pass.
echo "Building the Embedded Swift smoke executable using $SNAPSHOT."
swift build --swift-sdk "$SDK_ID" "$@"

MODULE_PATH="$(find .build -name "$MODULE_NAME" -type f | head -1)"
if [[ -z "$MODULE_PATH" ]]; then
	echo "Could not find $MODULE_NAME." >&2
	exit 1
fi

echo "Running $MODULE_PATH."
# NB: Node before 22 refuses to load node:wasi without the flag, and 22+ still accepts it.
OUTPUT="$(node --experimental-wasi-unstable-preview1 ./JavaScript/run.mjs "$MODULE_PATH" 2>&1)"
echo "$OUTPUT"

if [[ "$OUTPUT" != *"EDGE_TOOLS_EMBEDDED_SMOKE_OK"* ]]; then
	echo "The smoke executable did not report success." >&2
	exit 1
fi

echo ""
echo "Embedded Swift smoke test passed."
