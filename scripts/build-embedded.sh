#!/usr/bin/env bash

# Builds the traitless EdgeTools core in Embedded Swift mode against the WebAssembly Swift SDK.
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
BRANCH="${EMBEDDED_SWIFT_BRANCH:-6.4.x}"
SNAPSHOT="${EMBEDDED_SWIFT_SNAPSHOT:-}"
TARGET="${EMBEDDED_TARGET:-EdgeTools}"
FEED="https://www.swift.org/api/v1/install/dev/$BRANCH/wasm-sdk.json"
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

echo "Building $TARGET for Embedded Swift using $SNAPSHOT."

cd "$ROOT_DIRECTORY"
exec swift build \
	--swift-sdk "$SDK_ID" \
	--disable-default-traits \
	--target "$TARGET" \
	"$@"
