#!/usr/bin/env bash

# Builds the traitless EdgeTools core in Embedded Swift mode against the WebAssembly Swift SDK.
#
# Embedded stdlib modules are validated against the exact compiler that produced them, so the
# toolchain and the Swift SDK must come from the same snapshot.

set -euo pipefail

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${EMBEDDED_SWIFT_BRANCH:-6.4.x}"
SNAPSHOT="${EMBEDDED_SWIFT_SNAPSHOT:-}"
TARGET="${EMBEDDED_TARGET:-EdgeTools}"

if [[ -z "$SNAPSHOT" ]]; then
	SNAPSHOT="$(
		curl -fsSL "https://www.swift.org/api/v1/install/dev/$BRANCH/wasm-sdk.json" |
			jq -r '.[0].dir'
	)"
fi

SDK_ID="${SNAPSHOT}_wasm-embedded"

if ! swift sdk list | grep -qx "$SDK_ID"; then
	metadata="$(
		curl -fsSL "https://www.swift.org/api/v1/install/dev/$BRANCH/wasm-sdk.json" |
			jq -e --arg dir "$SNAPSHOT" '.[] | select(.dir == $dir)'
	)"
	download="$(jq -r .download <<<"$metadata")"
	checksum="$(jq -r .checksum <<<"$metadata")"
	swift sdk install \
		"https://download.swift.org/swift-$BRANCH-branch/wasm-sdk/$SNAPSHOT/$download" \
		--checksum "$checksum"
fi

cd "$ROOT_DIRECTORY"
exec swift build \
	--swift-sdk "$SDK_ID" \
	--disable-default-traits \
	--target "$TARGET" \
	"$@"
