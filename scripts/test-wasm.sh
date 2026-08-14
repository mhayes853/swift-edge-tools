#!/usr/bin/env bash

set -euo pipefail

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIRECTORY="$ROOT_DIRECTORY/swift/WASITests"
NEEDLE2_DIRECTORY="$ROOT_DIRECTORY/ts/needle2"
SWIFT_SDK_ID="${SWIFT_SDK_ID:-swift-6.3.2-RELEASE_wasm}"
BUILD_ONLY=0
FILTER=""
SWIFT_ARGUMENTS=()

while (($#)); do
	case "$1" in
	--filter)
		FILTER="$2"
		shift 2
		;;
	--build-only)
		BUILD_ONLY=1
		shift
		;;
	--)
		shift
		SWIFT_ARGUMENTS+=("$@")
		break
		;;
	*)
		SWIFT_ARGUMENTS+=("$1")
		shift
		;;
	esac
done

cd "$NEEDLE2_DIRECTORY"
npm ci --ignore-scripts
npm run build

cd "$TEST_DIRECTORY"
npm ci --ignore-scripts

COMMAND=(
	swift package
	--build-system native
	--swift-sdk "$SWIFT_SDK_ID"
	--disable-sandbox
)
if ((${#SWIFT_ARGUMENTS[@]})); then
	COMMAND+=("${SWIFT_ARGUMENTS[@]}")
fi
COMMAND+=(
	js test
	--prelude ./JavaScript/prelude.mjs
	--environment node
)

if [[ -n "$FILTER" ]]; then
	COMMAND+=(--filter "$FILTER")
fi
if ((BUILD_ONLY)); then
	COMMAND+=(--build-only)
fi

if [[ "$FILTER" == *Needle2JSEngine* ]]; then
	export EDGE_TOOLS_WASI_NEEDLE2_ONLY=1
fi

if command -v swiftly >/dev/null 2>&1; then
	swiftly run "${COMMAND[@]}"
else
	"${COMMAND[@]}"
fi
