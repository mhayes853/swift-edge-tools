#!/usr/bin/env bash

set -euo pipefail

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIRECTORY="$ROOT_DIRECTORY/swift/WASITests"
ONNX_PACKAGE="onnxruntime-node"
SWIFT_SDK_ID="${SWIFT_SDK_ID:-swift-6.3.2-RELEASE_wasm}"
BUILD_ONLY=0
FILTER=""
SWIFT_ARGUMENTS=()

while (($#)); do
	case "$1" in
	--onnx-package)
		ONNX_PACKAGE="$2"
		shift 2
		;;
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

case "$ONNX_PACKAGE" in
onnxruntime-node | onnxruntime-web) ;;
*)
	echo "Unsupported ONNX package: $ONNX_PACKAGE" >&2
	exit 2
	;;
esac

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

export EDGE_TOOLS_ONNX_PACKAGE="$ONNX_PACKAGE"

if command -v swiftly >/dev/null 2>&1; then
	swiftly run "${COMMAND[@]}"
else
	"${COMMAND[@]}"
fi
