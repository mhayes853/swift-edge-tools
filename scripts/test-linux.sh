#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

image="${SWIFT_DOCKER_IMAGE:-swift:6.3.2-jammy}"
traits="${SWIFT_TRAITS:-ONNX}"
filter="${SWIFT_TEST_FILTER:-CONNXRuntime tests}"
cache_volume="${SWIFT_DOCKER_CACHE_VOLUME:-swift-edge-tools-linux-build}"
build_only=0
disable_default_traits=1
extra_arguments=()

usage() {
	cat <<'EOF'
Usage: scripts/test-linux.sh [options] [-- <additional swift arguments>]  

Run a Swift build or test inside a Linux Docker container.

Options:
  --build-only              Build instead of testing.
  --traits TRAITS           Swift package traits (default: ONNX).
  --filter FILTER           Test filter or suite pattern (default: CONNXRuntime tests).
  --all-tests               Run all tests instead of applying a filter.
  --default-traits          Keep the package's default traits enabled.
  --image IMAGE             Docker image (default: swift:6.3.2-jammy).
  --cache-volume NAME       Docker build-cache volume.
  --no-cache-volume         Do not persist the Docker build cache.
  -h, --help                Show this help.

Environment overrides:
  SWIFT_DOCKER_IMAGE, SWIFT_DOCKER_CACHE_VOLUME, SWIFT_TRAITS,
  SWIFT_TEST_FILTER, DOCKER_PLATFORM

Examples:
  scripts/test-linux.sh
  scripts/test-linux.sh --traits ONNXCore --build-only
  scripts/test-linux.sh --traits ONNX --filter 'CONNXRuntime tests|NeedleONNXEngine core tests'
  scripts/test-linux.sh --traits ONNX --all-tests -- --configuration release
EOF
}

while (($# > 0)); do
	case "$1" in
	--build-only)
		build_only=1
		shift
		;;
	--traits)
		traits="${2:?--traits requires a value}"
		shift 2
		;;
	--filter)
		filter="${2:?--filter requires a value}"
		shift 2
		;;
	--all-tests)
		filter=""
		shift
		;;
	--default-traits)
		disable_default_traits=0
		shift
		;;
	--image)
		image="${2:?--image requires a value}"
		shift 2
		;;
	--cache-volume)
		cache_volume="${2:?--cache-volume requires a value}"
		shift 2
		;;
	--no-cache-volume)
		cache_volume=""
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	--)
		shift
		extra_arguments+=("$@")
		break
		;;
	*)
		echo "Unknown option: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

swift_command="test"
if ((build_only)); then
	swift_command="build"
fi

swift_arguments=(
	"$swift_command"
	--scratch-path /tmp/swift-edge-tools-build
	--disable-experimental-prebuilts
)
if ((disable_default_traits)); then
	swift_arguments+=(--disable-default-traits)
fi
if [[ -n "$traits" ]]; then
	swift_arguments+=(--traits "$traits")
fi
if [[ "$swift_command" == "test" && -n "$filter" ]]; then
	swift_arguments+=(--filter "$filter")
fi
if ((${#extra_arguments[@]} > 0)); then
	swift_arguments+=("${extra_arguments[@]}")
fi

docker_arguments=(
	run
	--rm
	--volume "$PWD:/workspace"
	--workdir /workspace
)
if [[ -n "$cache_volume" ]]; then
	docker_arguments+=(--volume "$cache_volume:/tmp/swift-edge-tools-build")
fi
if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
	docker_arguments+=(--platform "$DOCKER_PLATFORM")
fi

resolved_backup="$(mktemp)"
had_resolved=0
if [[ -f Package.resolved ]]; then
	had_resolved=1
	cp Package.resolved "$resolved_backup"
fi
restore_resolved() {
	if ((had_resolved)); then
		cp "$resolved_backup" Package.resolved
	else
		rm -f Package.resolved
	fi
	rm -f "$resolved_backup"
}
trap restore_resolved EXIT

echo "+ docker ${docker_arguments[*]} $image swift ${swift_arguments[*]}"
docker "${docker_arguments[@]}" "$image" swift "${swift_arguments[@]}"
