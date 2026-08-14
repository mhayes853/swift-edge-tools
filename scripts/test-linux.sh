#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

image="${SWIFT_DOCKER_IMAGE:-swift:6.3.2-jammy}"
traits="${SWIFT_TRAITS:-XGrammar}"
traits_explicit=0
filter="${SWIFT_TEST_FILTER:-}"
filter_explicit=0
cache_volume="${SWIFT_DOCKER_CACHE_VOLUME:-swift-edge-tools-linux-build}"
package_path="."
build_only=0
disable_default_traits=1
install_python_venv=0
install_needle2_runtime=0
extra_arguments=()

usage() {
	cat <<'EOF'
Usage: scripts/test-linux.sh [options] [-- <additional swift arguments>]

Run a Swift build or test inside a Linux Docker container.

Options:
  --build-only              Build instead of testing.
  --cli                     Target the swift/CLI package instead of the root package.
                             Traits don't apply to it (it has none of its own), so
                             --traits/--default-traits are skipped unless given explicitly,
                             and the default filter runs all CLI tests.
  --package-path PATH       Package directory to build/test, relative to the repo root
                             (default: "." or "swift/CLI" with --cli).
  --traits TRAITS           Swift package traits (default: XGrammar; root package only).
  --filter FILTER           Test filter or suite pattern (all tests by default).
  --all-tests               Run all tests instead of applying a filter.
  --python-venv             Install the Python package into python/.venv.
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
  scripts/test-linux.sh --traits XGrammar --build-only
  scripts/test-linux.sh --traits XGrammar --all-tests --python-venv
  scripts/test-linux.sh --cli
  scripts/test-linux.sh --cli --filter 'ModelDetection tests'
EOF
}

while (($# > 0)); do
	case "$1" in
	--build-only)
		build_only=1
		shift
		;;
	--cli)
		package_path="swift/CLI"
		shift
		;;
	--package-path)
		package_path="${2:?--package-path requires a value}"
		shift 2
		;;
	--traits)
		traits="${2:?--traits requires a value}"
		traits_explicit=1
		shift 2
		;;
	--filter)
		filter="${2:?--filter requires a value}"
		filter_explicit=1
		shift 2
		;;
	--all-tests)
		filter=""
		filter_explicit=1
		shift
		;;
	--python-venv)
		install_python_venv=1
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

if [[ "$package_path" != "." ]]; then
	((traits_explicit)) || traits=""
	((filter_explicit)) || filter=""
	disable_default_traits=0
fi

case ",$traits," in
*,Needle2,*) install_needle2_runtime=1 ;;
esac

swift_command="test"
if ((build_only)); then
	swift_command="build"
fi

swift_arguments=(
	"$swift_command"
	--scratch-path "/tmp/swift-edge-tools-build/$package_path"
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
	--workdir "/workspace/$package_path"
)
if [[ -n "$cache_volume" ]]; then
	docker_arguments+=(--volume "$cache_volume:/tmp/swift-edge-tools-build")
fi
if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
	docker_arguments+=(--platform "$DOCKER_PLATFORM")
fi

resolved_file="$package_path/Package.resolved"
resolved_backup="$(mktemp)"
had_resolved=0
if [[ -f "$resolved_file" ]]; then
	had_resolved=1
	cp "$resolved_file" "$resolved_backup"
fi
restore_resolved() {
	if ((had_resolved)); then
		cp "$resolved_backup" "$resolved_file"
	else
		rm -f "$resolved_file"
	fi
	rm -f "$resolved_backup"
}
trap restore_resolved EXIT

if ((install_python_venv || install_needle2_runtime)); then
	docker_arguments+=(
		--env "INSTALL_NEEDLE2_RUNTIME=$install_needle2_runtime"
		--env "INSTALL_PYTHON_VENV=$install_python_venv"
	)
	setup_script='set -euo pipefail
apt-get update
if [[ "$INSTALL_NEEDLE2_RUNTIME" == 1 ]]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libc++-dev libc++abi-dev
  needle2_library_path="$(dirname "$(find /usr/lib/llvm-* -name libc++.so -print -quit)")"
  export LIBRARY_PATH="$needle2_library_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
if [[ "$INSTALL_PYTHON_VENV" == 1 ]]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
  python3 -m venv /workspace/python/.venv
  /workspace/python/.venv/bin/python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
  /workspace/python/.venv/bin/python -m pip install --no-cache-dir -e /workspace/python
fi
	exec "$@"'
	echo "+ docker ${docker_arguments[*]} $image bash -c <dependency-setup> swift ${swift_arguments[*]}"
	docker "${docker_arguments[@]}" "$image" \
		bash -c "$setup_script" bash swift "${swift_arguments[@]}"
else
	echo "+ docker ${docker_arguments[*]} $image swift ${swift_arguments[*]}"
	docker "${docker_arguments[@]}" "$image" swift "${swift_arguments[@]}"
fi
