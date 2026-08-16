#!/usr/bin/env bash

set -euo pipefail

# Builds the cactus-patched llama.cpp static-library artifactbundle.
#
# The build clones llama.cpp at the tag pinned by cactus-hybrid, applies the cactus
# hybrid-inference patch series (handoff probe tensors, runtime, and C API), then the
# local patches in patches/Llama (C linkage for the probe API), and compiles a merged
# static library per target. Apple targets embed the Metal shader library; all other
# targets are CPU-only for now.
#
# LLAMA_TARGETS limits the built slices (space separated, e.g. "macos-arm64 ios-arm64").
# Cross targets require their toolchains: linux-* run through docker, android-* need
# ANDROID_NDK_HOME, and windows-* must be built on Windows.

llama_tag="b10076"
cactus_revision="cfea89ce32598254f6f18fef37801f479afe1553"
cactus_repository="https://raw.githubusercontent.com/cactus-compute/cactus-hybrid/$cactus_revision"
version="$llama_tag"

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(dirname "$(dirname "$script_directory")")"
if [[ $# -eq 0 ]]; then
  output="$repository_directory/bin/llama-$version.artifactbundle.zip"
elif [[ $1 = /* ]]; then
  output="$1"
else
  output="$(pwd)/$1"
fi

workspace="$(mktemp -d)"
bundle="$workspace/CLlama.artifactbundle"
checkout="$workspace/llama.cpp"
trap 'rm -rf "$workspace"' EXIT

targets=(
  macos-arm64
  macos-x86_64
  ios-arm64
  ios-sim-arm64
  linux-x86_64
  linux-arm64
  android-arm64
  windows-x86_64
  windows-arm64
)
if [[ -n "${LLAMA_TARGETS:-}" ]]; then
  read -r -a targets <<<"$LLAMA_TARGETS"
fi

headers=(
  llama.h
  ggml.h
  ggml-alloc.h
  ggml-backend.h
  ggml-cpu.h
  ggml-opt.h
  gguf.h
)

echo "Cloning llama.cpp $llama_tag"
git clone --quiet --depth 1 --branch "$llama_tag" \
  https://github.com/ggml-org/llama.cpp.git "$checkout"

echo "Applying cactus-hybrid patches at $cactus_revision"
mkdir -p "$workspace/patches"
for patch in \
  0001-gguf-py-add-gemma-4-e2b-it-hybrid-arch-with-handoff-.patch \
  0002-conversion-support-Gemma4E2BItHybridForCausalLM-gemm.patch \
  0003-llama-add-gemma-4-e2b-it-hybrid-arch-gemma4-handoff-.patch \
  0004-llama-add-handoff-probe-runtime-and-staging-API.patch \
  0005-server-report-handoff-probe-confidence-for-gemma-4-e.patch \
  0006-tests-add-gemma-4-e2b-it-hybrid-handoff-probe-golden.patch; do
  curl --fail --location --silent --show-error \
    "$cactus_repository/patches/llama.cpp/patches/$patch" \
    --output "$workspace/patches/$patch"
done
git -C "$checkout" -c user.email=build@edgetools -c user.name="EdgeTools Build" \
  am --quiet "$workspace/patches"/*.patch
git -C "$checkout" -c user.email=build@edgetools -c user.name="EdgeTools Build" \
  am --quiet "$repository_directory/patches/Llama"/*.patch

common_flags=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_COMMON=OFF
  -DLLAMA_BUILD_APP=OFF
  -DLLAMA_BUILD_MTMD=OFF
  -DLLAMA_CURL=OFF
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=OFF
)

apple_flags=(
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_ACCELERATE=ON
  -DGGML_BLAS=OFF
)

configure_flags() {
  case "$1" in
  macos-arm64)
    echo "${apple_flags[*]} -DCMAKE_OSX_ARCHITECTURES=arm64"
    ;;
  macos-x86_64)
    echo "${apple_flags[*]} -DCMAKE_OSX_ARCHITECTURES=x86_64 -DGGML_METAL=OFF"
    ;;
  ios-arm64)
    echo "${apple_flags[*]} -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0"
    ;;
  ios-sim-arm64)
    echo "${apple_flags[*]} -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0"
    ;;
  linux-x86_64 | linux-arm64)
    echo ""
    ;;
  android-arm64)
    echo "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28"
    ;;
  windows-x86_64 | windows-arm64)
    echo ""
    ;;
  *)
    echo "Unknown target $1" >&2
    return 1
    ;;
  esac
}

supported_triples() {
  case "$1" in
  macos-arm64) echo '"arm64-apple-macosx"' ;;
  macos-x86_64) echo '"x86_64-apple-macosx"' ;;
  ios-arm64) echo '"arm64-apple-ios"' ;;
  ios-sim-arm64) echo '"arm64-apple-ios-simulator"' ;;
  linux-x86_64) echo '"x86_64-unknown-linux-gnu"' ;;
  linux-arm64) echo '"aarch64-unknown-linux-gnu"' ;;
  android-arm64) echo '"aarch64-unknown-linux-android"' ;;
  windows-x86_64) echo '"x86_64-unknown-windows-msvc"' ;;
  windows-arm64) echo '"aarch64-unknown-windows-msvc"' ;;
  esac
}

merge_libraries() {
  local target="$1"
  local build="$2"
  local destination="$3"
  local libraries=()
  while IFS= read -r library; do
    case "$library" in
    *cpp-httplib* | *llama-common*) ;;
    *) libraries+=("$library") ;;
    esac
  done < <(find "$build" -name "*.a" | sort)

  case "$target" in
  macos-* | ios-*)
    libtool -static -no_warning_for_no_symbols -o "$destination" "${libraries[@]}"
    ;;
  *)
    local mri="$build/merge.mri"
    {
      echo "create $destination"
      for library in "${libraries[@]}"; do
        echo "addlib $library"
      done
      echo "save"
      echo "end"
    } >"$mri"
    "${AR:-ar}" -M <"$mri"
    ;;
  esac
}

build_target() {
  local target="$1"
  local build="$workspace/build-$target"
  local flags
  flags="$(configure_flags "$target")"

  case "$target" in
  linux-*)
    echo "Skipping $target: build it inside a Linux container with this script." >&2
    return 1
    ;;
  windows-*)
    echo "Skipping $target: build it on Windows with this script." >&2
    return 1
    ;;
  android-*)
    if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
      echo "Skipping $target: ANDROID_NDK_HOME is not set." >&2
      return 1
    fi
    ;;
  esac

  echo "Building $target"
  # shellcheck disable=SC2086
  cmake -G Ninja -S "$checkout" -B "$build" "${common_flags[@]}" $flags >/dev/null
  cmake --build "$build" >/dev/null
  mkdir -p "$bundle/$target"
  merge_libraries "$target" "$build" "$bundle/$target/libllama.a"
}

mkdir -p "$bundle/include"
built_targets=()
for target in "${targets[@]}"; do
  if build_target "$target"; then
    built_targets+=("$target")
  fi
done

if [[ ${#built_targets[@]} -eq 0 ]]; then
  echo "No targets were built." >&2
  exit 1
fi

for header in "${headers[@]}"; do
  if [[ -f "$checkout/include/$header" ]]; then
    cp "$checkout/include/$header" "$bundle/include/$header"
  else
    cp "$checkout/ggml/include/$header" "$bundle/include/$header"
  fi
done
cp "$script_directory/module.modulemap" "$bundle/include/module.modulemap"
cp "$checkout/LICENSE" "$bundle/LICENSE"

{
  echo '{'
  echo '  "artifacts": {'
  echo '    "CLlama": {'
  echo '      "type": "staticLibrary",'
  echo "      \"version\": \"$version\","
  echo '      "variants": ['
  for index in "${!built_targets[@]}"; do
    target="${built_targets[$index]}"
    separator=","
    if [[ $index -eq $((${#built_targets[@]} - 1)) ]]; then
      separator=""
    fi
    echo '        {'
    echo "          \"path\": \"$target/libllama.a\","
    echo '          "staticLibraryMetadata": { "headerPaths": ["include"] },'
    echo "          \"supportedTriples\": [$(supported_triples "$target")]"
    echo "        }$separator"
  done
  echo '      ]'
  echo '    }'
  echo '  },'
  echo '  "schemaVersion": "1.0"'
  echo '}'
} >"$bundle/info.json"

mkdir -p "$(dirname "$output")"
find "$bundle" -exec touch -t 202001010000 {} +
rm -f "$output"
(cd "$workspace" && COPYFILE_DISABLE=1 zip -X -q -r "$output" CLlama.artifactbundle)
swift package compute-checksum "$output"
echo "Built targets: ${built_targets[*]}"
