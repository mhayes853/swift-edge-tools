#!/usr/bin/env bash

set -euo pipefail

llama_tag="b10076"
cactus_revision="cfea89ce32598254f6f18fef37801f479afe1553"
cactus_repository="https://raw.githubusercontent.com/cactus-compute/cactus-hybrid/$cactus_revision"
version="$llama_tag"

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(dirname "$(dirname "$script_directory")")"
default_output="$repository_directory/bin/llama-$version.artifactbundle.zip"

targets=(
  macos-arm64
  macos-x86_64
  ios-arm64
  ios-sim-arm64
  tvos-arm64
  tvos-sim-arm64
  watchos-arm64
  watchos-arm64_32
  watchos-sim-arm64
  visionos-arm64
  visionos-sim-arm64
  linux-x86_64
  linux-arm64
  android-arm64
  windows-x86_64
  windows-arm64
)

headers=(
  llama.h
  ggml.h
  ggml-alloc.h
  ggml-backend.h
  ggml-cpu.h
  ggml-opt.h
  gguf.h
)

common_flags=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_COMMON=OFF
  -DLLAMA_BUILD_APP=OFF
  -DLLAMA_BUILD_MTMD=ON
  -DMTMD_VIDEO=OFF
  -DLLAMA_CURL=OFF
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=OFF
)

apple_gpu_flags=(
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_ACCELERATE=ON
  -DGGML_BLAS=OFF
)

apple_watch_flags=(
  -DGGML_METAL=OFF
  -DGGML_ACCELERATE=ON
  -DGGML_BLAS=OFF
  -DCMAKE_C_FLAGS=-D_DARWIN_C_SOURCE
  -DCMAKE_CXX_FLAGS=-D_DARWIN_C_SOURCE
)

usage() {
  cat <<'EOF'
Usage:
  scripts/llama/build-artifact.sh build-slices SLICE_DIRECTORY SLICE...
  scripts/llama/build-artifact.sh assemble SLICE_DIRECTORY [OUTPUT] [--without-windows]
  scripts/llama/build-artifact.sh list

Build slices on their required hosts, copy the resulting slice directories into one
SLICE_DIRECTORY, then run assemble. The assembler requires all listed slices unless
Windows is explicitly omitted.

Examples:
  scripts/llama/build-artifact.sh build-slices .build/llama-slices macos-arm64 ios-arm64
  scripts/llama/build-artifact.sh build-slices .build/llama-slices android-arm64
  scripts/llama/build-artifact.sh assemble .build/llama-slices --without-windows

Environment:
  ANDROID_NDK_HOME          Required for Android slices.
  LLAMA_BUILD_WORKSPACE     Optional persistent checkout and CMake build directory.
  LLAMA_WINDOWS_GENERATOR  CMake generator for Windows (default: Visual Studio 17 2022).
  LLAMA_LIBRARIAN          Optional Windows llvm-lib/lib.exe override.
EOF
}

target_is_known() {
  local requested="$1"
  local target
  for target in "${targets[@]}"; do
    if [[ "$target" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

supported_triple() {
  case "$1" in
  macos-arm64) echo "arm64-apple-macosx" ;;
  macos-x86_64) echo "x86_64-apple-macosx" ;;
  ios-arm64) echo "arm64-apple-ios" ;;
  ios-sim-arm64) echo "arm64-apple-ios-simulator" ;;
  tvos-arm64) echo "arm64-apple-tvos" ;;
  tvos-sim-arm64) echo "arm64-apple-tvos-simulator" ;;
  watchos-arm64) echo "arm64-apple-watchos" ;;
  watchos-arm64_32) echo "arm64_32-apple-watchos" ;;
  watchos-sim-arm64) echo "arm64-apple-watchos-simulator" ;;
  visionos-arm64) echo "arm64-apple-xros" ;;
  visionos-sim-arm64) echo "arm64-apple-xros-simulator" ;;
  linux-x86_64) echo "x86_64-unknown-linux-gnu" ;;
  linux-arm64) echo "aarch64-unknown-linux-gnu" ;;
  android-arm64) echo "aarch64-unknown-linux-android28" ;;
  windows-x86_64) echo "x86_64-unknown-windows-msvc" ;;
  windows-arm64) echo "aarch64-unknown-windows-msvc" ;;
  esac
}

supported_triples_json() {
  case "$1" in
  android-arm64)
    echo '["aarch64-unknown-linux-android28", "aarch64-unknown-linux-android29", "aarch64-unknown-linux-android30", "aarch64-unknown-linux-android31", "aarch64-unknown-linux-android32", "aarch64-unknown-linux-android33", "aarch64-unknown-linux-android34", "aarch64-unknown-linux-android35", "aarch64-unknown-linux-android36"]'
    ;;
  *) echo "[\"$(supported_triple "$1")\"]" ;;
  esac
}

library_name() {
  case "$1" in
  windows-*) echo "llama.lib" ;;
  *) echo "libllama.a" ;;
  esac
}

host_system() {
  case "$(uname -s)" in
  Darwin) echo "darwin" ;;
  Linux) echo "linux" ;;
  MINGW* | MSYS* | CYGWIN*) echo "windows" ;;
  *) echo "unknown" ;;
  esac
}

require_target_host() {
  local target="$1"
  local host
  host="$(host_system)"
  case "$target" in
  macos-* | ios-* | tvos-* | watchos-* | visionos-*)
    if [[ "$host" != "darwin" ]]; then
      echo "$target requires macOS and Xcode." >&2
      return 1
    fi
    ;;
  linux-x86_64)
    if [[ "$host" != "linux" || "$(uname -m)" != "x86_64" ]]; then
      echo "$target requires a native x86-64 Linux host." >&2
      return 1
    fi
    ;;
  linux-arm64)
    if [[ "$host" != "linux" || ! "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
      echo "$target requires a native ARM64 Linux host." >&2
      return 1
    fi
    ;;
  android-*)
    if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
      echo "$target requires ANDROID_NDK_HOME." >&2
      return 1
    fi
    ;;
  windows-*)
    if [[ "$host" != "windows" ]]; then
      echo "$target requires Windows." >&2
      return 1
    fi
    ;;
  esac
}

configure_target() {
  local target="$1"
  case "$target" in
  macos-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
    )
    ;;
  macos-x86_64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_OSX_ARCHITECTURES=x86_64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
    )
    ;;
  ios-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_SYSTEM_NAME=iOS
      -DCMAKE_OSX_SYSROOT=iphoneos
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0
    )
    ;;
  ios-sim-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_SYSTEM_NAME=iOS
      -DCMAKE_OSX_SYSROOT=iphonesimulator
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0
    )
    ;;
  tvos-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_SYSTEM_NAME=tvOS
      -DCMAKE_OSX_SYSROOT=appletvos
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0
    )
    ;;
  tvos-sim-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_SYSTEM_NAME=tvOS
      -DCMAKE_OSX_SYSROOT=appletvsimulator
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0
    )
    ;;
  watchos-arm64)
    target_flags=(
      "${apple_watch_flags[@]}"
      -DCMAKE_SYSTEM_NAME=watchOS
      -DCMAKE_OSX_SYSROOT=watchos
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.0
    )
    ;;
  watchos-arm64_32)
    target_flags=(
      "${apple_watch_flags[@]}"
      -DCMAKE_SYSTEM_NAME=watchOS
      -DCMAKE_OSX_SYSROOT=watchos
      -DCMAKE_OSX_ARCHITECTURES=arm64_32
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.0
    )
    ;;
  watchos-sim-arm64)
    target_flags=(
      "${apple_watch_flags[@]}"
      -DCMAKE_SYSTEM_NAME=watchOS
      -DCMAKE_OSX_SYSROOT=watchsimulator
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.0
    )
    ;;
  visionos-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_SYSTEM_NAME=visionOS
      -DCMAKE_OSX_SYSROOT=xros
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=1.0
    )
    ;;
  visionos-sim-arm64)
    target_flags=(
      "${apple_gpu_flags[@]}"
      -DCMAKE_SYSTEM_NAME=visionOS
      -DCMAKE_OSX_SYSROOT=xrsimulator
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=1.0
    )
    ;;
  linux-x86_64 | linux-arm64)
    target_flags=(
      -DCMAKE_C_COMPILER=clang
      -DCMAKE_CXX_COMPILER=clang++
      -DGGML_METAL=OFF
      -DGGML_ACCELERATE=OFF
      -DGGML_BLAS=OFF
    )
    ;;
  android-arm64)
    target_flags=(
      -DGGML_METAL=OFF
      -DGGML_ACCELERATE=OFF
      -DGGML_BLAS=OFF
      "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
      -DANDROID_ABI=arm64-v8a
      -DANDROID_PLATFORM=android-28
      -DANDROID_STL=c++_shared
    )
    ;;
  windows-x86_64)
    target_flags=(
      -A x64
      -DGGML_METAL=OFF
      -DGGML_ACCELERATE=OFF
      -DGGML_BLAS=OFF
    )
    ;;
  windows-arm64)
    target_flags=(
      -A ARM64
      -DGGML_METAL=OFF
      -DGGML_ACCELERATE=OFF
      -DGGML_BLAS=OFF
    )
    ;;
  esac
}

prepare_checkout() {
  local workspace="$1"
  local checkout="$workspace/llama.cpp"

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
  local patch_path
  for patch_path in "$workspace/patches"/*.patch "$repository_directory/patches/Llama"/*.patch; do
    git -C "$checkout" update-index --refresh
    git -C "$checkout" -c user.email=build@edgetools -c user.name="EdgeTools Build" \
      am --quiet "$patch_path"
  done
}

checkout_is_prepared() {
  local checkout="$1"
  [[ -d "$checkout/.git" ]] && \
    [[ "$(git -C "$checkout" log -1 --format=%s)" == \
      "llama : export the handoff probe API with C linkage" ]] && \
    git -C "$checkout" diff --quiet && \
    git -C "$checkout" diff --cached --quiet
}

copy_common_files() {
  local checkout="$1"
  local slices="$2"

  mkdir -p "$slices/include"
  local header
  for header in "${headers[@]}"; do
    if [[ -f "$checkout/include/$header" ]]; then
      cp "$checkout/include/$header" "$slices/include/$header"
    else
      cp "$checkout/ggml/include/$header" "$slices/include/$header"
    fi
  done
  cp "$checkout/tools/mtmd/mtmd.h" "$slices/include/mtmd.h"
  cp "$checkout/tools/mtmd/mtmd-helper.h" "$slices/include/mtmd-helper.h"
  cp "$script_directory/module.modulemap" "$slices/include/module.modulemap"
  cp "$checkout/LICENSE" "$slices/LICENSE"
  cat >"$slices/build-metadata" <<EOF
version=$version
cactus_revision=$cactus_revision
EOF
}

find_library() {
  local build="$1"
  local basename="$2"
  local extension="$3"
  local matches=()
  while IFS= read -r library; do
    matches+=("$library")
  done < <(
    find "$build" -mindepth 2 -type f -iname "$basename.$extension" \
      ! -path '*/merge.*/*' ! -path '*/merge-objects/*' | sort
  )
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "Expected one $basename.$extension in $build, found ${#matches[@]}." >&2
    return 1
  fi
  echo "${matches[0]}"
}

merge_libraries() {
  local target="$1"
  local build="$2"
  local destination="$3"
  local extension="a"
  local prefixes=(libllama libmtmd libggml libggml-base libggml-cpu)
  if [[ "$target" == windows-* ]]; then
    extension="lib"
    prefixes=(llama mtmd ggml ggml-base ggml-cpu)
  elif [[ "$target" != watchos-* && "$target" != linux-* && "$target" != android-* ]]; then
    prefixes+=(libggml-metal)
  fi

  local libraries=()
  local prefix
  for prefix in "${prefixes[@]}"; do
    libraries+=("$(find_library "$build" "$prefix" "$extension")")
  done

  case "$target" in
  macos-* | ios-* | tvos-* | watchos-* | visionos-*)
    libtool -static -no_warning_for_no_symbols -o "$destination" "${libraries[@]}"
    ;;
  windows-*)
    local librarian="${LLAMA_LIBRARIAN:-}"
    if [[ -z "$librarian" ]]; then
      if command -v llvm-lib >/dev/null; then
        librarian="llvm-lib"
      elif command -v lib.exe >/dev/null; then
        librarian="lib.exe"
      else
        echo "Windows archive merging requires llvm-lib or lib.exe." >&2
        return 1
      fi
    fi
    "$librarian" "/OUT:$destination" "${libraries[@]}"
    ;;
  *)
    local archiver="${AR:-}"
    if [[ -z "$archiver" && -f "$build/CMakeCache.txt" ]]; then
      archiver="$(sed -n 's/^CMAKE_AR:FILEPATH=//p' "$build/CMakeCache.txt" | head -1)"
    fi
    archiver="${archiver:-ar}"
    "$archiver" qcL "$destination" "${libraries[@]}"
    local ranlib=""
    if [[ -f "$build/CMakeCache.txt" ]]; then
      ranlib="$(sed -n 's/^CMAKE_RANLIB:FILEPATH=//p' "$build/CMakeCache.txt" | head -1)"
    fi
    "${ranlib:-ranlib}" "$destination"
    ;;
  esac
}

verify_symbols() {
  local target="$1"
  local build="$2"
  local library="$3"
  local symbols=(llama_backend_init llama_model_has_probe llama_probe_confidence mtmd_init_from_file)
  local listing
  case "$target" in
  windows-*)
    if command -v dumpbin.exe >/dev/null; then
      listing="$(dumpbin.exe /LINKERMEMBER:1 "$library")"
    elif command -v llvm-nm >/dev/null; then
      listing="$(llvm-nm --defined-only "$library")"
    else
      echo "Symbol verification requires dumpbin.exe or llvm-nm." >&2
      return 1
    fi
    ;;
  *)
    local nm_tool="${NM:-}"
    if [[ -z "$nm_tool" && -f "$build/CMakeCache.txt" ]]; then
      nm_tool="$(sed -n 's/^CMAKE_NM:FILEPATH=//p' "$build/CMakeCache.txt" | head -1)"
    fi
    if [[ -n "$nm_tool" ]]; then
      listing="$("$nm_tool" -g "$library")"
    elif [[ "$(host_system)" == "darwin" ]]; then
      listing="$(xcrun nm -g "$library")"
    else
      listing="$(nm -g "$library")"
    fi
    ;;
  esac

  local symbol
  for symbol in "${symbols[@]}"; do
    if [[ "$listing" != *"$symbol"* ]]; then
      echo "$target is missing required symbol $symbol." >&2
      return 1
    fi
  done
}

build_slice() {
  local checkout="$1"
  local workspace="$2"
  local slices="$3"
  local target="$4"
  local build="$workspace/build-$target"
  local library
  library="$(library_name "$target")"
  mkdir -p "$build"
  local merge_directory
  merge_directory="$(mktemp -d "$build/merge.XXXXXX")"
  local merged="$merge_directory/$library"

  require_target_host "$target"
  configure_target "$target"

  echo "Building $target"
  local generator_arguments=(-G Ninja)
  if [[ "$target" == windows-* ]]; then
    generator_arguments=(-G "${LLAMA_WINDOWS_GENERATOR:-Visual Studio 17 2022}")
  fi
  cmake "${generator_arguments[@]}" -S "$checkout" -B "$build" \
    "${common_flags[@]}" "${target_flags[@]}"
  if [[ "$target" == windows-* ]]; then
    cmake --build "$build" --config Release
  else
    cmake --build "$build"
  fi

  merge_libraries "$target" "$build" "$merged"
  verify_symbols "$target" "$build" "$merged"
  mkdir -p "$slices/$target"
  cp "$merged" "$slices/$target/$library"
  cat >"$slices/$target/slice-metadata" <<EOF
version=$version
cactus_revision=$cactus_revision
triple=$(supported_triple "$target")
library=$library
EOF
}

build_slices() (
  local slices="$1"
  shift
  if [[ $# -eq 0 ]]; then
    echo "build-slices requires at least one slice." >&2
    usage >&2
    return 2
  fi

  local target
  for target in "$@"; do
    if ! target_is_known "$target"; then
      echo "Unknown llama slice: $target" >&2
      return 2
    fi
  done

  local workspace
  if [[ -n "${LLAMA_BUILD_WORKSPACE:-}" ]]; then
    workspace="$LLAMA_BUILD_WORKSPACE"
    if [[ "$workspace" != /* ]]; then
      workspace="$(pwd)/$workspace"
    fi
    mkdir -p "$workspace"
  else
    workspace="$(mktemp -d)"
    trap 'rm -rf "$workspace"' EXIT
  fi
  if [[ ! -e "$workspace/llama.cpp" ]]; then
    prepare_checkout "$workspace"
  elif ! checkout_is_prepared "$workspace/llama.cpp"; then
    echo "The llama checkout in $workspace is incomplete or modified; use a clean workspace." >&2
    return 1
  fi
  copy_common_files "$workspace/llama.cpp" "$slices"
  for target in "$@"; do
    build_slice "$workspace/llama.cpp" "$workspace" "$slices" "$target"
  done
)

metadata_value() {
  local file="$1"
  local key="$2"
  sed -n "s/^$key=//p" "$file"
}

validate_slice() {
  local slices="$1"
  local target="$2"
  local metadata="$slices/$target/slice-metadata"
  if [[ ! -f "$metadata" ]]; then
    echo "Missing $target/slice-metadata." >&2
    return 1
  fi

  local expected_library
  expected_library="$(library_name "$target")"
  if [[ ! -f "$slices/$target/$expected_library" ]]; then
    echo "Missing $target/$expected_library." >&2
    return 1
  fi
  if [[ "$(metadata_value "$metadata" version)" != "$version" || \
        "$(metadata_value "$metadata" cactus_revision)" != "$cactus_revision" || \
        "$(metadata_value "$metadata" triple)" != "$(supported_triple "$target")" || \
        "$(metadata_value "$metadata" library)" != "$expected_library" ]]; then
    echo "$target metadata does not match this artifact build." >&2
    return 1
  fi
}

write_info_json() {
  local bundle="$1"
  shift
  local artifact_targets=("$@")
  {
    echo '{'
    echo '  "artifacts": {'
    echo '    "CLlama": {'
    echo '      "type": "staticLibrary",'
    echo "      \"version\": \"$version\","
    echo '      "variants": ['
    local index
    for index in "${!artifact_targets[@]}"; do
      local target="${artifact_targets[$index]}"
      local separator=","
      if [[ $index -eq $((${#artifact_targets[@]} - 1)) ]]; then
        separator=""
      fi
      echo '        {'
      echo "          \"path\": \"$target/$(library_name "$target")\","
      echo '          "staticLibraryMetadata": {'
      echo '            "headerPaths": ["include"],'
      echo '            "moduleMapPath": "include/module.modulemap"'
      echo '          },'
      echo "          \"supportedTriples\": $(supported_triples_json "$target")"
      echo "        }$separator"
    done
    echo '      ]'
    echo '    }'
    echo '  },'
    echo '  "schemaVersion": "1.0"'
    echo '}'
  } >"$bundle/info.json"
}

assemble() (
  local slices="$1"
  local output="$2"
  local include_windows="$3"
  local artifact_targets=()
  local target
  for target in "${targets[@]}"; do
    if [[ "$include_windows" == "false" && "$target" == windows-* ]]; then
      continue
    fi
    artifact_targets+=("$target")
  done
  local common_metadata="$slices/build-metadata"
  if [[ ! -f "$common_metadata" || \
        "$(metadata_value "$common_metadata" version)" != "$version" || \
        "$(metadata_value "$common_metadata" cactus_revision)" != "$cactus_revision" ]]; then
    echo "The common slice metadata does not match this artifact build." >&2
    return 1
  fi

  for target in "${artifact_targets[@]}"; do
    validate_slice "$slices" "$target"
  done
  local header
  for header in "${headers[@]}" mtmd.h mtmd-helper.h module.modulemap; do
    if [[ ! -f "$slices/include/$header" ]]; then
      echo "Missing include/$header." >&2
      return 1
    fi
  done
  if [[ ! -f "$slices/LICENSE" ]]; then
    echo "Missing LICENSE." >&2
    return 1
  fi

  local workspace
  workspace="$(mktemp -d)"
  trap 'rm -rf "$workspace"' EXIT
  local bundle="$workspace/CLlama.artifactbundle"
  mkdir -p "$bundle"
  cp -R "$slices/include" "$bundle/include"
  cp "$slices/LICENSE" "$bundle/LICENSE"
  for target in "${artifact_targets[@]}"; do
    mkdir -p "$bundle/$target"
    cp "$slices/$target/$(library_name "$target")" "$bundle/$target/$(library_name "$target")"
  done
  write_info_json "$bundle" "${artifact_targets[@]}"

  if [[ "$output" != /* ]]; then
    output="$(pwd)/$output"
  fi
  mkdir -p "$(dirname "$output")"
  find "$bundle" -exec touch -t 202001010000 {} +
  rm -f "$output"
  (cd "$workspace" && COPYFILE_DISABLE=1 zip -X -q -r "$output" CLlama.artifactbundle)
  echo "Monolithic checksum: $(swift package compute-checksum "$output")"
  "$repository_directory/scripts/partition-artifact-bundle.py" "$bundle" "$output"
  echo "Built $output with ${#artifact_targets[@]} slices."
)

case "${1:-}" in
build-slices)
  if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
  fi
  slice_directory="$2"
  if [[ "$slice_directory" != /* ]]; then
    slice_directory="$(pwd)/$slice_directory"
  fi
  shift 2
  build_slices "$slice_directory" "$@"
  ;;
assemble)
  if [[ $# -lt 2 || $# -gt 4 ]]; then
    usage >&2
    exit 2
  fi
  slice_directory="$2"
  if [[ "$slice_directory" != /* ]]; then
    slice_directory="$(pwd)/$slice_directory"
  fi
  shift 2
  output="$default_output"
  output_was_set=false
  include_windows=true
  for argument in "$@"; do
    if [[ "$argument" == "--without-windows" ]]; then
      include_windows=false
    elif [[ "$output_was_set" == "false" ]]; then
      output="$argument"
      output_was_set=true
    else
      usage >&2
      exit 2
    fi
  done
  assemble "$slice_directory" "$output" "$include_windows"
  ;;
list)
  printf '%s\n' "${targets[@]}"
  ;;
*)
  usage >&2
  exit 2
  ;;
esac
