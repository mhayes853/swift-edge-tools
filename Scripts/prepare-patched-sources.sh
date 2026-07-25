#!/bin/sh
set -eu

if [ "$#" -lt 4 ]; then
	echo "usage: $0 <source-root> <patch-file> <output-root> <relative-path>..." >&2
	exit 64
fi

source_root=$1
patch_file=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
output_root=$3
shift 3

if ! command -v git >/dev/null 2>&1; then
	echo "error: git is required to prepare patched sources" >&2
	exit 69
fi

rm -rf "$output_root"
mkdir -p "$output_root"

for relative_path in "$@"; do
	source_path="$source_root/$relative_path"
	output_path="$output_root/$relative_path"
	if [ ! -f "$source_path" ]; then
		echo "error: missing source file: $source_path" >&2
		exit 66
	fi
	mkdir -p "$(dirname "$output_path")"
	cp "$source_path" "$output_path"
done

git -C "$output_root" apply --check "$patch_file"
git -C "$output_root" apply --whitespace=error-all "$patch_file"
