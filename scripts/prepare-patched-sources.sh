#!/bin/sh
set -eu

if [ "$#" -lt 6 ]; then
	echo "usage: $0 <source-root> <output-root> --patch <patch-file>... [--prepend-include <header>] -- <relative-path>..." >&2
	exit 64
fi

source_root=$1
output_root=$2
shift 2

patch_list=$(mktemp)
trap 'rm -f "$patch_list"' EXIT HUP INT TERM
prepend_include=""
while [ "$#" -gt 0 ]; do
	case "$1" in
	--patch)
		[ "$#" -ge 2 ] || exit 64
		patch_file=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
		printf '%s\n' "$patch_file" >>"$patch_list"
		shift 2
		;;
	--prepend-include)
		[ "$#" -ge 2 ] || exit 64
		prepend_include=$2
		shift 2
		;;
	--)
		shift
		break
		;;
	*)
		echo "error: unknown option: $1" >&2
		exit 64
		;;
	esac
done

if [ ! -s "$patch_list" ] || [ "$#" -eq 0 ]; then
	echo "error: expected at least one patch and source path" >&2
	exit 64
fi

if ! command -v git >/dev/null 2>&1; then
	echo "error: git is required to prepare patched sources" >&2
	exit 69
fi

rm -rf "$output_root"
mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd)

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

# Isolate `git apply` from a parent worktree when the plugin output lives under `.build`.
git -C "$output_root" init --quiet
while IFS= read -r patch_file; do
	git -C "$output_root" apply --check "$patch_file"
	git -C "$output_root" apply --whitespace=error-all "$patch_file"
done <"$patch_list"
rm -rf "$output_root/.git"

if [ -n "$prepend_include" ]; then
	for relative_path in "$@"; do
		case "$relative_path" in
		*.c | *.cc | *.cpp | *.cxx)
			output_path="$output_root/$relative_path"
			temporary_path="$output_path.prepending"
			source_directory=$(dirname "$relative_path")
			case "$source_directory" in
			cpp)
				include_path=${prepend_include#cpp/}
				;;
			cpp/support)
				include_path=${prepend_include#cpp/support/}
				;;
			*)
				echo "error: unsupported source directory for prepended include: $source_directory" >&2
				exit 65
				;;
			esac
			{
				printf '#include "%s"\n\n' "$include_path"
				cat "$output_path"
			} >"$temporary_path"
			mv "$temporary_path" "$output_path"
			;;
		esac
	done
fi
