#!/usr/bin/env bash

# Reports Embedded Swift restriction violations in the traitless EdgeTools core.
#
# This is a fast pre-check that runs on any host without an embedded toolchain. It does not prove
# the module builds as Embedded Swift: it cannot see unavailable modules, key paths, or missing
# standard library types. Use scripts/build-embedded.sh for that.

set -euo pipefail

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

cd "$ROOT_DIRECTORY"

# NB: The native build system is required because the default one passes -suppress-warnings, and
# -Werror does not promote diagnostic groups, so violations have to be read out of the output.
swift build \
	--build-system native \
	--disable-default-traits \
	--target EdgeTools \
	-Xswiftc -Wwarning \
	-Xswiftc EmbeddedRestrictions \
	2>&1 | tee "$LOG_FILE"

VIOLATIONS="$(grep -E "^/.*: (warning|error): .*\[#EmbeddedRestrictions\]" "$LOG_FILE" | sort -u || true)"

if [[ -n "$VIOLATIONS" ]]; then
	COUNT="$(printf '%s\n' "$VIOLATIONS" | wc -l | tr -d ' ')"
	echo ""
	echo "Found $COUNT Embedded Swift restriction violation(s):"
	printf '%s\n' "$VIOLATIONS"
	exit 1
fi

echo ""
echo "No Embedded Swift restriction violations."
