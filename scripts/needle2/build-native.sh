#!/usr/bin/env bash

set -euo pipefail

root_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact="${1:-$root_directory/bin/needle2-2.0.3.artifactbundle.zip}"
output_directory="${2:-$root_directory/ts/needle2/dist/native}"

exec "$root_directory/ts/needle2/scripts/build-native.sh" \
  "$artifact" "$output_directory"
