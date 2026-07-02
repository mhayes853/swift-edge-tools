from __future__ import annotations

import argparse
import contextlib
import io
import sys
from typing import Sequence

from needle.coreai_export import DEFAULT_SOURCE, export_needle_coreai


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Needle CoreAI models")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True)
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    with contextlib.redirect_stdout(io.StringIO()):
        output_directory = export_needle_coreai(parsed.source, parsed.output)
    sys.stdout.write(f"{output_directory}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())