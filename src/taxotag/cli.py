"""Command-line entry point for taxotag."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .model import Gist, GistError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify text with taxotag.")
    parser.add_argument("text", help="text to classify")
    parser.add_argument("--top-k", type=int, default=3, help="maximum topics to print")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum topic score; the top topic is always retained",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="write topics as a JSON array",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="local model asset directory",
    )
    parser.add_argument(
        "--variant",
        choices=("multilingual", "english"),
        default="multilingual",
        help="model variant to use",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the taxotag command-line interface."""

    args = _parser().parse_args(argv)
    try:
        topics = Gist(directory=args.directory, variant=args.variant).classify(
            args.text,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    except (GistError, ValueError) as error:
        print(f"taxotag: {error}", file=sys.stderr)
        return 1

    if args.as_json:
        json.dump([asdict(topic) for topic in topics], sys.stdout)
        sys.stdout.write("\n")
    else:
        for topic in topics:
            print(f"{topic.slug}\t{topic.name}\t{topic.score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
