# Release wrapper copied from original source: scripts/paper_full/make_paper_figures.py
# Private local paths were sanitized for the GitHub release package.
from __future__ import annotations

import argparse
from common import make_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    make_paper_figures(clean=args.clean)


if __name__ == "__main__":
    main()




