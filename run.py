from __future__ import annotations

import sys

from cli import main


if __name__ == "__main__":
    argv = None if len(sys.argv) > 1 else ["--help"]
    raise SystemExit(main(argv))
