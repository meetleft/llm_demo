"""Initialize a novel long-term memory directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from novel_memory_store import NovelMemoryStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="novel_memory/demo")
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    if args.init:
        NovelMemoryStore(args.path).initialize()
        print(f"小说记忆项目已创建：{args.path}")
    else:
        print(Path(args.path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
