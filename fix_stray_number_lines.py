#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove stray indented-number-only lines that are outside code fences.

This fixes artifacts like:
    "        1  "
    "    349"
which often come from bad table-to-code conversions.

Rule:
- If a line matches `^\s+\d+\s*$` and it is NOT inside a fenced code block, delete it.

Usage:
  python3 fix_stray_number_lines.py
"""

from __future__ import annotations

import glob
import re

PAT = re.compile(r"^\s+\d+\s*$")


def process(md_path: str) -> bool:
    lines = open(md_path, "r", encoding="utf-8", errors="ignore").read().splitlines(True)

    out: list[str] = []
    in_fence = False
    changed = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue

        if (not in_fence) and PAT.match(line.rstrip("\n")):
            changed = True
            continue

        out.append(line)

    if changed:
        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(out)
    return changed


def main() -> None:
    files = glob.glob("source/_posts/**/*.md", recursive=True)
    changed_files = []
    for p in files:
        if process(p):
            changed_files.append(p)

    print("=" * 70)
    print("fix_stray_number_lines.py")
    print("=" * 70)
    print(f"Scanned: {len(files)}")
    print(f"Changed: {len(changed_files)}")
    for p in changed_files[:30]:
        print(f"- {p}")
    if len(changed_files) > 30:
        print(f"... and {len(changed_files) - 30} more")


if __name__ == "__main__":
    main()
