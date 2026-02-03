#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-fix Markdown image links in a reproducible way.

Scope:
- Only touches Markdown under `source/_posts/**`.
- Rewrites legacy domain image URLs to local `/images/...` when possible.
- Rewrites `./images/...` and `images/...` to `/images/...` if the target file exists in `source/images`.
- Rewrites `figures/...` to `./image-to-bird-eye-view/figures/...` for the specific post assets layout.

Does NOT:
- Modify theme files.
- Download missing assets.

Usage:
  python3 fix_md_assets.py
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass


LEGACY_PREFIXES = (
    "//caius-lu.github.io/",
    "https://caius-lu.github.io/",
    "http://caius-lu.github.io/",
)

MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class Replacement:
    md_path: str
    old: str
    new: str


def load_source_images_index() -> set[str]:
    names: set[str] = set()
    for path in glob.glob("source/images/**/*", recursive=True):
        if os.path.isfile(path):
            names.add(os.path.basename(path))
    return names


def normalize_legacy_url(token: str) -> str:
    for pref in LEGACY_PREFIXES:
        if token.startswith(pref):
            return "/" + token[len(pref) :].lstrip("/")
    return token


def collapse_post_images_to_root(path: str) -> str:
    # /2019/12/24/Post/images/foo.png -> /images/foo.png
    m = re.match(r"^/(\d{4}/\d{2}/\d{2}/[^/]+/)?images/(.+)$", path)
    if m:
        return "/images/" + m.group(2)
    return path


def apply_replacements(text: str, replacements: list[Replacement], md_path: str) -> tuple[str, int]:
    count = 0
    for r in replacements:
        if r.md_path != md_path:
            continue
        if r.old in text:
            text = text.replace(r.old, r.new)
            count += 1
    return text, count


def main() -> None:
    source_image_names = load_source_images_index()

    md_files = glob.glob("source/_posts/**/*.md", recursive=True)
    replacements: list[Replacement] = []
    missing: list[tuple[str, str]] = []

    for md in md_files:
        content = open(md, "r", encoding="utf-8", errors="ignore").read()
        for raw in MD_IMAGE_PATTERN.findall(content):
            # strip title, keep only first token
            token = raw.strip().split()[0]
            if token.startswith("http") or token.startswith("data:"):
                continue

            # 1) legacy domain -> local
            normalized = normalize_legacy_url(token)
            normalized = collapse_post_images_to_root(normalized)
            if normalized != token and normalized.startswith("/images/"):
                # only rewrite if target exists
                fname = os.path.basename(normalized)
                if fname in source_image_names:
                    replacements.append(Replacement(md, token, normalized))
                else:
                    missing.append((md, token))
                continue

            # 2) ./images/foo.png or images/foo.png -> /images/foo.png if exists
            if token.startswith("./images/") or token.startswith("images/"):
                fname = os.path.basename(token)
                if fname in source_image_names:
                    replacements.append(Replacement(md, token, f"/images/{fname}"))
                else:
                    missing.append((md, token))
                continue

            # 3) figures/... in `image-to-bird-eye-view.md` -> nested asset folder
            if token.startswith("figures/") and os.path.basename(md) == "image-to-bird-eye-view.md":
                # keep as relative to post folder which is already located at
                # source/_posts/image-to-bird-eye-view/figures/...
                replacements.append(
                    Replacement(md, token, f"./image-to-bird-eye-view/{token}")
                )
                continue

            # 4) if it is already absolute /images/... but missing, track it
            if token.startswith("/images/"):
                fname = os.path.basename(token)
                if fname not in source_image_names:
                    missing.append((md, token))

    changed_files = 0
    total_replacements = 0
    for md in md_files:
        original = open(md, "r", encoding="utf-8", errors="ignore").read()
        updated, c = apply_replacements(original, replacements, md)
        if updated != original:
            with open(md, "w", encoding="utf-8") as f:
                f.write(updated)
            changed_files += 1
            total_replacements += c

    print("=" * 70)
    print("fix_md_assets.py")
    print("=" * 70)
    print(f"Markdown files scanned: {len(md_files)}")
    print(f"Files changed: {changed_files}")
    print(f"Replacements applied: {total_replacements}")

    if missing:
        print("\nMissing targets (needs assets or manual decision):")
        for md, token in missing[:50]:
            print(f"- {md}: {token}")
        if len(missing) > 50:
            print(f"... and {len(missing) - 50} more")


if __name__ == "__main__":
    main()
