#!/usr/bin/env python3
"""
为 type: note 的文章追加 学习笔记 + 二级子分类(如 概念笔记)，
使 /categories/学习笔记/概念笔记/ 这种 URL 能正常工作。

Hexo 在单层 list 下，多个分类是平级；在嵌套 list 下，是父-子。
我们在保留原一级分类的同时，追加一个 [学习笔记, 概念笔记] 嵌套子列表。
"""
import os, re
from pathlib import Path

POSTS = Path("source/_posts")

NOTE_TYPE_CN = {
    "concept": "概念笔记",
    "paper": "论文笔记",
    "engineering": "工程笔记",
    "algorithm": "算法笔记",
}

def main():
    changed = 0
    for f in sorted(os.listdir(POSTS)):
        if not f.endswith(".md"): continue
        p = POSTS / f
        text = p.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
        if not m: continue
        fm_text, body = m.group(1), m.group(2)
        # 必须是 note 类型
        type_m = re.search(r"^type:\s*(\S+)", fm_text, re.MULTILINE)
        if not type_m or type_m.group(1) != "note": continue
        note_type_m = re.search(r"^note_type:\s*(\S+)", fm_text, re.MULTILINE)
        if not note_type_m: continue
        note_type = note_type_m.group(1)
        cn = NOTE_TYPE_CN.get(note_type)
        if not cn: continue
        # 已经有"学习笔记"子分类则跳过
        if "学习笔记" in fm_text and cn in fm_text:
            continue
        # 在 categories 块末尾追加嵌套子分类
        cats_match = re.search(r"^categories:\s*\n((?:\s+-\s+.*\n?)+)", fm_text, re.MULTILINE)
        if not cats_match: continue
        block = cats_match.group(0).rstrip()
        addition = f"\n  -\n    - 学习笔记\n    - {cn}\n"
        new_block = block + addition
        fm_new = fm_text.replace(cats_match.group(0).rstrip("\n"), new_block.rstrip("\n"))
        new_text = "---\n" + fm_new + "\n---\n" + body
        p.write_text(new_text, encoding="utf-8")
        changed += 1
    print(f"Added 学习笔记 sub-category in {changed} files")

if __name__ == "__main__":
    os.chdir("/Users/caius/Documents/alma/HEXO/caiusy.github.io")
    main()
