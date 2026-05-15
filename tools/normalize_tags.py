#!/usr/bin/env python3
"""
标签规范化：在原 tags 基础上 *追加*（不删除）规范化大标签，使旧文章在新标签体系下也能聚合。
原则：保守，不删除任何旧 tag。
"""
import os, re
from pathlib import Path

POSTS = Path("source/_posts")

# 触发关键词 → 要追加的规范化标签
# 用集合避免重复
ADDITIONS = [
    # (匹配文件名+title+原tags 的关键词, 要追加的规范化tag)
    (r"deep[\s\-_]?learning|深度学习|DeepLearning", "深度学习"),
    (r"computer[\s\-_]?vision|计算机视觉|\bCV\b", "计算机视觉"),
    (r"\bllm\b|large.language|大模型|chatgpt|gpt-?\d|llama|qwen|vicuna", "大模型"),
    (r"\bvlm\b|vision[\s\-_]?language|视觉语言|多模态|multimodal|llava|\bclip\b", "多模态"),
    (r"\bbev\b|bird.eye|俯视图|路侧|roadside|lss|bevdet|bevformer|petr|gauss.*lss", "BEV"),
    (r"radar|雷达|雷视|sensor.fusion", "雷达相机融合"),
    (r"camera.geometry|相机几何|相机模型|内参|外参|消失点|vanishing|射影", "相机几何"),
    (r"homograph|单应性|单应矩阵", "单应性矩阵"),
    (r"transformer|self.?attention|attention.机制", "Transformer"),
    (r"\blora\b", "LoRA"),
    (r"qlora", "QLoRA"),
    (r"\bmoe\b|mixture.of.expert", "MoE"),
    (r"\brag\b|retrieval.augment", "RAG"),
    (r"\bfine.tun|微调|sft\b|instruct.tun", "Fine-tuning"),
    (r"动态规划|dynamic.programming", "动态规划"),
    (r"图论|graph.theory|dfs|bfs|最短路|dijkstra|拓扑排序", "图论"),
    (r"链表|linkedlist", "链表"),
    (r"双指针|two.pointer", "双指针"),
    (r"二叉树|binary.tree", "二叉树"),
    (r"动态规划|背包|01背包", "动态规划"),
    (r"pytorch|torch\.|nn\.module", "PyTorch"),
    (r"opencv|cv2\.", "OpenCV"),
    (r"docker|dockerfile", "Docker"),
    (r"cuda|nvcc|cudnn", "CUDA"),
    (r"\bgit\b|github", "Git"),
    (r"\bpyqt\b|qt5|qwidget|qpainter", "PyQt"),
]

def parse_existing_tags(fm_text):
    """提取已有 tags 的 list 形式"""
    m = re.search(r"^tags:\s*\n((?:\s+-\s+.*\n?)+)", fm_text, re.MULTILINE)
    if m:
        tags = [re.sub(r"^\s*-\s+", "", l).strip() for l in m.group(1).split("\n") if l.strip()]
        return [t for t in tags if t]
    m = re.search(r"^tags:\s*(\S.*)$", fm_text, re.MULTILINE)
    if m:
        return [m.group(1).strip()]
    return []

def main():
    changed = 0
    for f in sorted(os.listdir(POSTS)):
        if not f.endswith(".md"): continue
        p = POSTS / f
        text = p.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
        if not m: continue
        fm_text = m.group(1)
        body = m.group(2)
        existing = parse_existing_tags(fm_text)
        existing_set = set(existing)
        # 拼接搜索文本
        search_text = (f + " " + fm_text + " " + body[:200]).lower()
        to_add = []
        for pattern, tag in ADDITIONS:
            if re.search(pattern, search_text, re.IGNORECASE) and tag not in existing_set:
                to_add.append(tag)
                existing_set.add(tag)
        if not to_add:
            continue
        # 把 to_add 追加到 tags 块
        if re.search(r"^tags:\s*\n((?:\s+-\s+.*\n?)+)", fm_text, re.MULTILINE):
            # list 格式 — 在末尾追加
            def repl(m):
                block = m.group(0).rstrip()
                add = "\n".join(f"  - {t}" for t in to_add)
                return block + "\n" + add + "\n"
            fm_new = re.sub(r"^tags:\s*\n((?:\s+-\s+.*\n?)+)", repl, fm_text, count=1, flags=re.MULTILINE)
        elif re.search(r"^tags:\s*(\S.*)$", fm_text, re.MULTILINE):
            def repl(m):
                first = m.group(1).strip()
                add = "\n".join(f"  - {t}" for t in to_add)
                return f"tags:\n  - {first}\n{add}"
            fm_new = re.sub(r"^tags:\s*(\S.*)$", repl, fm_text, count=1, flags=re.MULTILINE)
        else:
            add = "\n".join(f"  - {t}" for t in to_add)
            fm_new = fm_text.rstrip() + f"\ntags:\n{add}"
        new_text = "---\n" + fm_new + "\n---\n" + body if not fm_new.endswith("\n") else "---\n" + fm_new + "---\n" + body
        p.write_text(new_text, encoding="utf-8")
        changed += 1
    print(f"Tags normalized in {changed} files")

if __name__ == "__main__":
    os.chdir("/Users/caius/Documents/alma/HEXO/caiusy.github.io")
    main()
