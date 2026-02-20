#!/usr/bin/env python3
import os
import re

MAPPING = {
    'AI & Deep Learning': '深度学习',
    'AI Learning': '深度学习',
    'AI与大模型': '深度学习',
    '机器学习': '深度学习',
    'AI理论': '深度学习',
    '计算机视觉 - 计算机视觉': '计算机视觉',
}

def fix_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 处理数组格式: categories: [xxx]
    for old, new in MAPPING.items():
        content = re.sub(
            rf'categories:\s*\[{re.escape(old)}\]',
            f'categories: [{new}]',
            content
        )
    
    # 处理列表格式: categories:\n  - xxx
    for old, new in MAPPING.items():
        content = re.sub(
            rf'^(categories:\s*\n\s*-\s*){re.escape(old)}$',
            rf'\1{new}',
            content,
            flags=re.MULTILINE
        )
    
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

posts_dir = 'source/_posts'
changed = []

for f in os.listdir(posts_dir):
    if f.endswith('.md'):
        path = os.path.join(posts_dir, f)
        if fix_file(path):
            changed.append(f)

print(f"修改了 {len(changed)} 个文件")
for f in changed:
    print(f"  - {f}")
