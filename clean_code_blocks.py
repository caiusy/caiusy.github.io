#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理Markdown文件中代码块的空行
保持代码块外的内容不变
"""

import os
import re
from pathlib import Path

def clean_code_blocks(content):
    """清理代码块中的空行"""
    lines = content.split('\n')
    result = []
    in_code_block = False
    code_block_lines = []
    
    for line in lines:
        # 检测代码块开始或结束
        if line.strip().startswith('```'):
            if in_code_block:
                # 代码块结束，处理收集的代码行
                # 移除开头和结尾的空行
                while code_block_lines and not code_block_lines[0].strip():
                    code_block_lines.pop(0)
                while code_block_lines and not code_block_lines[-1].strip():
                    code_block_lines.pop()
                
                # 移除连续的空行（保留单个空行用于代码分段）
                cleaned_lines = []
                prev_empty = False
                for code_line in code_block_lines:
                    is_empty = not code_line.strip()
                    if is_empty and prev_empty:
                        # 跳过连续的空行
                        continue
                    cleaned_lines.append(code_line)
                    prev_empty = is_empty
                
                # 添加清理后的代码块
                result.extend(cleaned_lines)
                result.append(line)  # 添加结束的```
                code_block_lines = []
                in_code_block = False
            else:
                # 代码块开始
                result.append(line)
                in_code_block = True
        else:
            if in_code_block:
                # 在代码块内，收集行
                code_block_lines.append(line)
            else:
                # 在代码块外，直接添加
                result.append(line)
    
    return '\n'.join(result)


def process_markdown_file(file_path):
    """处理单个Markdown文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有代码块
        if '```' not in content:
            return False
        
        # 清理代码块
        cleaned_content = clean_code_blocks(content)
        
        # 检查是否有变化
        if cleaned_content == content:
            return False
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def main():
    """主函数"""
    posts_dir = Path('/Users/caius/Documents/alma/HEXO/caiusy.github.io/source/_posts')
    
    print("=" * 60)
    print("开始清理博客文章中代码块的空行")
    print("=" * 60)
    print()
    
    # 获取所有Markdown文件
    md_files = list(posts_dir.glob('*.md'))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    print()
    
    processed_count = 0
    
    for md_file in md_files:
        if process_markdown_file(md_file):
            processed_count += 1
            print(f"✓ 已清理: {md_file.name}")
    
    print()
    print("=" * 60)
    print(f"清理完成！共处理 {processed_count} 个文件")
    print("=" * 60)


if __name__ == '__main__':
    main()
