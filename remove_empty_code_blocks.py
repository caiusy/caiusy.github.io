#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除Markdown文件中只有行号的空代码块
"""

import os
import re
from pathlib import Path

def remove_empty_code_blocks(content):
    """删除只有行号或完全为空的代码块"""
    lines = content.split('\n')
    result = []
    in_code_block = False
    code_start_idx = -1
    code_lang = ""
    code_block_lines = []
    
    for i, line in enumerate(lines):
        # 检测代码块开始
        if line.strip().startswith('```'):
            if not in_code_block:
                # 代码块开始
                in_code_block = True
                code_start_idx = len(result)
                code_lang = line.strip()[3:].strip()
                result.append(line)
                code_block_lines = []
            else:
                # 代码块结束，检查内容
                # 判断是否是空代码块或只有行号的代码块
                non_empty_lines = [l for l in code_block_lines if l.strip()]
                
                # 检查是否只包含数字（行号）
                is_only_numbers = True
                if non_empty_lines:
                    for l in non_empty_lines:
                        # 如果不是纯数字，则不是"只有行号"的代码块
                        if not l.strip().isdigit():
                            is_only_numbers = False
                            break
                else:
                    # 完全空的代码块
                    is_only_numbers = True
                
                if is_only_numbers and len(non_empty_lines) > 2:
                    # 这是一个只有行号的空代码块，删除整个代码块
                    # 回退到代码块开始之前
                    result = result[:code_start_idx]
                else:
                    # 保留代码块
                    result.extend(code_block_lines)
                    result.append(line)
                
                in_code_block = False
                code_block_lines = []
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
            return False, 0
        
        # 统计代码块数量（处理前）
        code_blocks_before = content.count('```') // 2
        
        # 删除空代码块
        cleaned_content = remove_empty_code_blocks(content)
        
        # 统计代码块数量（处理后）
        code_blocks_after = cleaned_content.count('```') // 2
        
        removed = code_blocks_before - code_blocks_after
        
        # 检查是否有变化
        if cleaned_content == content:
            return False, 0
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True, removed
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False, 0


def main():
    """主函数"""
    posts_dir = Path('/Users/caius/Documents/alma/HEXO/caiusy.github.io/source/_posts')
    
    print("=" * 60)
    print("删除只有行号的空代码块")
    print("=" * 60)
    print()
    
    # 获取所有Markdown文件
    md_files = list(posts_dir.glob('*.md'))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    print()
    
    processed_count = 0
    total_removed = 0
    
    for md_file in md_files:
        changed, removed = process_markdown_file(md_file)
        if changed:
            processed_count += 1
            total_removed += removed
            print(f"✓ 已清理: {md_file.name:50s} (删除 {removed} 个空代码块)")
    
    print()
    print("=" * 60)
    print(f"清理完成！共处理 {processed_count} 个文件")
    print(f"共删除 {total_removed} 个空代码块")
    print("=" * 60)


if __name__ == '__main__':
    main()
