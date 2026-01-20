#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
博客格式修复脚本 V2 - 增强版
专门处理以下问题：
1. 删除纯行号块（1 2 3 4 ... 这种）
2. 修复表格包裹的代码块
3. 清理多余的空行
4. 确保代码块格式正确
"""

import os
import re
from pathlib import Path

def is_line_numbers_only(text):
    """检查文本是否只包含行号"""
    lines = text.strip().split('\n')
    if len(lines) < 5:  # 至少5行才算
        return False
    
    # 检查是否每行都是数字
    number_lines = 0
    for line in lines:
        line = line.strip()
        if line.isdigit():
            number_lines += 1
    
    # 如果超过80%的行都是数字，认为是行号块
    return number_lines / len(lines) > 0.8


def extract_code_from_table(text):
    """从表格格式中提取代码"""
    # 移除表格分隔符和管道符
    lines = text.split('\n')
    code_lines = []
    
    for line in lines:
        # 跳过表格分隔符行
        if re.match(r'^[\s|:-]+$', line):
            continue
        
        # 移除开头和结尾的管道符
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        
        line = line.strip()
        if line and not re.match(r'^-+$', line):
            code_lines.append(line)
    
    return '\n'.join(code_lines)


def detect_code_language(code):
    """检测代码语言"""
    code_lower = code.lower()
    
    # Python特征
    if any(keyword in code_lower for keyword in ['import ', 'def ', 'class ', 'print(', 'from ', '__init__']):
        return 'python'
    
    # JavaScript特征
    if any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
        return 'javascript'
    
    # Java特征
    if any(keyword in code_lower for keyword in ['public class', 'private ', 'protected ', 'void ', 'static ']):
        return 'java'
    
    # C/C++特征
    if any(keyword in code_lower for keyword in ['#include', 'int main', 'printf', 'cout', 'std::']):
        return 'cpp'
    
    # Shell特征
    if any(keyword in code_lower for keyword in ['#!/bin/bash', 'sudo ', 'apt-get', 'yum ']):
        return 'bash'
    
    return ''


def fix_markdown_file(file_path):
    """修复单个Markdown文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # === 步骤1: 删除纯行号块 ===
    # 匹配连续的行号（可能在代码块中）
    pattern_line_numbers = r'```\s*\n\s*((?:\d+\s*\n)+)\s*```'
    
    def remove_line_numbers(match):
        block_content = match.group(1)
        if is_line_numbers_only(block_content):
            changes_made.append("删除了纯行号块")
            return ''  # 完全删除
        return match.group(0)  # 保留原样
    
    content = re.sub(pattern_line_numbers, remove_line_numbers, content, flags=re.MULTILINE)
    
    # 也处理没有代码块标记的纯数字段落
    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检查是否是连续数字行的开始
        if line.isdigit() and i + 5 < len(lines):
            # 查看接下来的行
            consecutive_numbers = 1
            j = i + 1
            while j < len(lines) and lines[j].strip().isdigit():
                consecutive_numbers += 1
                j += 1
            
            # 如果有超过10行连续数字，认为是行号，删除
            if consecutive_numbers > 10:
                changes_made.append(f"删除了{consecutive_numbers}行连续数字")
                i = j
                continue
        
        new_lines.append(lines[i])
        i += 1
    
    content = '\n'.join(new_lines)
    
    # === 步骤2: 修复表格包裹的代码 ===
    # 匹配表格格式的代码块
    pattern_table_code = r'\|\s*\n((?:.*\n)+?)\s*---\|---'
    
    def fix_table_code(match):
        table_content = match.group(1)
        
        # 检查是否看起来像代码
        if any(keyword in table_content for keyword in ['import ', 'def ', 'class ', 'function', '#include', 'public ']):
            code = extract_code_from_table(table_content)
            language = detect_code_language(code)
            
            changes_made.append(f"将表格格式转换为{language}代码块")
            return f'\n```{language}\n{code}\n```\n'
        
        return match.group(0)
    
    content = re.sub(pattern_table_code, fix_table_code, content, flags=re.MULTILINE)
    
    # === 步骤3: 清理代码块中的多余空行 ===
    def clean_code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        
        # 移除开头和结尾的空行
        code_lines = code.split('\n')
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        
        code = '\n'.join(code_lines)
        return f'```{lang}\n{code}\n```'
    
    pattern_code_block = r'```(\w*)\n(.*?)```'
    content = re.sub(pattern_code_block, clean_code_block, content, flags=re.DOTALL)
    
    # === 步骤4: 清理多余的空行（不超过2个连续空行）===
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # 保存文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes_made
    
    return False, []


def main():
    """主函数"""
    posts_dir = Path('/Users/caius/Documents/alma/HEXO/caiusy.github.io/source/_posts')
    
    if not posts_dir.exists():
        print(f"错误：目录不存在: {posts_dir}")
        return
    
    print("=" * 80)
    print("博客格式修复脚本 V2 - 开始执行")
    print("=" * 80)
    print()
    
    fixed_files = []
    skipped_files = []
    
    # 遍历所有md文件
    for md_file in posts_dir.rglob('*.md'):
        try:
            modified, changes = fix_markdown_file(md_file)
            
            if modified:
                fixed_files.append(md_file)
                print(f"✓ 已修复: {md_file.name}")
                for change in changes:
                    print(f"  - {change}")
            else:
                skipped_files.append(md_file)
                
        except Exception as e:
            print(f"✗ 处理失败: {md_file.name}")
            print(f"  错误: {e}")
    
    # 输出统计
    print()
    print("=" * 80)
    print("修复完成！")
    print("=" * 80)
    print(f"修复文件数: {len(fixed_files)}")
    print(f"跳过文件数: {len(skipped_files)}")
    print(f"总文件数: {len(fixed_files) + len(skipped_files)}")
    print()
    
    if fixed_files:
        print("已修复的文件列表：")
        for f in fixed_files:
            print(f"  - {f.name}")


if __name__ == '__main__':
    main()
