#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复所有博客文件中的代码块格式问题
- 将表格格式的代码转换为正确的Markdown代码块
- 修复图片路径问题
"""

import os
import re
from pathlib import Path

def detect_language_from_content(code):
    """从代码内容检测编程语言"""
    # Python
    if any(keyword in code for keyword in ['import ', 'def ', 'class ', 'print(', 'torch.', 'np.', 'cv2.']):
        return 'python'
    
    # Shell/Bash
    if any(keyword in code for keyword in ['#!/bin/bash', 'echo ', 'cd ', '$', 'sh-5.1#']):
        return 'bash'
    
    # C/C++
    if any(keyword in code for keyword in ['#include', 'int main', 'std::', 'cout']):
        return 'cpp'
    
    # JavaScript
    if any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ', '=>']):
        return 'javascript'
    
    # MATLAB
    if '%' in code and ('function ' in code or 'end' in code):
        return 'matlab'
    
    return ''


def advanced_fix_code_blocks(content):
    """
    高级代码块修复 - 处理表格代码格式
    """
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检测表格代码块开始
        # 特征：连续的纯数字行（行号）如 "    1  "
        if re.match(r'^\s{4,}\d+\s*$' , line):
            # 收集所有行号
            line_numbers = []
            start_idx = i
            
            while i < len(lines) and re.match(r'^\s{4,}\d+\s*$', lines[i]):
                line_numbers.append(lines[i])
                i += 1
            
            # 跳过空行和分隔符
            while i < len(lines) and (lines[i].strip() == '' or lines[i].strip() in ['|', '  ', '    ']):
                i += 1
            
            # 收集代码内容（直到遇到 ---|--- 或下一个表格）
            code_lines = []
            while i < len(lines):
                if '---|---' in lines[i]:
                    i += 1
                    break
                if re.match(r'^\s{4}\d+\s{2}$', lines[i]):
                    # 遇到下一个表格，退出
                    break
                if lines[i].strip() and lines[i].strip() not in ['|', '  ', '    ']:
                    code_lines.append(lines[i])
                i += 1
            
            # 如果收集到了代码
            if code_lines:
                # 清理代码
                cleaned_code = []
                for code_line in code_lines:
                    # 移除行尾的多余空格
                    cleaned = code_line.rstrip()
                    if cleaned:
                        cleaned_code.append(cleaned)
                
                # 生成代码块
                code_text = '\n'.join(cleaned_code)
                lang = detect_language_from_content(code_text)
                
                result.append(f'\n```{lang}')
                result.extend(cleaned_code)
                result.append('```\n')
            
            continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def fix_image_paths(content):
    """修复图片路径问题"""
    # 修复 Windows 绝对路径
    content = re.sub(
        r'!\[(.*?)\]\(D:\\HEXO\\source\\_posts\\(.*?)\\(.*?)\)',
        r'![\1](./images/\3)',
        content
    )
    
    # 修复其他可能的绝对路径
    content = re.sub(
        r'!\[(.*?)\]\([A-Z]:\\.*?\\(.*?\.(?:png|jpg|jpeg|gif|svg))\)',
        r'![\1](./images/\2)',
        content
    )
    
    return content


def process_file(file_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用所有修复
        content = advanced_fix_code_blocks(content)
        content = fix_image_paths(content)
        
        # 检查是否有变化
        if content != original_content:
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, file_path.name
        
        return False, None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False, None


def main():
    """主函数"""
    posts_dir = Path('/Users/caius/Documents/alma/HEXO/caiusy.github.io/source/_posts')
    
    print("=" * 70)
    print("批量修复博客代码块和图片格式问题")
    print("=" * 70)
    print()
    
    # 获取所有Markdown文件
    md_files = list(posts_dir.glob('**/*.md'))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    print()
    
    processed_files = []
    
    for md_file in md_files:
        changed, filename = process_file(md_file)
        if changed:
            processed_files.append(filename)
            print(f"✓ 已修复: {filename}")
    
    print()
    print("=" * 70)
    print(f"修复完成！共处理 {len(processed_files)} 个文件")
    print("=" * 70)
    
    if processed_files:
        print("\n已修复的文件：")
        for filename in processed_files[:20]:  # 显示前20个
            print(f"  - {filename}")
        if len(processed_files) > 20:
            print(f"  ... 以及其他 {len(processed_files) - 20} 个文件")


if __name__ == '__main__':
    main()
