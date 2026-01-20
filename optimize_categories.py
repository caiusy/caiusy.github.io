#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化博客文章的分类和标签
根据文章内容自动分配合理的categories和tags
"""

import os
import re
from pathlib import Path

# 分类规则定义
CATEGORY_RULES = {
    '算法': {
        'keywords': ['algorithm', 'leetcode', 'BFS', 'DFS', '算法', '二叉树', '链表', '栈', '队列', 
                    '回溯', 'backtrack', 'binary-tree', 'binary-search', 'dynamic', 'dp', 
                    'heap', 'graph', 'sliding-window', 'two-pointers', 'stack', 'linked-list',
                    '斐波那契', '跳台阶', '矩形覆盖', '二进制', '查找', '排序', 'Sort', 
                    'Duplicate', '反转', '合并'],
        'subcategory': {
            'LeetCode': ['leetcode'],
        }
    },
    '深度学习': {
        'keywords': ['MobileNet', 'squeezenet', 'ssd', 'GAN', 'RNN', 'MLP', 'Back-Propagation',
                    'SENet', 'attention', 'object-detection', 'Semantic-segmentation',
                    'Scene-Text-Detection', 'AUTOAUGMENT', 'deep learning', 'pytorch',
                    'tensorflow', 'neural', '神经网络', 'mosse', 'HMM'],
    },
    '计算机视觉': {
        'keywords': ['透视变换', '仿射变换', '图像处理', '图像增强', 'opencv', 'PIL', 'skimage',
                    'camera-geometry', 'bird-eye-view', 'ocr', 'park', 'SpatialSense',
                    '文本检测', 'LNMS', 'Hungary'],
    },
    'Python': {
        'keywords': ['python', 'PIL模块', 'opencv模块', 'logging模块', 'glob模块', 'shutil模块',
                    'skimage模块', 'numpy', 'pythonplot', 'matplotlib', 'csv', 'xml',
                    'kmeans', 'PCA', 'knn', 'SVM', '支持向量机'],
    },
    'C++': {
        'keywords': ['STL', 'stack', 'queue', 'deque', '容器', 'c++'],
    },
    '工具': {
        'keywords': ['docker', 'cmd', 'Hexo', 'mx-maskrcnn', '环境搭建'],
    },
    '其他': {
        'keywords': ['firstblog', 'zhongjie', '采坑', 'matlab'],
    }
}

# 标签规则定义
TAG_RULES = {
    'LeetCode': ['leetcode'],
    '算法': ['algorithm', 'BFS', 'DFS', '算法', '动态规划', 'dynamic'],
    '深度学习': ['deep learning', 'pytorch', 'tensorflow', 'neural', 'GAN', 'RNN'],
    '计算机视觉': ['computer vision', '计算机视觉', 'opencv', 'image processing'],
    '目标检测': ['object detection', 'ssd', 'detection'],
    '目标跟踪': ['object track', 'mosse'],
    'Python': ['python'],
    'C++': ['c++', 'STL'],
    '机器学习': ['machine learning', 'kmeans', 'PCA', 'SVM', 'knn'],
    '工具': ['tools', 'docker', 'hexo'],
    'MATLAB': ['matlab'],
}


def detect_category(filename, title):
    """根据文件名和标题检测分类"""
    text = f"{filename} {title}".lower()
    
    # 优先匹配更具体的分类
    for category, rules in CATEGORY_RULES.items():
        for keyword in rules['keywords']:
            if keyword.lower() in text:
                # 检查是否有子分类
                if 'subcategory' in rules:
                    for subcat, subkeywords in rules['subcategory'].items():
                        for subkey in subkeywords:
                            if subkey.lower() in text:
                                return [category, subcat]
                return [category]
    
    return ['技术']  # 默认分类


def detect_tags(filename, title, category):
    """根据文件名、标题和分类检测标签"""
    text = f"{filename} {title}".lower()
    tags = set()
    
    for tag, keywords in TAG_RULES.items():
        for keyword in keywords:
            if keyword.lower() in text:
                tags.add(tag)
    
    # 如果没有检测到标签，使用分类作为标签
    if not tags:
        if isinstance(category, list):
            tags.update(category)
        else:
            tags.add(category)
    
    return sorted(list(tags))


def update_frontmatter(content, filename):
    """更新文章的front matter"""
    # 提取标题
    title_match = re.search(r'^title:\s*(.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filename
    
    # 检测分类和标签
    category = detect_category(filename, title)
    tags = detect_tags(filename, title, category)
    
    # 检查是否已有categories
    has_categories = re.search(r'^categories:', content, re.MULTILINE)
    has_tags = re.search(r'^tags:', content, re.MULTILINE)
    
    if has_categories:
        # 更新现有的categories
        if len(category) == 1:
            content = re.sub(
                r'^categories:.*$',
                f'categories: {category[0]}',
                content,
                flags=re.MULTILINE
            )
        else:
            # 多级分类
            cat_str = '\n'.join([f'  - {c}' for c in category])
            content = re.sub(
                r'^categories:.*$',
                f'categories:\n{cat_str}',
                content,
                flags=re.MULTILINE
            )
    else:
        # 在front matter中添加categories（在title后面）
        if len(category) == 1:
            cat_line = f'categories: {category[0]}'
        else:
            cat_str = '\n'.join([f'  - {c}' for c in category])
            cat_line = f'categories:\n{cat_str}'
        
        content = re.sub(
            r'(^title:.*$)',
            f'\\1\n{cat_line}',
            content,
            flags=re.MULTILINE
        )
    
    if has_tags:
        # 更新现有的tags
        if len(tags) == 1:
            content = re.sub(
                r'^tags:.*$',
                f'tags: {tags[0]}',
                content,
                flags=re.MULTILINE
            )
        else:
            tags_str = '\n'.join([f'  - {t}' for t in tags])
            content = re.sub(
                r'^tags:.*$',
                f'tags:\n{tags_str}',
                content,
                flags=re.MULTILINE
            )
    else:
        # 在categories后面添加tags
        if len(tags) == 1:
            tag_line = f'tags: {tags[0]}'
        else:
            tags_str = '\n'.join([f'  - {t}' for t in tags])
            tag_line = f'tags:\n{tags_str}'
        
        # 找到categories的结束位置
        if len(category) == 1:
            content = re.sub(
                r'(^categories:.*$)',
                f'\\1\n{tag_line}',
                content,
                flags=re.MULTILINE
            )
        else:
            # 多行categories，需要找到最后一个 - 开头的行
            lines = content.split('\n')
            new_lines = []
            in_categories = False
            categories_ended = False
            
            for line in lines:
                new_lines.append(line)
                if line.startswith('categories:'):
                    in_categories = True
                elif in_categories and not line.strip().startswith('-') and not categories_ended:
                    # categories结束
                    new_lines.insert(-1, tag_line)
                    categories_ended = True
                    in_categories = False
            
            if in_categories:  # categories是最后一个字段
                new_lines.append(tag_line)
            
            content = '\n'.join(new_lines)
    
    return content, category, tags


def process_file(file_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新front matter
        new_content, category, tags = update_frontmatter(content, file_path.stem)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return category, tags
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, None


def main():
    """主函数"""
    posts_dir = Path('/Users/caius/Documents/alma/HEXO/caiusy.github.io/source/_posts')
    
    print("=" * 80)
    print("博客分类和标签优化")
    print("=" * 80)
    print()
    
    # 获取所有Markdown文件
    md_files = sorted(posts_dir.glob('*.md'))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    print()
    
    category_stats = {}
    tag_stats = {}
    processed = 0
    
    for md_file in md_files:
        category, tags = process_file(md_file)
        if category:
            processed += 1
            cat_key = ' -> '.join(category) if isinstance(category, list) else category
            category_stats[cat_key] = category_stats.get(cat_key, 0) + 1
            
            for tag in tags:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1
            
            print(f"✓ {md_file.name[:50]:50s} | {cat_key:20s} | {', '.join(tags[:3])}")
    
    print()
    print("=" * 80)
    print(f"处理完成！共优化 {processed} 个文件")
    print()
    print("分类统计：")
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} ({count:3d})")
    print()
    print("标签统计（前10）：")
    for tag, count in sorted(tag_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tag:20s} ({count:3d})")
    print("=" * 80)


if __name__ == '__main__':
    main()
