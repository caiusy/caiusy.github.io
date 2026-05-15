#!/usr/bin/env python3
"""
增量梳理 _posts front-matter
原则:
1. 不改文件名 (永久链接安全)
2. 不删除原有 categories / tags
3. 只追加 type / note_type / difficulty / review_status
4. 重复的一级分类(深度学习/计算机视觉/算法等)归类到新的一级分类体系
"""
import os, re, sys
from pathlib import Path

POSTS = Path("source/_posts")

# 关键词 → (新一级分类, type, note_type, difficulty)
# 顺序很重要 — 越特定的规则越靠前
RULES = [
    # --- 多模态大模型 (最特定) ---
    (r"llava|clip|llm|gpt|bert|attention|transformer|多模态|视觉语言|vision-language|大模型|huggingface|qwen|vlm|qlora|\blora\b|mixture.of.expert|\bmoe\b|\brag\b|prompt.engineer|instruct|微调|fine.tun|预训练|pre.train|chatgpt|llama|vicuna", "多模态大模型", "deep-dive", None, "intermediate"),
    # --- BEV / 路侧 / 相机几何 ---
    (r"bev|lss\b|bevdet|petr|detr3d|路侧|roadside|gauss.*lss|nuscenes|kitti|相机几何|camera.geometry|bird.eye|俯视图|消失点|vanishing|homograph|单应|仿射|内参|外参|标定|calibrat|\bipm\b|相机模型|射影|透视变换", "BEV感知", "deep-dive", None, "intermediate"),
    # --- 雷达 / 雷视融合 ---
    (r"radar|雷达|雷视|sensor.fusion|融合|millimeter|毫米波|fmcw", "雷达相机融合", "deep-dive", None, "intermediate"),
    # --- 算法 / leetcode / 数据结构 (非常宽匹配 - 早期文章很多算法题) ---
    (r"leetcode|动态规划|dynamic.programming|图论|graph.theory|链表|linkedlist|双指针|two.pointer|单调栈|monotonic|二叉树|binary.tree|\bdfs\b|\bbfs\b|背包|\bkmp\b|二分查找|二分搜索|排序|sorting|快排|归并|堆排|sliding.window|滑动窗口|回溯|backtrack|哈希|hash.table|跳台阶|斐波那契|fibonacci|矩形覆盖|二进制中.*个数|树的子结构|替换空格|从尾到头|hungary|匈牙利|算法面试|顺序容器|stl|\bdeque\b|\bvector\b|\bqueue\b|\bstack\b|\bset\b|\bmap\b|拓扑|最短路|dijkstra|floyd|krusk|prim", "算法基础", "note", "algorithm", "intermediate"),
    # --- OpenCV / 图像处理 / 传统CV ---
    (r"opencv|图像处理|图像增强|形态学|边缘|hough|sift|surf|\borb\b|特征点|tracking|跟踪|目标检测|object.detect|reid|yolo|faster.rcnn|fcos|\banchor\b|\bnms\b|\biou\b|mosse|\bkcf\b|\bsort\b|bytetrack|deepsort|分割|segmentation|相机标定|opencv模块|图像金字塔|高斯滤波|中值滤波|直方图|轮廓|霍夫|角点|harris|fast.特征|\bpil\b|skimage|图片处理|图像识别|ocr|答题卡|车牌识别|颜色空间|阈值|二值化|灰度化", "计算机视觉", "deep-dive", None, "intermediate"),
    # --- 深度学习基础 / 模型架构 / 训练 ---
    (r"\bgan\b|\bcnn\b|\brnn\b|\blstm\b|resnet|\bvgg\b|inception|batch.norm|dropout|activation|\brelu\b|sigmoid|softmax|\bloss\b|交叉熵|cross.entropy|kl.散度|kl.divergence|backprop|反向传播|梯度|gradient|optimizer|\badam\b|\bsgd\b|tensorflow|tensorboard|\bsvm\b|支持向量机|\bknn\b|决策树|聚类|kmeans|\bpca\b|降维|无监督|监督学习|半监督|强化学习|reinforce|神经网络|neural.network|训练.*模型|超参|hyper.param|过拟合|正则化|regulariz|epoch|batch.size|学习率|learning.rate|数据增强|data.augment|imagenet|ssd源码|目标检测.*模型", "深度学习", "note", "concept", "intermediate"),
    # --- PyTorch ---
    (r"pytorch|torch\.|nn\.module|dataloader|\btensor\b", "深度学习", "note", "engineering", "intermediate"),
    # --- 工程实践 / 环境 / 工具 ---
    (r"docker|cuda|conda|\bpip\b|安装|install|部署|deploy|nginx|linux|shell|bash|tmux|vscode|jupyter|\bgit\b|github|\bssh\b|rsync|server|服务器|远程|terminal|\bcmd\b|环境配置|采坑|踩坑|报错|错误|debug|调试", "工程实践", "note", "engineering", "intermediate"),
    # --- Python 语言 / 库 ---
    (r"python|pandas|numpy|matplotlib|seaborn|scikit|sklearn|pyqt|\bqt\b|tkinter|flask|django|requests|asyncio|generator|装饰器|decorator|闭包|迭代器|魔法方法|数据类|dataclass|shutil|遍历文件|文件夹|python手册|\bpil\b模块|skimage模块|\.py\b", "工程实践", "note", "engineering", "beginner"),
    # --- C++ ---
    (r"\bc\+\+|\bstl\b|template|多态|虚函数|继承|operator.重载", "工程实践", "note", "engineering", "intermediate"),
    # --- MCP / Skills / Claude / Alma 工具 ---
    (r"\bmcp\b|skill|claude|alma|cursor|copilot|telegram|discord|feishu", "工程实践", "note", "engineering", "intermediate"),
    # --- 数学基础 ---
    (r"线性代数|矩阵|matrix|特征值|eigenvalue|\bsvd\b|向量|vector|概率|probability|统计|高斯|gaussian|分布|distribution|贝叶斯|bayes|信息论|熵|entropy|kl[\.\s\-_]?散度|kl[\.\s\-_]?divergence|散度.*kl|ransac|hmm|马尔可夫|markov", "深度学习", "note", "concept", "intermediate"),
    # --- 经典轻量网络 / mobile / squeeze 等 → 深度学习 ---
    (r"mobilenet|squeezenet|shufflenet|efficientnet|mobile.net|depth.wise|depthwise", "深度学习", "note", "paper", "intermediate"),
    # --- mmdetection / mmcv / detectron 等检测框架 ---
    (r"mmdetection|mmcv|detectron|paddle|mxnet|maskrcnn|mask.rcnn|faster.rcnn|retinanet", "深度学习", "note", "engineering", "intermediate"),
    # --- 经典数据结构题(只有 2 条) ---
    (r"二维数组|两个栈|实现队列|分治|knn.*手写|手写.*数字|xml.解析|xml.生成|csv.读取|从csv|adaptive.pose", "算法基础", "note", "algorithm", "beginner"),
    # --- 文本检测 / 场景检测 ---
    (r"文本检测|scene.text|spatial.sense|场景检测", "计算机视觉", "deep-dive", None, "intermediate"),
    # --- matlab ---
    (r"matlab", "工程实践", "note", "engineering", "beginner"),
    # --- hexo / blog 工具 ---
    (r"hexo|blog|博客|markdown", "工程实践", "note", "engineering", "beginner"),
    # --- adversarial / 对抗 ---
    (r"adversarial|对抗", "深度学习", "note", "paper", "intermediate"),
    # --- 算法竞赛 / dynamic planning ---
    (r"算法竞赛|入门经典|dynamic.plan|lnms", "算法基础", "note", "algorithm", "beginner"),
    # --- xml ---
    (r"xml", "工程实践", "note", "engineering", "beginner"),
    # --- xmind / 项目类 ---
    (r"xmind|智慧交通|track.*xmind", "计算机视觉", "deep-dive", None, "intermediate"),
    # --- spatial sense ---
    (r"spatial.sense|spatialsense", "计算机视觉", "note", "paper", "intermediate"),
    # --- PIL ---
    (r"\bpil\b模块|pil模块", "工程实践", "note", "engineering", "beginner"),
]

def classify(filename, title, existing_cats):
    """返回 (new_cat, type, note_type, difficulty)"""
    text = (filename + " " + (title or "") + " " + " ".join(str(c) for c in (existing_cats or []))).lower()
    for pattern, cat, typ, note_type, diff in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return cat, typ, note_type, diff
    # 默认
    return "历史归档", "archive", None, "beginner"

def parse_fm(text):
    """简单解析 front-matter，返回 (fm_dict_lines, body)"""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
    if not m: return None, text
    return m.group(1), m.group(2)

def get_field(fm_text, key):
    m = re.search(rf"^{key}:\s*(.*)$", fm_text, re.MULTILINE)
    return m.group(1).strip() if m else None

def has_field(fm_text, key):
    return re.search(rf"^{key}:", fm_text, re.MULTILINE) is not None

def get_categories_list(fm_text):
    """提取 categories 一级分类列表(尽量简单)"""
    # 多行格式 categories:\n  - xx\n  - yy
    m = re.search(r"^categories:\s*\n((?:\s+-\s+.*\n?)+)", fm_text, re.MULTILINE)
    if m:
        return [re.sub(r"^\s*-\s+", "", l).strip() for l in m.group(1).strip().split("\n") if l.strip()]
    # 单行 categories: xxx
    m = re.search(r"^categories:\s*(\S.*)$", fm_text, re.MULTILINE)
    if m:
        return [m.group(1).strip()]
    return []

def main():
    changed = 0
    skipped = 0
    summary = {"deep-dive": 0, "note": 0, "archive": 0, "experiment": 0, "project": 0}
    cat_summary = {}
    
    for f in sorted(os.listdir(POSTS)):
        if not f.endswith(".md"): continue
        p = POSTS / f
        text = p.read_text(encoding="utf-8")
        fm_text, body = parse_fm(text)
        if fm_text is None:
            skipped += 1
            continue
        if has_field(fm_text, "type"):
            # 已经处理过
            continue
        title = get_field(fm_text, "title") or f
        existing_cats = get_categories_list(fm_text)
        new_cat, typ, note_type, diff = classify(f, title, existing_cats)
        
        # 准备追加字段
        addons = []
        addons.append(f"type: {typ}")
        if note_type:
            addons.append(f"note_type: {note_type}")
        addons.append(f"difficulty: {diff}")
        addons.append(f"review_status: {'archived' if typ == 'archive' else 'reviewing'}")
        
        # 追加新一级分类(如果不存在)
        cats_section_match = re.search(r"^categories:\s*\n((?:\s+-\s+.*\n?)+)", fm_text, re.MULTILINE)
        if cats_section_match:
            existing_block = cats_section_match.group(0)
            if new_cat not in [c.strip("- ").strip() for c in existing_block.split("\n")]:
                # 在 categories 块顶部追加新一级
                new_block = f"categories:\n  - {new_cat}\n" + "\n".join(cats_section_match.group(1).rstrip().split("\n")) + "\n"
                fm_text_new = fm_text.replace(existing_block, new_block)
            else:
                fm_text_new = fm_text
        elif "categories:" in fm_text:
            # 单行格式: 替换为 list 并 prepend
            fm_text_new = re.sub(r"^categories:\s*(\S.*)$", lambda m: f"categories:\n  - {new_cat}\n  - {m.group(1).strip()}", fm_text, flags=re.MULTILINE)
        else:
            # 没有 categories
            fm_text_new = fm_text + f"\ncategories:\n  - {new_cat}"
        
        fm_text_new = fm_text_new.rstrip() + "\n" + "\n".join(addons) + "\n"
        new_text = "---\n" + fm_text_new + "---\n" + body
        p.write_text(new_text, encoding="utf-8")
        changed += 1
        summary[typ] = summary.get(typ, 0) + 1
        cat_summary[new_cat] = cat_summary.get(new_cat, 0) + 1
    
    print(f"Updated: {changed} files, Skipped (no fm or already typed): {skipped}")
    print("\nBy type:")
    for k, v in summary.items():
        print(f"  {k:14s}  {v}")
    print("\nBy new top category:")
    for k, v in sorted(cat_summary.items(), key=lambda x: -x[1]):
        print(f"  {k:14s}  {v}")

if __name__ == "__main__":
    os.chdir("/Users/caius/Documents/alma/HEXO/caiusy.github.io")
    main()
