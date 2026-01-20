#!/bin/bash

echo "======================================"
echo "相机几何可视化 - 批量运行脚本"
echo "======================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查依赖
echo "[1/5] 检查依赖..."
python3 -c "import numpy, matplotlib, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
fi
echo "✓ 依赖检查完成"
echo ""

# 创建输出目录
mkdir -p images
echo "✓ 创建输出目录: images/"
echo ""

# 运行可视化脚本
echo "[2/5] 运行坐标系统可视化..."
python3 visualize_coordinates.py
echo ""

echo "[3/5] 运行旋转矩阵可视化..."
python3 rotation_visualization.py
echo ""

echo "[4/5] 运行单应矩阵可视化..."
python3 homography_visualization.py
echo ""

echo "[5/5] 运行完整演示..."
python3 camera_geometry_demo.py
echo ""

# 移动生成的图片
echo "移动生成的图片到 images/ 目录..."
mv *.png images/ 2>/dev/null

echo ""
echo "======================================"
echo "✓ 所有可视化完成！"
echo "======================================"
echo ""
echo "生成的图片保存在 images/ 目录"
ls -lh images/*.png 2>/dev/null
echo ""
