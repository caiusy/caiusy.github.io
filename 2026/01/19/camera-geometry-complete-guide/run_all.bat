@echo off
echo ======================================
echo 相机几何可视化 - 批量运行脚本
echo ======================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    pause
    exit /b 1
)

REM 检查依赖
echo [1/5] 检查依赖...
python -c "import numpy, matplotlib, scipy" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
)
echo √ 依赖检查完成
echo.

REM 创建输出目录
if not exist images mkdir images
echo √ 创建输出目录: images/
echo.

REM 运行可视化脚本
echo [2/5] 运行坐标系统可视化...
python visualize_coordinates.py
echo.

echo [3/5] 运行旋转矩阵可视化...
python rotation_visualization.py
echo.

echo [4/5] 运行单应矩阵可视化...
python homography_visualization.py
echo.

echo [5/5] 运行完整演示...
python camera_geometry_demo.py
echo.

REM 移动生成的图片
echo 移动生成的图片到 images/ 目录...
move *.png images\ >nul 2>&1

echo.
echo ======================================
echo √ 所有可视化完成！
echo ======================================
echo.
echo 生成的图片保存在 images/ 目录
dir /b images\*.png
echo.
pause
