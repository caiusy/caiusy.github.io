#!/bin/bash

echo "======================================"
echo "🚀 HEXO博客发布脚本"
echo "======================================"
echo ""

cd /Users/caius/Documents/alma/HEXO/caiusy.github.io

# 检查是否有未提交的更改
if [[ -n $(git status -s) ]]; then
    echo "📝 发现未提交的更改"
    git status -s
    echo ""
    read -p "是否提交这些更改？(y/n): " confirm
    
    if [[ $confirm == "y" ]]; then
        echo "请输入提交信息:"
        read commit_msg
        
        git add .
        git commit -m "$commit_msg"
        echo "✅ 更改已提交"
    fi
else
    echo "✅ 没有未提交的更改"
fi

echo ""
echo "======================================"
echo "📦 生成静态文件..."
echo "======================================"
npx hexo clean
npx hexo generate

echo ""
echo "======================================"
echo "🚀 部署到GitHub Pages..."
echo "======================================"
npx hexo deploy

echo ""
echo "======================================"
echo "✅ 博客发布完成！"
echo "======================================"
echo ""
echo "🌐 访问你的博客: https://caiusy.github.io"
echo ""
