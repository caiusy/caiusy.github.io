---
title: shutil模块
date: 2019-08-28 00:00:00
categories:
  - 其他
tags:
  - 编程
---


    1  
    2  
    3  
    4  
    5  
    6  
    7  
      
  
| 
    
    
    shutil.copyfile("old","new") 　　　　  # 复制文件，都只能是文件  
      
    shutil.copytree("old","new")　　　　 # 复制文件夹，都只能是目录，且new必须不存在  
      
    shutil.copy("old","new")　　　　       # 复制文件/文件夹，复制 old 为 new（new是文件，若不存在，即新建），复制 old 为至 new 文件夹（文件夹已存在）  
      
    shutil.move("old","new")  　　　　    # 移动文件/文件夹至 new 文件夹中  
      
  
---|---
