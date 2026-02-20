---
title: PIL模块
date: 2019-08-28 00:00:00
categories:
  - 计算机视觉
tags:
  - Python
  - OpenCV
---

## 导入
    
    
    1  
    

| 
    
    
    from PIL import Image  
      
  
---|---  
  
## 读取
    
    
    1  
    

| 
    
    
    img = Image.open(filepath)  
      
  
---|---  
  
## 显示
    
    
    1  
    

| 
    
    
    img.show()  
      
  
---|---  
  
## 与 numpy 数组的互相转换

PIL Image 转 numpy 数组  

    
    
    1  
    

| 
    
    
    img_to_array = np.array(img)  
      
  
---|---  
  
numpy 数组转 PIL Image (注意要确保数组内的值符合 PIL 的要求)  

    
    
    1  
    

| 
    
    
    array_to_img = Image.fromarray(img_to_array)  
      
  
---|---  
  
## PIL 与 cv2 格式互相转换

PIL.Image读入的图片数据类型不是 numpy 数组, 它的size属性为 (w, h), 利用np.array转换成 numpy 数组后, 它的通道顺序为 (r, g, b)  

    
    
    1  
    2  
    3  
    4  
    5  
    6  
    7  
    8  
    9  
    10  
    11  
    

| 
    
    
    from PIL import Image  
    import numpy as np  
      
    # PIL to cv2  
    pil_img = Image.open(img_path)  
    print(pil_img.size) # (w, h)  
    np_img = np.array(pil_img)  
    cv2_img = np_img[:, :, ::-1] # 交换通道  
      
    # cv2 to PIL  
    pil_img = Image.fromarray(cv2_img[:, :, ::-1])  
      
  
---|---
