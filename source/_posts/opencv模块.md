---
title: opencv模块
date: 2019-08-28 00:00:00
categories:
  - 计算机视觉
tags:
  - Python
---
## opencv 基础知识  
  
cv2.imread 读入的图片, 其shape为(h, w, c), 颜色通道顺序为 (b, g, r)

## 常用颜色

## 读取图片
    
    
    1  
    

| 
    
    
    img = cv2.imread(img_path)  
      
  
---|---  
  
## 保存图片
    
    
    1  
    

| 
    
    
    cv2.imwrite(save_path, img)  
      
  
---|---  
  
## 文本

(startX, startY) 为左上角坐标  

    
    
    1  
    

| 
    
    
    cv2.putText(img, "text test", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, font_size, (B,G,R), thickness)  
      
  
---|---  
  
## 画框

(x,y) 为左上角坐标  
(x+h,y+w) 为右下角坐标  

    
    
    1  
    

| 
    
    
    cv2.rectangle(img,(x,y), (x+h,y+w), (0,255,0), thickness)  
      
  
---|---  
  
## waitKey()
    
    
    1  
    2  
    3  
    4  
    5  
    

| 
    
    
    keypress = cv2.waitKey(200) # 200为当前图片的显示持续时间  
      
    if keypress == ord('c') # keypress为按键的整数形式, 所以需要用ord将字符类型转换  
      
    if cv2.waitKey(200) == 27: # Decimal 27 = Esc  
      
  
---|---  
  
## opencv与numpy

opencv的基础类型为numpy.ndarray, 因此可以直接使用 ndarray 的一些属性的方法  

    
    
    1  
    2  
    3  
    4  
    

| 
    
    
    import cv2  
    img = cv2.imread('./test.jpg')  
    print(type(img)) # <class 'numpy.ndarray'>  
    print(img.shape) # (500, 1069, 3)  (高, 宽, 通道)  
      
  
---|---  
  
利用 cv2.merge 方法将 numpy.ndarray 数据转换成opencv的图片数据:  

    
    
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
    

| 
    
    
    # 图片的分辨率为300*200(宽*高)，这里b, g, r设为随机值，注意dtype属性  
    b = np.random.randint(0, 255, (200, 300), dtype=np.uint8)  
    g = np.random.randint(0, 255, (200, 300), dtype=np.uint8)  
    r = np.random.randint(0, 255, (200, 300), dtype=np.uint8)  
    # 合并通道，形成图片  
    img = cv2.merge([b, g, r])  # opencv的通道是b在最前,r在最后  
    # 显示图片  
    cv2.imshow('test', img)  
    cv2.waitKey(0)  
    cv2.destroyWindow('test')  
      
  
---|---  
  
## 通道的拆分与合并

拆分: cv2.split  
合并: cv2.merge  

    
    
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
    12  
    

| 
    
    
    # 图片的分辨率为800*200(宽*高)，这里b, g, r设为随机值，注意dtype属性  
    b = np.random.randint(0, 255, (200, 800), dtype=np.uint8)  
    g = np.random.randint(0, 255, (200, 800), dtype=np.uint8)  
    r = np.random.randint(0, 255, (200, 800), dtype=np.uint8)  
    # 合并通道，形成图片  
    img = cv2.merge([b, g, r])  # opencv的通道是b在最前,r在最后  
    # 显示图片  
    cv2.imshow('test', img)  
    cv2.waitKey(0)  
    cv2.destroyWindow('test')  
    # 拆分通道, 每个通道都变成了单通道数组  
    [blue, green, red] = cv2.split(img)  
      
  
---|---  
  
## 将 BGR 转换成 RGB 通道顺序
    
    
    1  
    2  
    3  
    4  
    5  
    

| 
    
    
    # 方法一:  
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # 方法二:  
    rgb_img = img[:, :, [2, 1, 0]]   # img[h,w,v]  
    rgb_img = img[:, :, ::-1]  
      
  
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
  
## 用matplotlib显示图像
    
    
    1  
    2  
    3  
    4  
    

| 
    
    
    b,g,r=cv2.split(img)  
    img2=cv2.merge([r,g,b])  
    plt.imshow(img2)  
    plt.show()  
      
  
---|---  
  
## 截取子图
    
    
    1  
    2  
    

| 
    
    
    # 已知子图左上角坐标 (x1, y1), 右下角坐标(x2, y2)  
    crop_img = img[y1:y2, x1:x2, :]  
      
  
---|---  
  
## opencv 核心算法

## cv2
    
    
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
    12  
    13  
    14  
    15  
    16  
    

| 
    
    
    import cv2  
    image_path = './test.jpg'  
    src_image = cv2.imread(image_path) # 读取图片  
      
    size = src_image.shape  # 获取图片的尺寸, 返回一个元组: (height, width, depth)  
      
    copy_image = src_image.copy() # 复制图片  
      
    cv2.imwrite('./dst_test.jpg', copy_image) # 保存图片  
      
    cv2.imshow('image', src_image) # 显示图片  
      
    # 利用下标访问指定像素  
    for x in range(src_image.shape[0]): # 以行为主, 行数=图片height  
      for y in range(src_image.shape[1]):  # 列数 = 图片width  
        src_image[x,y] = (255,0,255)   # (blue, green, red)  值越高表示对应颜色越显著, 全0为黑, 全255为白  
      
  
---|---
