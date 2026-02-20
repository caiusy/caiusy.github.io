---
title: numpy实现神经网络
date: 2019-08-28 00:00:00
categories:
  - Python
tags:
  - Python
---
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
    17  
    18  
    19  
    20  
    21  
    22  
    23  
    24  
    25  
    26  
    27  
    28  
    29  
    30  
    31  
    32  
    33  
    34  
    35  
    36  
    37  
    

| 
    
    
    import numpy as np  
      
    # N 为batch size, D_in 为输入维度  
    # H 为隐藏层的维度, D_out 为输出的维度  
    N, D_in, H, D_out = 64, 1000, 100, 10  
      
    # 创建随机的输入和输出数据  
    x = np.random.randn(N, D_in) # N × D_in 的矩阵  
    y = np.random.randn(N, D_out) # N × D_out 的矩阵  
      
    # 对两个隐藏层w1,w2进行初始化  
    w1 = np.random.randn(D_in, H)  
    w2 = np.random.randn(H, D_out)  
      
    # 设置学习率  
    learning_rate = 1e-6  
    for t in range(500):  
        # 前向传播: 计算预测结果 y_pred  
        h = x.dot(w1) # x维度为64 × 1000, w1维度为 1000 × 100, 计算完以后, h维度为 64 × 100  
        h_relu = np.maximum(h,0)  
        y_pred = h_relu.dot(w2) # h_relu维度为 64×100, w2维度为100×10, y的维度为64×10  
      
        # 计算损失  
        loss = np.square(y_pred - y).sum()  
        print(t, loss)  
      
        # 反向传播根据loss更新w1和w2的值  
        grad_y_pred = 2.0*(y_pred - y) # 对y_pred求导  
        grad_w2 = h_relu.T.dot(grad_y_pred) # 对w2求导, 微分矩阵应该与w2的size相同  
        grad_h_relu = grad_y_pred.dot(w2.T) # 对h_relu求导  
        grad_h = grad_h_relu.copy()  
        grad_h[h < 0] = 0 # 经过relu, 将小于0的梯度归0  
        grad_w1 = x.T.dot(grad_h)  
      
        # Update weights  
        w1 = w1 - learning_rate * grad_w1  
        w2 = w2 - learning_rate * grad_w2  
      
  
---|---
