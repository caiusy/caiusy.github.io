---
title: DFS_BFS
date: 2019-11-17 00:00:00
categories:
  - 算法
tags:
  - Python
  - BFS
  - DFS
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
    38  
    39  
    40  
    41  
    42  
    43  
    44  
    45  
    46  
    47  
    48  
    49  
    50  
    51  
    52  
    53  
    54  
    55  
    56  
    57  
    58  
    59  
    60  
    61  
    62  
    63  
    64  
    65  
    66  
    67  
    68  
      
  
| 
    
    
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 10:28  
    # @Author : caius  
    # @Site :   
    # @File : BFS.py  
    # @Software: PyCharm  
      
    graph = {  
        "A":{"B", "C"},  
        "B":{"A","C","D"},  
        "C":{"A","B","D","E"},  
        "D":{"B","C","E","F"},  
        "E":{"C","D"},  
        "F":{"D"}  
    }  
    # 字典的基本用法  
    # keys: A B C D E F  
    # graph["E"} "c". "D"  
      
      
    def BFS(graph, s):  
        # 队列先进先出  
        queue=[]  
        queue.append(s)  
        seen = set()# 代表这个东西是个set  
        seen.add(s)  
        parrent ={}  
        parrent={s:None}  
        while(len(queue)>0):  
            vertex = queue.pop(0)  
            nodes = graph[vertex]  
            for w in nodes:  
                if w not in seen:  
                    queue.append(w)  
                    seen.add(w)  
                    parrent[w] = vertex  
            print(vertex)  
        return parrent  
      
    def DFS(graph, s):  
        # 队列先进先出  
        stack=[]  
        stack.append(s)  
        seen = set()# 代表这个东西是个set  
        seen.add(s)  
        while(len(stack)>0):  
            vertex = stack.pop()  
            nodes = graph[vertex]  
            for w in nodes:  
                if w not in seen:  
                    stack.append(w)  
                    seen.add(w)  
            print(vertex)  
      
      
    DFS(graph,"E")  
    parrent = BFS(graph,'E')  
    for key in parrent:  
        print(key,parrent[key])  
    v = "B"  
    count=-1  
    while v!= None:  
        print(v)  
        v = parrent[v]  
        count+=1  
      
    print("count: {} 次".format(count))  
      
  
---|---
