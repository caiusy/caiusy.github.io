---
title: python_learn
date: 2019-11-17 00:00:00
categories:
  - 深度学习
tags:
  - Python
---
## 查找图中两个节点的最小的距离  
  
这里面使用了python的优先队列，这里的队列按照后面的数值大小进行排序，而不是像普通的队列一样先进先出。后面的数值，是节点到出发节点的距离长度。  

    
    
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
    

| 
    
    
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 11:50  
    # @Author : caius  
    # @Site :   
    # @File : find_min_bfs.py  
    # @Software: PyCharm  
      
    import  heapq  
    import  math  
    # pqueue = []  
    # heapq.heappush(pqueue,(1,"A"))  
    # heapq.heappush(pqueue,(7,"B"))  
    # heapq.heappush(pqueue,(3,"C"))  
    # heapq.heappush(pqueue,(6,"D"))  
    # heapq.heappush(pqueue,(2,"E"))  
      
    graph = {  
        "A":{"B": 5, "C": 1},  
        "B":{"A": 5,"C": 2,"D": 1},  
        "C":{"A": 1,"B": 2,"D": 4,"E": 8},  
        "D":{"B": 1,"C": 4,"E": 3,"F": 6},  
        "E":{"C": 8,"D": 3},  
        "F":{"D": 6}  
    }  
      
    def init_distance(graph,s):  
        distance = {s:0}  
        for vertex in graph:  
            if vertex != s:  
                distance[vertex] = math.inf  
        return distance  
      
      
    def dijkstra(graph, s):  
        pqueue = []  
        heapq.heappush(pqueue,(0,s))  
        seen =set()  
        parent ={s:None}  
        distance = init_distance(graph,s)  
        while(len(pqueue)>0):  
            pair = heapq.heappop(pqueue) # 拿到一对点，pair  
            dist = pair[0]  
            vertex = pair[1]  
            seen.add(vertex)  
      
            nodes = graph[vertex].keys()  
            for w in nodes:  
                if w not in seen:  
                    if dist+graph[vertex][w] <distance[w]:  
                        heapq.heappush(pqueue,(dist+graph[vertex][w],w))  
                        parent[w] = vertex  
                        distance[w] = dist+graph[vertex][w]  
      
        return parent,distance  
      
    parent, distance  = dijkstra(graph,"A")  
    print(parent)  
    print(distance)  
      
  
---|---  
  
### python 装饰器

装饰器(Decorators)是 Python 的一个重要部分。简单地说：他们是修改其他函数的功能的函数。他们有助于让我们的代码更简短，也更Pythonic（Python范儿）。装饰器可以让你的代码更简洁。  

    
    
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
    

| 
    
    
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 9:59  
    # @Author : caius  
    # @Site :   
    # @File : deco.py.py  
    # @Software: PyCharm  
    import  time  
      
    # 装饰器  
    def display_time(func):  
        def wrapper(*args):  
            t1 = time.time()  
            result = func(*args)  
            t2 = time.time()  
            print("Total time: {:.4} s".format(t2-t1))  
            return result  
        return wrapper  
      
    # 输出质数  
    def is_prime(num):  
        if num<2:  
            return False  
        elif num==2:  
            return True  
        else:  
            for i in range(2, num):  
                if num%i ==0:  
                    return False  
            return True  
      
    @display_time  
    def prime_nums():  
      
        for i in  range(2,10000):  
            if is_prime(i):  
                print(i)  
      
      
    @display_time  
    def count_prime_nums(maxnum):  
        count = 0  
        for i in range(2, maxnum):  
            if is_prime(i):  
                count += 1  
        return count  
      
    count = count_prime_nums(5000)  
    print(count)  
    ~  
      
  
---|---  
  
### 用turtle 画图
    
    
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
    69  
    70  
    71  
    72  
    73  
    74  
    75  
    76  
    

| 
    
    
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 13:27  
    # @Author : caius  
    # @Site :   
    # @File : draw.py  
    # @Software: PyCharm  
    from turtle import  *  
    import math  
    # turtle 画图  
    # forward(100)  
    # left(90)  
    # forward(100)  
    # left(90)  
    # forward(100)  
    # left(90)  
    # forward(100)  
    # left(90)  
    # exitonclick()# 不点击窗口的话就不会退出  
      
    # # 画等边三角形  
    # forward(100)  
    # left(120)  
    # forward(100)  
    # left(120)  
    # forward(100)  
    # left(120)  
    # exitonclick()# 不点击窗口的话就不会退出  
      
    # 画五角星  
    # forward(100)  
    # right(180-36)  
    # forward(100)  
    # right(180-36)  
    # forward(100)  
    # right(180-36)  
    # forward(100)  
    # right(180-36)  
    # forward(100)  
    # right(180-36)  
    # for i in range(5):  
    #     forward(100)  
    #     right(180-36)  
    #  
    angle = 360/8  
    length = 100  
    speed(0)  
    for i in range(8):  
        if i %2==0:  
            color('yellow')  
        else:  
            color('red')  
        begin_fill()  
        forward(100)  
        left(angle)  
        forward(length)  
        left(180-angle)  
        forward(length)  
        left(angle)  
        forward(length)  
        left(180-angle)  
        end_fill()  
        left(angle)  
    forward(length)  
    left(180-(180-angle)/2)  
      
    alpha = angle*3.1415926536 /180  
    step = 2*length*math.sin(alpha/2)  
    color('blue')  
    begin_fill()  
    for i in range(8):  
        forward(step)  
        left(angle)  
    end_fill()  
      
    exitonclick()# 不点击窗口的话就不会退出  
      
  
---|---  
  
![h1](//caius-lu.github.io/2019/11/17/python-learn/images/20191117_python-learn_h1.png)  

    
    
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
    

| 
    
    
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 14:20  
    # @Author : caius  
    # @Site :   
    # @File : Lsystem2.py  
    # @Software: PyCharm  
    from turtle import  *  
    length = 7  
    angle = 60  
      
      
    def split_path(path):  
        i = 0  
        list = []  
        while i <len(path):  
            if path[i] == "F":  
                list.append(path[i:i+2])  
                i = i+2  
            else:  
                list.append(path[i])  
                i = i+1  
        return list  
      
    def apply_rule(path, rules):  
        lst = split_path(path)  
        for i in range(len(lst)):  
            symbol = lst[i]  
            if symbol in rules:  
                lst[i] = rules[symbol]  
        path ="".join(symbol for symbol in lst)  
        return path  
    rules={  
        "Fl": "Fr+Fl+Fr",  
        "Fr":"Fl-Fr-Fl"  
    }  
    def draw_patj(path):  
        lst = split_path(path)  
        for symbol in lst:  
            if symbol =="Fl"  or symbol=='Fr':  
                forward(length)  
            elif symbol=="-":  
                left(angle)  
            elif symbol=='+':  
                right(angle)  
      
    speed(0)  
    path = 'Fr'  
    # speed(0)  
    #  
    #lst = split_path(path)  
    for i in range(6):  
        path = apply_rule(path,rules)  
    print(path)  
    draw_patj(path)  
    exitonclick()  
    ~  
      
  
---|---  
  
![h2](//caius-lu.github.io/2019/11/17/python-learn/images/20191117_python-learn_h2.png)

### python 类的构造
    
    
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
    

| 
    
    
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 14:52  
    # @Author : caius  
    # @Site :   
    # @File : Bank.py  
    # @Software: PyCharm  
    class BankAccount:  
        # Constructor 构造器  
        def __init__(self,accountNumber, accountName, balance):  
            self.accountNumber = accountNumber  
            self.accountName = accountName  
            self.balance = balance  
      
        def __str__(self):  
            return "(name: {},  balance: {})".format(self.accountName,self.balance)  
      
    #!/usr/bin/env python  
    # -*- coding: utf-8 -*-  
    # @Time : 2019/11/17 14:56  
    # @Author : caius  
    # @Site :   
    # @File : main.py  
    # @Software: PyCharm  
    from Bank import BankAccount  
      
    b1 = BankAccount("56789","Tony", 100.0)  
    print((b1))  
      
  
---|---
