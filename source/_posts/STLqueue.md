---
title: c++中级教程  STL queue
date: 2019-07-26 00:00:00
categories:
  - C++
tags:
  - C++
---
## STL queue

  * 队列： FIFO 先进先出
  * 自适应容器（容器适配器）
  * 栈适配器 STL queue
        
        1  
        2  
        3  
        4  
        5  
        6  
        7  
        8  
        

| 
        
        queue<int, deque<int>>   q;  
        queue<int, list<int>>   q;  
        q.empty()  
        q.size()  
        q.front()  
        q.back()  
        q.pop()  
        q.push(item)  
          
  
---|---  

### 可以用list和deque做queue

先进先出，后进后出  

    
    
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
    

| 
    
    
    // queue.cpp : 定义控制台应用程序的入口点。  
    //  
      
    #include "stdafx.h"  
    #include<iostream>  
    #include<queue>  
    #include<list>  
    #include<deque>  
    using namespace std;  
      
      
    int main()  
    {  
    	queue<int, deque<int>> a;  
    	queue<int, list<int>> b;  
    	//queue<int, vector<int>> c;  不可以，因为vector不能进行两端操作  
    	//队列有什么用途？？？  
    	queue<int> q;  
      
    	q.push(10);  
    	q.push(5);  
    	q.push(-1);  
    	q.push(20);  
    	cout << "现在队列里有" << q.size() << "个数据 " << endl;  
    	cout << "队首的数据：" << q.front() << endl;  
    	cout << "队尾的数据：" << q.back() << endl;  
    	q.pop();  
    	cout << "新的队首的数据：" << q.front() << endl;  
    	while (q.size() != 0)  
    	{  
    		cout << " 删除" << a.front() << endl;  
    		q.pop();  
    	}  
    	if (q.empty())  
    	{  
    		cout << "队列为空！"<<endl;  
    	}  
    	system("pause");  
        return 0;  
    }  
      
  
---|---  
  
## 优先级队列 priority_queue

  * 自适应容器（容器适配器）：不能使用list
  * 最大值优先级队列、最小值优先级队列(值越大，优先级越高，值越小优先级越高)
  * 优先级队列适配器 STL priority_queue

* * *
    
    
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
    
    
    priority_queue<int, deque<int>> pg1;  
    //对队列里的数据进行随机操作，所以不能使用list  
    priority_queue<int, vector<int>> pg2; //vector是默认的  
    								//谓词  
    priority_queue<int, vector<int>, greater<int> pg2; //vector是默认的，最小优先队列  
    pg.empty()  
    pg.size()  
    pg.top()  
    pg.pop()  
    pg.push(item)  
    ~~~~~~~~~~  
      
  
---|---  
  
* * *
    
    
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
    

| 
    
    
    // priority_queue.cpp : 定义控制台应用程序的入口点。  
    //  
      
    #include "stdafx.h"  
    #include<iostream>  
    #include<deque>  
    #include<vector>  
    #include<queue>  
      
    using namespace std;  
    int main()  
    {  
    	priority_queue<int, deque<int>> pg1;  
        //对队列里的数据进行随机操作，所以不能使用list  
    	priority_queue<int, vector<int>> pg2; //vector是默认的  
    									//谓词  
    	priority_queue<int, vector<int>, greater<int> pg2; //vector是默认的，最小优先队列  
      
    	pg2.push(10);  
    	pg2.push(5);  
    	pg2.push(-1);  
    	pg2.push(20);  
    	cout << "优先级队列一共有： " << pg2.size() << "个数据" << endl;  
    	cout << pg2.top() << endl;  
    	while (!pg2.empty())  
    	{  
    		cout << "从优先级队列里删除： " << pg2.top() << endl;  
    		pg2.pop();  
    	}  
    	system("pause");  
        return 0;  
    }  
    优先级队列一共有： 4个数据  
    20  
    从优先级队列里删除： 20  
    从优先级队列里删除： 10  
    从优先级队列里删除： 5  
    从优先级队列里删除： -1  
    请按任意键继续. . .  
      
    ~~  
      
  
---|---  
  
* * *
