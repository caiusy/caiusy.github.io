---
title: c++中级教程  STL stack
date: 2019-07-25 00:00:00
categories:
  - C++
tags:
  - C++
---

## STL stack

  * (堆) 栈： LIFO 后进先出
  * 自适应容器（容器适配器）
  * 栈适配器 STL stack
        
        1  
        2  
        3  
        4  
        5  
        6  
        7  
        

| 
        
        stack<int, deque<int>>   s;  
        stack<int, vector<int>>   s;  
        stack<int, list<int>>   s;  
        s.empty()  
        s.size()  
        s.pop()  
        s.push(item)  
          
  
---|---  

    
    
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
    

| 
    
    
    // stack1.cpp : 定义控制台应用程序的入口点。  
    //  
      
    #include "stdafx.h"  
    #include <iostream>  
    #include <vector>  
    #include <list>  
    #include <stack>  
    using namespace std;  
    int main()  
    {  
    	stack<int, deque<int>>   a;  
    	stack<int, vector<int>>  b;  
    	stack<int, list<int>>    c;  
    	stack<int>               d; //默认用deque  
    	//什么是堆栈？ 先进后出，后进先出  
      
    	d.push(25);  
    	d.push(10);  
    	d.push(1);  
    	d.push(5);  
    	int x = 0;  
    	cout << "现在栈里一共有：" << d.size() << "个数据。" << endl;  
    	while (d.empty() == false)  
    	{  
    		x = d.top(); //查看数据并且返回  
    		d.pop();//删除，不返回   
    		cout << x << endl;  
    	}  
    	  
      
    	//x = d.top(); //查看数据并且返回  
    	//d.pop();//删除，不返回   
    	//cout << x << endl;  
    	cout << "现在栈里一共有：" << d.size() << "个数据。" << endl;  
    	system("pause");  
        return 0;  
    }  
    现在栈里一共有：4个数据。  
    5  
    1  
    10  
    25  
    现在栈里一共有：0个数据。  
    请按任意键继续. . .  
      
  
---|---
