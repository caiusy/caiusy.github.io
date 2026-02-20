---
title: c++中级教程(一)
date: 2019-07-25 00:00:00
categories:
  - 算法
tags:
  - C++
---

## 顺序容器 STL deque 类

  * deque是一个动态数组

  * deque与vector 非常类似

  * deque可以在数组开头和末尾插入和删除数据
        
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
        

| 
        
        // demo3.cpp : 定义控制台应用程序的入口点。  
        //  
          
        #include "stdafx.h"  
        #include <iostream>  
        #include <deque>  
        #include <algorithm>  
        using namespace std;  
          
        int main()  
        {  
        	deque<int> a;  
        	a.push_back(3);  
        	a.push_back(4);  
        	a.push_back(5);  
        	a.push_back(6);  
        	//vector只能push_back  
        	a.push_front(2);  
        	a.push_front(1);  
        	a.push_front(9);  
        	a.push_front(8);  
          
        	for (size_t nCount = 0; nCount < a.size(); ++nCount)  
        	{  
        		cout <<"a["<<nCount<<"]" << "= " << a[nCount] << endl;  
        	}  
        	cout << endl<<endl;  
          
        	a.pop_front();// 前面删除  
        	a.pop_back();// 后面删除  
        	cout << "删除之后：" << endl;  
        	/*for (size_t nCount = 0; nCount < a.size(); ++nCount)  
        	{  
        		cout << "a[" << nCount << "]" << a[nCount] << endl;  
        	}*/  
        	deque<int>::iterator iElementLocater; //这边使用了迭代器  distence 可以计算当前  
        	for (iElementLocater = a.begin();  
        		iElementLocater != a.end();  
        		++iElementLocater)  
        	{  
        		size_t nOffset = distance(a.begin(), iElementLocater);//distence 可以计算当前下标与begin开始的，距离正好是下标  
        		cout << "a[" << nOffset << "]" << "= "<<*iElementLocater << endl;  
        	}  
        	system("pause");  
            return 0;  
        }  
        a[0]= 8  
        a[1]= 9  
        a[2]= 1  
        a[3]= 2  
        a[4]= 3  
        a[5]= 4  
        a[6]= 5  
        a[7]= 6  
          
        删除之后：  
        a[0]= 9  
        a[1]= 1  
        a[2]= 2  
        a[3]= 3  
        a[4]= 4  
        a[5]= 5  
        请按任意键继续. . .  
          
  
---|---  

* * *

順序容器 STL list 類

  * 实例化std::list对象
  * 在list开头插入元素
  * 在list末尾插入元素
  * 在list中间插入元素
  * 删除list中的元素
  * 对list中的元素进行反转和排序
        
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
        

| 
        
        // list.cpp : 定义控制台应用程序的入口点。  
        //  
          
        #include "stdafx.h"  
        #include<iostream>  
        #include<list>  
        using namespace std;  
          
        void PrintListContent(const list<int>& listInput);  
        int main()  
        {  
        	list<int> a;  
        	list<int> b;  
          
        	b.push_back(100);  
        	b.push_back(200);  
        	b.push_back(300);  
        	b.push_back(400);  
        	b.push_back(500);  
        	PrintListContent(b);  
          
        	cout << endl;  
          
          
        	a.push_front(4);  
        	a.push_front(3);  
        	a.push_front(2);  
          
        	a.push_front(1);  
        	a.push_back(5);  
          
        	//使用链表数据，不能使用下标，只能使用迭代器  
        	list<int>::iterator iter;  
        	iter = a.begin();  
        	a.insert(iter, 10);// 在begin前面插入10，第一个参数迭代器，指定插入的位置  
        	a.insert(a.end(),10);  
          
        	PrintListContent(a);  
        	//将b插入到a之中  
        	a.insert(a.begin(), b.begin(), b.end());  
        	a.insert(iter,++b.begin(),--b.end())  
        	  
        	PrintListContent(a);  
        	  
        	system("pause");  
            return 0;  
        }  
          
          
        void PrintListContent(const list<int>& listInput)  
        {  
        	//会是一个底层 const，即其所指对象可以改变，但不能改变其所指对象的值。  
        	list<int>::const_iterator iter;  
        	for (iter = listInput.begin(); iter != listInput.end(); ++iter)  
        	{  
        		cout << *iter << endl;  
        	}  
        }  
          
  
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
    42  
    43  
    44  
    45  
    46  
    47  
    48  
    49  
    

| 
    
    
    // listdelet.cpp : 定义控制台应用程序的入口点。  
    //  
      
    #include "stdafx.h"  
    #include<iostream>  
    #include<list>  
    using namespace std;  
    void PrintListContent(const list<int>& listInput);  
      
      
    int main()  
    {  
    	list<int> a;  
    	a.push_front(4);  
    	a.push_front(3);  
      
    	list<int>::iterator iElementValueTwo;  
    	iElementValueTwo = a.insert(a.begin(),2); // inset 才返回迭代器迭代器指向这个位置  
    	a.push_front(1);  
    	a.push_back(0);  
    	cout << "删除之前" << endl;  
    	PrintListContent(a);  
    	// 删除2  
    	cout << "删除之后" << endl;  
      
    	a.erase(iElementValueTwo);  
    	//a.erase(a.beigin(),iElementValueTwo); 删除从第一个迭代器到第二个迭代器所有的数据  
    	PrintListContent(a);  
    	system("pause");  
        return 0;  
    }  
      
      
    void PrintListContent(const list<int>& listInput)  
    {  
    	//会是一个底层 const，即其所指对象可以改变，但不能改变其所指对象的值。  
    	cout << "{";  
    	list<int>::const_iterator iter;  
    	for (iter = listInput.begin(); iter != listInput.end(); ++iter)  
    	{  
    		cout << *iter << " ";  
    	}  
    	cout << "}" << endl;  
    }  
    删除之前  
    {1 2 3 4 0 }  
    删除之后  
    {1 3 4 0 }  
    请按任意键继续. . .  
      
  
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
    

| 
    
    
    // list3.cpp : 定义控制台应用程序的入口点。  
    //  
      
    #include "stdafx.h"  
    #include<iostream>  
    #include<list>  
    using namespace std;  
    void PrintListContent(const list<int>& listInput);  
      
    int main()  
    {  
    	list<int> a;  
    	a.push_front(4);  
    	a.push_front(3);  
    	a.push_front(2);  
      
    	a.push_front(1);  
      
    	PrintListContent(a);  
    	cout << "反转之后的数据：" << endl;  
    	a.reverse();  
    	PrintListContent(a);  
    	list<int> b;  
    	b.push_front(4);  
    	b.push_front(53);  
    	b.push_front(24);  
      
    	b.push_front(132);  
      
    	PrintListContent(b);  
    	cout << "排序之后的数据：" << endl;  
    	b.sort();  
    	PrintListContent(b);  
    	system("pause");  
      
        return 0;  
    }  
      
    void PrintListContent(const list<int>& listInput)  
    {  
    	//会是一个底层 const，即其所指对象可以改变，但不能改变其所指对象的值。  
    	cout << "{";  
    	list<int>::const_iterator iter;  
    	for (iter = listInput.begin(); iter != listInput.end(); ++iter)  
    	{  
    		cout << *iter << " ";  
    	}  
    	cout << "}" << endl;  
    }  
    {1 2 3 4 }  
    反转之后的数据：  
    {4 3 2 1 }  
    {132 24 53 4 }  
    排序之后的数据：  
    {4 24 53 132 }  
    请按任意键继续. . .  
      
  
---|---  
  
* * *
