---
title: 链表中倒数第k个结点
date: 2019-09-10 00:00:00
categories:
  - 算法
tags:
  - 编程
---

时间限制：1秒 空间限制：32768K 热度指数：829537  
本题知识点： 链表  
  
## 题目描述

输入一个链表，输出该链表中倒数第k个结点。
    
    
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
    

| 
    
    
    /*  
    struct ListNode {  
    	int val;  
    	struct ListNode *next;  
    	ListNode(int x) :  
    			val(x), next(NULL) {  
    	}  
    };*/  
    class Solution {  
    public:  
        ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {  
      
      
            if (k == 0)  
              return NULL;//如果K为0，返回NULL  
            queue<ListNode*> que;  
            ListNode *node = pListHead;  
            while (node != NULL)  
            {  
                if (que.size() == k)  
                {  
                    que.pop();  
                }  
                que.push(node);  
                node = node->next;  
            }  
            if (que.size() == k)  
                return que.front();  
            else  
                return NULL;//如果k大于链表的最大长度，返回NULL  
        }  
    };  
      
  
---|---  
  
运行时间：3ms  
占用内存：472K
