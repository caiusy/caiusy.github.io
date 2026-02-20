---
title: 算法竞赛入门经典第二章
date: 2019-08-15 00:00:00
categories:
  - 其他
tags:
  - C++
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
    69  
    70  
    71  
    72  
    73  
    74  
    75  
    76  
    77  
    78  
    79  
    80  
    81  
    82  
    83  
    84  
    85  
    86  
    87  
    88  
    89  
    90  
    91  
    92  
    93  
    94  
    95  
    96  
    97  
    98  
    99  
    100  
    101  
    102  
    103  
    104  
    105  
    106  
    107  
    108  
    109  
    110  
    111  
    112  
    113  
    114  
    115  
    116  
    117  
    118  
    119  
    120  
    121  
    122  
    123  
    124  
    125  
    126  
    127  
    128  
    129  
    130  
    131  
    132  
    133  
    134  
    135  
    136  
    137  
    138  
    139  
    140  
    141  
    142  
    143  
    144  
    145  
    146  
    147  
    148  
    149  
    150  
    151  
    152  
    153  
    154  
    155  
    156  
    157  
    158  
    159  
    160  
    161  
    162  
    163  
    164  
    165  
    166  
    167  
    168  
    169  
    170  
    171  
    172  
    173  
    174  
    175  
    176  
    177  
    178  
    179  
    180  
    181  
    182  
    183  
    184  
    185  
    186  
    187  
    

| 
    
    
    // t2.cpp : 定义控制台应用程序的入口点。  
    //  
    #include "stdafx.h"  
    #include "stdio.h"  
    #include <stdlib.h>  
    #include<math.h>  
    #include<time.h>  
      
    int main()  
    {  
    	////3n+1问题、  
    	//int n, count = 0; //当n过大的时候 2*n溢出  
    	//  
    	//scanf_s("%d", &n);  
    	//long long n2 = n;  
    	//while (n2 > 1)  
    	//{  
    	//	if (n2 % 2 == 1) n2 = n2 * 3 + 1;  
    	//	else n2 /= 2;  
    	//	count++;  
    	//}  
    	//printf_s("%d\n", count);  
      
      
    	//近似计算  
    	/*double sum = 0;  
    	for (int i = 0;; i++) {  
    		double term = 1.0 / (i * 2 + 1);  
    		if (i % 2 == 0) sum += term;  
    		else sum -= term;  
    		if (term < 1e-6) break;  
    	}  
    	printf_s("%.6f\n", sum);*/  
    	// 阶乘之和 只保存后六位  
    	//int n, S = 0;  
    	//scanf_s("%d", &n);  
    	//for (int i = 1; i <= n; i++)  
    	//{  
    	//	int factorial = 1;  
    	//	for (int j = 1; j <= i; j++)  
    	//		factorial *= j;  
    	//	S += factorial;  
      
      
    	//}  
    	//printf_s("%d\n", S % 1000000);  
    	// 阶乘之和2, 优化版本  
    	//int n, S = 0;  
    	//const int MOD = 1000000;  
    	//scanf_s("%d", &n);  
    	//for (int i = 1; i <= n; i++)  
    	//{  
    	//	int factorial = 1;  
    	//	for (int j = 1; j <= i; j++)  
    	//		factorial = (factorial *j)%MOD;  
    	//	S += factorial;  
      
      
    	//}  
    	//printf_s("%d\n", S %MOD);  
    	//printf_s("Time used = %.2f\n", (double)clock() / CLOCKS_PER_SEC); // 得到程序运行的时间 单位：秒  
    	// 数据统计  
        //习题2.1 水仙花数目  
    	/*int sum;  
    	for (int i = 1; i < 10; i++)  
    		for (int k = 0; k < 10; k++)  
    			for (int j = 0; j < 10; j++)  
    			{  
    				sum = i * 100 + 10 * k + j;  
    				if(sum == i*i*i+j*j*j + k*k*k)  
    					printf_s("%d ", sum);  
    			}*/  
    	//习题2.2 韩信点兵 相传韩信才智过人，从不直接清点自己军队的人数，只要让士兵先后以三人一排、五人一排、七人一排地变换队形，  
    	//而他每次只掠一眼队伍的排尾就知道总人数了。输入包含多组数据，每组数据包含3个非负整数a，b，c，表示每种队形排尾的人数（a＜3，b＜5，c＜7），  
    	//输出总人数的最小值（或报告无解）。已知总人数不小于10，不超过100。输入到文件结束为止。  
    	//int i, a, b, c;  
    	//scanf_s("%d%d%d", &a, &b, &c);  
    	//for (i = 0; i <= 100; i++) {  
    	//	if (i % 3 == a && i % 5 == b && i % 7 == c)  
    	//		printf_s("%d\n", i);  
    	//	  
    	//}  
    	//if (i % 3 != a && i % 5 != b && i % 7 != c && i>100)  
    	//		printf_s("No answer\n");  
        //习题2.3 倒三角形  
    	//int n, /* 输出n行; n<=20 */  
    	//	i, /* 打印第i行 */  
    	//	j;  
    	//scanf_s("%d", &n);  
    	//for (i = 1; i <= n; i = i + 1) {  
    	//	/* 在第i行，打印(i-1)个空格 */  
    	//	for (j = 1; j <= i - 1; j = j + 1)       printf_s(" ");  
    	//	/* 在第i行，打印(2*n-2*i+1)个# */  
    	//	for (j = 1; j <= (2 * n - 2 * i + 1); j = j + 1)  printf_s("#");  
    	//	printf_s("\n");  /* 输出结束后换行，否则所有的#号在同一行输出 */  
    	//}  
    	//习题2.4 子序列的和 输入两个正整数n＜m＜10 6 ，输出 ，保留5位小数。输入包含多组数据， 注：陷阱就是在n特别大时如果直接n*n就会溢出，所以只能连除两次  
    	//int count = 0;  
    	//while (1) {  
    	//	int n = 0;  
    	//	int m = 0;  
    	//	scanf_s("%d", &n);  
    	//	scanf_s("%d", &m);  
    	//	if (n == m&&n == 0) {  
    	//		break;  
    	//	}  
    	//	count++;  
    	//	double sum = 0;  
    	//	for (int i = n; i <= m; i++) {  
    	//		sum += 1.0 / i / i;  
    	//	}  
    	//	printf_s("Case %d:%.5f\n", count, sum);  
    	//}  
        //习题2.5 分数化小数（decimal）   
        //输入正整数a，b，c，输出a / b的小数形式，精确到小数点后c位。a，b≤10 ^ 6，c≤100。输入包含多组数据，结束标记为a＝b＝c＝0。  
    		  
     //   int count = 0;  
    	//while (1) {  
    	//	int a, b, c;  
    	//	int k, d, i;  
    	//	scanf_s("%d", &a);  
    	//	scanf_s("%d", &b);  
    	//	scanf_s("%d", &c);  
    	//	if (a == 0&&b == 0 &&c==0 ) {  
    	//		break;  
    	//	}  
    	//	count++;  
    	//	for (i = 0; i<c - 1; i++)  
    	//	{  
    	//		/*机智地把余数放大十倍，使之除以b并取模*/  
    	//		k = (k%b) * 10;  
    	//		printf_s("Case %d:%.5f\n", count, k / b);  
    	//	}  
    	//	k = (k%b) * 10;  
    	//	d = (k%b) * 10 / b;  
    	//	if (d >= 5)//判断第c+1位小数是否大于等于5，if yes,第c位小数要进1   
    	//	{  
    	//		printf_s("Case %d:%.5f\n", count, k / b + 1);  
    	//	}  
    	//	else  
    	//	{  
    	//		printf_s("Case %d:%.5f\n", count, k / b);  
    	//	}  
    	//	  
    	//}  
        int abc,def,ghi;  
        int a[10],count=0;  
      
        memset(a,0,sizeof(a)); // 将a数组中的值全部设置为0  
      
        for (abc = 123;abc < 333;abc ++) { // 基本可以确定abc的最小值和最大值  
            def = 2 * abc;  
            ghi = 3 * abc;  
      
            // 设置数组中所有对应的9位数字位置的值1  
            a[abc/100] = 1; // a  
            a[abc/10%10] = 1; // b  
            a[abc%10] = 1; // c  
      
            a[def/100] = 1; // d  
            a[def/10%10] = 1; // e  
            a[def%10] = 1; // f  
      
            a[ghi/100] = 1; // g  
            a[ghi/10%10] = 1; // h  
            a[ghi%10] = 1; // i  
      
            int i;  
            for (i=1;i<=9;i++) {  
                count += a[i];  
            }  
      
            if (count == 9) {  
                printf_s("%d %d %d\n",abc,def,ghi);  
            }  
      
            // 重置count 和a数组  
            count = 0;  
            memset(a,0,sizeof(a));  
        }  
      
      
    	system("pause");  
        return 0;  
    	  
      
    }  
      
  
---|---
