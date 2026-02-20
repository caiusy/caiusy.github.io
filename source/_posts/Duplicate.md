---
title: Duplicate
date: 2019-11-17 00:00:00
categories:
  - 算法
tags:
  - Python
---
是否存在相同元素，python3用字典的方式解决  
  
      
    
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
    

| 
    
    
    class Solution(object):  
        def containsDuplicate(self, nums):  
            """  
            :type nums: List[int]  
            :rtype: bool  
            """  
            # 方法3：数字存字典  
            dic = {}  
            for i in nums:  
                dic[i] = dic.get(i, 0) + 1  
                if dic[i] > 1:  
                    return True  
            return False  
      
  
---|---  
  
最长回文子串  

    
    
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
    

| 
    
    
    class Solution:  
        def longestPalindrome(self, s: str) -> str:  
            # 两种判断条件  
            # DP 动态规划  
            # CABAC  
            # B  
            # ABA  
            # CABAC  
            palindrome = ''  
            for i in range(len(s)):  
                aa1 = self.getlongestpalindrome(s,i,i)  
                len1 = len(aa1)  
                if len1>len(palindrome):  
                    palindrome = aa1  
                aa2 = self.getlongestpalindrome(s,i,i+1)  
                len2 = len(aa2)  
                if len2>len(palindrome):  
                    palindrome = aa2  
              
            return palindrome  
      
          
        def getlongestpalindrome(self, s, l, r):  
            while l >= 0 and r<len(s) and s[l]==s[r]:  
                l -= 1  
                r += 1  
            return s[l+1:r]  
      
    ~  
      
  
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
    

| 
    
    
    class Solution:  
        def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:  
            # 用dictionary来做比较容易一点  
            lookup = {}  
            for i , num in enumerate(nums):  
                if num not in lookup:  
                    lookup[num] = i # 将num存到dic里面  
                else:  
                    if i-lookup[num]<=k:  
                        return True  
                    lookup[num] = i  
            return False  
              
      
    ~  
      
  
---|---
