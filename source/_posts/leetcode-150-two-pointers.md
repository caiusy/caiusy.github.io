---
title: 👆 LeetCode 150 - 双指针专题
date: 2026-01-18
updated: 2026-01-18
categories:
  - 算法基础
  - 算法
  -
    - 学习笔记
    - 算法笔记
tags:
  - LeetCode
  - 双指针
description: LeetCode 面试 150 题之双指针专题，含图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 👆 双指针专题 (5题)

> 🎯 **核心技巧**：相向双指针、同向双指针、快慢指针

---

## 🗺️ 双指针三大模式

```
┌─────────────────────────────────────────────────────────────┐
│                      双指针模式                              │
├───────────────────┬───────────────────┬─────────────────────┤
│    相向双指针      │    同向双指针      │     背向双指针       │
│   (对撞指针)       │   (快慢指针)       │   (中心扩展)        │
├───────────────────┼───────────────────┼─────────────────────┤
│  L ──────▶ ◀── R  │  S ──▶ F ──▶     │     ◀── C ──▶      │
├───────────────────┼───────────────────┼─────────────────────┤
│ • 两数之和        │ • 移除元素         │ • 最长回文子串       │
│ • 盛水容器        │ • 删除重复         │ • 回文判断          │
│ • 三数之和        │ • 链表快慢         │                    │
└───────────────────┴───────────────────┴─────────────────────┘
```

---

## 1️⃣ LC 125. 验证回文串 🟢

### 题目描述
判断字符串是否为回文串（只考虑字母和数字，忽略大小写）。

### 🎨 图解思路

```
相向双指针，跳过非字母数字字符

s = "A man, a plan, a canal: Panama"
     ↑                           ↑
     L                           R

Step 1: 'A' vs 'a' → 相等，L++, R--
Step 2: ' ' 跳过，L++
Step 3: 'm' vs 'm' → 相等
...
最终 L >= R，是回文串 ✓
```

### 💻 代码实现

```python
def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    
    while left < right:
        # 跳过非字母数字
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # 比较（忽略大小写）
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

### 🧠 记忆口诀
> **"两头往中走，不同就说No"**

---

## 2️⃣ LC 392. 判断子序列 🟢

### 题目描述
判断 `s` 是否为 `t` 的子序列。

### 🎨 图解思路

```
同向双指针：i 遍历 s，j 遍历 t

s = "ace", t = "abcde"
     ↑          ↑
     i          j

j=0: t[0]='a' = s[0], i++, j++
j=1: t[1]='b' ≠ s[1]='c', j++
j=2: t[2]='c' = s[1], i++, j++
j=3: t[3]='d' ≠ s[2]='e', j++
j=4: t[4]='e' = s[2], i++, j++

i = 3 = len(s) ✓ 是子序列
```

### 💻 代码实现

```python
def isSubsequence(s: str, t: str) -> bool:
    i, j = 0, 0
    
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    
    return i == len(s)
```

### 🧠 记忆口诀
> **"s 指针只在匹配时前进"**

---

## 3️⃣ LC 167. 两数之和 II - 输入有序数组 🟡

### 题目描述
在有序数组中找两个数，使它们的和等于目标值。

### 🎨 图解思路

```
有序数组 + 两数之和 = 相向双指针！

numbers = [2, 7, 11, 15], target = 9
           ↑          ↑
           L          R

sum = 2 + 15 = 17 > 9 → R-- (和太大，减小右边)
sum = 2 + 11 = 13 > 9 → R--
sum = 2 + 7 = 9 = target ✓

返回 [1, 2] (1-indexed)
```

### 💻 代码实现

```python
def twoSum(numbers: list, target: int) -> list:
    left, right = 0, len(numbers) - 1
    
    while left < right:
        total = numbers[left] + numbers[right]
        
        if total == target:
            return [left + 1, right + 1]  # 1-indexed
        elif total < target:
            left += 1   # 和太小，增大左边
        else:
            right -= 1  # 和太大，减小右边
    
    return []
```

### 🧠 记忆口诀
> **"小了左移，大了右移"**

---

## 4️⃣ LC 11. 盛最多水的容器 🟡

### 题目描述
找两条线，使构成的容器盛水最多。

### 🎨 图解思路

```
面积 = min(左高, 右高) × 宽度

    │           │
    │     │     │
    │  │  │  │  │
    │  │  │  │  │
   [1, 8, 6, 2, 5, 4, 8, 3, 7]
    ↑                       ↑
    L                       R

关键洞察：移动较矮的那边！
为什么？因为宽度必然减小，只有高度增加才可能面积增大
而高度由矮的决定，所以移动矮的才有可能找到更高的

Step 1: height[L]=1 < height[R]=7
        area = 1 × 8 = 8
        移动 L (因为左边矮)

Step 2: height[L]=8 > height[R]=7  
        area = 7 × 7 = 49
        移动 R

... 最大面积 = 49
```

### 💻 代码实现

```python
def maxArea(height: list) -> int:
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        # 计算当前面积
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)
        
        # 移动较矮的一边
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
```

### 🧠 记忆口诀
> **"矮的先走，高的等着"**

---

## 5️⃣ LC 15. 三数之和 🟡

### 题目描述
找出所有和为 0 的三元组（不能重复）。

### 🎨 图解思路

```
排序 + 固定一个数 + 双指针找另外两个

nums = [-4, -1, -1, 0, 1, 2]
        ↑
        i (固定)
            ↑           ↑
            L           R

target = -nums[i] = 4

找 L + R = 4:
  -1 + 2 = 1 < 4 → L++
  -1 + 2 = 1 < 4 → L++
  0 + 2 = 2 < 4 → L++
  1 + 2 = 3 < 4 → L++
  L >= R，结束

i++ 继续...

去重技巧：
1. nums[i] == nums[i-1] 时跳过
2. 找到解后，L++/R-- 跳过重复值
```

### 💻 代码实现

```python
def threeSum(nums: list) -> list:
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # 去重：跳过重复的第一个数
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # 双指针找另外两个数
        left, right = i + 1, n - 1
        target = -nums[i]
        
        while left < right:
            total = nums[left] + nums[right]
            
            if total == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # 去重：跳过重复的第二、三个数
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif total < target:
                left += 1
            else:
                right -= 1
    
    return result
```

### 🧠 记忆口诀
> **"排序固定一，双指针找二三，遇重复就跳过"**

---

## 📊 本章总结

### 双指针模板速查

```python
# 模板1: 相向双指针
def two_pointer_opposite(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        if condition:
            left += 1
        else:
            right -= 1

# 模板2: 同向双指针
def two_pointer_same(arr):
    slow = 0
    for fast in range(len(arr)):
        if condition:
            arr[slow] = arr[fast]
            slow += 1
    return slow

# 模板3: 背向双指针（中心扩展）
def expand_from_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return s[left+1:right]
```

### 题目模式识别

| 特征 | 使用模式 | 典型题目 |
|------|----------|----------|
| 有序数组求和 | 相向双指针 | 167, 15 |
| 回文判断 | 相向双指针 | 125 |
| 最大/最小容器 | 相向双指针 | 11 |
| 子序列匹配 | 同向双指针 | 392 |
| 原地删除元素 | 同向双指针 | 26, 27 |
| 回文子串 | 背向双指针 | 5, 647 |

### 🧠 全章记忆口诀

```
验判两盛三，双指针五关

验 - 验证回文串 (125)
判 - 判断子序列 (392)
两 - 两数之和 II (167)
盛 - 盛最多水的容器 (11)
三 - 三数之和 (15)
```

---

> 📖 **下一篇**：[滑动窗口专题](/2026/01/18/leetcode-150-sliding-window/)

