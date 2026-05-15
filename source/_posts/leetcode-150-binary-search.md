---
title: 🔢 LeetCode 150 - 二分查找专题
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
description: LeetCode 面试 150 题之二分查找专题，含红蓝染色法图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 🔢 二分查找专题 (7题)

> 🎯 **核心思想**：每次排除一半的搜索空间，时间复杂度 O(log n)

---

## 🗺️ 二分查找的本质

```
┌─────────────────────────────────────────────────────────────┐
│                   二分查找的本质                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  二分查找的本质是：在有序（或部分有序）的空间中               │
│  找到满足某个条件的边界点                                    │
│                                                             │
│  🔵🔵🔵🔵🔵🔴🔴🔴🔴🔴🔴                                    │
│  ↑         ↑                                                │
│  蓝色区域   红色区域                                         │
│  (不满足)   (满足)                                          │
│                                                             │
│  目标：找到第一个红色（或最后一个蓝色）                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 二分查找模板

### 模板1：找第一个满足条件的位置

```python
def binary_search_first(nums, target):
    left, right = 0, len(nums)  # 左闭右开
    
    while left < right:
        mid = left + (right - left) // 2
        
        if condition(mid):    # 满足条件
            right = mid       # 答案在 [left, mid]
        else:
            left = mid + 1    # 答案在 [mid+1, right)
    
    return left  # 第一个满足条件的位置
```

### 模板2：找最后一个满足条件的位置

```python
def binary_search_last(nums, target):
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left + 1) // 2  # 向上取整
        
        if condition(mid):    # 满足条件
            left = mid        # 答案在 [mid, right)
        else:
            right = mid - 1   # 答案在 [left, mid-1]
    
    return left  # 最后一个满足条件的位置
```

---

## 1️⃣ LC 35. 搜索插入位置 🟢

### 题目描述
在排序数组中找到目标值的位置，如果不存在则返回应该插入的位置。

### 🎨 图解思路

```
nums = [1, 3, 5, 6], target = 5

找第一个 >= target 的位置

  1   3   5   6
  ↑       ↑
 <5      >=5

二分查找：
初始: left=0, right=4
mid=2, nums[2]=5 >= 5, right=2
mid=1, nums[1]=3 < 5, left=2
left == right, 返回 2
```

### 💻 代码实现

```python
def searchInsert(nums: list, target: int) -> int:
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] >= target:
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 🧠 记忆口诀
> **"找第一个不小于目标的位置"**

---

## 2️⃣ LC 74. 搜索二维矩阵 🟡

### 题目描述
在行列都有序的二维矩阵中搜索目标值。

### 🎨 图解思路

```
matrix:
[1,  3,  5,  7]
[10, 11, 16, 20]
[23, 30, 34, 60]

将 2D 矩阵看作 1D 数组:
[1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]

坐标转换:
index → (index // n, index % n)
```

### 💻 代码实现

```python
def searchMatrix(matrix: list, target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n
    
    while left < right:
        mid = left + (right - left) // 2
        row, col = mid // n, mid % n
        
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid
    
    return False
```

### 🧠 记忆口诀
> **"2D变1D，除法取行，余数取列"**

---

## 3️⃣ LC 162. 寻找峰值 🟡

### 题目描述
找到数组中任意一个峰值元素的索引（比左右邻居都大）。

### 🎨 图解思路

```
nums = [1, 2, 1, 3, 5, 6, 4]

峰值: 索引 1 (值为2) 或 索引 5 (值为6)

二分思路:
- 如果 mid 在上坡，峰值在右边
- 如果 mid 在下坡，峰值在左边


```python
   5 \4
  /
 / \
1   1
mid 处于上坡 → 往右找
mid 处于下坡 → 往左找（包含mid）
```
### 💻 代码实现
```python
def findPeakElement(nums: list) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < nums[mid + 1]:
            # 上坡，峰值在右边
            left = mid + 1
        else:
            # 下坡或峰值，峰值在左边（包含mid）
            right = mid
    return left
```
### 🧠 记忆口诀
> **"上坡往右，下坡往左"**
---
## 4️⃣ LC 33. 搜索旋转排序数组 🟡
### 题目描述
在旋转后的有序数组中搜索目标值。
### 🎨 图解思路
```
nums = [4, 5, 6, 7, 0, 1, 2], target = 0
旋转后的数组特点:
   6  \
  5    0
 4      \
二分策略:
1. 判断 mid 在左半段还是右半段
2. 判断 target 在 mid 的哪边
```
### 💻 代码实现
```python
def search(nums: list, target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        # 判断 mid 在左半段还是右半段
        if nums[left] <= nums[mid]:
            # mid 在左半段（有序）
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # mid 在右半段（有序）
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```
### 🧠 记忆口诀
> **"先判断哪边有序，再判断目标在哪边"**
---
## 5️⃣ LC 34. 查找元素的第一个和最后一个位置 🟡
### 题目描述
在排序数组中找到目标值的起始和结束位置。
### 🎨 图解思路
```
nums = [5, 7, 7, 8, 8, 10], target = 8
找第一个 8: 索引 3
找最后一个 8: 索引 4
  5   7   7   8   8   10
              ↑   ↑
            first last
```
### 💻 代码实现
```python
def searchRange(nums: list, target: int) -> list:
    def find_first():
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        return left
    def find_last():
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        return left - 1
    first = find_first()
    if first == len(nums) or nums[first] != target:
        return [-1, -1]
    last = find_last()
    return [first, last]
```
### 🧠 记忆口诀
> **"找第一个>=，找第一个>再减1"**
---
## 6️⃣ LC 153. 寻找旋转排序数组中的最小值 🟡
### 题目描述
在旋转后的有序数组中找到最小值。
### 🎨 图解思路
```
nums = [3, 4, 5, 1, 2]
   4  \
  3    1
        \
最小值是旋转点
比较 nums[mid] 和 nums[right]:
- nums[mid] > nums[right]: 最小值在右边
- nums[mid] <= nums[right]: 最小值在左边（包含mid）
```
### 💻 代码实现
```python
def findMin(nums: list) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            # 最小值在右边
            left = mid + 1
        else:
            # 最小值在左边（包含mid）
            right = mid
    return nums[left]
```
### 🧠 记忆口诀
> **"比右边大就往右，否则往左"**
---
## 7️⃣ LC 4. 寻找两个正序数组的中位数 🔴
### 题目描述
找到两个正序数组的中位数，要求时间复杂度 O(log(m+n))。
### 🎨 图解思路
```
nums1 = [1, 3], nums2 = [2]
合并后: [1, 2, 3]
中位数: 2
二分思路:
在较短的数组上二分，找到一个划分点 i
使得 nums1[0:i] 和 nums2[0:j] 的总数 = (m+n+1)//2
       nums1:  1 | 3
       nums2:  2 |
               ↑
              划分点
左半边最大值 <= 右半边最小值
```
### 💻 代码实现
```python
def findMedianSortedArrays(nums1: list, nums2: list) -> float:
    # 确保 nums1 是较短的数组
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i
        # 边界处理
        nums1_left = float('-inf') if i == 0 else nums1[i - 1]
        nums1_right = float('inf') if i == m else nums1[i]
        nums2_left = float('-inf') if j == 0 else nums2[j - 1]
        nums2_right = float('inf') if j == n else nums2[j]
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # 找到正确的划分
            if (m + n) % 2 == 1:
                return max(nums1_left, nums2_left)
            else:
                return (max(nums1_left, nums2_left) +
                        min(nums1_right, nums2_right)) / 2
        elif nums1_left > nums2_right:
            # nums1 划分点太靠右
            right = i - 1
        else:
            # nums1 划分点太靠左
            left = i + 1
    return 0.0
```
### 🧠 记忆口诀
> **"短数组二分，找正确划分"**
---
## 📊 本章总结
### 二分查找场景
| 场景 | 关键点 | 典型题目 |
```

| 有序数组查找 | 直接二分 | 35, 74 |
| 旋转数组 | 判断有序段 | 33, 153 |
| 峰值问题 | 比较相邻元素 | 162 |
| 边界问题 | >=和>的区别 | 34 |
| 双数组 | 在短数组二分 | 4 |

### 🧠 全章记忆口诀

```
插矩峰旋范围最中
二分七题要记清

插 - 搜索插入位置 (35)
矩 - 搜索二维矩阵 (74)
峰 - 寻找峰值 (162)
旋 - 搜索旋转排序数组 (33)
范围 - 查找元素的第一个和最后一个位置 (34)
最 - 寻找旋转排序数组中的最小值 (153)
中 - 寻找两个正序数组的中位数 (4)
```

---

> 📖 **下一篇**：[位运算专题](/2026/01/18/leetcode-150-bit/)

