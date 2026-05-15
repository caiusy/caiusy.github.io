---
title: 📚 LeetCode 150 - 数组与字符串专题
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
description: LeetCode 面试 150 题之数组与字符串专题，含图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 📚 数组与字符串专题 (15题)

> 🎯 **核心技巧**：双指针、前缀和、贪心、模拟

---

## 🗺️ 知识图谱

```
                    ┌─────────────────┐
                    │   数组与字符串    │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  双指针   │      │   贪心    │      │  前缀和   │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
    ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
    │合并数组  │       │跳跃游戏  │       │除自身乘积│
    │移除元素  │       │买卖股票  │       │         │
    │删除重复  │       │分发糖果  │       │         │
    └─────────┘       └─────────┘       └─────────┘
```

---

## 1️⃣ LC 88. 合并两个有序数组 🟢

### 题目描述
将两个有序数组 `nums1` 和 `nums2` 合并到 `nums1` 中，使其有序。

### 🎨 图解思路

```
从后往前填充，避免覆盖！

nums1 = [1, 2, 3, 0, 0, 0]  nums2 = [2, 5, 6]
              ↑                        ↑
              p1                       p2
                                              ↑
                                              p (填充位置)

Step 1: 比较 3 vs 6 → 填 6
nums1 = [1, 2, 3, 0, 0, 6]

Step 2: 比较 3 vs 5 → 填 5  
nums1 = [1, 2, 3, 0, 5, 6]

Step 3: 比较 3 vs 2 → 填 3
nums1 = [1, 2, 3, 3, 5, 6]

... 最终结果 [1, 2, 2, 3, 5, 6]
```

### 💻 代码实现

```python
def merge(nums1, m, nums2, n):
    p1, p2, p = m - 1, n - 1, m + n - 1
    
    while p2 >= 0:  # nums2 还有元素
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
```

### 🧠 记忆口诀
> **"从后往前填，大的先落位"**

---

## 2️⃣ LC 27. 移除元素 🟢

### 题目描述
原地移除数组中等于 `val` 的元素，返回新长度。

### 🎨 图解思路

```
快慢指针：慢指针标记有效位置，快指针遍历

val = 3
      s
      f
[3, 2, 2, 3, 4]

f=0: nums[0]=3, 跳过
f=1: nums[1]=2≠3, nums[s]=2, s++
      s
         f
[2, 2, 2, 3, 4]

f=2: nums[2]=2≠3, nums[s]=2, s++
         s
            f
[2, 2, 2, 3, 4]

f=3: nums[3]=3, 跳过
f=4: nums[4]=4≠3, nums[s]=4, s++

最终: [2, 2, 4, _, _], 返回 3
```

### 💻 代码实现

```python
def removeElement(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

### 🧠 记忆口诀
> **"快指针探路，慢指针收货"**

---

## 3️⃣ LC 26. 删除有序数组中的重复项 🟢

### 🎨 图解思路

```
有序数组，相同元素必相邻！

      s
      f
[1, 1, 2, 2, 3]

f=1: nums[1]=nums[0], 跳过
f=2: nums[2]≠nums[1], s++, nums[s]=2
         s
            f
[1, 2, 2, 2, 3]

f=4: nums[4]≠nums[3], s++, nums[s]=3
            s
[1, 2, 3, _, _]

返回 s+1 = 3
```

### 💻 代码实现

```python
def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

---

## 4️⃣ LC 80. 删除有序数组中的重复项 II 🟡

### 题目描述
每个元素最多出现 **两次**。

### 🎨 通用模板

```python
def removeDuplicates(nums, k=2):
    """允许每个元素最多出现 k 次"""
    slow = 0
    for num in nums:
        if slow < k or num != nums[slow - k]:
            nums[slow] = num
            slow += 1
    return slow
```

### 🧠 记忆口诀
> **"检查 k 位之前，不同才能进"**

---

## 5️⃣ LC 169. 多数元素 🟢

### 题目描述
找出出现次数超过 `n/2` 的元素。

### 🎨 Boyer-Moore 投票算法

```
把多数元素看作 +1，其他元素看作 -1
最终 +1 一定比 -1 多！

nums = [2, 2, 1, 1, 1, 2, 2]

candidate=2, count=1  → [2]
candidate=2, count=2  → [2,2]
candidate=2, count=1  → 遇到1，抵消
candidate=2, count=0  → 遇到1，抵消
candidate=1, count=1  → count=0时换人
candidate=1, count=0  → 遇到2，抵消
candidate=2, count=1  → count=0时换人

最终 candidate = 2 ✓
```

### 💻 代码实现

```python
def majorityElement(nums):
    candidate, count = None, 0
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    return candidate
```

### 🧠 记忆口诀
> **"同加异减，归零换帅"**

---

## 6️⃣ LC 189. 轮转数组 🟡

### 题目描述
将数组向右轮转 `k` 位。

### 🎨 三次翻转法

```
nums = [1,2,3,4,5,6,7], k = 3

Step 1: 整体翻转
[7,6,5,4,3,2,1]

Step 2: 翻转前 k 个
[5,6,7,4,3,2,1]

Step 3: 翻转后 n-k 个  
[5,6,7,1,2,3,4] ✓
```

### 💻 代码实现

```python
def rotate(nums, k):
    n = len(nums)
    k %= n  # 处理 k > n 的情况
    
    def reverse(left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left, right = left + 1, right - 1
    
    reverse(0, n - 1)      # 整体翻转
    reverse(0, k - 1)      # 前k个
    reverse(k, n - 1)      # 后n-k个
```

### 🧠 记忆口诀
> **"全转前转后，三步到位"**

---

## 7️⃣ LC 121. 买卖股票的最佳时机 🟢

### 题目描述
只能买卖一次，求最大利润。

### 🎨 图解思路

```
维护「历史最低价」，计算当天卖出利润

prices = [7, 1, 5, 3, 6, 4]
          │  │  │  │  │  │
min_price │  1  1  1  1  1
profit    0  0  4  2  5  3
                       ↑
                    最大利润 = 5
```

### 💻 代码实现

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

### 🧠 记忆口诀
> **"记住最低点，时刻算差价"**

---

## 8️⃣ LC 55. 跳跃游戏 🟡

### 题目描述
判断能否跳到最后一个位置。

### 🎨 贪心思路

```
维护能到达的最远位置 max_reach

nums = [2, 3, 1, 1, 4]
        ↑
i=0: max_reach = max(0, 0+2) = 2
i=1: max_reach = max(2, 1+3) = 4 ≥ 4 ✓ 可达！

nums = [3, 2, 1, 0, 4]
i=0: max_reach = 3
i=1: max_reach = 3
i=2: max_reach = 3
i=3: max_reach = 3 < 4 
i=4: i > max_reach，无法到达 ✗
```

### 💻 代码实现

```python
def canJump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True
```

### 🧠 记忆口诀
> **"走一步算一步，能到就更新"**

---

## 9️⃣ LC 45. 跳跃游戏 II 🟡

### 题目描述
求到达最后位置的最少跳跃次数。

### 🎨 BFS 思想

```
把每一跳能到的范围看作一层

nums = [2, 3, 1, 1, 4]
        ↑
层0: 位置0，能到 [1,2]
层1: 位置1-2，能到 [2,3,4]  → 到达终点！

跳跃次数 = 2
```

### 💻 代码实现

```python
def jump(nums):
    jumps = 0
    cur_end = 0      # 当前跳跃能到的边界
    cur_farthest = 0 # 下一跳能到的最远
    
    for i in range(len(nums) - 1):
        cur_farthest = max(cur_farthest, i + nums[i])
        if i == cur_end:  # 到达边界，必须跳
            jumps += 1
            cur_end = cur_farthest
    
    return jumps
```

---

## 🔟 LC 238. 除自身以外数组的乘积 🟡

### 题目描述
返回数组，`answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

### 🎨 前缀积 × 后缀积

```
nums =    [1,  2,  3,  4]
前缀积 =   [1,  1,  2,  6]   (不含当前)
后缀积 =   [24, 12, 4,  1]   (不含当前)
结果 =    [24, 12, 8,  6]   (前缀 × 后缀)
```

### 💻 代码实现 (O(1) 空间)

```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    
    # 计算前缀积
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    
    # 计算后缀积并相乘
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result
```

### 🧠 记忆口诀
> **"左边乘一遍，右边乘一遍"**

---

## 📊 本章总结

### 核心模式速查表

| 模式 | 适用场景 | 典型题目 |
|------|----------|----------|
| **快慢指针** | 原地修改数组 | 26, 27, 80 |
| **前后指针** | 有序数组合并 | 88 |
| **贪心** | 最优解问题 | 55, 45, 121 |
| **前缀和/积** | 区间计算 | 238 |
| **投票算法** | 多数元素 | 169 |
| **翻转技巧** | 轮转/翻转 | 189 |

### 🧠 全章记忆口诀

```
合移删删多，轮买买跳跳
H插除加糖，数组十五妙

合 - 合并数组 (88)
移 - 移除元素 (27)  
删删 - 删除重复 I/II (26, 80)
多 - 多数元素 (169)
轮 - 轮转数组 (189)
买买 - 买卖股票 I/II (121, 122)
跳跳 - 跳跃游戏 I/II (55, 45)
H - H指数 (274)
插 - O(1)插入删除 (380)
除 - 除自身乘积 (238)
加 - 加油站 (134)
糖 - 分发糖果 (135)
```

---

> 📖 **下一篇**：[双指针专题](/2026/01/18/leetcode-150-two-pointers/)

