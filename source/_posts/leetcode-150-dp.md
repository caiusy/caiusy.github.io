---
title: 📈 LeetCode 150 - 动态规划专题
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
  - 动态规划
description: LeetCode 面试 150 题之动态规划专题，含状态转移图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 📈 动态规划专题 (11题)

> 🎯 **核心思想**：将大问题分解为小问题，记录子问题的解，避免重复计算

---

## 🗺️ 动态规划解题框架

```
┌─────────────────────────────────────────────────────────────┐
│                    DP 解题四步法                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 定义状态                                           │
│          dp[i] 表示什么？dp[i][j] 表示什么？                 │
│                                                             │
│  Step 2: 状态转移方程                                        │
│          dp[i] = f(dp[i-1], dp[i-2], ...)                  │
│                                                             │
│  Step 3: 初始化                                             │
│          边界条件是什么？dp[0], dp[1] 等于多少？             │
│                                                             │
│  Step 4: 遍历顺序                                            │
│          正序还是倒序？先行后列还是先列后行？                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 DP 代码模板

```python
def dp_template(nums):
    n = len(nums)
    
    # Step 1: 定义 dp 数组
    dp = [0] * n  # 或 [[0]*m for _ in range(n)]
    
    # Step 3: 初始化
    dp[0] = base_case
    
    # Step 4: 遍历顺序
    for i in range(1, n):
        # Step 2: 状态转移
        dp[i] = transition(dp[i-1], ...)
    
    return dp[n-1]  # 或其他目标
```

---

## 1️⃣ LC 70. 爬楼梯 🟢

### 题目描述
每次可以爬 1 或 2 个台阶，爬到第 n 阶有多少种方法？

### 🎨 图解思路

```
到达第 n 阶的方法 = 从第 n-1 阶爬 1 步 + 从第 n-2 阶爬 2 步

        ┌───┐
        │ n │ ← 目标
        └───┘
       ↗     ↖
    ┌───┐   ┌───┐
    │n-1│   │n-2│
    └───┘   └───┘
      ↑       ↑
     1步     2步

dp[n] = dp[n-1] + dp[n-2]  (斐波那契数列!)
```

### 💻 代码实现

```python
def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    
    # 空间优化：只需记录前两个状态
    prev, curr = 1, 2
    
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    
    return curr
```

### 🧠 记忆口诀
> **"爬楼梯就是斐波那契"**

---

## 2️⃣ LC 198. 打家劫舍 🟡

### 题目描述
不能偷相邻的房子，求能偷到的最大金额。

### 🎨 图解思路

```
nums = [2, 7, 9, 3, 1]

对于每个房子，两个选择：
1. 偷：dp[i] = dp[i-2] + nums[i]
2. 不偷：dp[i] = dp[i-1]

dp[i] = max(dp[i-2] + nums[i], dp[i-1])

i:      0   1   2   3   4
nums:   2   7   9   3   1
dp:     2   7  11  11  12
        ↑   ↑   ↑   ↑   ↑
       偷  偷 偷0+9 不偷 偷2+1
```

### 💻 代码实现

```python
def rob(nums: list) -> int:
    if len(nums) == 1:
        return nums[0]
    
    # 空间优化
    prev, curr = nums[0], max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        prev, curr = curr, max(curr, prev + nums[i])
    
    return curr
```

### 🧠 记忆口诀
> **"偷不偷，取最大"**

---

## 3️⃣ LC 139. 单词拆分 🟡

### 题目描述
判断字符串是否可以被拆分为字典中的单词。

### 🎨 图解思路

```
s = "leetcode", wordDict = ["leet", "code"]

dp[i] 表示 s[0:i] 是否可以被拆分

dp[0] = True (空字符串)
dp[4] = dp[0] and "leet" in dict → True
dp[8] = dp[4] and "code" in dict → True

   l  e  e  t  c  o  d  e
   0  1  2  3  4  5  6  7  8
dp T  F  F  F  T  F  F  F  T
               ↑           ↑
            "leet"      "code"
```

### 💻 代码实现

```python
def wordBreak(s: str, wordDict: list) -> bool:
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

### 🧠 记忆口诀
> **"前面能拆，后面在字典，就能拆"**

---

## 4️⃣ LC 322. 零钱兑换 🟡

### 题目描述
用最少的硬币凑出目标金额。

### 🎨 图解思路

```
coins = [1, 2, 5], amount = 11

dp[i] = 凑出金额 i 需要的最少硬币数

对于每个金额 i，尝试每个硬币 c：
dp[i] = min(dp[i], dp[i-c] + 1)

amount:  0  1  2  3  4  5  6  7  8  9  10  11
dp:      0  1  1  2  2  1  2  2  3  3   2   3
            ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑   ↑   ↑
           +1 +2 +1 +2 +5 +1 +2 +1 +2  +5  +5
```

### 💻 代码实现

```python
def coinChange(coins: list, amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 🧠 记忆口诀
> **"每个硬币试一试，取最小"**

---

## 5️⃣ LC 300. 最长递增子序列 🟡

### 题目描述
找出数组中最长的严格递增子序列的长度。

### 🎨 图解思路

```
nums = [10, 9, 2, 5, 3, 7, 101, 18]

dp[i] = 以 nums[i] 结尾的 LIS 长度

对于 nums[i]，找所有 j < i 且 nums[j] < nums[i]：
dp[i] = max(dp[j] + 1)

i:      0   1   2   3   4   5   6    7
nums:  10   9   2   5   3   7  101  18
dp:     1   1   1   2   2   3    4   4
                    ↑   ↑   ↑    ↑   ↑
                   2+1 2+1 5+1  7+1 7+1

LIS = 4 (如 [2, 3, 7, 101])
```

### 💻 代码实现

```python
# O(n²) 解法
def lengthOfLIS(nums: list) -> int:
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
```

### 🔥 O(n log n) 二分解法

```python
def lengthOfLIS(nums: list) -> int:
    # tails[i] = 长度为 i+1 的 LIS 的最小结尾元素
    tails = []
    
    for num in nums:
        # 二分查找第一个 >= num 的位置
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)
```

### 🧠 记忆口诀
> **"前面比我小的，加1取最大"**

---

## 6️⃣ LC 120. 三角形最小路径和 🟡

### 题目描述
从顶部到底部的最小路径和。

### 🎨 图解思路

```
    [2]
   [3,4]
  [6,5,7]
 [4,1,8,3]

自底向上计算：
dp[i][j] = min(dp[i+1][j], dp[i+1][j+1]) + triangle[i][j]

第3层: [4, 1, 8, 3]
第2层: [6+1, 5+1, 7+3] = [7, 6, 10]
第1层: [3+6, 4+6] = [9, 10]
第0层: [2+9] = [11]

最小路径和 = 11
```

### 💻 代码实现

```python
def minimumTotal(triangle: list) -> int:
    n = len(triangle)
    # 从最后一行开始
    dp = triangle[-1][:]
    
    # 自底向上
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]
    
    return dp[0]
```

### 🧠 记忆口诀
> **"自底向上，取小加自己"**

---

## 7️⃣ LC 64. 最小路径和 🟡

### 题目描述
从左上角到右下角的最小路径和。

### 🎨 图解思路

```
grid:
[1, 3, 1]
[1, 5, 1]
[4, 2, 1]

dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

dp:
[1, 4, 5]
[2, 7, 6]
[6, 8, 7]

最小路径和 = 7 (1→3→1→1→1)
```

### 💻 代码实现

```python
def minPathSum(grid: list) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                dp[i][j] = grid[i][j]
            elif i == 0:
                dp[i][j] = dp[i][j-1] + grid[i][j]
            elif j == 0:
                dp[i][j] = dp[i-1][j] + grid[i][j]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[m-1][n-1]
```

### 🧠 记忆口诀
> **"上左取小，加自己"**

---

## 8️⃣ LC 63. 不同路径 II 🟡

### 题目描述
有障碍物的网格，从左上到右下的路径数。

### 🎨 图解思路

```
grid:              dp:
[0, 0, 0]         [1, 1, 1]
[0, 1, 0]    →    [1, 0, 1]
[0, 0, 0]         [1, 1, 2]

障碍物位置 dp = 0
其他位置 dp = dp[上] + dp[左]
```

### 💻 代码实现

```python
def uniquePathsWithObstacles(obstacleGrid: list) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    if obstacleGrid[0][0] == 1:
        return 0
    
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    # 初始化第一列
    for i in range(1, m):
        if obstacleGrid[i][0] == 0:
            dp[i][0] = dp[i-1][0]
    
    # 初始化第一行
    for j in range(1, n):
        if obstacleGrid[0][j] == 0:
            dp[0][j] = dp[0][j-1]
    
    # 填充 dp
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

### 🧠 记忆口诀
> **"障碍为0，否则上加左"**

---

## 9️⃣ LC 5. 最长回文子串 🟡

### 题目描述
找出字符串中最长的回文子串。

### 🎨 图解思路

```
s = "babad"

dp[i][j] = s[i:j+1] 是否为回文

条件: s[i] == s[j] and dp[i+1][j-1]

填表顺序：按长度从小到大

长度1: 全为 True
长度2: s[i] == s[i+1]
长度3+: s[i] == s[j] and dp[i+1][j-1]
```

### 💻 代码实现

```python
def longestPalindrome(s: str) -> str:
    n = len(s)
    if n < 2:
        return s
    
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    
    # 所有长度为 1 的子串都是回文
    for i in range(n):
        dp[i][i] = True
    
    # 按长度填表
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]
            
            if dp[i][j] and length > max_len:
                start, max_len = i, length
    
    return s[start:start + max_len]
```

### 🔥 中心扩展法 (更优)

```python
def longestPalindrome(s: str) -> str:
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    result = ""
    for i in range(len(s)):
        # 奇数长度
        odd = expand(i, i)
        # 偶数长度
        even = expand(i, i + 1)
        
        result = max(result, odd, even, key=len)
    
    return result
```

### 🧠 记忆口诀
> **"首尾相同，中间也是回文"**

---

## 🔟 LC 97. 交错字符串 🟡

### 题目描述
判断 s3 是否由 s1 和 s2 交错组成。

### 🎨 图解思路

```
s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"

dp[i][j] = s1[0:i] 和 s2[0:j] 能否交错组成 s3[0:i+j]

状态转移:
dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or
           (dp[i][j-1] and s2[j-1]==s3[i+j-1])
```

### 💻 代码实现

```python
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    
    if m + n != len(s3):
        return False
    
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # 初始化第一列
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    # 初始化第一行
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # 填充 dp
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = ((dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                        (dp[i][j-1] and s2[j-1] == s3[i+j-1]))
    
    return dp[m][n]
```

### 🧠 记忆口诀
> **"上或左能到，且字符匹配"**

---

## 1️⃣1️⃣ LC 72. 编辑距离 🟡

### 题目描述
将 word1 转换成 word2 所使用的最少操作数。

### 🎨 图解思路

```
word1 = "horse", word2 = "ros"

dp[i][j] = word1[0:i] 转换为 word2[0:j] 的最少操作

三种操作:
1. 插入: dp[i][j-1] + 1
2. 删除: dp[i-1][j] + 1
3. 替换: dp[i-1][j-1] + (0 if 相同 else 1)

dp 表:
      ""  r  o  s
  ""   0  1  2  3
  h    1  1  2  3
  o    2  2  1  2
  r    3  2  2  2
  s    4  3  3  2
  e    5  4  4  3
```

### 💻 代码实现

```python
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充 dp
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # 删除
                    dp[i][j-1],      # 插入
                    dp[i-1][j-1]     # 替换
                )
    
    return dp[m][n]
```

### 🧠 记忆口诀
> **"相同不变，不同取三者最小加1"**

---

## 📊 本章总结

### DP 问题分类

```
┌──────────────────────────────────────────────────┐
│                  动态规划分类                     │
├──────────────────────────────────────────────────┤
│                                                  │
│  线性 DP                                         │
│  ├─ 单序列: 爬楼梯, 打家劫舍, LIS                │
│  └─ 双序列: 编辑距离, 交错字符串                  │
│                                                  │
│  区间 DP                                         │
│  └─ 回文子串                                     │
│                                                  │
│  背包 DP                                         │
│  └─ 零钱兑换, 单词拆分                           │
│                                                  │
│  网格 DP                                         │
│  └─ 最小路径和, 不同路径                         │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 🧠 全章记忆口诀

```
爬楼劫舍单词拆
零钱递增三角来
路径网格回文判
交错编辑全都会
```

---

> 📖 **下一篇**：[图论专题](/2026/01/18/leetcode-150-graph/)

