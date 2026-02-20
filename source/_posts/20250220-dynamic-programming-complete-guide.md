---
title: 动态规划完全指南：从入门到精通
date: 2025-01-20
categories: 
  - 算法
tags:
  - 动态规划
  - DP
  - LeetCode
  - 算法面试
---

# 动态规划完全指南：从入门到精通

> 用费曼学习法，彻底搞懂动态规划的本质

## 一、什么是动态规划？

### 1.1 一句话定义

**动态规划 (Dynamic Programming, DP)** = **记住已经解决过的子问题的答案，避免重复计算**。

就这么简单。

### 1.2 为什么叫"动态规划"？

这个名字其实很有误导性。Richard Bellman 在 1950 年代发明这个方法时，故意起了个听起来很高大上的名字，因为他的老板不喜欢"数学研究"。"Dynamic" 听起来很酷，"Programming" 在当时指的是"规划/优化"，不是写代码。

所以别被名字吓到，它的本质就是：**用空间换时间，记住中间结果**。

### 1.3 核心直觉：斐波那契数列

让我们从最经典的例子开始：

```
fib(0) = 0
fib(1) = 1
fib(n) = fib(n-1) + fib(n-2)
```

**暴力递归写法：**

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

看起来很简洁对吧？但是有个致命问题：

![暴力递归 vs 动态规划](/images/dp/01_dp_vs_recursion.png)

看到了吗？`fib(3)` 被计算了 2 次，`fib(2)` 被计算了 3 次！当 n=50 时，这个递归树会有 2^50 个节点，你的电脑会直接卡死。

**时间复杂度：O(2^n)** —— 指数级爆炸！

### 1.4 DP 的解决方案

**方案一：记忆化递归 (Top-Down)**

```python
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

**方案二：迭代填表 (Bottom-Up)**

```python
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**方案三：空间优化**

```python
def fib(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

| 方案 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 暴力递归 | O(2^n) | O(n) 栈空间 |
| 记忆化递归 | O(n) | O(n) |
| 迭代填表 | O(n) | O(n) |
| 空间优化 | O(n) | O(1) |

**这就是 DP 的威力：从指数级降到线性级！**


---

## 二、DP 的三大核心要素

![DP三要素](/images/dp/02_dp_three_elements.png)

### 2.1 状态定义 (State Definition)

**这是最关键的一步！** 状态定义错了，后面全白搭。

**问自己：`dp[i]` 或 `dp[i][j]` 代表什么？**

常见的状态定义模式：

| 问题类型 | 状态定义 | 例子 |
|---------|---------|------|
| 线性序列 | `dp[i]` = 以第i个元素结尾的xxx | 最长递增子序列 |
| 线性序列 | `dp[i]` = 前i个元素的xxx | 打家劫舍 |
| 双序列 | `dp[i][j]` = s1前i个和s2前j个的xxx | LCS, 编辑距离 |
| 背包 | `dp[i][w]` = 前i个物品、容量w的xxx | 0-1背包 |
| 区间 | `dp[i][j]` = 区间[i,j]的xxx | 戳气球 |

### 2.2 状态转移方程 (Transition)

**问自己：当前状态和哪些之前的状态有关？**

这是 DP 的"递推公式"，决定了如何从小问题推导出大问题。

```
dp[i] = f(dp[i-1], dp[i-2], ..., dp[0])
```

**技巧：画图！** 把状态之间的依赖关系画出来，转移方程自然就出来了。

### 2.3 边界条件 (Base Case)

**问自己：最小的子问题，答案是什么？**

边界条件就是递推的起点。没有正确的边界条件，整个 DP 就会崩溃。

常见边界：
- `dp[0] = 0` 或 `dp[0] = 1`
- `dp[0][j] = xxx`, `dp[i][0] = xxx`
- 空集、空串的情况

---

## 三、两种实现方式对比

![Top-Down vs Bottom-Up](/images/dp/05_topdown_vs_bottomup.png)

### 3.1 Top-Down：记忆化递归

```python
def solve(n, memo={}):
    # 1. 检查缓存
    if n in memo:
        return memo[n]
    
    # 2. Base case
    if n <= 1:
        return base_value
    
    # 3. 递归 + 记忆化
    memo[n] = f(solve(n-1), solve(n-2), ...)
    return memo[n]
```

**优点：**
- 思路自然，从目标出发
- 只计算需要的子问题
- 代码更接近数学定义

**缺点：**
- 递归栈开销
- Python 默认递归深度限制 (1000)

### 3.2 Bottom-Up：迭代填表

```python
def solve(n):
    # 1. 初始化 DP 数组
    dp = [0] * (n + 1)
    
    # 2. Base case
    dp[0] = base_value
    
    # 3. 按顺序填表
    for i in range(1, n + 1):
        dp[i] = f(dp[i-1], dp[i-2], ...)
    
    # 4. 返回结果
    return dp[n]
```

**优点：**
- 无递归栈开销
- 更容易做空间优化
- 通常更快（无函数调用开销）

**缺点：**
- 需要想清楚遍历顺序
- 可能计算不需要的子问题

### 3.3 如何选择？

| 场景 | 推荐方式 |
|------|---------|
| 面试时快速写出 | Top-Down（更直观）|
| 追求性能 | Bottom-Up |
| 需要空间优化 | Bottom-Up |
| 状态转移复杂 | Top-Down（更容易调试）|


---

## 四、空间优化技巧

![空间优化](/images/dp/06_space_optimization.png)

### 4.1 滚动数组

当 `dp[i]` 只依赖 `dp[i-1]` 时，不需要保存整个数组：

```python
# 优化前：O(n) 空间
dp = [0] * n
for i in range(1, n):
    dp[i] = dp[i-1] + something

# 优化后：O(1) 空间
prev = 0
for i in range(1, n):
    curr = prev + something
    prev = curr
```

### 4.2 二维降一维

当 `dp[i][j]` 只依赖 `dp[i-1][...]` 时：

```python
# 优化前：O(m*n) 空间
dp = [[0] * n for _ in range(m)]
for i in range(1, m):
    for j in range(1, n):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]

# 优化后：O(n) 空间
dp = [0] * n
for i in range(1, m):
    for j in range(1, n):
        dp[j] = dp[j] + dp[j-1]  # dp[j] 就是原来的 dp[i-1][j]
```

**注意遍历顺序！** 如果依赖左上角，需要从右往左遍历。

---

## 五、经典问题详解

### 5.1 爬楼梯 (LeetCode 70)

![爬楼梯](/images/dp/03_classic_problems.png)

**问题：** 每次可以爬 1 或 2 个台阶，爬到第 n 阶有多少种方法？

**状态定义：** `dp[i]` = 爬到第 i 阶的方法数

**转移方程：** `dp[i] = dp[i-1] + dp[i-2]`
- 从第 i-1 阶爬 1 步上来
- 从第 i-2 阶爬 2 步上来

**边界条件：** `dp[0] = 1, dp[1] = 1`

```python
def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

**复杂度：** 时间 O(n)，空间 O(1)


### 5.2 打家劫舍 (LeetCode 198)

**问题：** 不能偷相邻的房子，求能偷到的最大金额。

**状态定义：** `dp[i]` = 考虑前 i 个房子能偷到的最大金额

**转移方程：** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
- 不偷第 i 个：`dp[i-1]`
- 偷第 i 个：`dp[i-2] + nums[i]`（不能偷 i-1）

```python
def rob(nums: list[int]) -> int:
    if len(nums) <= 2:
        return max(nums) if nums else 0
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
    return prev1
```

**复杂度：** 时间 O(n)，空间 O(1)

### 5.3 最大子数组和 (LeetCode 53)

![Kadane算法](/images/dp/08_kadane.png)

**问题：** 找出和最大的连续子数组。

**状态定义：** `dp[i]` = 以 nums[i] 结尾的最大子数组和

**转移方程：** `dp[i] = max(nums[i], dp[i-1] + nums[i])`
- 要么从 nums[i] 重新开始
- 要么接上前面的子数组

```python
def maxSubArray(nums: list[int]) -> int:
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

**复杂度：** 时间 O(n)，空间 O(1)


### 5.4 最长递增子序列 LIS (LeetCode 300)

**问题：** 找出最长严格递增子序列的长度。

**状态定义：** `dp[i]` = 以 nums[i] 结尾的 LIS 长度

**转移方程：** `dp[i] = max(dp[j] + 1)` for all j < i where nums[j] < nums[i]

```python
def lengthOfLIS(nums: list[int]) -> int:
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

**复杂度：** 时间 O(n²)，空间 O(n)

**优化版本（二分查找）：** O(n log n)

```python
import bisect

def lengthOfLIS(nums: list[int]) -> int:
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### 5.5 最长公共子序列 LCS (LeetCode 1143)

![LCS](/images/dp/04_lcs.png)

**问题：** 两个字符串的最长公共子序列长度。

**状态定义：** `dp[i][j]` = s1 前 i 个字符和 s2 前 j 个字符的 LCS 长度

**转移方程：**
```
if s1[i-1] == s2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

**复杂度：** 时间 O(mn)，空间 O(mn)，可优化到 O(n)


### 5.6 编辑距离 (LeetCode 72)

![编辑距离](/images/dp/09_edit_distance.png)

**问题：** 将 word1 转换成 word2 所需的最少操作数（插入、删除、替换）。

**状态定义：** `dp[i][j]` = word1 前 i 个字符转换成 word2 前 j 个字符的最少操作数

**转移方程：**
```
if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  # 不需要操作
else:
    dp[i][j] = 1 + min(
        dp[i-1][j-1],  # 替换
        dp[i-1][j],    # 删除
        dp[i][j-1]     # 插入
    )
```

```python
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**复杂度：** 时间 O(mn)，空间 O(mn)


### 5.7 0-1 背包问题

**问题：** n 个物品，每个有重量 w[i] 和价值 v[i]，背包容量 W，求最大价值。

**状态定义：** `dp[i][w]` = 前 i 个物品、容量 w 时的最大价值

**转移方程：**
```
dp[i][w] = max(
    dp[i-1][w],           # 不选第 i 个
    dp[i-1][w-w[i]] + v[i]  # 选第 i 个（如果装得下）
)
```

```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]  # 不选
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    
    return dp[n][W]
```

**空间优化版本：**

```python
def knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):  # 从右往左！
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

**为什么要从右往左？** 因为 `dp[w]` 依赖 `dp[w-weights[i]]`，如果从左往右，`dp[w-weights[i]]` 已经被更新过了，相当于同一个物品被选了多次（变成完全背包了）。

**复杂度：** 时间 O(nW)，空间 O(W)


### 5.8 完全背包问题

**问题：** 每个物品可以选无限次。

**和 0-1 背包的区别：** 遍历顺序从左往右！

```python
def unboundedKnapsack(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(weights[i], W + 1):  # 从左往右！
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

### 5.9 零钱兑换 (LeetCode 322)

**问题：** 用最少的硬币凑出金额 amount。

**状态定义：** `dp[i]` = 凑出金额 i 需要的最少硬币数

**转移方程：** `dp[i] = min(dp[i - coin] + 1)` for all coins

```python
def coinChange(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

**复杂度：** 时间 O(amount × n)，空间 O(amount)


### 5.10 不同路径 (LeetCode 62)

**问题：** m×n 网格，从左上到右下，只能向右或向下，有多少条路径？

**状态定义：** `dp[i][j]` = 到达 (i,j) 的路径数

**转移方程：** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`

```python
def uniquePaths(m: int, n: int) -> int:
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[n-1]
```

**复杂度：** 时间 O(mn)，空间 O(n)

---

## 六、DP 问题分类

![DP分类](/images/dp/07_dp_categories.png)

### 6.1 线性 DP

特点：状态沿着一个维度线性递推

| 题目 | 状态定义 | 转移方程 |
|-----|---------|---------|
| 爬楼梯 | dp[i]=到第i阶方法数 | dp[i]=dp[i-1]+dp[i-2] |
| 打家劫舍 | dp[i]=前i家最大金额 | dp[i]=max(dp[i-1], dp[i-2]+nums[i]) |
| 最大子数组和 | dp[i]=以i结尾的最大和 | dp[i]=max(nums[i], dp[i-1]+nums[i]) |
| LIS | dp[i]=以i结尾的LIS长度 | dp[i]=max(dp[j]+1) for j<i |

### 6.2 序列 DP（双序列）

特点：两个序列之间的关系

| 题目 | 状态定义 | 关键点 |
|-----|---------|-------|
| LCS | dp[i][j]=s1前i和s2前j的LCS | 字符相等时+1 |
| 编辑距离 | dp[i][j]=最少操作数 | 三种操作取min |
| 不同子序列 | dp[i][j]=s中t出现次数 | 选或不选当前字符 |

### 6.3 背包 DP

特点：选择物品，满足约束，优化目标

| 类型 | 特点 | 遍历顺序 |
|-----|------|---------|
| 0-1背包 | 每个物品最多选1次 | 容量从大到小 |
| 完全背包 | 每个物品可选无限次 | 容量从小到大 |
| 多重背包 | 每个物品有数量限制 | 二进制优化 |


### 6.4 区间 DP

特点：在区间上进行决策

```python
# 区间DP模板
for length in range(2, n + 1):      # 枚举区间长度
    for i in range(n - length + 1):  # 枚举起点
        j = i + length - 1           # 计算终点
        for k in range(i, j):        # 枚举分割点
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost)
```

经典题目：戳气球、石子合并、矩阵链乘法

### 6.5 树形 DP

特点：在树结构上进行 DP

```python
def dfs(node):
    if not node:
        return 0
    left = dfs(node.left)
    right = dfs(node.right)
    # 根据子树结果计算当前节点
    return f(left, right, node.val)
```

经典题目：打家劫舍III、二叉树最大路径和

### 6.6 状态压缩 DP

特点：用二进制表示状态集合

```python
# 枚举所有子集
for mask in range(1 << n):
    for i in range(n):
        if mask & (1 << i):  # 第i位是否为1
            # 处理
```

经典题目：旅行商问题 TSP

---

## 七、时间空间复杂度分析

### 7.1 时间复杂度

**公式：状态数量 × 每个状态的转移代价**

| 问题 | 状态数 | 转移代价 | 总复杂度 |
|-----|-------|---------|---------|
| 斐波那契 | O(n) | O(1) | O(n) |
| LIS (朴素) | O(n) | O(n) | O(n²) |
| LCS | O(mn) | O(1) | O(mn) |
| 0-1背包 | O(nW) | O(1) | O(nW) |
| 区间DP | O(n²) | O(n) | O(n³) |

### 7.2 空间复杂度

**基本空间 = DP 数组大小**

优化技巧：
1. **滚动数组**：O(n) → O(1)
2. **降维**：O(mn) → O(n)
3. **只保留必要状态**


---

## 八、DP 解题模板

![解题模板](/images/dp/10_dp_template.png)

### 8.1 五步法

```python
def dp_template(input):
    # Step 1: 定义状态
    # dp[i] = ???
    
    # Step 2: 初始化（边界条件）
    dp = [0] * n
    dp[0] = base_case
    
    # Step 3: 确定遍历顺序
    for i in range(1, n):
        # Step 4: 状态转移
        dp[i] = f(dp[i-1], ...)
    
    # Step 5: 返回结果
    return dp[n-1]
```

### 8.2 Debug 技巧

1. **打印 DP 数组**：看中间状态是否正确
2. **手算小例子**：n=3,4 时手动验证
3. **检查边界**：i=0, j=0 的情况
4. **检查遍历顺序**：依赖的状态是否已计算


---

## 九、LeetCode 经典题目练习

### 9.1 入门级（Easy）

| 题号 | 题目 | 核心思路 |
|-----|------|---------|
| 70 | 爬楼梯 | dp[i]=dp[i-1]+dp[i-2] |
| 121 | 买卖股票最佳时机 | 记录最小价格，更新最大利润 |
| 53 | 最大子数组和 | dp[i]=max(nums[i], dp[i-1]+nums[i]) |
| 746 | 使用最小花费爬楼梯 | dp[i]=min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2]) |

### 9.2 中等级（Medium）

| 题号 | 题目 | 核心思路 |
|-----|------|---------|
| 198 | 打家劫舍 | dp[i]=max(dp[i-1], dp[i-2]+nums[i]) |
| 300 | 最长递增子序列 | dp[i]=max(dp[j]+1) |
| 322 | 零钱兑换 | 完全背包变形 |
| 62 | 不同路径 | dp[i][j]=dp[i-1][j]+dp[i][j-1] |
| 64 | 最小路径和 | 同上，取min |
| 139 | 单词拆分 | dp[i]=any(dp[j] and s[j:i] in dict) |
| 152 | 乘积最大子数组 | 同时维护最大和最小 |
| 1143 | 最长公共子序列 | 双序列DP经典 |

### 9.3 困难级（Hard）

| 题号 | 题目 | 核心思路 |
|-----|------|---------|
| 72 | 编辑距离 | 三种操作取min |
| 312 | 戳气球 | 区间DP |
| 10 | 正则表达式匹配 | 双序列DP + 特殊字符处理 |
| 32 | 最长有效括号 | dp[i]=以i结尾的最长有效长度 |
| 115 | 不同的子序列 | dp[i][j]=s前i个中t前j个出现次数 |


---

## 十、费曼总结：用一句话解释 DP

### 10.1 给小学生解释

**"做作业时，把算过的题目答案记在草稿纸上，下次遇到一样的题直接抄答案，不用重新算。"**

### 10.2 给程序员解释

**"用哈希表缓存递归的中间结果，或者用数组从小到大迭代填表。"**

### 10.3 给面试官解释

**"DP 是一种通过将问题分解为重叠子问题，并存储子问题的解来避免重复计算的优化技术。它适用于具有最优子结构和重叠子问题性质的问题。"**

### 10.4 核心公式

```
DP = 递归 + 记忆化 = 分治 + 缓存
```

### 10.5 判断是否能用 DP

问自己两个问题：

1. **最优子结构**：大问题的最优解能否由小问题的最优解推导出来？
2. **重叠子问题**：在求解过程中，是否会重复计算相同的子问题？

如果两个都是 Yes，就可以用 DP！


---

## 十一、常见错误与陷阱

### 11.1 状态定义不清

**错误：** 模糊地定义 `dp[i]` 是"第 i 个的答案"

**正确：** 明确是"以第 i 个结尾"还是"前 i 个"

### 11.2 边界条件遗漏

```python
# 错误：忘记处理空数组
def maxSubArray(nums):
    dp = [0] * len(nums)  # 如果 nums 为空会报错
    ...

# 正确：先检查边界
def maxSubArray(nums):
    if not nums:
        return 0
    ...
```

### 11.3 遍历顺序错误

```python
# 0-1背包：必须从右往左
for w in range(W, weight - 1, -1):  # 正确
for w in range(weight, W + 1):       # 错误！变成完全背包了
```

### 11.4 返回值搞错

```python
# LIS：返回 max(dp)，不是 dp[n-1]
return max(dp)  # 正确
return dp[n-1]  # 错误！
```


---

## 十二、进阶技巧

### 12.1 状态压缩

当状态是一个集合时，用二进制数表示：

```python
# 表示选了哪些元素
mask = 0b1011  # 选了第0、1、3个元素

# 检查第i位
if mask & (1 << i): ...

# 设置第i位
mask |= (1 << i)

# 清除第i位
mask &= ~(1 << i)
```

### 12.2 单调队列优化

当转移方程形如 `dp[i] = max(dp[j]) + cost` 且 j 在滑动窗口内时：

```python
from collections import deque

def solve(nums, k):
    q = deque()  # 存储下标，保持单调递减
    dp = [0] * n
    for i in range(n):
        while q and q[0] < i - k:
            q.popleft()
        dp[i] = nums[q[0]] + cost if q else cost
        while q and nums[q[-1]] <= nums[i]:
            q.pop()
        q.append(i)
```

### 12.3 斜率优化

当转移方程形如 `dp[i] = min(dp[j] + f(i,j))` 且 f 可以分离变量时，用凸包优化到 O(n)。


---

## 十三、实战代码模板汇总

### 13.1 线性 DP 模板

```python
def linear_dp(nums):
    n = len(nums)
    if n == 0: return 0
    
    dp = [0] * n
    dp[0] = nums[0]  # base case
    
    for i in range(1, n):
        dp[i] = f(dp[i-1], nums[i])
    
    return dp[n-1]  # 或 max(dp)
```

### 13.2 双序列 DP 模板

```python
def two_seq_dp(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    # 边界初始化
    for i in range(m+1): dp[i][0] = init_val
    for j in range(n+1): dp[0][j] = init_val
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### 13.3 背包 DP 模板

```python
# 0-1 背包
def knapsack_01(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i]-1, -1):  # 从右往左
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[W]

# 完全背包
def knapsack_complete(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(weights[i], W+1):  # 从左往右
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[W]
```

### 13.4 区间 DP 模板

```python
def interval_dp(nums):
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    # 长度从小到大
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i,j))
    
    return dp[0][n-1]
```


---

## 十四、面试高频问题速查

### 14.1 股票买卖系列

| 题号 | 限制 | 状态定义 |
|-----|------|---------|
| 121 | 只能买卖1次 | 记录最小价格 |
| 122 | 无限次 | 贪心：所有上涨都吃 |
| 123 | 最多2次 | dp[i][k][0/1] |
| 188 | 最多k次 | 同上 |
| 309 | 有冷冻期 | 增加冷冻状态 |
| 714 | 有手续费 | 卖出时减fee |

### 14.2 子序列系列

| 题目 | 关键区别 |
|-----|---------|
| 子序列 | 可以不连续 |
| 子数组 | 必须连续 |
| 子串 | 必须连续（字符串版） |

### 14.3 路径系列

| 题号 | 变形 |
|-----|------|
| 62 | 基础版 |
| 63 | 有障碍物 |
| 64 | 带权重（最小路径和）|
| 120 | 三角形 |
| 931 | 下降路径最小和 |

---

## 十五、总结思维导图

```
动态规划
├── 核心思想
│   ├── 最优子结构
│   ├── 重叠子问题
│   └── 空间换时间
├── 三要素
│   ├── 状态定义 dp[i] = ?
│   ├── 转移方程 dp[i] = f(dp[...])
│   └── 边界条件 dp[0] = ?
├── 实现方式
│   ├── Top-Down 记忆化递归
│   └── Bottom-Up 迭代填表
├── 优化技巧
│   ├── 滚动数组
│   ├── 降维
│   └── 单调队列/斜率优化
└── 问题分类
    ├── 线性DP
    ├── 序列DP
    ├── 背包DP
    ├── 区间DP
    ├── 树形DP
    └── 状压DP
```

---

## 参考资料

1. 《算法导论》第15章 - 动态规划
2. LeetCode 动态规划专题
3. [OI Wiki - 动态规划](https://oi-wiki.org/dp/)

---

> 💡 **费曼学习法核心**：如果你能把 DP 解释给一个完全不懂编程的人听，你就真正理解了它。
>
> 记住：**DP 不是背模板，而是理解"如何把大问题拆成小问题，并记住小问题的答案"。**

